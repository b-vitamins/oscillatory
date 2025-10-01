"""Unit tests for group-theoretic attention operations."""

import pytest
import torch

from oscillatory.nn.functional.gta import (
    make_2d_coord,
    make_so2_matrices,
    apply_group_action,
    apply_group_action_qkv,
    apply_group_action_qk,
    embed_block_diagonal,
)
from tests.conftest import assert_shape, assert_device, assert_dtype


class TestMake2DCoord:
    """Test make_2d_coord function."""

    def test_basic_shape(self):
        coords = make_2d_coord(4, 4)
        assert_shape(coords, (4, 4, 2))

    def test_different_dimensions(self):
        coords = make_2d_coord(3, 5)
        assert_shape(coords, (3, 5, 2))

    def test_coordinate_values_unnormalized(self):
        coords = make_2d_coord(3, 3, normalize=False)
        # First row should have y=0
        assert torch.allclose(coords[0, :, 0], torch.zeros(3))
        # First column should have x=0
        assert torch.allclose(coords[:, 0, 1], torch.zeros(3))
        # Last row should have y=2
        assert torch.allclose(coords[2, :, 0], torch.full((3,), 2.0))

    def test_coordinate_values_normalized(self):
        coords = make_2d_coord(4, 4, normalize=True)
        # Values should be in [0, 1]
        assert (coords >= 0).all() and (coords < 1).all()
        # First row should have y=0
        assert torch.allclose(coords[0, :, 0], torch.zeros(4))
        # Last row should have y=3/4
        assert torch.allclose(coords[3, :, 0], torch.full((4,), 3.0 / 4))

    @pytest.mark.parametrize("device_str", ["cpu"])
    def test_device(self, device_str):
        device = torch.device(device_str)
        coords = make_2d_coord(4, 4, device=device)
        assert_device(coords, device)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype):
        coords = make_2d_coord(4, 4, dtype=dtype)
        assert_dtype(coords, dtype)

    def test_indexing_order(self):
        coords = make_2d_coord(2, 3, normalize=False)
        # Check that indexing is ij (row-major)
        # coords[i, j] should be (i, j)
        assert coords[0, 0, 0] == 0 and coords[0, 0, 1] == 0
        assert coords[0, 2, 0] == 0 and coords[0, 2, 1] == 2
        assert coords[1, 0, 0] == 1 and coords[1, 0, 1] == 0


class TestMakeSO2Matrices:
    """Test make_so2_matrices function."""

    def test_basic_shape(self):
        coords = torch.randn(4, 4, 2)
        matrices = make_so2_matrices(coords, n_freqs=8)
        # Should be [H, W, n_freqs, dim, 2, 2]
        assert_shape(matrices, (4, 4, 8, 2, 2, 2))

    def test_rotation_matrix_properties(self):
        coords = torch.randn(3, 3, 2)
        matrices = make_so2_matrices(coords, n_freqs=4)
        # Each 2x2 matrix should be a rotation matrix
        # det(R) = 1 for rotation matrices
        dets = torch.linalg.det(matrices)
        assert torch.allclose(dets, torch.ones_like(dets), atol=1e-6)

    def test_orthogonality(self):
        coords = torch.randn(3, 3, 2)
        matrices = make_so2_matrices(coords, n_freqs=4)
        # R @ R^T should be identity
        # Shape is [3, 3, n_freqs=4, dim=2, 2, 2]
        identity = torch.eye(2, device=matrices.device, dtype=matrices.dtype)
        for i in range(3):
            for j in range(3):
                for f in range(4):
                    for d in range(2):
                        R = matrices[i, j, f, d]
                        RRt = R @ R.T
                        assert torch.allclose(RRt, identity, atol=1e-6)

    def test_base_parameter(self):
        coords = torch.ones(2, 2, 2)
        mat1 = make_so2_matrices(coords, n_freqs=4, base=10000.0)
        mat2 = make_so2_matrices(coords, n_freqs=4, base=5000.0)
        # Different bases should produce different matrices
        assert not torch.allclose(mat1, mat2)

    def test_frequency_progression(self):
        coords = torch.ones(1, 1, 2)
        matrices = make_so2_matrices(coords, n_freqs=4, base=10000.0)
        # Lower frequency indices should have smaller angles (slower rotation)
        # Extract rotation angles from the matrices
        angles = torch.atan2(matrices[0, 0, 0, :, 1, 0], matrices[0, 0, 0, :, 0, 0])
        # Angles should generally increase in magnitude
        assert (angles.abs()[:-1] <= angles.abs()[1:]).all()


class TestApplyGroupAction:
    """Test apply_group_action function."""

    def test_basic_shape_preservation(self):
        # rep should have shape [seq_len, D/2, 2, 2]
        # x should have shape [batch, seq_len, D]
        rep = torch.randn(8, 8, 2, 2)  # seq_len=8, D/2=8
        x = torch.randn(2, 8, 16)  # batch=2, seq_len=8, D=16
        result = apply_group_action(rep, x)
        assert_shape(result, x.shape)

    def test_identity_rotation(self):
        # Identity rotation should preserve input
        rep = torch.eye(2).unsqueeze(0).unsqueeze(0).expand(1, 1, 2, 2)
        x = torch.randn(1, 4)
        result = apply_group_action(rep, x)
        assert torch.allclose(result, x, atol=1e-6)

    def test_gradient_flow(self, gradient_checker):
        # rep: [seq, D/2, 2, 2], x: [batch, seq, D]
        rep = torch.randn(4, 4, 2, 2, requires_grad=True)
        x = torch.randn(1, 4, 8, requires_grad=True)

        def func(rep, x):
            return apply_group_action(rep, x)

        output = func(rep, x)
        output.sum().backward()
        assert rep.grad is not None
        assert x.grad is not None

    def test_batch_independence(self):
        # Results for different batch elements should be independent
        rep = torch.randn(1, 1, 2, 2)
        x1 = torch.randn(1, 4)
        x2 = torch.randn(1, 4)
        x_batch = torch.cat([x1, x2], dim=0)

        result1 = apply_group_action(rep, x1)
        result2 = apply_group_action(rep, x2)
        result_batch = apply_group_action(rep, x_batch)

        assert torch.allclose(result_batch[0], result1[0])
        assert torch.allclose(result_batch[1], result2[0])


class TestApplyGroupActionQKV:
    """Test apply_group_action_qkv function."""

    def test_shape_preservation(self):
        # rep: [seq, D/2, 2, 2], q/k/v: [batch, seq, D]
        rep = torch.randn(8, 8, 2, 2)
        q = torch.randn(2, 8, 16)
        k = torch.randn(2, 8, 16)
        v = torch.randn(2, 8, 16)

        q_rot, k_rot, v_rot = apply_group_action_qkv(rep, q, k, v)
        assert_shape(q_rot, q.shape)
        assert_shape(k_rot, k.shape)
        assert_shape(v_rot, v.shape)

    def test_gradient_flow(self):
        # rep: [seq, D/2, 2, 2], q/k/v: [batch, seq, D]
        rep = torch.randn(4, 4, 2, 2, requires_grad=True)
        q = torch.randn(1, 4, 8, requires_grad=True)
        k = torch.randn(1, 4, 8, requires_grad=True)
        v = torch.randn(1, 4, 8, requires_grad=True)

        q_rot, k_rot, v_rot = apply_group_action_qkv(rep, q, k, v)
        loss = q_rot.sum() + k_rot.sum() + v_rot.sum()
        loss.backward()

        assert rep.grad is not None
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


class TestApplyGroupActionQK:
    """Test apply_group_action_qk function."""

    def test_shape_preservation(self):
        # rep: [seq, D/2, 2, 2], q/k: [batch, seq, D]
        rep = torch.randn(8, 8, 2, 2)
        q = torch.randn(2, 8, 16)
        k = torch.randn(2, 8, 16)

        q_rot, k_rot = apply_group_action_qk(rep, q, k)
        assert_shape(q_rot, q.shape)
        assert_shape(k_rot, k.shape)


class TestEmbedBlockDiagonal:
    """Test embed_block_diagonal function."""

    def test_basic_shape(self):
        # Input: [HW, D/2, 2, 2], n_blocks=2
        # Output: [HW, D/(2*n_blocks), 4, 4]
        matrices = torch.randn(16, 8, 2, 2)
        result = embed_block_diagonal(matrices, n_blocks=2)
        assert_shape(result, (16, 4, 4, 4))

    def test_block_diagonal_structure(self):
        # Create simple input matrices
        matrices = torch.eye(2).unsqueeze(0).unsqueeze(0).expand(1, 2, 2, 2)
        result = embed_block_diagonal(matrices, n_blocks=2)

        # Result should have 2x2 identity blocks on diagonal
        expected = torch.block_diag(torch.eye(2), torch.eye(2))
        assert torch.allclose(result[0, 0], expected, atol=1e-6)

    def test_divisibility_check(self):
        # Should raise error if D/2 not divisible by n_blocks
        matrices = torch.randn(1, 5, 2, 2)
        with pytest.raises(AssertionError, match="must be divisible"):
            embed_block_diagonal(matrices, n_blocks=2)

    def test_matrix_size_check(self):
        # Should raise error if input matrices are not 2x2
        matrices = torch.randn(1, 4, 3, 3)
        with pytest.raises(AssertionError, match="must be 2x2"):
            embed_block_diagonal(matrices, n_blocks=2)

    def test_different_n_blocks(self):
        matrices = torch.randn(1, 8, 2, 2)
        result1 = embed_block_diagonal(matrices, n_blocks=2)
        result2 = embed_block_diagonal(matrices, n_blocks=4)

        assert_shape(result1, (1, 4, 4, 4))
        assert_shape(result2, (1, 2, 8, 8))

    def test_zero_off_diagonals(self):
        # Off-diagonal blocks should be zero
        matrices = torch.randn(1, 4, 2, 2)
        result = embed_block_diagonal(matrices, n_blocks=2)

        # Check that off-diagonal 2x2 blocks are zero
        assert torch.allclose(result[0, 0, 0:2, 2:4], torch.zeros(2, 2), atol=1e-6)
        assert torch.allclose(result[0, 0, 2:4, 0:2], torch.zeros(2, 2), atol=1e-6)


class TestGTAIntegration:
    """Integration tests for GTA operations."""

    def test_full_pipeline(self):
        # Test complete GTA encoding pipeline
        H, W = 8, 8
        n_freqs = 4
        batch_size = 2
        embed_dim = 16

        # Create coordinates
        coords = make_2d_coord(H, W)
        assert_shape(coords, (H, W, 2))

        # Create SO(2) matrices
        matrices = make_so2_matrices(coords, n_freqs)
        # Shape is [H, W, n_freqs, dim, 2, 2]
        assert_shape(matrices, (H, W, n_freqs, 2, 2, 2))

        # Flatten for sequence: [H, W, n_freqs, dim, 2, 2] -> [H*W, n_freqs*dim, 2, 2]
        matrices_flat = matrices.flatten(0, 1).flatten(1, 2)[:, : embed_dim // 2, :, :]
        assert_shape(matrices_flat, (H * W, embed_dim // 2, 2, 2))

        # Apply to features
        q = torch.randn(batch_size, H * W, embed_dim)
        k = torch.randn(batch_size, H * W, embed_dim)
        v = torch.randn(batch_size, H * W, embed_dim)

        q_rot, k_rot, v_rot = apply_group_action_qkv(matrices_flat, q, k, v)
        assert_shape(q_rot, (batch_size, H * W, embed_dim))
        assert_shape(k_rot, (batch_size, H * W, embed_dim))
        assert_shape(v_rot, (batch_size, H * W, embed_dim))

    def test_deterministic_with_seed(self):
        # Same seed should produce same results
        torch.manual_seed(42)
        coords1 = make_2d_coord(4, 4)
        mat1 = make_so2_matrices(coords1, n_freqs=4)

        torch.manual_seed(42)
        coords2 = make_2d_coord(4, 4)
        mat2 = make_so2_matrices(coords2, n_freqs=4)

        assert torch.allclose(mat1, mat2)

    def test_dtype_consistency(self):
        # All operations should preserve dtype
        coords = make_2d_coord(4, 4, dtype=torch.float64)
        matrices = make_so2_matrices(coords, n_freqs=4)
        assert_dtype(matrices, torch.float64)

        x = torch.randn(1, 8, dtype=torch.float64)
        rep = matrices[0, 0, 0, 0].unsqueeze(0).unsqueeze(0)
        result = apply_group_action(rep, x)
        assert_dtype(result, torch.float64)
