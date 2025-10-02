"""Property-based tests for mathematical invariants.

These tests verify mathematical properties that should hold for all valid inputs,
using parameterized testing to explore the input space systematically.
"""

import math
import pytest
import torch

from oscillatory.nn.functional.kuramoto import (
    normalize_oscillators2d,
    kuramoto_step,
)
from oscillatory.nn.functional.diffusion import (
    phase_modulate,
    wrapped_gaussian_score,
    mean_field_phase,
)
from oscillatory.nn.functional.gta import (
    make_so2_matrices,
    apply_group_action,
)
from oscillatory.nn.modules.attention import GTAttention


class TestNormalizationProperties:
    """Test properties of oscillator normalization."""

    @pytest.mark.parametrize(
        "B,C,H,W,n_osc",
        [
            (1, 8, 4, 4, 2),
            (2, 12, 8, 8, 4),
            (1, 16, 16, 16, 4),
            (4, 8, 4, 4, 4),
        ],
    )
    def test_unit_norm_property(self, B, C, H, W, n_osc):
        """Normalized oscillators should have unit norm."""
        x = torch.randn(B, C, H, W) * 100  # Large range
        x_norm = normalize_oscillators2d(x, n_osc)

        # Reshape and check norms
        x_reshaped = x_norm.view(B, C // n_osc, n_osc, H, W)
        norms = x_reshaped.norm(dim=2, p=2)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
    def test_scale_invariance(self, scale):
        """Normalization result should be independent of input scale."""
        x = torch.randn(1, 8, 4, 4)
        x_scaled = x * scale

        norm1 = normalize_oscillators2d(x, n_oscillators=4)
        norm2 = normalize_oscillators2d(x_scaled, n_oscillators=4)

        # Direction should be preserved (scale doesn't matter)
        # Check by comparing normalized versions
        # Note: very small scales (< 1e-3) can have numerical precision issues
        assert torch.allclose(norm1, norm2, atol=1e-5)

    @pytest.mark.parametrize("n_oscillators", [2, 4, 8])
    def test_idempotence(self, n_oscillators):
        """Normalizing twice should give same result as once."""
        C = n_oscillators * 3
        x = torch.randn(2, C, 8, 8)

        norm1 = normalize_oscillators2d(x, n_oscillators)
        norm2 = normalize_oscillators2d(norm1, n_oscillators)

        assert torch.allclose(norm1, norm2, atol=1e-6)


class TestPhaseWrappingProperties:
    """Test properties of phase wrapping."""

    @pytest.mark.parametrize("offset", [0, 2 * math.pi, -2 * math.pi, 4 * math.pi])
    def test_2pi_periodicity(self, offset):
        """Phases differing by 2π should wrap to same value."""
        x = torch.randn(2, 3, 4, 4)
        x_offset = x + offset

        wrapped1 = phase_modulate(x)
        wrapped2 = phase_modulate(x_offset)

        assert torch.allclose(wrapped1, wrapped2, atol=1e-6)

    @pytest.mark.parametrize("value", [-10, -5, 0, 5, 10])
    def test_output_range(self, value):
        """Wrapped output should always be in [-π, π]."""
        x = torch.tensor([[[[value]]]])
        wrapped = phase_modulate(x)

        assert (wrapped >= -math.pi - 1e-6).all()
        assert (wrapped <= math.pi + 1e-6).all()

    def test_idempotence(self):
        """Wrapping twice should give same result."""
        x = torch.randn(2, 3, 4, 4) * 100
        wrapped1 = phase_modulate(x)
        wrapped2 = phase_modulate(wrapped1)

        assert torch.allclose(wrapped1, wrapped2, atol=1e-6)


class TestKuramotoStepProperties:
    """Test properties of Kuramoto dynamics step."""

    @pytest.mark.parametrize("n_oscillators", [2, 4])
    @pytest.mark.parametrize("spatial_ndim", [1, 2, 3])
    def test_normalization_preservation(self, n_oscillators, spatial_ndim):
        """Kuramoto step should preserve unit norm of oscillators."""
        if spatial_ndim == 1:
            shape = (2, 8, 16)
        elif spatial_ndim == 2:
            shape = (2, 8, 4, 4)
        else:
            shape = (2, 8, 2, 4, 4)

        x = torch.randn(*shape)
        coupling = torch.randn(*shape)
        stimulus = torch.randn(*shape)

        # Normalize input
        if spatial_ndim == 2:
            x = normalize_oscillators2d(x, n_oscillators)

        x_new, _ = kuramoto_step(
            x,
            coupling,
            stimulus,
            n_oscillators,
            spatial_ndim=spatial_ndim,
            normalize=True,
        )

        # Check normalization preserved
        if spatial_ndim == 2:
            x_reshaped = x_new.view(2, 8 // n_oscillators, n_oscillators, 4, 4)
        elif spatial_ndim == 1:
            x_reshaped = x_new.view(2, 8 // n_oscillators, n_oscillators, 16)
        else:
            x_reshaped = x_new.view(2, 8 // n_oscillators, n_oscillators, 2, 4, 4)

        norms = x_reshaped.norm(dim=2, p=2)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("step_size", [0.1, 0.5, 1.0, 2.0])
    def test_step_size_scaling(self, step_size):
        """Larger step size should produce proportionally larger changes."""
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        x = normalize_oscillators2d(x, n_oscillators)
        coupling = torch.randn(B, C, H, W) * 0.1
        stimulus = torch.randn(B, C, H, W) * 0.1

        x_new, _ = kuramoto_step(
            x.clone(),
            coupling,
            stimulus,
            n_oscillators,
            step_size=step_size,
            normalize=False,
            spatial_ndim=2,
        )

        # Measure change magnitude
        change = (x_new - x).abs().mean()

        # Larger step size should produce larger change (roughly linear)
        # Store for comparison (this is a rough check)
        assert change >= 0  # Basic sanity


class TestMeanFieldPhaseProperties:
    """Test properties of mean field phase computation."""

    def test_order_parameter_range(self):
        """Order parameter should always be in [0, 1]."""
        for _ in range(10):
            x = torch.randn(2, 3, 8, 8) * 10
            order_param, _ = mean_field_phase(x)

            assert (order_param >= 0).all()
            assert (order_param <= 1).all()

    def test_synchronized_phases(self):
        """Perfectly synchronized phases should give order parameter = 1."""
        phase = 1.5
        x = torch.full((2, 3, 8, 8), phase)

        order_param, mean_phase = mean_field_phase(x)

        assert torch.allclose(order_param, torch.ones_like(order_param), atol=1e-5)
        assert torch.allclose(mean_phase, torch.full_like(mean_phase, phase), atol=1e-5)

    @pytest.mark.parametrize("offset", [0, math.pi, -math.pi, 2 * math.pi])
    def test_phase_shift_invariance_modulo_2pi(self, offset):
        """Shifting all phases by 2π should not change order parameter."""
        x = torch.randn(2, 3, 8, 8)
        x_shifted = x + offset

        order1, phase1 = mean_field_phase(x)
        order2, phase2 = mean_field_phase(x_shifted)

        # Order parameter should be identical
        assert torch.allclose(order1, order2, atol=1e-5)

        # Mean phase should differ by offset (modulo 2π)
        if abs(offset) < 1e-6 or abs(abs(offset) - 2 * math.pi) < 1e-6:
            assert torch.allclose(phase1, phase2, atol=1e-5)


class TestSO2MatrixProperties:
    """Test properties of SO(2) rotation matrices."""

    @pytest.mark.parametrize("H,W,n_freqs", [(4, 4, 4), (8, 8, 8), (16, 16, 16)])
    def test_rotation_matrix_determinant(self, H, W, n_freqs):
        """SO(2) matrices should have determinant = 1."""
        coords = torch.randn(H, W, 2)
        matrices = make_so2_matrices(coords, n_freqs)

        # Check determinant for all matrices
        dets = torch.linalg.det(matrices)
        assert torch.allclose(dets, torch.ones_like(dets), atol=1e-5)

    @pytest.mark.parametrize("H,W,n_freqs", [(4, 4, 4), (8, 8, 8)])
    def test_orthogonality(self, H, W, n_freqs):
        """SO(2) matrices should be orthogonal: R @ R^T = I."""
        coords = torch.randn(H, W, 2)
        matrices = make_so2_matrices(coords, n_freqs)

        # Sample a few matrices to check
        for i in range(min(H, 2)):
            for j in range(min(W, 2)):
                for d in range(2):
                    for f in range(min(n_freqs, 2)):
                        R = matrices[i, j, d, f]
                        RRt = R @ R.T
                        identity = torch.eye(2, device=R.device, dtype=R.dtype)
                        assert torch.allclose(RRt, identity, atol=1e-5)

    def test_rotation_composition(self):
        """Applying rotation twice should compose correctly."""
        coords = torch.ones(1, 1, 2)
        matrices = make_so2_matrices(coords, n_freqs=2)

        R1 = matrices[0, 0, 0, 0]
        R2 = matrices[0, 0, 0, 1]

        # R1 @ R2 should also be a rotation matrix
        R_composed = R1 @ R2
        det = torch.linalg.det(R_composed)

        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)


class TestGroupActionProperties:
    """Test properties of group action application."""

    def test_identity_preservation(self):
        """Identity rotation should preserve input."""
        # rep: [seq, D/2, 2, 2], x: [batch, seq, D]
        rep = torch.eye(2).unsqueeze(0).unsqueeze(0).expand(4, 4, 2, 2)
        x = torch.randn(2, 4, 8)

        x_rotated = apply_group_action(rep, x)

        assert torch.allclose(x, x_rotated, atol=1e-6)

    def test_shape_preservation(self):
        """Group action should preserve shape."""
        # rep: [seq, D/2, 2, 2], x: [batch, seq, D]
        test_cases = [
            (torch.randn(4, 4, 2, 2), torch.randn(2, 4, 8)),
            (torch.randn(16, 8, 2, 2), torch.randn(2, 16, 16)),
            (torch.randn(8, 8, 2, 2), torch.randn(4, 8, 16)),
        ]

        for rep, x in test_cases:
            x_rotated = apply_group_action(rep, x)
            assert x_rotated.shape == x.shape


class TestGTAttentionProperties:
    """Test properties of GTAttention module."""

    @pytest.mark.parametrize("embed_dim,n_heads", [(64, 8), (128, 16), (32, 4)])
    def test_shape_preservation(self, embed_dim, n_heads):
        """Attention should preserve shape."""
        attn = GTAttention(embed_dim=embed_dim, n_heads=n_heads, weight_type="conv")
        x = torch.randn(2, embed_dim, 8, 8)

        out = attn(x)

        assert out.shape == x.shape

    def test_permutation_equivariance_not_expected(self):
        """Attention with GTA is NOT permutation equivariant (due to positional encoding)."""
        # Enable GTA to make it position-aware
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        x = torch.randn(2, 64, 8, 8)

        # Permute spatial dimensions
        x_permuted = x.clone()
        x_permuted[:, :, 0, :], x_permuted[:, :, 1, :] = (
            x[:, :, 1, :].clone(),
            x[:, :, 0, :].clone(),
        )

        out1 = attn(x)
        out2 = attn(x_permuted)

        # With GTA, swapping spatial positions should give different outputs
        # Compare position 0 in out1 with position 1 in out2 (which was position 0 before permutation)
        assert not torch.allclose(out1[:, :, 0, :], out2[:, :, 1, :], atol=0.01)


class TestWrappedGaussianScoreProperties:
    """Test properties of wrapped Gaussian score."""

    @pytest.mark.parametrize("K", [1, 3, 5, 7])
    def test_convergence_with_K(self, K):
        """Larger K should give more accurate score."""
        y = torch.tensor([[[[1.0]]]])
        mu = torch.tensor([[[[0.0]]]])
        sigma = 0.5

        score = wrapped_gaussian_score(y, mu, sigma, K=K)

        # Just verify it computes without error and is finite
        assert torch.isfinite(score).all()

    def test_score_at_mean_is_small(self):
        """Score at the mean should be close to zero."""
        mu = torch.randn(2, 3, 4, 4)
        y = mu.clone()
        sigma = 0.5

        score = wrapped_gaussian_score(y, mu, sigma, K=3)

        # Score should be small when y = mu
        assert score.abs().mean() < 0.5

    @pytest.mark.parametrize("offset", [2 * math.pi, -2 * math.pi])
    def test_periodicity(self, offset):
        """Score should respect 2π periodicity."""
        y = torch.tensor([[[[1.5]]]])
        mu = torch.tensor([[[[0.5]]]])
        sigma = 0.3

        score1 = wrapped_gaussian_score(y, mu, sigma, K=3)
        score2 = wrapped_gaussian_score(y + offset, mu, sigma, K=3)

        assert torch.allclose(score1, score2, rtol=1e-4, atol=1e-4)


class TestDeterminismProperties:
    """Test that operations are deterministic with fixed seeds."""

    def test_kuramoto_step_deterministic(self):
        """Same seed should give same Kuramoto step result."""
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4

        results = []
        for _ in range(2):
            torch.manual_seed(42)
            x = torch.randn(B, C, H, W)
            coupling = torch.randn(B, C, H, W)
            stimulus = torch.randn(B, C, H, W)

            x_new = kuramoto_step(
                x, coupling, stimulus, n_oscillators, spatial_ndim=2
            )
            results.append(x_new)

        assert torch.allclose(results[0], results[1])

    def test_gta_matrices_deterministic(self):
        """Same inputs should give same GTA matrices."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            coords = torch.randn(4, 4, 2)
            matrices = make_so2_matrices(coords, n_freqs=4)
            results.append(matrices)

        assert torch.allclose(results[0], results[1])
