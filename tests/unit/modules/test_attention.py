"""Unit tests for GTAttention module."""

import pytest
import torch
import torch.nn as nn

from oscillatory.nn.modules.attention import GTAttention
from tests.conftest import assert_shape, assert_finite


class TestGTAttentionBasic:
    """Basic functionality tests for GTAttention."""

    def test_initialization_conv(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        assert attn.embed_dim == 64
        assert attn.n_heads == 8
        assert attn.head_dim == 8

    def test_initialization_fc(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="fc", hw=(8, 8))
        assert attn.embed_dim == 64
        assert attn.n_heads == 8

    def test_embed_dim_divisibility(self):
        with pytest.raises(AssertionError, match="must be divisible"):
            GTAttention(embed_dim=65, n_heads=8)

    def test_invalid_weight_type(self):
        with pytest.raises(ValueError, match="must be 'conv' or 'fc'"):
            GTAttention(embed_dim=64, n_heads=8, weight_type="invalid")  # type: ignore[arg-type]

    def test_gta_rope_mutual_exclusion(self):
        with pytest.raises(AssertionError, match="Cannot use both"):
            GTAttention(embed_dim=64, n_heads=8, use_gta=True, use_rope=True, hw=(8, 8))

    def test_gta_requires_hw(self):
        with pytest.raises(ValueError, match="hw must be provided"):
            GTAttention(embed_dim=64, n_heads=8, use_gta=True)

    def test_rope_requires_hw(self):
        with pytest.raises(ValueError, match="hw must be provided"):
            GTAttention(embed_dim=64, n_heads=8, use_rope=True)


class TestGTAttentionForwardConv:
    """Test forward pass with convolutional weights."""

    def test_basic_forward_4d(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))

    def test_different_spatial_sizes(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        for h, w in [(4, 4), (8, 8), (16, 16), (8, 16)]:
            x = torch.randn(2, 64, h, w)
            out = attn(x)
            assert_shape(out, (2, 64, h, w))

    def test_stride(self):
        # Stride is applied in convolutions, reducing spatial dims
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv", stride=2)
        x = torch.randn(2, 64, 16, 16)
        out = attn(x)
        # With stride=2, 16x16 -> 4x4 (reduced by factor of 4)
        assert_shape(out, (2, 64, 4, 4))

    def test_kernel_size_padding(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", kernel_size=3, padding=1
        )
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))

    def test_gradient_flow(self, gradient_checker):
        attn = GTAttention(embed_dim=32, n_heads=4, weight_type="conv")
        x = torch.randn(1, 32, 4, 4, requires_grad=True)
        gradient_checker.check_gradients(attn, x)


class TestGTAttentionForwardFC:
    """Test forward pass with fully-connected weights."""

    def test_basic_forward_3d(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="fc", hw=(8, 8))
        x = torch.randn(2, 64, 64)  # [B, L, C]
        out = attn(x)
        assert_shape(out, (2, 64, 64))

    def test_forward_4d_input(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="fc", hw=(8, 8))
        x = torch.randn(2, 64, 8, 8)  # [B, C, H, W]
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))

    def test_different_sequence_lengths(self):
        for L in [16, 64, 256]:
            H = W = int(L**0.5)
            attn = GTAttention(embed_dim=64, n_heads=8, weight_type="fc", hw=(H, W))
            x = torch.randn(2, L, 64)
            out = attn(x)
            assert_shape(out, (2, L, 64))

    def test_gradient_flow(self, gradient_checker):
        attn = GTAttention(embed_dim=32, n_heads=4, weight_type="fc", hw=(4, 4))
        x = torch.randn(1, 16, 32, requires_grad=True)
        gradient_checker.check_gradients(attn, x)


class TestGTAttentionGTA:
    """Test Group-Theoretic Attention mode."""

    def test_gta_initialization(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        assert hasattr(attn, "mat_q")
        assert hasattr(attn, "mat_k")
        assert hasattr(attn, "mat_v")
        assert hasattr(attn, "mat_o")
        # GTA matrices should be parameters
        assert isinstance(attn.mat_q, nn.Parameter)

    def test_gta_forward(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))
        assert_finite(out)

    def test_gta_different_spatial_sizes(self):
        # GTA with same size as input (no rescaling needed)
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(16, 16)
        )
        x = torch.randn(2, 64, 16, 16)
        out = attn(x)
        assert_shape(out, (2, 64, 16, 16))

    def test_gta_gradient_flow(self):
        attn = GTAttention(
            embed_dim=32, n_heads=4, weight_type="conv", use_gta=True, hw=(4, 4)
        )
        x = torch.randn(1, 32, 4, 4, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert attn.mat_q.grad is not None
        assert attn.mat_k.grad is not None
        assert attn.mat_v.grad is not None
        assert attn.mat_o.grad is not None


class TestGTAttentionRoPE:
    """Test Rotary Position Embeddings mode."""

    def test_rope_initialization(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_rope=True, hw=(8, 8)
        )
        assert hasattr(attn, "mat_q")
        assert hasattr(attn, "mat_k")
        # RoPE matrices should be buffers, not parameters
        assert not isinstance(attn.mat_q, nn.Parameter)

    def test_rope_forward(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_rope=True, hw=(8, 8)
        )
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))
        assert_finite(out)

    def test_rope_no_mat_v(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_rope=True, hw=(8, 8)
        )
        # RoPE should not have mat_v (only applies to Q and K)
        assert not hasattr(attn, "mat_v")

    def test_rope_gradient_flow(self):
        attn = GTAttention(
            embed_dim=32, n_heads=4, weight_type="conv", use_rope=True, hw=(4, 4)
        )
        x = torch.randn(1, 32, 4, 4, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        # RoPE matrices are buffers, not parameters
        assert attn.mat_q.grad is None or not attn.mat_q.grad.any()


class TestGTAttentionDropout:
    """Test dropout functionality."""

    def test_dropout_training_mode(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv", dropout=0.5)
        attn.train()

        x = torch.randn(2, 64, 8, 8)
        # With dropout in training, outputs should vary
        out1 = attn(x)
        out2 = attn(x)

        # Different due to dropout randomness
        assert not torch.allclose(out1, out2)

    def test_dropout_eval_mode(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv", dropout=0.5)
        attn.eval()

        torch.manual_seed(42)
        x = torch.randn(2, 64, 8, 8)
        out1 = attn(x)

        torch.manual_seed(42)
        x = torch.randn(2, 64, 8, 8)
        out2 = attn(x)

        # Should be identical in eval mode
        assert torch.allclose(out1, out2)


class TestGTAttentionRescaling:
    """Test matrix rescaling functionality."""

    def test_rescaling_upscale(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(4, 4)
        )
        # Forward with larger size
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))

    def test_rescaling_downscale(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        # Forward with smaller size
        x = torch.randn(2, 64, 4, 4)
        out = attn(x)
        assert_shape(out, (2, 64, 4, 4))

    def test_no_rescaling_when_same_size(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        # Same size should not rescale
        x = torch.randn(2, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (2, 64, 8, 8))


class TestGTAttentionBatchSizes:
    """Test with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        x = torch.randn(batch_size, 64, 8, 8)
        out = attn(x)
        assert_shape(out, (batch_size, 64, 8, 8))


class TestGTAttentionHeadConfigurations:
    """Test with different head configurations."""

    @pytest.mark.parametrize("n_heads", [1, 2, 4, 8, 16])
    def test_different_head_counts(self, n_heads):
        embed_dim = 64
        attn = GTAttention(embed_dim=embed_dim, n_heads=n_heads, weight_type="conv")
        assert attn.head_dim == embed_dim // n_heads

        x = torch.randn(2, embed_dim, 8, 8)
        out = attn(x)
        assert_shape(out, (2, embed_dim, 8, 8))


class TestGTAttentionNumericalStability:
    """Test numerical stability."""

    def test_large_values(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        x = torch.randn(2, 64, 8, 8) * 1000  # Large values
        out = attn(x)
        assert_finite(out)

    def test_small_values(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        x = torch.randn(2, 64, 8, 8) * 1e-6  # Small values
        out = attn(x)
        assert_finite(out)

    def test_zero_input(self):
        attn = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        x = torch.zeros(2, 64, 8, 8)
        out = attn(x)
        assert_finite(out)


class TestGTAttentionStateDictCompatibility:
    """Test state dict save/load."""

    def test_save_load_state_dict(self):
        attn1 = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        state_dict = attn1.state_dict()

        attn2 = GTAttention(embed_dim=64, n_heads=8, weight_type="conv")
        attn2.load_state_dict(state_dict)

        x = torch.randn(2, 64, 8, 8)
        out1 = attn1(x)
        out2 = attn2(x)

        assert torch.allclose(out1, out2)

    def test_gta_state_dict(self):
        attn = GTAttention(
            embed_dim=64, n_heads=8, weight_type="conv", use_gta=True, hw=(8, 8)
        )
        state_dict = attn.state_dict()

        # GTA parameters should be in state dict
        assert "mat_q" in state_dict
        assert "mat_k" in state_dict
        assert "mat_v" in state_dict
        assert "mat_o" in state_dict
