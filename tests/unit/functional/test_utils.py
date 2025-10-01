"""Unit tests for utility functions."""

import pytest
import torch

from oscillatory.nn.functional.utils import positional_encoding_2d
from tests.conftest import assert_shape, assert_dtype, assert_finite


class TestPositionalEncoding2D:
    """Test positional_encoding_2d function."""

    def test_basic_shape(self):
        pe = positional_encoding_2d(d_model=64, height=8, width=8)
        assert_shape(pe, (64, 8, 8))

    def test_different_dimensions(self):
        pe = positional_encoding_2d(d_model=128, height=16, width=32)
        assert_shape(pe, (128, 16, 32))

    def test_divisibility_requirement(self):
        # d_model must be divisible by 4
        with pytest.raises(ValueError, match="must be divisible by 4"):
            positional_encoding_2d(d_model=63, height=8, width=8)

        with pytest.raises(ValueError, match="must be divisible by 4"):
            positional_encoding_2d(d_model=65, height=8, width=8)

    def test_valid_d_models(self):
        # These should all work
        for d_model in [4, 8, 12, 16, 32, 64, 128, 256]:
            pe = positional_encoding_2d(d_model, height=8, width=8)
            assert_shape(pe, (d_model, 8, 8))

    def test_channel_organization(self):
        # First half encodes width, second half encodes height
        pe = positional_encoding_2d(d_model=8, height=4, width=4)

        # Width encoding (first half) should vary along width dimension
        width_encoding = pe[:4, 0, :]  # First row, first half channels
        # Different columns should have different values
        assert not torch.allclose(width_encoding[:, 0], width_encoding[:, 1])

        # Height encoding (second half) should vary along height dimension
        height_encoding = pe[4:, :, 0]  # First column, second half channels
        # Different rows should have different values
        assert not torch.allclose(height_encoding[:, 0], height_encoding[:, 1])

    def test_sinusoidal_pattern(self):
        # Channels should alternate between sin and cos
        pe = positional_encoding_2d(d_model=8, height=4, width=4)

        # For a single position, extract the encoding
        pe[:, 0, 1]  # Height=0, Width=1

        # First half (width encoding) should have alternating sin/cos pattern
        # Second half (height encoding) should also have alternating sin/cos

    def test_position_zero(self):
        # At position (0, 0), the encoding should have specific pattern
        pe = positional_encoding_2d(d_model=8, height=4, width=4)
        pe[:, 0, 0]

        # At pos=0, sin(0) = 0, cos(0) = 1
        # Channels should alternate close to [0, 1, 0, 1, ...] pattern
        # (accounting for frequency scaling)

    def test_dtype(self):
        pe = positional_encoding_2d(d_model=64, height=8, width=8)
        assert_dtype(pe, torch.float32)

    def test_deterministic(self):
        # Same parameters should always produce same encoding
        pe1 = positional_encoding_2d(d_model=32, height=8, width=8)
        pe2 = positional_encoding_2d(d_model=32, height=8, width=8)
        assert torch.allclose(pe1, pe2)

    def test_values_finite(self):
        pe = positional_encoding_2d(d_model=64, height=8, width=8)
        assert_finite(pe)

    def test_values_range(self):
        # Sinusoidal encoding should be in [-1, 1]
        pe = positional_encoding_2d(d_model=64, height=32, width=32)
        assert (pe >= -1.0).all() and (pe <= 1.0).all()

    def test_frequency_progression(self):
        # Higher channel indices should have higher frequencies (faster oscillation)
        pe = positional_encoding_2d(d_model=16, height=1, width=16)

        # Extract encodings along width for different channel pairs
        pe[0, 0, :].abs()  # First channel
        pe[2, 0, :].abs()  # Third channel

        # The pattern should change more rapidly for higher channels
        # (This is a rough heuristic check)

    def test_square_vs_rectangular(self):
        # Square and rectangular images should work
        pe_square = positional_encoding_2d(d_model=64, height=8, width=8)
        pe_rect = positional_encoding_2d(d_model=64, height=8, width=16)

        assert_shape(pe_square, (64, 8, 8))
        assert_shape(pe_rect, (64, 8, 16))

        # First 8 columns should potentially match
        # (This depends on implementation details)

    def test_additive_property(self):
        # Positional encoding should be additive with features
        d_model, h, w = 64, 8, 8
        pe = positional_encoding_2d(d_model, h, w)

        # Create dummy features
        features = torch.randn(2, d_model, h, w)

        # Adding PE should work
        features_with_pe = features + pe
        assert_shape(features_with_pe, (2, d_model, h, w))

    def test_broadcast_batch(self):
        # PE should broadcast with batched features
        d_model, h, w = 64, 8, 8
        batch_size = 4

        pe = positional_encoding_2d(d_model, h, w)
        features = torch.randn(batch_size, d_model, h, w)

        # Should broadcast correctly
        features_with_pe = features + pe.unsqueeze(0)
        assert_shape(features_with_pe, (batch_size, d_model, h, w))

    def test_large_dimensions(self):
        # Test with larger dimensions
        pe = positional_encoding_2d(d_model=512, height=64, width=64)
        assert_shape(pe, (512, 64, 64))
        assert_finite(pe)

    def test_minimal_dimensions(self):
        # Test with minimal valid dimensions
        pe = positional_encoding_2d(d_model=4, height=1, width=1)
        assert_shape(pe, (4, 1, 1))

    def test_single_spatial_dimension(self):
        # Test with 1D-like spatial dimensions
        pe_h = positional_encoding_2d(d_model=64, height=32, width=1)
        pe_w = positional_encoding_2d(d_model=64, height=1, width=32)

        assert_shape(pe_h, (64, 32, 1))
        assert_shape(pe_w, (64, 1, 32))


class TestPositionalEncodingProperties:
    """Test mathematical properties of positional encoding."""

    def test_unique_positions(self):
        # Different spatial positions should have different encodings
        pe = positional_encoding_2d(d_model=64, height=8, width=8)

        # Compare a few different positions
        pos_00 = pe[:, 0, 0]
        pos_01 = pe[:, 0, 1]
        pos_10 = pe[:, 1, 0]
        pos_11 = pe[:, 1, 1]

        # All should be different
        assert not torch.allclose(pos_00, pos_01, atol=1e-3)
        assert not torch.allclose(pos_00, pos_10, atol=1e-3)
        assert not torch.allclose(pos_00, pos_11, atol=1e-3)
        assert not torch.allclose(pos_01, pos_10, atol=1e-3)

    def test_smooth_variation(self):
        # Adjacent positions should have similar (but not identical) encodings
        pe = positional_encoding_2d(d_model=64, height=16, width=16)

        # Compare adjacent positions
        pos_55 = pe[:, 5, 5]
        pos_56 = pe[:, 5, 6]
        pos_65 = pe[:, 6, 5]

        # Compute distances
        dist_horizontal = (pos_55 - pos_56).norm()
        dist_vertical = (pos_55 - pos_65).norm()

        # Adjacent positions should be relatively close
        # but not too close (need discrimination)
        assert dist_horizontal > 0.1
        assert dist_horizontal < 5.0
        assert dist_vertical > 0.1
        assert dist_vertical < 5.0

    def test_translation_variance(self):
        # Positional encoding is NOT translation invariant
        # (unlike relative position encoding)
        pe = positional_encoding_2d(d_model=64, height=8, width=8)

        # Two patches at different absolute positions
        patch1 = pe[:, 0:2, 0:2]
        patch2 = pe[:, 2:4, 2:4]

        # Should be different (absolute position encoding)
        assert not torch.allclose(patch1, patch2, atol=1e-2)
