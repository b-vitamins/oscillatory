"""Unit tests for Kuramoto oscillator dynamics."""

import pytest
import torch

from oscillatory.nn.functional.kuramoto import (
    normalize_oscillators,
    normalize_oscillators1d,
    normalize_oscillators2d,
    normalize_oscillators3d,
    kuramoto_step,
)
from tests.conftest import assert_shape, assert_unit_norm, assert_finite


class TestNormalizeOscillators:
    """Test normalize_oscillators function (dimension-agnostic)."""

    def test_basic_normalization(self):
        x = torch.randn(2, 12)  # [B, C]
        n_oscillators = 4
        x_norm = normalize_oscillators(x, n_oscillators)

        # Check shape preservation
        assert_shape(x_norm, x.shape)

        # Check unit norm constraint
        x_reshaped = x_norm.view(2, 3, 4)  # [B, n_groups, n_oscillators]
        assert_unit_norm(x_reshaped, dim=2, eps=1e-6)

    def test_divisibility_check(self):
        x = torch.randn(2, 10)
        with pytest.raises(ValueError, match="must be divisible"):
            normalize_oscillators(x, n_oscillators=3)

    def test_already_normalized(self):
        # Input already normalized should stay normalized
        x = torch.randn(2, 8)
        x_reshaped = x.view(2, 2, 4)
        x_reshaped = x_reshaped / x_reshaped.norm(dim=2, keepdim=True)
        x = x_reshaped.view(2, 8)

        x_norm = normalize_oscillators(x, n_oscillators=4)
        assert torch.allclose(x, x_norm, atol=1e-6)

    def test_gradient_flow(self):
        x = torch.randn(2, 8, requires_grad=True)
        x_norm = normalize_oscillators(x, n_oscillators=4)
        loss = x_norm.sum()
        loss.backward()
        assert x.grad is not None

    def test_zero_handling(self):
        # Should handle near-zero vectors gracefully
        x = torch.zeros(2, 8)
        x_norm = normalize_oscillators(x, n_oscillators=4, eps=1e-12)
        assert_finite(x_norm)


class TestNormalizeOscillators1D:
    """Test normalize_oscillators1d function."""

    def test_basic_normalization(self):
        x = torch.randn(2, 12, 16)  # [B, C, L]
        x_norm = normalize_oscillators1d(x, n_oscillators=4)

        assert_shape(x_norm, x.shape)
        x_reshaped = x_norm.view(2, 3, 4, 16)
        assert_unit_norm(x_reshaped, dim=2, eps=1e-6)

    def test_different_sequence_lengths(self):
        for L in [8, 16, 32]:
            x = torch.randn(2, 12, L)
            x_norm = normalize_oscillators1d(x, n_oscillators=4)
            assert_shape(x_norm, (2, 12, L))

    def test_divisibility_check(self):
        x = torch.randn(2, 10, 16)
        with pytest.raises(ValueError, match="must be divisible"):
            normalize_oscillators1d(x, n_oscillators=3)


class TestNormalizeOscillators2D:
    """Test normalize_oscillators2d function."""

    def test_basic_normalization(self):
        x = torch.randn(2, 12, 8, 8)  # [B, C, H, W]
        x_norm = normalize_oscillators2d(x, n_oscillators=4)

        assert_shape(x_norm, x.shape)
        x_reshaped = x_norm.view(2, 3, 4, 8, 8)
        assert_unit_norm(x_reshaped, dim=2, eps=1e-6)

    def test_different_spatial_sizes(self):
        for H, W in [(4, 4), (8, 8), (16, 16), (8, 16)]:
            x = torch.randn(2, 12, H, W)
            x_norm = normalize_oscillators2d(x, n_oscillators=4)
            assert_shape(x_norm, (2, 12, H, W))

    def test_single_channel_groups(self):
        x = torch.randn(2, 4, 8, 8)
        x_norm = normalize_oscillators2d(x, n_oscillators=4)
        assert_shape(x_norm, x.shape)

    def test_divisibility_check(self):
        x = torch.randn(2, 10, 8, 8)
        with pytest.raises(ValueError, match="must be divisible"):
            normalize_oscillators2d(x, n_oscillators=3)


class TestNormalizeOscillators3D:
    """Test normalize_oscillators3d function."""

    def test_basic_normalization(self):
        x = torch.randn(2, 12, 4, 8, 8)  # [B, C, D, H, W]
        x_norm = normalize_oscillators3d(x, n_oscillators=4)

        assert_shape(x_norm, x.shape)
        x_reshaped = x_norm.view(2, 3, 4, 4, 8, 8)
        assert_unit_norm(x_reshaped, dim=2, eps=1e-6)

    def test_different_spatial_sizes(self):
        x = torch.randn(2, 8, 4, 8, 16)
        x_norm = normalize_oscillators3d(x, n_oscillators=4)
        assert_shape(x_norm, (2, 8, 4, 8, 16))

    def test_divisibility_check(self):
        x = torch.randn(2, 10, 4, 8, 8)
        with pytest.raises(ValueError, match="must be divisible"):
            normalize_oscillators3d(x, n_oscillators=3)


class TestKuramotoStep:
    """Test kuramoto_step function."""

    def test_basic_2d_step(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        x_new = kuramoto_step(
            x=x,
            coupling=coupling,
            stimulus=stimulus,
            n_oscillators=n_oscillators,
            omega=None,
            step_size=1.0,
            spatial_ndim=2,
        )

        # Check shapes
        assert_shape(x_new, (B, C, H, W))

        # Check normalization
        x_reshaped = x_new.view(B, C // n_oscillators, n_oscillators, H, W)
        assert_unit_norm(x_reshaped, dim=2, eps=1e-6)

    def test_1d_step(self):
        B, C, L = 2, 8, 16
        n_oscillators = 4
        x = torch.randn(B, C, L)
        coupling = torch.randn(B, C, L)
        stimulus = torch.randn(B, C, L)

        x_new = kuramoto_step(
            x=x,
            coupling=coupling,
            stimulus=stimulus,
            n_oscillators=n_oscillators,
            spatial_ndim=1,
        )

        assert_shape(x_new, (B, C, L))

    def test_3d_step(self):
        B, C, D, H, W = 2, 8, 4, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, D, H, W)
        coupling = torch.randn(B, C, D, H, W)
        stimulus = torch.randn(B, C, D, H, W)

        x_new = kuramoto_step(
            x=x,
            coupling=coupling,
            stimulus=stimulus,
            n_oscillators=n_oscillators,
            spatial_ndim=3,
        )

        assert_shape(x_new, (B, C, D, H, W))

    def test_omega_scalar(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)
        omega = torch.tensor(1.0)

        x_new = kuramoto_step(
            x, coupling, stimulus, n_oscillators, omega=omega, spatial_ndim=2
        )

        assert_shape(x_new, (B, C, H, W))

    def test_omega_per_group(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        n_groups = C // n_oscillators
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)
        omega = torch.randn(n_groups)

        x_new = kuramoto_step(
            x, coupling, stimulus, n_oscillators, omega=omega, spatial_ndim=2
        )

        assert_shape(x_new, (B, C, H, W))

    def test_step_size_scaling(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        # Larger step size should produce larger changes
        x_new1, _ = kuramoto_step(
            x.clone(), coupling, stimulus, n_oscillators, step_size=0.1, spatial_ndim=2
        )
        x_new2, _ = kuramoto_step(
            x.clone(), coupling, stimulus, n_oscillators, step_size=1.0, spatial_ndim=2
        )

        diff1 = (x_new1 - x).abs().mean()
        diff2 = (x_new2 - x).abs().mean()
        assert diff2 > diff1

    def test_without_projection(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        x_new = kuramoto_step(
            x,
            coupling,
            stimulus,
            n_oscillators,
            apply_projection=False,
            spatial_ndim=2,
        )

        assert_shape(x_new, (B, C, H, W))

    def test_without_normalization(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        x_new = kuramoto_step(
            x, coupling, stimulus, n_oscillators, normalize=False, spatial_ndim=2
        )

        assert_shape(x_new, (B, C, H, W))
        # Should NOT be normalized
        x_reshaped = x_new.view(B, C // n_oscillators, n_oscillators, H, W)
        norms = x_reshaped.norm(dim=2, p=2)
        # At least some norms should deviate from 1
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

    def test_gradient_flow(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W, requires_grad=True)
        coupling = torch.randn(B, C, H, W, requires_grad=True)
        stimulus = torch.randn(B, C, H, W, requires_grad=True)

        x_new = kuramoto_step(
            x, coupling, stimulus, n_oscillators, spatial_ndim=2
        )

        loss = x_new.sum()
        loss.backward()

        assert x.grad is not None
        assert coupling.grad is not None
        assert stimulus.grad is not None

    def test_omega_odd_oscillators_error(self):
        # Omega with odd oscillators should raise error
        B, C, H, W = 2, 9, 4, 4
        n_oscillators = 3
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)
        omega = torch.tensor(1.0)

        with pytest.raises(NotImplementedError, match="odd n_oscillators"):
            kuramoto_step(
                x, coupling, stimulus, n_oscillators, omega=omega, spatial_ndim=2
            )

    def test_invalid_spatial_ndim(self):
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        with pytest.raises(ValueError, match="spatial_ndim must be"):
            kuramoto_step(x, coupling, stimulus, n_oscillators, spatial_ndim=4)

    def test_deterministic(self):
        # Same inputs should produce same outputs
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4

        torch.manual_seed(42)
        x1 = torch.randn(B, C, H, W)
        coupling1 = torch.randn(B, C, H, W)
        stimulus1 = torch.randn(B, C, H, W)

        torch.manual_seed(42)
        x2 = torch.randn(B, C, H, W)
        coupling2 = torch.randn(B, C, H, W)
        stimulus2 = torch.randn(B, C, H, W)

        x_new1 = kuramoto_step(
            x1, coupling1, stimulus1, n_oscillators, spatial_ndim=2
        )
        x_new2 = kuramoto_step(
            x2, coupling2, stimulus2, n_oscillators, spatial_ndim=2
        )

        assert torch.allclose(x_new1, x_new2)


class TestKuramotoDynamicsProperties:
    """Test mathematical properties of Kuramoto dynamics."""

    def test_tangent_space_projection(self):
        # After projection, update should be orthogonal to state
        B, C, H, W = 1, 4, 2, 2
        n_oscillators = 2
        x = torch.randn(B, C, H, W)
        x = normalize_oscillators2d(x, n_oscillators)  # Normalize first

        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        x_new = kuramoto_step(
            x,
            coupling,
            stimulus,
            n_oscillators,
            omega=None,
            apply_projection=True,
            normalize=False,  # Don't normalize to see raw update
            spatial_ndim=2,
        )

        # Compute update
        delta = x_new - x

        # Reshape for checking
        x_reshaped = x.view(B, C // n_oscillators, n_oscillators, H, W)
        delta_reshaped = delta.view(B, C // n_oscillators, n_oscillators, H, W)

        # Compute inner product <delta, x>
        inner_product = (delta_reshaped * x_reshaped).sum(dim=2)

        # Should be close to zero (tangent space property)
        assert torch.allclose(inner_product, torch.zeros_like(inner_product), atol=1e-5)

    def test_multiple_steps_preserve_normalization(self):
        # Running multiple steps should maintain unit norm
        B, C, H, W = 2, 8, 4, 4
        n_oscillators = 4
        x = torch.randn(B, C, H, W)
        x = normalize_oscillators2d(x, n_oscillators)

        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        for _ in range(5):
            x = kuramoto_step(x, coupling, stimulus, n_oscillators, spatial_ndim=2)

            # Check normalization maintained
            x_reshaped = x.view(B, C // n_oscillators, n_oscillators, H, W)
            assert_unit_norm(x_reshaped, dim=2, eps=1e-5)
