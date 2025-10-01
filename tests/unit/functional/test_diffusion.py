"""Unit tests for circular diffusion operations."""

import math
import torch

from oscillatory.nn.functional.diffusion import (
    phase_modulate,
    wrapped_gaussian_score,
    mean_field_phase,
    cossin_to_angle_score,
)
from tests.conftest import assert_shape, assert_finite


class TestPhaseModulate:
    """Test phase_modulate function."""

    def test_wrapping_positive(self):
        # Values > π should wrap to negative
        x = torch.tensor([4.0, 5.0, 6.0])
        result = phase_modulate(x)
        assert (result >= -math.pi).all() and (result <= math.pi).all()

    def test_wrapping_negative(self):
        # Values < -π should wrap to positive
        x = torch.tensor([-4.0, -5.0, -6.0])
        result = phase_modulate(x)
        assert (result >= -math.pi).all() and (result <= math.pi).all()

    def test_range_preservation(self):
        # Values already in range should stay unchanged
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        result = phase_modulate(x)
        assert torch.allclose(result, x, atol=1e-6)

    def test_exact_boundaries(self):
        # Test exactly at boundaries
        x = torch.tensor([math.pi, -math.pi, 0.0])
        result = phase_modulate(x)
        assert (result >= -math.pi).all() and (result <= math.pi).all()

    def test_2pi_equivalence(self):
        # Values differing by 2π should map to same result
        x1 = torch.tensor([1.0])
        x2 = torch.tensor([1.0 + 2 * math.pi])
        x3 = torch.tensor([1.0 - 2 * math.pi])

        r1 = phase_modulate(x1)
        r2 = phase_modulate(x2)
        r3 = phase_modulate(x3)

        assert torch.allclose(r1, r2, atol=1e-6)
        assert torch.allclose(r1, r3, atol=1e-6)

    def test_batch_preservation(self):
        x = torch.randn(2, 3, 32, 32) * 10.0  # Large range
        result = phase_modulate(x)
        assert_shape(result, x.shape)
        assert (result >= -math.pi).all() and (result <= math.pi).all()

    def test_gradient_flow(self):
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        result = phase_modulate(x)
        loss = result.sum()
        loss.backward()
        assert x.grad is not None


class TestWrappedGaussianScore:
    """Test wrapped_gaussian_score function."""

    def test_basic_shape(self):
        y = torch.randn(2, 3, 8, 8)
        mu = torch.randn(2, 3, 8, 8)
        sigma = 0.5
        score = wrapped_gaussian_score(y, mu, sigma, K=3)
        assert_shape(score, y.shape)

    def test_scalar_sigma(self):
        y = torch.randn(2, 3, 8, 8)
        mu = torch.randn(2, 3, 8, 8)
        score = wrapped_gaussian_score(y, mu, sigma=1.0, K=3)
        assert_finite(score)

    def test_tensor_sigma(self):
        y = torch.randn(2, 3, 8, 8)
        mu = torch.randn(2, 3, 8, 8)
        sigma = torch.rand(2, 3, 8, 8) * 0.5 + 0.3  # Avoid small sigma (min 0.3)
        score = wrapped_gaussian_score(y, mu, sigma, K=3)
        assert_finite(score)

    def test_wrapping_parameter_K(self):
        y = torch.randn(2, 3, 4, 4)
        mu = torch.randn(2, 3, 4, 4)
        sigma = 0.5

        score1 = wrapped_gaussian_score(y, mu, sigma, K=1)
        score3 = wrapped_gaussian_score(y, mu, sigma, K=3)
        score5 = wrapped_gaussian_score(y, mu, sigma, K=5)

        # Higher K should give more accurate scores (not necessarily larger)
        assert_finite(score1)
        assert_finite(score3)
        assert_finite(score5)

    def test_zero_difference(self):
        # When y == mu, score should be close to zero
        mu = torch.randn(2, 3, 4, 4)
        y = mu.clone()
        sigma = 0.5
        score = wrapped_gaussian_score(y, mu, sigma, K=3)
        # Score should be small when y = mu
        assert score.abs().mean() < 1.0

    def test_periodicity(self):
        # Score should respect 2π periodicity
        y = torch.tensor([[[[1.0]]]])
        mu = torch.tensor([[[[0.5]]]])
        sigma = 0.3

        score1 = wrapped_gaussian_score(y, mu, sigma, K=3)

        # Shift y by 2π
        y2 = y + 2 * math.pi
        score2 = wrapped_gaussian_score(y2, mu, sigma, K=3)

        assert torch.allclose(score1, score2, rtol=1e-5, atol=1e-5)

    def test_gradient_flow(self):
        y = torch.randn(2, 3, 4, 4, requires_grad=True)
        mu = torch.randn(2, 3, 4, 4, requires_grad=True)
        sigma = 0.5

        score = wrapped_gaussian_score(y, mu, sigma, K=3)
        loss = score.sum()
        loss.backward()

        assert y.grad is not None
        assert mu.grad is not None

    def test_sign_convention(self):
        # Verify sign convention: (mu - 2πk) not (mu + 2πk)
        # When y > mu (in wrapped sense), score should push y down
        y = torch.tensor([[[[1.0]]]])
        mu = torch.tensor([[[[0.0]]]])
        sigma = 0.5
        score = wrapped_gaussian_score(y, mu, sigma, K=3)
        # Score should be negative (pointing from y toward mu)
        assert score.item() < 0


class TestMeanFieldPhase:
    """Test mean_field_phase function."""

    def test_basic_shape(self):
        x = torch.randn(2, 3, 8, 8)
        order_param, mean_phase = mean_field_phase(x, keepdim=True)

        assert_shape(order_param, (2, 1, 1, 1))
        assert_shape(mean_phase, (2, 1, 1, 1))

    def test_shape_without_keepdim(self):
        x = torch.randn(2, 3, 8, 8)
        order_param, mean_phase = mean_field_phase(x, keepdim=False)

        assert_shape(order_param, (2,))
        assert_shape(mean_phase, (2,))

    def test_order_parameter_range(self):
        x = torch.randn(2, 3, 8, 8) * math.pi
        order_param, _ = mean_field_phase(x)

        # Order parameter R should be in [0, 1]
        assert (order_param >= 0).all() and (order_param <= 1).all()

    def test_mean_phase_range(self):
        x = torch.randn(2, 3, 8, 8) * math.pi
        _, mean_phase = mean_field_phase(x)

        # Mean phase should be in [-π, π]
        assert (mean_phase >= -math.pi).all() and (mean_phase <= math.pi).all()

    def test_uniform_phases_low_order(self):
        # Uniformly distributed random phases should have low order parameter
        x = torch.rand(1, 100, 100, 100) * 2 * math.pi - math.pi
        order_param, _ = mean_field_phase(x)

        # With many random phases, order parameter should be close to 0
        assert order_param.item() < 0.3

    def test_synchronized_phases_high_order(self):
        # All same phase should have order parameter = 1
        x = torch.ones(1, 10, 10, 10) * 1.5
        order_param, mean_phase = mean_field_phase(x)

        assert torch.allclose(order_param, torch.ones_like(order_param), atol=1e-6)
        assert torch.allclose(mean_phase, torch.full_like(mean_phase, 1.5), atol=1e-6)

    def test_gradient_flow(self):
        x = torch.randn(2, 3, 4, 4, requires_grad=True)
        order_param, mean_phase = mean_field_phase(x)
        loss = order_param.sum() + mean_phase.sum()
        loss.backward()

        assert x.grad is not None

    def test_different_dimensions(self):
        # Test with 3D input
        x = torch.randn(2, 3, 4)
        order_param, mean_phase = mean_field_phase(x)
        assert order_param.shape[0] == 2
        assert mean_phase.shape[0] == 2


class TestCossinToAngleScore:
    """Test cossin_to_angle_score function."""

    def test_basic_shape(self):
        angle = torch.randn(2, 3, 8, 8)
        score = torch.randn(2, 6, 8, 8)  # 2*C channels
        result = cossin_to_angle_score(score, angle)
        assert_shape(result, angle.shape)

    def test_channel_reduction(self):
        # Should reduce from 2*C to C channels
        angle = torch.randn(2, 5, 8, 8)
        score = torch.randn(2, 10, 8, 8)
        result = cossin_to_angle_score(score, angle)
        assert_shape(result, (2, 5, 8, 8))

    def test_zero_score(self):
        # Zero score should give zero result
        angle = torch.randn(2, 3, 4, 4)
        score = torch.zeros(2, 6, 4, 4)
        result = cossin_to_angle_score(score, angle)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_gradient_flow(self):
        angle = torch.randn(2, 3, 4, 4, requires_grad=True)
        score = torch.randn(2, 6, 4, 4, requires_grad=True)
        result = cossin_to_angle_score(score, angle)
        loss = result.sum()
        loss.backward()

        assert angle.grad is not None
        assert score.grad is not None

    def test_chain_rule_application(self):
        # Verify the chain rule formula is applied correctly
        angle = torch.tensor([[[[1.0]]]])
        # Create score with known values
        score_cos = torch.tensor([[[[2.0]]]])  # d/dcos
        score_sin = torch.tensor([[[[3.0]]]])  # d/dsin
        score = torch.cat([score_cos, score_sin], dim=1)

        result = cossin_to_angle_score(score, angle)

        # Expected: -score_cos * sin(angle) + score_sin * cos(angle)
        expected = -2.0 * math.sin(1.0) + 3.0 * math.cos(1.0)
        assert torch.allclose(result, torch.tensor([[[[expected]]]]), atol=1e-6)

    def test_different_batch_sizes(self):
        for B in [1, 2, 4]:
            angle = torch.randn(B, 3, 8, 8)
            score = torch.randn(B, 6, 8, 8)
            result = cossin_to_angle_score(score, angle)
            assert_shape(result, (B, 3, 8, 8))


class TestDiffusionIntegration:
    """Integration tests for diffusion operations."""

    def test_full_diffusion_pipeline(self):
        # Simulate a simple diffusion step
        B, C, H, W = 2, 3, 16, 16

        # Initial clean angles
        x0 = torch.randn(B, C, H, W) * math.pi
        x0 = phase_modulate(x0)

        # Add some noise
        noise = torch.randn_like(x0) * 0.1
        xt = phase_modulate(x0 + noise)

        # Compute order parameter
        order_param, mean_phase = mean_field_phase(xt)
        assert (order_param >= 0).all() and (order_param <= 1).all()

        # Compute coupling term (simplified)
        coupling = order_param * torch.sin(mean_phase - xt)
        assert_finite(coupling)

        # Network predicts score in (cos, sin) space
        predicted_score_cossin = torch.randn(B, 2 * C, H, W)

        # Convert to angle space
        predicted_score = cossin_to_angle_score(predicted_score_cossin, xt)
        assert_shape(predicted_score, (B, C, H, W))

        # Compute target score
        target_score = wrapped_gaussian_score(xt, x0, sigma=0.1, K=3)
        assert_shape(target_score, (B, C, H, W))

    def test_wrapped_score_convergence(self):
        # Score should point toward mean
        mu = torch.zeros(1, 1, 1, 1)

        # Test points around the circle
        angles = torch.linspace(-math.pi + 0.1, math.pi - 0.1, 10)

        for angle in angles:
            y = torch.tensor([[[[angle.item()]]]])
            score = wrapped_gaussian_score(y, mu, sigma=0.5, K=5)

            # Score should point toward mu (0)
            # If angle > 0, score should be negative
            # If angle < 0, score should be positive
            if angle > 0.1:
                assert score.item() < 0, f"Angle {angle}, score {score.item()}"
            elif angle < -0.1:
                assert score.item() > 0, f"Angle {angle}, score {score.item()}"
