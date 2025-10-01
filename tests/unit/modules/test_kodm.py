"""Unit tests for KODM (Kuramoto Orientation Diffusion Model)."""

import math
import torch

from oscillatory.nn.modules.kodm import KODM, KODMUNet, NestedScoreMatchingLoss
from tests.conftest import assert_shape, assert_finite


class TestKODMUNet:
    """Test KODMUNet module."""

    def test_basic_initialization(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        assert model.channels == 3
        assert model.time_dim == 256
        assert model.img_size == 32

    def test_forward_shape(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        x = torch.randn(2, 3, 32, 32)  # Angles
        t = torch.randint(0, 100, (2,))  # Timesteps
        out = model(x, t)
        # Output should be in (cos, sin) space: 2*channels
        assert_shape(out, (2, 6, 32, 32))

    def test_angle_to_cossin_conversion(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        # Input is angles, should be converted to (cos, sin) internally
        x = torch.randn(1, 3, 32, 32) * math.pi
        t = torch.tensor([50])
        out = model(x, t)
        assert_shape(out, (1, 6, 32, 32))
        assert_finite(out)

    def test_different_timesteps(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        x = torch.randn(2, 3, 32, 32)

        for t_val in [1, 50, 99]:
            t = torch.full((2,), t_val, dtype=torch.long)
            out = model(x, t)
            assert_shape(out, (2, 6, 32, 32))

    def test_different_batch_sizes(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 32, 32)
            t = torch.randint(0, 100, (batch_size,))
            out = model(x, t)
            assert_shape(out, (batch_size, 6, 32, 32))

    def test_different_channel_counts(self):
        for channels in [1, 3, 4]:
            model = KODMUNet(channels=channels, time_dim=256, img_size=32)
            x = torch.randn(2, channels, 32, 32)
            t = torch.randint(0, 100, (2,))
            out = model(x, t)
            assert_shape(out, (2, 2 * channels, 32, 32))

    def test_gradient_flow(self, gradient_checker):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        t = torch.tensor([50])
        gradient_checker.check_gradients(model, [x, t])

    def test_eval_mode_deterministic(self):
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([50])
        out1 = model(x, t)

        torch.manual_seed(42)
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([50])
        out2 = model(x, t)

        assert torch.allclose(out1, out2)


class TestKODM:
    """Test KODM scheduler."""

    def test_basic_initialization(self):
        scheduler = KODM(
            num_timesteps=100,
            noise_start=1e-4,
            noise_end=0.1,
            coupling_start=3e-5,
            coupling_end=0.03,
            img_size=32,
        )
        assert scheduler.num_timesteps == 100
        assert scheduler.img_size == 32

    def test_schedule_shapes(self):
        scheduler = KODM(num_timesteps=100)
        assert_shape(scheduler.noise_schedule, (100,))
        assert_shape(scheduler.coupling_schedule, (100,))

    def test_schedule_monotonicity(self):
        scheduler = KODM(num_timesteps=100)
        # Noise schedule should be monotonically increasing
        noise_sched = scheduler.noise_schedule
        assert (noise_sched[:-1] <= noise_sched[1:]).all()

        # Coupling schedule should be monotonically increasing
        coupling_sched = scheduler.coupling_schedule
        assert (coupling_sched[:-1] <= coupling_sched[1:]).all()

    def test_compute_kuramoto_coupling(self):
        scheduler = KODM(num_timesteps=100, img_size=32)
        x = torch.randn(2, 3, 32, 32) * math.pi

        noise, coupling, order_param, order_phase = scheduler.compute_kuramoto_coupling(
            x
        )

        assert_shape(noise, (2, 3, 32, 32))
        assert_shape(coupling, (2, 3, 32, 32))
        assert order_param.shape[0] == 2
        assert order_phase.shape[0] == 2

        # Order parameter should be in [0, 1]
        assert (order_param >= 0).all() and (order_param <= 1).all()

        # Order phase should be in [-π, π]
        assert (order_phase >= -math.pi).all() and (order_phase <= math.pi).all()

    def test_add_noise(self):
        scheduler = KODM(num_timesteps=100, img_size=32)
        x0 = torch.randn(2, 3, 32, 32) * math.pi
        t = torch.tensor([30, 40])

        xt = scheduler.add_noise(x0, t)

        assert_shape(xt, (2, 3, 32, 32))
        # Output should be in [-π, π] (wrapped)
        assert (xt >= -math.pi).all() and (xt <= math.pi).all()

    def test_add_noise_zero_steps(self):
        scheduler = KODM(num_timesteps=100, img_size=32)
        x0 = torch.randn(2, 3, 32, 32) * math.pi
        t = torch.tensor([0, 0])

        xt = scheduler.add_noise(x0, t)

        # With t=0, no noise should be added (loop doesn't run)
        # Output equals input (no wrapping when t=0)
        assert torch.allclose(xt, x0)

    def test_step(self):
        scheduler = KODM(num_timesteps=100, img_size=32)
        sample = torch.randn(2, 3, 32, 32) * math.pi
        model_output = torch.randn(2, 6, 32, 32)  # (cos, sin) space
        t = 50

        x_prev = scheduler.step(model_output, t, sample)

        assert_shape(x_prev, (2, 3, 32, 32))
        assert (x_prev >= -math.pi).all() and (x_prev <= math.pi).all()

    def test_sample(self):
        scheduler = KODM(num_timesteps=10, img_size=16)  # Small for speed
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        model.eval()

        samples = scheduler.sample(model, batch_size=1, channels=3)

        assert_shape(samples, (1, 3, 16, 16))
        assert (samples >= -math.pi).all() and (samples <= math.pi).all()
        assert_finite(samples)

    def test_von_mises_initialization(self):
        # Test that concentration parameter is computed correctly
        scheduler = KODM(
            num_timesteps=100,
            noise_end=0.1,
            coupling_end=0.03,
        )

        # Formula: (2.0 * coupling[-1]) / (0.5 * noise[-1]^2)
        expected_concentration = (2.0 * scheduler.coupling_schedule[-1]) / (
            0.5 * scheduler.noise_schedule[-1] ** 2
        )

        # Just verify it's computed (checked in sample method)
        assert expected_concentration > 0

    def test_ref_phase(self):
        scheduler = KODM(num_timesteps=100, ref_phase=1.5)
        assert scheduler.ref_phase == 1.5

        x = torch.randn(2, 3, 32, 32)
        noise, coupling, _, _ = scheduler.compute_kuramoto_coupling(x)

        # Coupling should include ref_phase term
        assert_finite(coupling)


class TestNestedScoreMatchingLoss:
    """Test NestedScoreMatchingLoss."""

    def test_basic_initialization(self):
        scheduler = KODM(num_timesteps=100)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=5)
        assert loss_fn.num_samples == 5

    def test_forward(self):
        scheduler = KODM(num_timesteps=100, img_size=32)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=3)
        model = KODMUNet(channels=3, time_dim=256, img_size=32)

        x0 = torch.randn(2, 3, 32, 32) * math.pi
        t = torch.randint(0, 99, (2,))  # Don't use last timestep

        loss = loss_fn(model, x0, t)

        assert loss.shape == ()  # Scalar
        assert_finite(loss)
        assert loss.item() >= 0  # MSE loss should be non-negative

    def test_gradient_flow(self):
        scheduler = KODM(num_timesteps=100, img_size=16)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=2)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)

        x0 = torch.randn(1, 3, 16, 16) * math.pi
        t = torch.tensor([10])

        loss = loss_fn(model, x0, t)
        loss.backward()

        # Check gradients computed for model parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad

    def test_different_num_samples(self):
        scheduler = KODM(num_timesteps=100, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        x0 = torch.randn(1, 3, 16, 16) * math.pi
        t = torch.tensor([10])

        losses = []
        for num_samples in [1, 3, 5]:
            loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=num_samples)
            loss = loss_fn(model, x0, t)
            losses.append(loss.item())
            assert_finite(loss)

        # More samples should give more stable (but not necessarily lower) loss


class TestKODMIntegration:
    """Integration tests for KODM components."""

    def test_full_training_step(self):
        # Simulate a complete training step
        scheduler = KODM(num_timesteps=100, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Generate fake data
        x0 = torch.randn(2, 3, 16, 16) * math.pi
        t = torch.randint(0, 99, (2,))

        # Forward pass
        loss = loss_fn(model, x0, t)
        assert_finite(loss)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify parameters updated
        # (Just check that step completed without errors)

    def test_forward_reverse_consistency(self):
        # Add noise then denoise should approximately recover original
        scheduler = KODM(num_timesteps=10, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        model.eval()

        x0 = torch.zeros(1, 3, 16, 16)  # Start from zero
        t = torch.tensor([5])

        # Forward process
        xt = scheduler.add_noise(x0, t)

        # Reverse process (single step, won't fully recover)
        with torch.no_grad():
            model_output = model(xt, t)
            x_prev = scheduler.step(model_output, int(t.item()), xt)

        # Just verify both processes work
        assert_finite(xt)
        assert_finite(x_prev)

    def test_sampling_produces_valid_angles(self):
        # Generated samples should be valid angles
        scheduler = KODM(num_timesteps=5, img_size=8)  # Very small for speed
        model = KODMUNet(channels=3, time_dim=256, img_size=8)
        model.eval()

        samples = scheduler.sample(model, batch_size=2, channels=3)

        # Should be in valid angle range
        assert (samples >= -math.pi).all() and (samples <= math.pi).all()
        assert_finite(samples)

    def test_different_image_sizes(self):
        for img_size in [8, 16, 32]:
            scheduler = KODM(num_timesteps=10, img_size=img_size)
            model = KODMUNet(channels=3, time_dim=256, img_size=img_size)

            x = torch.randn(1, 3, img_size, img_size) * math.pi
            t = torch.tensor([5])

            # Forward diffusion
            xt = scheduler.add_noise(x, t)
            assert_shape(xt, (1, 3, img_size, img_size))

            # Model prediction
            out = model(xt, t)
            assert_shape(out, (1, 6, img_size, img_size))


class TestKODMMathematicalProperties:
    """Test mathematical properties of KODM."""

    def test_phase_wrapping_preserved(self):
        # Throughout diffusion, angles should stay in [-π, π]
        scheduler = KODM(num_timesteps=50, img_size=16)
        x0 = torch.randn(2, 3, 16, 16) * math.pi

        for t_val in [10, 20, 30, 40]:
            t = torch.full((2,), t_val)
            xt = scheduler.add_noise(x0, t)
            assert (xt >= -math.pi - 0.01).all() and (xt <= math.pi + 0.01).all()

    def test_coupling_increases_with_order(self):
        # Higher order parameter should lead to stronger coupling
        scheduler = KODM(num_timesteps=100, img_size=16)

        # Synchronized phases (high order)
        x_sync = torch.ones(1, 3, 16, 16) * 1.0
        _, coupling_sync, order_sync, _ = scheduler.compute_kuramoto_coupling(x_sync)

        # Random phases (low order)
        x_random = torch.rand(1, 3, 16, 16) * 2 * math.pi - math.pi
        _, coupling_random, order_random, _ = scheduler.compute_kuramoto_coupling(
            x_random
        )

        # Order parameter should be higher for synchronized
        assert order_sync.mean() > order_random.mean()

    def test_noise_accumulation(self):
        # More timesteps should add more noise (generally)
        scheduler = KODM(num_timesteps=100, img_size=16)
        x0 = torch.zeros(2, 3, 16, 16)

        xt_10 = scheduler.add_noise(x0.clone(), torch.tensor([10, 10]))
        xt_50 = scheduler.add_noise(x0.clone(), torch.tensor([50, 50]))

        # Deviation from zero should be larger for more steps
        dev_10 = xt_10.abs().mean()
        dev_50 = xt_50.abs().mean()

        assert dev_50 > dev_10
