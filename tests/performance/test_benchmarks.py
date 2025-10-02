"""Performance benchmarks and regression tests.

These tests ensure that operations maintain acceptable performance
and detect regressions in speed or memory usage.
"""

import time
import torch

from oscillatory.nn.functional.kuramoto import kuramoto_step, normalize_oscillators2d
from oscillatory.nn.functional.gta import make_so2_matrices, apply_group_action
from oscillatory.nn.modules.attention import GTAttention
from oscillatory.nn.modules.akorn import AKOrNHierarchical, AKOrNDense, AKOrNGrid
from oscillatory.nn.modules.kodm import KODM, KODMUNet


# Benchmark thresholds (in seconds)
# These are generous to avoid flaky tests, but catch major regressions
THRESHOLDS = {
    "kuramoto_step_small": 0.1,
    "kuramoto_step_large": 0.5,
    "gta_matrices": 0.1,
    "attention_forward": 0.5,
    "akorn_hierarchical": 2.0,
    "akorn_dense": 2.0,
    "akorn_grid": 1.0,
    "kodm_forward": 0.5,
    "kodm_sampling": 5.0,
}


class TestFunctionalPerformance:
    """Performance tests for functional operations."""

    def test_kuramoto_step_performance_small(self):
        """Benchmark Kuramoto step with small input."""
        B, C, H, W = 2, 8, 8, 8
        n_oscillators = 4

        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        # Warmup
        for _ in range(3):
            _ = kuramoto_step(x, coupling, stimulus, n_oscillators, spatial_ndim=2)

        # Benchmark
        start = time.time()
        for _ in range(10):
            x_new = kuramoto_step(
                x, coupling, stimulus, n_oscillators, spatial_ndim=2
            )
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < THRESHOLDS["kuramoto_step_small"], (
            f"Kuramoto step too slow: {avg_time:.4f}s > {THRESHOLDS['kuramoto_step_small']}s"
        )

    def test_kuramoto_step_performance_large(self):
        """Benchmark Kuramoto step with large input."""
        B, C, H, W = 4, 64, 32, 32
        n_oscillators = 4

        x = torch.randn(B, C, H, W)
        coupling = torch.randn(B, C, H, W)
        stimulus = torch.randn(B, C, H, W)

        # Warmup
        for _ in range(2):
            _ = kuramoto_step(x, coupling, stimulus, n_oscillators, spatial_ndim=2)

        # Benchmark
        start = time.time()
        for _ in range(5):
            x_new = kuramoto_step(
                x, coupling, stimulus, n_oscillators, spatial_ndim=2
            )
        elapsed = time.time() - start

        avg_time = elapsed / 5
        assert avg_time < THRESHOLDS["kuramoto_step_large"], (
            f"Kuramoto step (large) too slow: {avg_time:.4f}s > {THRESHOLDS['kuramoto_step_large']}s"
        )

    def test_normalization_performance(self):
        """Benchmark oscillator normalization."""
        B, C, H, W = 4, 64, 32, 32
        n_oscillators = 4

        x = torch.randn(B, C, H, W)

        # Warmup
        for _ in range(5):
            _ = normalize_oscillators2d(x, n_oscillators)

        # Benchmark
        start = time.time()
        for _ in range(100):
            normalize_oscillators2d(x, n_oscillators)
        elapsed = time.time() - start

        avg_time = elapsed / 100
        assert avg_time < 0.01, f"Normalization too slow: {avg_time:.4f}s"

    def test_gta_matrices_performance(self):
        """Benchmark SO(2) matrix generation."""
        H, W = 32, 32
        n_freqs = 16

        coords = torch.randn(H, W, 2)

        # Warmup
        for _ in range(3):
            _ = make_so2_matrices(coords, n_freqs)

        # Benchmark
        start = time.time()
        for _ in range(10):
            make_so2_matrices(coords, n_freqs)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < THRESHOLDS["gta_matrices"], (
            f"GTA matrix generation too slow: {avg_time:.4f}s > {THRESHOLDS['gta_matrices']}s"
        )

    def test_group_action_performance(self):
        """Benchmark group action application."""
        # rep: [seq_len, D/2, 2, 2], x: [batch, seq_len, D]
        rep = torch.randn(64, 64, 2, 2)
        x = torch.randn(4, 64, 128)

        # Warmup
        for _ in range(5):
            _ = apply_group_action(rep, x)

        # Benchmark
        start = time.time()
        for _ in range(100):
            apply_group_action(rep, x)
        elapsed = time.time() - start

        avg_time = elapsed / 100
        assert avg_time < 0.005, f"Group action too slow: {avg_time:.4f}s"


class TestAttentionPerformance:
    """Performance tests for attention module."""

    def test_gtattn_forward_performance(self):
        """Benchmark GTAttention forward pass."""
        attn = GTAttention(embed_dim=256, n_heads=8, weight_type="conv")
        x = torch.randn(4, 256, 16, 16)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = attn(x)

        # Benchmark
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                attn(x)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < THRESHOLDS["attention_forward"], (
            f"Attention too slow: {avg_time:.4f}s > {THRESHOLDS['attention_forward']}s"
        )

    def test_gta_vs_vanilla_attention(self):
        """Compare GTA attention with vanilla attention performance."""
        # GTA attention
        attn_gta = GTAttention(
            embed_dim=256, n_heads=8, weight_type="conv", use_gta=True, hw=(16, 16)
        )

        # Vanilla attention
        attn_vanilla = GTAttention(
            embed_dim=256, n_heads=8, weight_type="conv", use_gta=False
        )

        x = torch.randn(2, 256, 16, 16)

        # Benchmark vanilla
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = attn_vanilla(x)
        time_vanilla = time.time() - start

        # Benchmark GTA
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = attn_gta(x)
        time_gta = time.time() - start

        # GTA should not be much slower (allow 2x overhead)
        assert time_gta < time_vanilla * 2.5, (
            f"GTA too slow vs vanilla: {time_gta:.4f}s vs {time_vanilla:.4f}s"
        )


class TestModelPerformance:
    """Performance tests for complete models."""

    def test_hierarchical_forward_performance(self):
        """Benchmark hierarchical model forward pass."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=3,
            n_oscillators=2,
            n_timesteps=3,
        )
        model.eval()

        x = torch.randn(2, 3, 32, 32)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        start = time.time()
        for _ in range(5):
            with torch.no_grad():
                model(x)
        elapsed = time.time() - start

        avg_time = elapsed / 5
        assert avg_time < THRESHOLDS["akorn_hierarchical"], (
            f"Hierarchical model too slow: {avg_time:.4f}s > {THRESHOLDS['akorn_hierarchical']}s"
        )

    def test_dense_forward_performance(self):
        """Benchmark dense model forward pass."""
        model = AKOrNDense(
            image_size=128,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=3,
        )
        model.eval()

        x = torch.randn(2, 3, 128, 128)

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        start = time.time()
        for _ in range(5):
            with torch.no_grad():
                model(x)
        elapsed = time.time() - start

        avg_time = elapsed / 5
        assert avg_time < THRESHOLDS["akorn_dense"], (
            f"Dense model too slow: {avg_time:.4f}s > {THRESHOLDS['akorn_dense']}s"
        )

    def test_grid_forward_performance(self):
        """Benchmark grid model forward pass."""
        model = AKOrNGrid(
            grid_size=(9, 9),
            vocab_size=10,
            embed_dim=64,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=16,
        )
        model.eval()

        x = torch.randint(0, 10, (2, 9, 9))

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model(x)

        # Benchmark
        start = time.time()
        for _ in range(5):
            with torch.no_grad():
                model(x)
        elapsed = time.time() - start

        avg_time = elapsed / 5
        assert avg_time < THRESHOLDS["akorn_grid"], (
            f"Grid model too slow: {avg_time:.4f}s > {THRESHOLDS['akorn_grid']}s"
        )

    def test_kodm_forward_performance(self):
        """Benchmark KODM UNet forward pass."""
        model = KODMUNet(channels=3, time_dim=256, img_size=32)
        model.eval()

        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([50, 60])

        # Warmup
        for _ in range(2):
            with torch.no_grad():
                _ = model(x, t)

        # Benchmark
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                model(x, t)
        elapsed = time.time() - start

        avg_time = elapsed / 10
        assert avg_time < THRESHOLDS["kodm_forward"], (
            f"KODM forward too slow: {avg_time:.4f}s > {THRESHOLDS['kodm_forward']}s"
        )

    def test_kodm_sampling_performance(self):
        """Benchmark KODM sampling."""
        scheduler = KODM(num_timesteps=20, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        model.eval()

        # Benchmark
        start = time.time()
        scheduler.sample(model, batch_size=2, channels=3)
        elapsed = time.time() - start

        assert elapsed < THRESHOLDS["kodm_sampling"], (
            f"KODM sampling too slow: {elapsed:.4f}s > {THRESHOLDS['kodm_sampling']}s"
        )


class TestMemoryUsage:
    """Memory usage tests."""

    def test_hierarchical_memory_usage(self):
        """Test hierarchical model doesn't use excessive memory."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=3,
            n_oscillators=2,
            n_timesteps=3,
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        # Should be reasonable (< 100M parameters for this config)
        assert n_params < 100_000_000, f"Too many parameters: {n_params:,}"

        # Test memory during forward pass
        x = torch.randn(4, 3, 32, 32)

        with torch.no_grad():
            model(x)

        # Just verify it completes without OOM

    def test_dense_memory_usage(self):
        """Test dense model memory usage."""
        model = AKOrNDense(
            image_size=128,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=8,
        )

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 100_000_000, f"Too many parameters: {n_params:,}"

    def test_kodm_memory_usage(self):
        """Test KODM model memory usage."""
        model = KODMUNet(channels=3, time_dim=256, img_size=32)

        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 100_000_000, f"Too many parameters: {n_params:,}"


class TestScalability:
    """Test model scalability with different input sizes."""

    def test_hierarchical_scalability(self):
        """Test hierarchical model scales reasonably with input size."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )
        model.eval()

        times = []
        sizes = [16, 32, 64]

        for size in sizes:
            x = torch.randn(1, 3, size, size)

            # Warmup
            with torch.no_grad():
                _ = model(x)

            # Benchmark
            start = time.time()
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            elapsed = time.time() - start

            times.append(elapsed / 3)

        # Time should scale roughly quadratically with spatial size
        # (but this is a soft constraint)
        # Just verify all sizes complete reasonably fast
        for t in times:
            assert t < 2.0, f"Scaling test failed: {t:.4f}s"

    def test_batch_size_scalability(self):
        """Test model scales with batch size."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        times = []

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32)

            # Warmup
            with torch.no_grad():
                _ = model(x)

            # Benchmark
            start = time.time()
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            elapsed = time.time() - start

            times.append(elapsed / 3)

        # Time should scale roughly linearly with batch size
        # Verify all complete reasonably
        for t in times:
            assert t < 2.0


class TestGradientComputationPerformance:
    """Test gradient computation performance."""

    def test_hierarchical_backward_performance(self):
        """Benchmark backward pass for hierarchical model."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )

        x = torch.randn(2, 3, 32, 32, requires_grad=True)

        # Warmup
        for _ in range(2):
            out = model(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad()

        # Benchmark
        start = time.time()
        for _ in range(5):
            out = model(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad()
        elapsed = time.time() - start

        avg_time = elapsed / 5
        assert avg_time < 2.0, f"Backward pass too slow: {avg_time:.4f}s"

    def test_kodm_loss_backward_performance(self):
        """Benchmark KODM loss backward pass."""
        from oscillatory.nn.modules.kodm import NestedScoreMatchingLoss

        scheduler = KODM(num_timesteps=10, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=2)

        x0 = torch.randn(2, 3, 16, 16)
        t = torch.tensor([5, 6])

        # Warmup
        for _ in range(2):
            loss = loss_fn(model, x0, t)
            loss.backward()
            model.zero_grad()

        # Benchmark
        start = time.time()
        for _ in range(3):
            loss = loss_fn(model, x0, t)
            loss.backward()
            model.zero_grad()
        elapsed = time.time() - start

        avg_time = elapsed / 3
        assert avg_time < 3.0, f"Loss backward too slow: {avg_time:.4f}s"
