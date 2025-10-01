"""End-to-end integration tests for complete pipelines."""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from oscillatory.nn.modules.akorn import (
    AKOrNHierarchical,
    AKOrNDense,
    AKOrNGrid,
)
from oscillatory.nn.modules.heads import (
    ClassificationHead,
    ObjectDiscoveryHead,
    SudokuHead,
)
from oscillatory.nn.modules.kodm import KODM, KODMUNet, NestedScoreMatchingLoss
from tests.conftest import assert_shape, assert_finite


class TestImageClassificationPipeline:
    """Test complete image classification pipeline."""

    def test_hierarchical_classification_training_step(self):
        """Test a complete training step for hierarchical classification."""
        # Create model
        # With base_channels=64, num_layers=2: output is 64 * 2 = 128 channels
        head = ClassificationHead(in_channels=128, num_classes=10)
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
            final_head=head,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fake batch
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))

        # Training step
        model.train()
        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        assert_finite(loss)

        loss.backward()
        optimizer.step()

        # Verify gradients computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_hierarchical_classification_inference(self):
        """Test inference mode for hierarchical classification."""
        # With base_channels=64, num_layers=2: output is 64 * 2 = 128 channels
        head = ClassificationHead(in_channels=128, num_classes=10)
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
            final_head=head,
        )

        model.eval()
        images = torch.randn(4, 3, 32, 32)

        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        assert_shape(logits, (4, 10))
        assert_shape(preds, (4,))
        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))

    def test_dense_classification_pipeline(self):
        """Test dense model for image classification."""
        head = ClassificationHead(in_channels=256, num_classes=10)
        model = AKOrNDense(
            image_size=128,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=2,
            final_head=head,
        )

        images = torch.randn(2, 3, 128, 128)
        labels = torch.randint(0, 10, (2,))

        model.train()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        assert_finite(loss)

        loss.backward()

        # Check gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestObjectDiscoveryPipeline:
    """Test object discovery pipeline."""

    def test_object_discovery_training(self):
        """Test object discovery training step."""
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, return_slots=True
        )
        model = AKOrNDense(
            image_size=128,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=2,
            final_head=head,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        images = torch.randn(2, 3, 128, 128)

        model.train()
        optimizer.zero_grad()

        slots = model(images)  # [B, num_slots, slot_features]

        # Simple reconstruction loss (placeholder)
        target_slots = torch.randn_like(slots)
        loss = F.mse_loss(slots, target_slots)

        assert_finite(loss)

        loss.backward()
        optimizer.step()

    def test_object_discovery_slot_extraction(self):
        """Test extracting object slots."""
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, return_slots=True
        )
        model = AKOrNDense(
            image_size=128,
            patch_size=4,
            in_channels=3,
            embed_dim=256,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=2,
            final_head=head,
        )

        model.eval()
        images = torch.randn(2, 3, 128, 128)

        with torch.no_grad():
            slots = model(images)

        assert slots.shape[1] == 4  # num_slots
        assert_finite(slots)


class TestSudokuSolverPipeline:
    """Test Sudoku solver pipeline."""

    def test_sudoku_training_step(self):
        """Test Sudoku solver training step."""
        head = SudokuHead(in_channels=64, num_digits=9, permute_output=True)
        model = AKOrNGrid(
            grid_size=(9, 9),
            vocab_size=10,
            embed_dim=64,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=4,
            final_head=head,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fake Sudoku puzzles
        puzzles = torch.randint(0, 10, (2, 9, 9))
        solutions = torch.randint(0, 9, (2, 9, 9))  # 0-8 for 9 classes

        model.train()
        optimizer.zero_grad()

        logits = model(puzzles)  # [B, H, W, num_digits]

        # Compute loss
        logits_flat = logits.reshape(-1, 9)
        solutions_flat = solutions.reshape(-1)
        loss = F.cross_entropy(logits_flat, solutions_flat)

        assert_finite(loss)

        loss.backward()
        optimizer.step()

    def test_sudoku_with_mask(self):
        """Test Sudoku solver with masked input."""
        head = SudokuHead(in_channels=64, num_digits=9, permute_output=True)
        model = AKOrNGrid(
            grid_size=(9, 9),
            vocab_size=10,
            embed_dim=64,
            num_layers=1,
            n_oscillators=4,
            n_timesteps=4,
            final_head=head,
        )

        model.eval()

        # Puzzle with mask (known cells)
        puzzle = torch.randint(0, 10, (1, 9, 9))
        mask = torch.rand(1, 9, 9) > 0.7  # ~30% cells known

        with torch.no_grad():
            logits = model(puzzle, mask=mask)

        assert_shape(logits, (1, 9, 9, 9))

        # Get predictions
        preds = logits.argmax(dim=-1)
        assert_shape(preds, (1, 9, 9))


class TestKODMDiffusionPipeline:
    """Test KODM diffusion pipeline."""

    def test_kodm_training_step(self):
        """Test KODM training step."""
        scheduler = KODM(num_timesteps=10, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)
        loss_fn = NestedScoreMatchingLoss(scheduler, num_samples=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Fake angle images
        x0 = torch.randn(2, 3, 16, 16) * math.pi

        model.train()
        optimizer.zero_grad()

        # Random timesteps
        t = torch.randint(0, 9, (2,))

        loss = loss_fn(model, x0, t)
        assert_finite(loss)

        loss.backward()
        optimizer.step()

    def test_kodm_sampling_pipeline(self):
        """Test complete sampling pipeline."""
        scheduler = KODM(num_timesteps=5, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)

        model.eval()

        # Generate samples
        samples = scheduler.sample(model, batch_size=2, channels=3)

        assert_shape(samples, (2, 3, 16, 16))
        assert (samples >= -math.pi).all()
        assert (samples <= math.pi).all()
        assert_finite(samples)

    def test_kodm_forward_reverse_cycle(self):
        """Test forward diffusion then reverse denoising."""
        scheduler = KODM(num_timesteps=10, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)

        model.eval()

        # Start with clean image
        x0 = torch.zeros(1, 3, 16, 16)

        # Forward diffusion
        t = torch.tensor([5])
        xt = scheduler.add_noise(x0, t)

        # Denoise one step
        with torch.no_grad():
            model_output = model(xt, t)
            x_prev = scheduler.step(model_output, int(t.item()), xt)

        assert_shape(x_prev, (1, 3, 16, 16))
        assert_finite(x_prev)


class TestMultiTaskScenarios:
    """Test models with multiple heads/tasks."""

    def test_feature_extraction_mode(self):
        """Test using model for feature extraction without head."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
            final_head=None,  # No head
        )

        model.eval()
        images = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            features = model(images)

        # Should return raw features
        # With base_channels=64, num_layers=2: output is 64 * 2 = 128 channels
        assert features.shape[0] == 2
        assert features.shape[1] == 128
        assert_finite(features)

    def test_custom_head_attachment(self):
        """Test attaching custom head to model."""

        # Create custom head
        class CustomHead(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.conv = nn.Conv2d(in_channels, 64, 1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 5)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return self.fc(x)

        # With base_channels=64, num_layers=2: output is 64 * 2 = 128 channels
        custom_head = CustomHead(in_channels=128)

        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=64,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
            final_head=custom_head,
        )

        model.eval()
        images = torch.randn(2, 3, 32, 32)

        with torch.no_grad():
            output = model(images)

        assert_shape(output, (2, 5))


class TestStateRetrieval:
    """Test retrieval of intermediate states."""

    def test_hierarchical_state_retrieval(self):
        """Test retrieving intermediate states from hierarchical model."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=3,
            return_all_states=True,
        )

        images = torch.randn(1, 3, 32, 32)
        output, all_xs, all_es = model(images)

        # Should have states for each layer
        assert len(all_xs) == 2
        assert len(all_es) == 2

        # Each layer should have states for each timestep
        for xs in all_xs:
            assert len(xs) == 3  # n_timesteps

        # Energies include initial (0) + n_timesteps
        for es in all_es:
            assert len(es) == 4  # 1 + n_timesteps

    def test_dense_state_retrieval(self):
        """Test retrieving states from dense model."""
        model = AKOrNDense(
            image_size=64,
            patch_size=4,
            in_channels=3,
            embed_dim=64,
            num_layers=2,
            n_oscillators=4,
            n_timesteps=3,
            return_all_states=True,
        )

        images = torch.randn(1, 3, 64, 64)
        output, all_xs, all_es = model(images)

        assert len(all_xs) == 2
        assert len(all_es) == 2

    def test_grid_state_retrieval(self):
        """Test retrieving states from grid model."""
        model = AKOrNGrid(
            grid_size=(6, 6),
            vocab_size=10,
            embed_dim=32,
            num_layers=2,
            n_oscillators=4,
            n_timesteps=3,
            return_all_states=True,
        )

        grids = torch.randint(0, 10, (1, 6, 6))
        output, all_xs, all_es = model(grids)

        assert len(all_xs) == 2
        assert len(all_es) == 2


class TestBatchProcessing:
    """Test models with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_hierarchical_batch_processing(self, batch_size):
        """Test hierarchical model with different batch sizes."""
        model = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )

        model.eval()
        images = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(images)

        assert output.shape[0] == batch_size

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_kodm_batch_processing(self, batch_size):
        """Test KODM with different batch sizes."""
        scheduler = KODM(num_timesteps=5, img_size=16)
        model = KODMUNet(channels=3, time_dim=256, img_size=16)

        model.eval()
        samples = scheduler.sample(model, batch_size=batch_size, channels=3)

        assert_shape(samples, (batch_size, 3, 16, 16))


class TestModelSaveLoad:
    """Test model serialization and loading."""

    def test_hierarchical_save_load(self):
        """Test saving and loading hierarchical model."""
        model1 = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )

        # Save state dict
        state_dict = model1.state_dict()

        # Create new model and load
        model2 = AKOrNHierarchical(
            input_size=32,
            in_channels=3,
            base_channels=32,
            num_layers=2,
            n_oscillators=2,
            n_timesteps=2,
        )
        model2.load_state_dict(state_dict)

        # Test that outputs match
        model1.eval()
        model2.eval()

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1, out2)

    def test_kodm_save_load(self):
        """Test saving and loading KODM model."""
        model1 = KODMUNet(channels=3, time_dim=256, img_size=16)
        state_dict = model1.state_dict()

        model2 = KODMUNet(channels=3, time_dim=256, img_size=16)
        model2.load_state_dict(state_dict)

        model1.eval()
        model2.eval()

        x = torch.randn(1, 3, 16, 16)
        t = torch.tensor([5])

        with torch.no_grad():
            out1 = model1(x, t)
            out2 = model2(x, t)

        assert torch.allclose(out1, out2)
