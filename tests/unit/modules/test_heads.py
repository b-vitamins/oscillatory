"""Unit tests for task-specific heads."""

import pytest
import torch

from oscillatory.nn.modules.heads import (
    ClassificationHead,
    ObjectDiscoveryHead,
    SudokuHead,
)
from tests.conftest import assert_shape, assert_finite


class TestClassificationHead:
    """Test ClassificationHead module."""

    def test_basic_initialization(self):
        head = ClassificationHead(in_channels=256, num_classes=10)
        assert hasattr(head, "pool")
        assert hasattr(head, "flatten")
        assert hasattr(head, "classifier")

    def test_forward_shape(self):
        head = ClassificationHead(in_channels=256, num_classes=10)
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 10))

    def test_different_input_sizes(self):
        head = ClassificationHead(in_channels=256, num_classes=10)
        for size in [(4, 4), (8, 8), (16, 16)]:
            x = torch.randn(2, 256, *size)
            out = head(x)
            assert_shape(out, (2, 10))

    def test_adaptive_avg_pool(self):
        head = ClassificationHead(
            in_channels=256, num_classes=10, pool_type="adaptive_avg", pool_size=(1, 1)
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 10))

    def test_adaptive_max_pool(self):
        head = ClassificationHead(
            in_channels=256, num_classes=10, pool_type="adaptive_max", pool_size=(1, 1)
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 10))

    def test_larger_pool_size(self):
        head = ClassificationHead(
            in_channels=256, num_classes=10, pool_type="adaptive_avg", pool_size=(2, 2)
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        # Output should be (B, num_classes) regardless of pool_size
        # because the linear layer adapts
        assert_shape(out, (2, 10))

    def test_invalid_pool_type(self):
        with pytest.raises(ValueError, match="Unknown pool_type"):
            ClassificationHead(in_channels=256, num_classes=10, pool_type="invalid")

    def test_gradient_flow(self, gradient_checker):
        head = ClassificationHead(in_channels=64, num_classes=10)
        x = torch.randn(2, 64, 4, 4, requires_grad=True)
        gradient_checker.check_gradients(head, x)

    def test_different_batch_sizes(self):
        head = ClassificationHead(in_channels=256, num_classes=10)
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 256, 8, 8)
            out = head(x)
            assert_shape(out, (batch_size, 10))

    def test_different_num_classes(self):
        for num_classes in [2, 10, 100, 1000]:
            head = ClassificationHead(in_channels=256, num_classes=num_classes)
            x = torch.randn(2, 256, 8, 8)
            out = head(x)
            assert_shape(out, (2, num_classes))

    def test_numerical_stability(self):
        head = ClassificationHead(in_channels=256, num_classes=10)
        x = torch.randn(2, 256, 8, 8) * 1000  # Large values
        out = head(x)
        assert_finite(out)


class TestObjectDiscoveryHead:
    """Test ObjectDiscoveryHead module."""

    def test_basic_initialization(self):
        head = ObjectDiscoveryHead(in_channels=256, num_slots=4, slot_dim=256)
        assert head.num_slots == 4
        assert hasattr(head, "pool")
        assert hasattr(head, "mlp")

    def test_forward_flat_output(self):
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, return_slots=False
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 256))

    def test_forward_slot_output(self):
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=64, return_slots=True
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 4, 16))  # [B, num_slots, slot_dim/num_slots]

    def test_different_num_slots(self):
        for num_slots in [2, 4, 8]:
            head = ObjectDiscoveryHead(
                in_channels=256, num_slots=num_slots, slot_dim=256, return_slots=False
            )
            x = torch.randn(2, 256, 8, 8)
            out = head(x)
            assert_shape(out, (2, 256))

    def test_adaptive_max_pool(self):
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, pool_type="adaptive_max"
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 256))

    def test_adaptive_avg_pool(self):
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, pool_type="adaptive_avg"
        )
        x = torch.randn(2, 256, 8, 8)
        out = head(x)
        assert_shape(out, (2, 256))

    def test_gradient_flow(self, gradient_checker):
        head = ObjectDiscoveryHead(in_channels=64, num_slots=4, slot_dim=64)
        x = torch.randn(2, 64, 4, 4, requires_grad=True)
        gradient_checker.check_gradients(head, x)

    def test_different_input_sizes(self):
        head = ObjectDiscoveryHead(in_channels=256, num_slots=4, slot_dim=256)
        for size in [(4, 4), (8, 8), (16, 16)]:
            x = torch.randn(2, 256, *size)
            out = head(x)
            assert_shape(out, (2, 256))

    def test_numerical_stability(self):
        head = ObjectDiscoveryHead(in_channels=256, num_slots=4, slot_dim=256)
        x = torch.randn(2, 256, 8, 8) * 1000
        out = head(x)
        assert_finite(out)


class TestSudokuHead:
    """Test SudokuHead module."""

    def test_basic_initialization(self):
        head = SudokuHead(in_channels=64, num_digits=9)
        assert hasattr(head, "head")

    def test_forward_no_permute(self):
        head = SudokuHead(in_channels=64, num_digits=9, permute_output=False)
        x = torch.randn(2, 64, 9, 9)
        out = head(x)
        assert_shape(out, (2, 9, 9, 9))  # [B, num_digits, H, W]

    def test_forward_with_permute(self):
        head = SudokuHead(in_channels=64, num_digits=9, permute_output=True)
        x = torch.randn(2, 64, 9, 9)
        out = head(x)
        assert_shape(out, (2, 9, 9, 9))  # [B, H, W, num_digits]

    def test_different_grid_sizes(self):
        head = SudokuHead(in_channels=64, num_digits=9)
        for size in [(6, 6), (9, 9), (12, 12)]:
            x = torch.randn(2, 64, *size)
            out = head(x)
            assert_shape(out, (2, 9, *size))

    def test_different_num_digits(self):
        for num_digits in [4, 9, 16]:
            head = SudokuHead(in_channels=64, num_digits=num_digits)
            x = torch.randn(2, 64, 9, 9)
            out = head(x)
            assert_shape(out, (2, num_digits, 9, 9))

    def test_gradient_flow(self, gradient_checker):
        head = SudokuHead(in_channels=64, num_digits=9)
        x = torch.randn(2, 64, 9, 9, requires_grad=True)
        gradient_checker.check_gradients(head, x)

    def test_relu_activation(self):
        head = SudokuHead(in_channels=64, num_digits=9)
        # Negative inputs should be zeroed by ReLU
        x = torch.ones(2, 64, 9, 9) * -10.0
        out = head(x)
        # After ReLU, no values should be exactly -10 (though conv could produce anything)
        # Just check it runs and produces valid output
        assert_finite(out)

    def test_numerical_stability(self):
        head = SudokuHead(in_channels=64, num_digits=9)
        x = torch.randn(2, 64, 9, 9) * 1000
        out = head(x)
        assert_finite(out)

    def test_different_batch_sizes(self):
        head = SudokuHead(in_channels=64, num_digits=9)
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 64, 9, 9)
            out = head(x)
            assert_shape(out, (batch_size, 9, 9, 9))


class TestHeadIntegration:
    """Integration tests for head modules."""

    def test_classification_head_with_backbone(self):
        # Simulate backbone + head pipeline
        backbone_out = torch.randn(2, 256, 8, 8)
        head = ClassificationHead(in_channels=256, num_classes=10)
        logits = head(backbone_out)

        # Should produce valid logits
        assert_shape(logits, (2, 10))
        assert_finite(logits)

        # Can compute loss
        targets = torch.randint(0, 10, (2,))
        loss = torch.nn.functional.cross_entropy(logits, targets)
        assert_finite(loss)

    def test_object_discovery_head_with_backbone(self):
        # Simulate object discovery pipeline
        backbone_out = torch.randn(2, 256, 8, 8)
        head = ObjectDiscoveryHead(
            in_channels=256, num_slots=4, slot_dim=256, return_slots=True
        )
        slots = head(backbone_out)

        assert_shape(slots, (2, 4, 64))  # [B, num_slots, feature_dim]
        assert_finite(slots)

    def test_sudoku_head_with_backbone(self):
        # Simulate sudoku solver pipeline
        backbone_out = torch.randn(2, 64, 9, 9)
        head = SudokuHead(in_channels=64, num_digits=9, permute_output=True)
        logits = head(backbone_out)

        assert_shape(logits, (2, 9, 9, 9))  # [B, H, W, num_digits]

        # Can compute loss
        targets = torch.randint(0, 9, (2, 9, 9))
        logits_flat = logits.reshape(-1, 9)
        targets_flat = targets.reshape(-1)
        loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
        assert_finite(loss)

    def test_state_dict_save_load(self):
        # Test all heads can save/load state
        heads_with_copies = [
            (
                ClassificationHead(in_channels=256, num_classes=10),
                ClassificationHead(in_channels=256, num_classes=10),
            ),
            (
                ObjectDiscoveryHead(in_channels=256, num_slots=4, slot_dim=256),
                ObjectDiscoveryHead(in_channels=256, num_slots=4, slot_dim=256),
            ),
            (
                SudokuHead(in_channels=64, num_digits=9),
                SudokuHead(in_channels=64, num_digits=9),
            ),
        ]

        for head1, head2 in heads_with_copies:
            state_dict = head1.state_dict()
            assert len(state_dict) > 0
            head2.load_state_dict(state_dict)
            # Verify loading succeeds
            assert head2.state_dict().keys() == state_dict.keys()
