"""Task-specific heads for AKOrN models."""

import torch.nn as nn


class ClassificationHead(nn.Module):
    """Simple classification head with global pooling.

    Similar to torchvision models: pool -> flatten -> linear.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pool_type: str = "adaptive_avg",
        pool_size: tuple = (1, 1),
    ):
        super().__init__()

        if pool_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d(pool_size)
        else:
            raise ValueError(f"Unknown pool_type: {pool_type}")

        self.flatten = nn.Flatten()

        pool_h, pool_w = pool_size
        flat_dim = in_channels * pool_h * pool_w

        self.classifier = nn.Sequential(nn.Linear(flat_dim, num_classes))

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class ObjectDiscoveryHead(nn.Module):
    """Object discovery head for unsupervised object detection.

    Pool -> flatten -> MLP to predict slot representations.
    """

    def __init__(
        self,
        in_channels: int,
        num_slots: int = 4,
        slot_dim: int = 256,
        pool_type: str = "adaptive_max",
        return_slots: bool = False,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.return_slots = return_slots

        if pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, num_slots * in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_slots * in_channels, slot_dim),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.mlp(x)

        if self.return_slots:
            B = x.size(0)
            x = x.view(B, self.num_slots, -1)

        return x


class SudokuHead(nn.Module):
    """Sudoku solver head.

    Simple ReLU -> Conv projection to digit logits.
    """

    def __init__(
        self,
        in_channels: int,
        num_digits: int = 9,
        permute_output: bool = False,
    ):
        super().__init__()

        self.permute_output = permute_output

        self.head = nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(in_channels, num_digits, 1, 1, 0)
        )

    def forward(self, x):
        x = self.head(x)
        if self.permute_output:
            x = x.permute(0, 2, 3, 1)
        return x
