"""Utility functions for neural network operations."""

import math
import torch
from torch import Tensor


__all__ = ["positional_encoding_2d"]


def positional_encoding_2d(d_model: int, height: int, width: int) -> Tensor:
    """Generate 2D sinusoidal positional encoding.

    Args:
        d_model: Embedding dimension (must be divisible by 4)
        height: Height of spatial grid
        width: Width of spatial grid

    Returns:
        Positional encoding tensor of shape (d_model, height, width)

    Note:
        First half of channels encode width position, second half encode height.
        Uses standard Transformer-style sinusoidal encoding.
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {d_model}")

    d_model_half = d_model // 2

    # Compute frequency scaling: 1 / (10000^(2i/d))
    div_term = torch.exp(
        -(math.log(10000.0) / d_model_half)
        * torch.arange(0, d_model_half, 2, dtype=torch.float32)
    ).reshape(1, -1)

    pos_h = torch.arange(height, dtype=torch.float32).reshape(-1, 1)
    pos_w = torch.arange(width, dtype=torch.float32).reshape(-1, 1)

    # Compute sin/cos for each position
    sin_w = torch.sin(pos_w @ div_term)
    cos_w = torch.cos(pos_w @ div_term)
    sin_h = torch.sin(pos_h @ div_term)
    cos_h = torch.cos(pos_h @ div_term)

    # Pre-allocate output
    pe = torch.empty(d_model, height, width, dtype=torch.float32)

    # Interleave sin/cos: first half for width, second half for height
    pe[0:d_model_half:2] = sin_w.T[:, None, :]
    pe[1:d_model_half:2] = cos_w.T[:, None, :]
    pe[d_model_half::2] = sin_h.T[:, :, None]
    pe[d_model_half + 1 :: 2] = cos_h.T[:, :, None]

    return pe


# Deprecated alias for backward compatibility
positionalencoding2d = positional_encoding_2d
