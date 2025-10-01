"""Group-theoretic attention operations."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "make_2d_coord",
    "make_so2_matrices",
    "apply_group_action",
    "apply_group_action_qkv",
    "apply_group_action_qk",
    "embed_block_diagonal",
]


def make_2d_coord(
    height: int,
    width: int,
    normalize: bool = False,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Create 2D coordinate grid.

    Args:
        height: Grid height.
        width: Grid width.
        normalize: Normalize coordinates to [0, 1].
        device: Device for tensor.
        dtype: Data type for tensor.

    Returns:
        Coordinate grid [H, W, 2].
    """
    dtype = dtype or torch.float32

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    if normalize:
        y = y / height
        x = x / width

    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([y_grid, x_grid], dim=-1)

    return coords


def make_so2_matrices(
    coords: Tensor,
    n_freqs: int,
    base: float = 10000.0,
) -> Tensor:
    """Generate SO(2) rotation matrices for positional encoding.

    Args:
        coords: Coordinate tensor [..., dim].
        n_freqs: Number of frequency bands.
        base: Base for geometric frequency progression.

    Returns:
        SO(2) matrices [..., dim, n_freqs, 2, 2].
    """
    *batch_dims, dim = coords.shape
    device = coords.device
    dtype = coords.dtype

    exponent = torch.arange(0, 2 * n_freqs, 2, device=device, dtype=dtype)
    freqs = torch.exp(-exponent * (math.log(base) / (2 * n_freqs)))

    angles_list = []
    for d in range(dim):
        angles = torch.einsum("...i,j->...j", coords[..., d : d + 1], freqs)
        angles_list.append(angles)

    matrices = []
    for angles in angles_list:
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        mat = torch.stack(
            [
                torch.stack([cos_angles, -sin_angles], dim=-1),
                torch.stack([sin_angles, cos_angles], dim=-1),
            ],
            dim=-2,
        )
        matrices.append(mat)

    result = torch.stack(matrices, dim=-3)

    return result


@torch.jit.script
def apply_group_action(rep: Tensor, x: Tensor) -> Tensor:
    """Apply group representation to feature vector.

    Args:
        rep: Representation matrices [..., 2, 2].
        x: Feature vectors [..., D] where D is even.

    Returns:
        Rotated features [..., D].
    """
    shape = x.shape
    d = rep.shape[-1]

    x_paired = x.unflatten(-1, (-1, d))
    rep_expanded = rep.unsqueeze(0).unsqueeze(0)
    x_expanded = x_paired.unsqueeze(-1)
    rotated = (rep_expanded @ x_expanded).squeeze(-1)

    return rotated.view(shape)


@torch.jit.script
def apply_group_action_qkv(
    rep: Tensor, q: Tensor, k: Tensor, v: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Apply group action to query, key, and value.

    Args:
        rep: Representation matrices.
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.

    Returns:
        Tuple of (rotated_q, rotated_k, rotated_v).
    """
    return (
        apply_group_action(rep, q),
        apply_group_action(rep, k),
        apply_group_action(rep, v),
    )


@torch.jit.script
def apply_group_action_qk(rep: Tensor, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
    """Apply group action to query and key only.

    Args:
        rep: Representation matrices.
        q: Query tensor.
        k: Key tensor.

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    return (
        apply_group_action(rep, q),
        apply_group_action(rep, k),
    )


def embed_block_diagonal(
    matrices: Tensor,
    n_blocks: int,
) -> Tensor:
    """Embed 2x2 matrices as blocks in larger block-diagonal matrices.

    Args:
        matrices: Input matrices [HW, D/2, 2, 2].
        n_blocks: Number of blocks per output matrix.

    Returns:
        Block-diagonal matrices [HW, D/(2*n_blocks), 2*n_blocks, 2*n_blocks].
    """
    hw, d_half, h, w = matrices.shape
    assert h == 2 and w == 2, "Input must be 2x2 matrices"
    assert d_half % n_blocks == 0, (
        f"D/2={d_half} must be divisible by n_blocks={n_blocks}"
    )

    d_out = d_half // n_blocks
    size_out = 2 * n_blocks

    device = matrices.device
    dtype = matrices.dtype
    output = torch.zeros(
        (hw, d_out, size_out, size_out),
        device=device,
        dtype=dtype,
    )

    for t in range(hw):
        for d in range(d_out):
            blocks = [matrices[t, n_blocks * d + i] for i in range(n_blocks)]
            output[t, d] = torch.block_diag(*blocks)

    return output
