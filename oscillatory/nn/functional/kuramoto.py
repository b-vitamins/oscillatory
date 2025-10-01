"""Kuramoto oscillator dynamics for neural networks.

This module implements the core mathematical operations for Adaptive Kuramoto
Oscillatory Recurrent Networks (AKOrN), including oscillator normalization,
tangent space projection, and coupled oscillator dynamics.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor


__all__ = [
    "kuramoto_step",
    "normalize_oscillators",
    "normalize_oscillators1d",
    "normalize_oscillators2d",
    "normalize_oscillators3d",
]


def _normalize_reshaped(x_reshaped: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize oscillators along dimension 2 (the oscillator dimension).

    Args:
        x_reshaped: Tensor with oscillator dimension at index 2
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    norm_sq = x_reshaped.pow(2).sum(dim=2, keepdim=True)
    return x_reshaped * norm_sq.clamp(min=eps).rsqrt()


def normalize_oscillators2d(
    x: Tensor,
    n_oscillators: int,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize 2D oscillator states to unit sphere.

    Args:
        x: Input tensor of shape (B, C, H, W)
        n_oscillators: Number of oscillators per group (C must be divisible by this)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.shape

    if C % n_oscillators != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by n_oscillators ({n_oscillators})"
        )

    x_reshaped = x.view(B, C // n_oscillators, n_oscillators, H, W)
    x_normalized = _normalize_reshaped(x_reshaped, eps)
    return x_normalized.view(B, C, H, W)


def normalize_oscillators1d(
    x: Tensor,
    n_oscillators: int,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize 1D oscillator states to unit sphere.

    Args:
        x: Input tensor of shape (B, C, L)
        n_oscillators: Number of oscillators per group (C must be divisible by this)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of shape (B, C, L)
    """
    B, C, L = x.shape

    if C % n_oscillators != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by n_oscillators ({n_oscillators})"
        )

    x_reshaped = x.view(B, C // n_oscillators, n_oscillators, L)
    x_normalized = _normalize_reshaped(x_reshaped, eps)
    return x_normalized.view(B, C, L)


def normalize_oscillators3d(
    x: Tensor,
    n_oscillators: int,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize 3D oscillator states to unit sphere.

    Args:
        x: Input tensor of shape (B, C, D, H, W)
        n_oscillators: Number of oscillators per group (C must be divisible by this)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of shape (B, C, D, H, W)
    """
    B, C, D, H, W = x.shape

    if C % n_oscillators != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by n_oscillators ({n_oscillators})"
        )

    x_reshaped = x.view(B, C // n_oscillators, n_oscillators, D, H, W)
    x_normalized = _normalize_reshaped(x_reshaped, eps)
    return x_normalized.view(B, C, D, H, W)


def normalize_oscillators(
    x: Tensor,
    n_oscillators: int,
    eps: float = 1e-12,
) -> Tensor:
    """Normalize oscillator states to unit sphere (dimension-agnostic version).

    Args:
        x: Input tensor of shape (*batch_dims, C)
        n_oscillators: Number of oscillators per group (C must be divisible by this)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    *prefix_shape, C = x.shape

    if C % n_oscillators != 0:
        raise ValueError(
            f"Number of channels ({C}) must be divisible by n_oscillators ({n_oscillators})"
        )

    x_reshaped = x.view(*prefix_shape, C // n_oscillators, n_oscillators)
    norm_sq = x_reshaped.pow(2).sum(dim=-1, keepdim=True)
    x_normalized = x_reshaped * norm_sq.clamp(min=eps).rsqrt()
    return x_normalized.view(*prefix_shape, C)


def _project_to_tangent_space(
    y_reshaped: Tensor,
    x_reshaped: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Project y onto tangent space of unit sphere at x.

    Computes: y - <y, x> * x (Riemannian projection onto tangent space)

    Args:
        y_reshaped: Update vector with oscillator dimension at index 2
        x_reshaped: Current state with oscillator dimension at index 2

    Returns:
        Tuple of (projected update, element-wise similarity)
    """
    full_similarity = y_reshaped * x_reshaped
    inner_product = full_similarity.sum(dim=2, keepdim=True)
    y_projected = torch.addcmul(y_reshaped, inner_product, x_reshaped, value=-1)
    return y_projected, full_similarity


def _apply_rotational_dynamics2d(
    x_reshaped: Tensor,
    omega: Tensor,
    n_oscillators: int,
) -> Tensor:
    """Apply SO(2) rotational dynamics to 2D oscillator pairs.

    Args:
        x_reshaped: Oscillator states of shape (B, n_groups, n_oscillators, H, W)
        omega: Rotation frequency, shape (n_groups,) or scalar
        n_oscillators: Number of oscillators per group (must be even)

    Returns:
        Rotated oscillator states with same shape as input
    """
    if n_oscillators % 2 != 0:
        raise NotImplementedError(
            "Rotational dynamics for odd n_oscillators not yet supported"
        )

    B, n_groups, _, H, W = x_reshaped.shape

    if omega.numel() == 1:
        omega = omega.expand(n_groups)
    elif omega.shape[0] != n_groups:
        raise ValueError(
            f"omega must have shape ({n_groups},) or be scalar, got {omega.shape}"
        )

    x_pairs = x_reshaped.view(B, n_groups, n_oscillators // 2, 2, H, W)
    omega_reshaped = omega.view(1, n_groups, 1, 1, 1)

    x_rotated = torch.empty_like(x_pairs)
    x_rotated[:, :, :, 0, :, :] = omega_reshaped * x_pairs[:, :, :, 1, :, :]
    x_rotated[:, :, :, 1, :, :] = -omega_reshaped * x_pairs[:, :, :, 0, :, :]

    return x_rotated.view(B, n_groups, n_oscillators, H, W)


def _apply_rotational_dynamics1d(
    x_reshaped: Tensor,
    omega: Tensor,
    n_oscillators: int,
) -> Tensor:
    """Apply SO(2) rotational dynamics to 1D oscillator pairs."""
    if n_oscillators % 2 != 0:
        raise NotImplementedError(
            "Rotational dynamics for odd n_oscillators not yet supported"
        )

    B, n_groups, _, L = x_reshaped.shape

    if omega.numel() == 1:
        omega = omega.expand(n_groups)
    elif omega.shape[0] != n_groups:
        raise ValueError(
            f"omega must have shape ({n_groups},) or be scalar, got {omega.shape}"
        )

    x_pairs = x_reshaped.view(B, n_groups, n_oscillators // 2, 2, L)
    omega_reshaped = omega.view(1, n_groups, 1, 1)

    x_rotated = torch.empty_like(x_pairs)
    x_rotated[:, :, :, 0, :] = omega_reshaped * x_pairs[:, :, :, 1, :]
    x_rotated[:, :, :, 1, :] = -omega_reshaped * x_pairs[:, :, :, 0, :]

    return x_rotated.view(B, n_groups, n_oscillators, L)


def _apply_rotational_dynamics3d(
    x_reshaped: Tensor,
    omega: Tensor,
    n_oscillators: int,
) -> Tensor:
    """Apply SO(2) rotational dynamics to 3D oscillator pairs."""
    if n_oscillators % 2 != 0:
        raise NotImplementedError(
            "Rotational dynamics for odd n_oscillators not yet supported"
        )

    B, n_groups, _, D, H, W = x_reshaped.shape

    if omega.numel() == 1:
        omega = omega.expand(n_groups)
    elif omega.shape[0] != n_groups:
        raise ValueError(
            f"omega must have shape ({n_groups},) or be scalar, got {omega.shape}"
        )

    x_pairs = x_reshaped.view(B, n_groups, n_oscillators // 2, 2, D, H, W)
    omega_reshaped = omega.view(1, n_groups, 1, 1, 1, 1)

    x_rotated = torch.empty_like(x_pairs)
    x_rotated[:, :, :, 0, :, :, :] = omega_reshaped * x_pairs[:, :, :, 1, :, :, :]
    x_rotated[:, :, :, 1, :, :, :] = -omega_reshaped * x_pairs[:, :, :, 0, :, :, :]

    return x_rotated.view(B, n_groups, n_oscillators, D, H, W)


def kuramoto_step(
    x: Tensor,
    coupling: Tensor,
    stimulus: Tensor,
    n_oscillators: int,
    omega: Optional[Tensor] = None,
    step_size: float = 1.0,
    apply_projection: bool = True,
    normalize: bool = True,
    spatial_ndim: int = 2,
) -> Tuple[Tensor, Tensor]:
    """Perform one step of Kuramoto oscillator dynamics.

    Implements: x_{t+1} = normalize(x_t + step_size * (omega_term + proj(coupling + stimulus)))

    Args:
        x: Current oscillator state
        coupling: Coupling term from connectivity (e.g., attention or convolution)
        stimulus: External stimulus/input signal
        n_oscillators: Number of oscillators per group (must divide channels evenly)
        omega: Optional rotational frequency (natural frequency of oscillators)
        step_size: Integration step size (gamma parameter)
        apply_projection: Whether to project updates onto tangent space
        normalize: Whether to normalize result to unit sphere
        spatial_ndim: Number of spatial dimensions (1, 2, or 3)

    Returns:
        Tuple of (updated state, energy)
    """
    y = coupling + stimulus
    B, C = x.shape[0], x.shape[1]
    n_groups = C // n_oscillators

    if spatial_ndim == 1:
        L = x.shape[2]
        x_reshaped = x.view(B, n_groups, n_oscillators, L)
        y_reshaped = y.view(B, n_groups, n_oscillators, L)
        norm_fn = _normalize_reshaped
        rot_fn = _apply_rotational_dynamics1d
        final_shape = (B, C, L)

    elif spatial_ndim == 2:
        H, W = x.shape[2], x.shape[3]
        x_reshaped = x.view(B, n_groups, n_oscillators, H, W)
        y_reshaped = y.view(B, n_groups, n_oscillators, H, W)
        norm_fn = _normalize_reshaped
        rot_fn = _apply_rotational_dynamics2d
        final_shape = (B, C, H, W)

    elif spatial_ndim == 3:
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x_reshaped = x.view(B, n_groups, n_oscillators, D, H, W)
        y_reshaped = y.view(B, n_groups, n_oscillators, D, H, W)
        norm_fn = _normalize_reshaped
        rot_fn = _apply_rotational_dynamics3d
        final_shape = (B, C, D, H, W)
    else:
        raise ValueError(f"spatial_ndim must be 1, 2, or 3, got {spatial_ndim}")

    # Apply rotational dynamics if omega provided
    if omega is not None:
        omega_term_reshaped = rot_fn(x_reshaped, omega, n_oscillators)
    else:
        omega_term_reshaped = torch.zeros_like(x_reshaped)

    # Project onto tangent space if requested
    if apply_projection:
        y_projected_reshaped, full_similarity_reshaped = _project_to_tangent_space(
            y_reshaped, x_reshaped
        )
    else:
        full_similarity_reshaped = y_reshaped * x_reshaped
        y_projected_reshaped = y_reshaped

    # Update and normalize
    delta_x_reshaped = omega_term_reshaped + y_projected_reshaped
    x_new_reshaped = torch.add(x_reshaped, delta_x_reshaped, alpha=step_size)

    if normalize:
        x_new_reshaped = norm_fn(x_new_reshaped)

    x_new = x_new_reshaped.view(*final_shape)
    full_similarity = full_similarity_reshaped.view(*final_shape)
    energy = -full_similarity.view(B, -1).sum(dim=-1)

    return x_new, energy
