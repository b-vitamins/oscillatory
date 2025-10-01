"""Circular diffusion operations for orientation fields."""

import math
from typing import Tuple

import torch
from torch import Tensor


def phase_modulate(x: Tensor) -> Tensor:
    """Wrap angles to [-π, π] range.

    Args:
        x: Angles in radians

    Returns:
        Wrapped angles in [-π, π]
    """
    return (x + math.pi) % (2 * math.pi) - math.pi


def wrapped_gaussian_score(
    y: Tensor,
    mu: Tensor,
    sigma: Tensor | float,
    K: int = 3,
) -> Tensor:
    """Score function for wrapped Gaussian distribution on circle.

    The wrapped Gaussian sums Gaussians centered at mu + 2πk for k ∈ [-K, K],
    wrapping the real line onto the circle.

    Args:
        y: Target angles in [-π, π]
        mu: Mean angles in [-π, π]
        sigma: Standard deviation (scalar or per-sample tensor)
        K: Number of wraps to sum (default: 3)

    Returns:
        Score (gradient of log probability) at y
    """
    k = torch.arange(-K, K + 1, device=y.device, dtype=torch.float32).view(
        1, 1, 1, 1, -1
    )

    diff = y.unsqueeze(-1) - (mu.unsqueeze(-1) - 2 * math.pi * k)

    if isinstance(sigma, (int, float)):
        sigma_sq = sigma**2
        exp_terms = torch.exp(-0.5 * diff**2 / sigma_sq)
    else:
        sigma_sq = sigma.unsqueeze(-1) ** 2
        exp_terms = torch.exp(-0.5 * diff**2 / sigma_sq)

    denom = exp_terms.sum(dim=-1)
    numer = (diff * exp_terms).sum(dim=-1)

    if isinstance(sigma, (int, float)):
        return -(1.0 / sigma_sq) * (numer / denom)
    else:
        return -(1.0 / sigma**2) * (numer / denom)


def mean_field_phase(x: Tensor, keepdim: bool = True) -> Tuple[Tensor, Tensor]:
    """Compute Kuramoto order parameter and mean phase.

    Maps phases to complex unit circle, computes mean, extracts magnitude
    (order parameter R ∈ [0,1]) and angle (mean phase Φ).

    Args:
        x: Phase angles [B, C, H, W] or [B, C, ...]
        keepdim: Keep spatial dimensions in output

    Returns:
        Tuple of (order_parameter, mean_phase):
            - order_parameter: Magnitude R ∈ [0,1]
            - mean_phase: Angle Φ ∈ [-π, π]
    """
    complex_phases = torch.complex(x.cos(), x.sin())

    if x.dim() == 4:
        mean_vector = complex_phases.mean(dim=[1, 2, 3], keepdim=keepdim)
    else:
        mean_vector = complex_phases.mean(dim=list(range(1, x.dim())), keepdim=keepdim)

    order_parameter = mean_vector.abs()
    mean_phase = torch.atan2(mean_vector.imag, mean_vector.real)

    return order_parameter, mean_phase


def cossin_to_angle_score(score: Tensor, angle: Tensor) -> Tensor:
    """Convert score from (cos, sin) space to angle space via chain rule.

    Given ∂L/∂cos and ∂L/∂sin, computes ∂L/∂θ using:
    ∂L/∂θ = -∂L/∂cos · sin(θ) + ∂L/∂sin · cos(θ)

    Args:
        score: Network output [B, 2C, H, W] where first C channels are
               ∂/∂cos and last C channels are ∂/∂sin
        angle: Angles [B, C, H, W] in radians

    Returns:
        Score in angle space [B, C, H, W]
    """
    C = angle.size(1)
    return -score[:, :C] * torch.sin(angle) + score[:, C:] * torch.cos(angle)
