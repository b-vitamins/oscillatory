"""Kuramoto Orientation Diffusion Model (KODM).

Treats pixels as phase angles on S¹ circle, combines Gaussian noise with
Kuramoto mean-field coupling in diffusion process.

Reference:
    Song et al. "Kuramoto Orientation Diffusion Models." NeurIPS 2025.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..functional.diffusion import (
    cossin_to_angle_score,
    mean_field_phase,
    phase_modulate,
    wrapped_gaussian_score,
)


class _SelfAttention(nn.Module):
    """Self-attention block for UNet."""

    def __init__(self, channels: int, size: int):
        super().__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attn
        x = x + self.ff(x)
        return x.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class _DoubleConv(nn.Module):
    """Double convolution block with optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        mid_channels = mid_channels or out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class _Down(nn.Module):
    """Downsampling block with time embedding."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, in_channels, residual=True),
            _DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].expand(
            -1, -1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class _Up(nn.Module):
    """Upsampling block with skip connections and time embedding."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int = 256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            _DoubleConv(in_channels, in_channels, residual=True),
            _DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x: Tensor, skip_x: Tensor, t: Tensor) -> Tensor:
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].expand(
            -1, -1, x.shape[-2], x.shape[-1]
        )
        return x + emb


class KODMUNet(nn.Module):
    """UNet for circular orientation diffusion.

    Handles circular (cos, sin) representation internally. Input and output
    are angles in radians, automatically converted to/from (cos, sin) pairs.

    Args:
        channels: Number of input/output channels (angles)
        time_dim: Dimension of time embedding
        img_size: Spatial size of input images
    """

    def __init__(self, channels: int = 3, time_dim: int = 256, img_size: int = 32):
        super().__init__()
        self.channels = channels
        self.time_dim = time_dim
        self.img_size = img_size

        # Encoder (input: 2*channels for cos/sin)
        self.inc = _DoubleConv(channels * 2, 64)
        self.down1 = _Down(64, 128, time_dim)
        self.sa1 = _SelfAttention(128, img_size // 2)
        self.down2 = _Down(128, 256, time_dim)
        self.sa2 = _SelfAttention(256, img_size // 4)
        self.down3 = _Down(256, 256, time_dim)
        self.sa3 = _SelfAttention(256, img_size // 8)

        # Bottleneck
        self.bot1 = _DoubleConv(256, 512)
        self.bot2 = _DoubleConv(512, 512)
        self.bot3 = _DoubleConv(512, 256)

        # Decoder
        self.up1 = _Up(512, 128, time_dim)
        self.sa4 = _SelfAttention(128, img_size // 4)
        self.up2 = _Up(256, 64, time_dim)
        self.sa5 = _SelfAttention(64, img_size // 2)
        self.up3 = _Up(128, 64, time_dim)
        self.sa6 = _SelfAttention(64, img_size)

        # Output: 2*channels for cos/sin gradients
        self.outc = nn.Conv2d(64, channels * 2, kernel_size=1)

    def _pos_encoding(self, t: Tensor, channels: int) -> Tensor:
        """Sinusoidal positional encoding for time."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=t.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Angles [B, C, H, W] in radians
            t: Timesteps [B] as integers

        Returns:
            Score in (cos, sin) space [B, 2*C, H, W]
        """
        # Convert angles to (cos, sin) representation
        x = torch.cat([x.cos(), x.sin()], dim=1)

        # Time embedding
        t = self._pos_encoding(t.unsqueeze(-1).float(), self.time_dim)

        # Encoder
        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        # Bottleneck
        x4 = self.bot3(self.bot2(self.bot1(x4)))

        # Decoder
        x = self.sa4(self.up1(x4, x3, t))
        x = self.sa5(self.up2(x, x2, t))
        x = self.sa6(self.up3(x, x1, t))

        return self.outc(x)


class KODM(nn.Module):
    """Kuramoto Orientation Diffusion Model.

    Diffusion process combining Gaussian noise with Kuramoto mean-field
    coupling for circular variables on S¹.

    Args:
        num_timesteps: Number of diffusion steps
        noise_start: Initial noise level
        noise_end: Final noise level
        coupling_start: Initial coupling strength
        coupling_end: Final coupling strength
        img_size: Spatial size of images
        ref_phase: Reference phase for Kuramoto coupling
    """

    def __init__(
        self,
        num_timesteps: int = 100,
        noise_start: float = 1e-4,
        noise_end: float = 0.1,
        coupling_start: float = 3e-5,
        coupling_end: float = 0.03,
        img_size: int = 32,
        ref_phase: float = 0.0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.img_size = img_size
        self.ref_phase = ref_phase

        # Register schedules as buffers
        self.register_buffer(
            "noise_schedule",
            torch.linspace(noise_start, noise_end, num_timesteps).sqrt(),
        )
        self.register_buffer(
            "coupling_schedule",
            torch.linspace(coupling_start, coupling_end, num_timesteps),
        )

    noise_schedule: Tensor
    coupling_schedule: Tensor

    def compute_kuramoto_coupling(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute Kuramoto mean-field coupling term.

        Args:
            x: Phase angles [B, C, H, W]

        Returns:
            Tuple of (noise, coupling, order_parameter, order_phase):
                - noise: Gaussian noise sample [B, C, H, W]
                - coupling: Kuramoto coupling term [B, C, H, W]
                - order_parameter: R ∈ [0,1]
                - order_phase: Φ ∈ [-π, π]
        """
        noise = torch.randn_like(x)
        order_parameter, order_phase = mean_field_phase(x)

        # Kuramoto coupling: R·sin(Φ - θ) + γ·sin(θ₀ - θ)
        coupling = order_parameter * torch.sin(order_phase - x) + 1.5 * torch.sin(
            self.ref_phase - x
        )

        return noise, coupling, order_parameter, order_phase

    def add_noise(self, x0: Tensor, t: Tensor) -> Tensor:
        """Forward diffusion: add noise and coupling for t steps.

        Args:
            x0: Clean angles [B, C, H, W]
            t: Number of steps [B] as integers

        Returns:
            Noisy angles [B, C, H, W]
        """
        x = x0.clone()

        for i in range(int(t.max().item())):
            mask = (t > i).float()[:, None, None, None]
            noise, coupling, _, _ = self.compute_kuramoto_coupling(x)
            x = x + mask * (
                self.coupling_schedule[i] * coupling + self.noise_schedule[i] * noise
            )
            x = phase_modulate(x)

        return x

    def step(
        self,
        model_output: Tensor,
        t: int,
        sample: Tensor,
    ) -> Tensor:
        """Single reverse diffusion step.

        Args:
            model_output: Model prediction [B, 2*C, H, W] in (cos, sin) space
            t: Current timestep (scalar)
            sample: Current noisy angles [B, C, H, W]

        Returns:
            Denoised angles [B, C, H, W]
        """
        # Convert model output to angle space score
        score = cossin_to_angle_score(model_output, sample)

        # Compute coupling term
        noise, coupling, _, _ = self.compute_kuramoto_coupling(sample)

        # Reverse step: x_{t-1} = x_t - ε_coupling·coupling + σ²·score + σ·noise
        x = (
            sample
            - self.coupling_schedule[t] * coupling
            + self.noise_schedule[t] ** 2 * score
            + self.noise_schedule[t] * noise
        )

        return phase_modulate(x)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        batch_size: int,
        channels: int = 3,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Generate samples via reverse diffusion.

        Args:
            model: Denoising model (e.g., KODMUNet)
            batch_size: Number of samples
            channels: Number of angle channels
            device: Device to generate on

        Returns:
            Generated angles [B, C, H, W]
        """
        model.eval()
        device = device or next(model.parameters()).device

        # Initialize from Von Mises distribution
        # Original formula: ((0.5 + 1.5) * coupling[-1]) / (0.5 * noise[-1]^2)
        concentration = (2.0 * self.coupling_schedule[-1]) / (
            0.5 * self.noise_schedule[-1] ** 2
        )
        x = (
            torch.distributions.VonMises(
                torch.tensor(0.0, device=device), concentration
            )
            .sample((batch_size, channels, self.img_size, self.img_size))
            .squeeze(-1)
            .to(device)
        )

        # Reverse diffusion
        for i in reversed(range(1, self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            model_output = model(x, t)
            x = self.step(model_output, i, x)

        return x


class NestedScoreMatchingLoss(nn.Module):
    """Nested score matching loss for KODM training.

    Averages over multiple noise realizations at each step to reduce variance.

    Args:
        scheduler: KODM diffusion scheduler
        num_samples: Number of noise realizations to average
    """

    def __init__(self, scheduler: KODM, num_samples: int = 5):
        super().__init__()
        self.scheduler = scheduler
        self.num_samples = num_samples

    def forward(self, model: nn.Module, x0: Tensor, t: Tensor) -> Tensor:
        """Compute nested score matching loss.

        Args:
            model: Denoising model
            x0: Clean angles [B, C, H, W]
            t: Timesteps [B] as integers

        Returns:
            Scalar loss
        """
        # Forward diffusion to timestep t
        xt = self.scheduler.add_noise(x0, t)

        # Get schedule values (move to same device as t)
        device = t.device
        coupling_sched = self.scheduler.coupling_schedule.to(device)[t][
            :, None, None, None
        ]
        noise_sched = self.scheduler.noise_schedule.to(device)[t][:, None, None, None]
        noise_sched_next = self.scheduler.noise_schedule.to(device)[t + 1][
            :, None, None, None
        ]

        # Average over noise realizations
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        for _ in range(self.num_samples):
            noise, coupling, _, _ = self.scheduler.compute_kuramoto_coupling(xt)

            # Forward step: F(x_t) = x_t + ε·coupling
            Fxt = phase_modulate(xt + coupling_sched * coupling)

            # Add noise: x_{t+1} = F(x_t) + σ·η
            xt_plus = phase_modulate(Fxt + noise_sched * noise)

            # Predict score at x_{t+1}
            model_output = model(xt_plus, t + 1)
            score_pred = cossin_to_angle_score(model_output, xt_plus)

            # Target score from wrapped Gaussian
            score_target = wrapped_gaussian_score(xt_plus, Fxt, noise_sched)

            # Weighted MSE
            weight = noise_sched_next**2 / 2
            loss = loss + F.mse_loss(score_pred * weight, score_target * weight)

        return loss / self.num_samples
