"""AKOrN model implementations."""

from typing import List, Optional, Tuple, Union

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from oscillatory.nn.functional import kuramoto as K
from oscillatory.nn.functional.utils import positional_encoding_2d
from oscillatory.nn.modules.attention import GTAttention


__all__ = ["AKOrNHierarchical", "AKOrNDense", "AKOrNGrid", "ConvReadout"]


class _ReadoutBlock(nn.Module):
    """Readout block with optional skip connection.

    Args:
        first: Primary transformation module.
        second: Optional feedforward module for skip connection.
        third: Optional final transformation module.
    """

    def __init__(
        self,
        first: nn.Module,
        second: Optional[nn.Module] = None,
        third: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.first = first
        self.second = second
        self.third = third

    def forward(self, x: Tensor) -> Tensor:
        c = self.first(x)
        if self.second is not None and self.third is not None:
            c = c + self.second(c)
            c = self.third(c)
        return c


class ConvReadout(nn.Module):
    """Convolutional readout with oscillator norm.

    Projects oscillator state via convolution then computes L2 norm over
    oscillators: ||Conv(x)||_2 + bias

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        n_oscillators: Number of oscillators per group.
        kernel_size: Convolutional kernel size.
        stride: Convolutional stride.
        padding: Convolutional padding.
        bias: Whether to add bias after norm.
        device: Device for parameters.
        dtype: Data type for parameters.
    """

    __constants__ = ["in_channels", "out_channels", "n_oscillators"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_oscillators: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_oscillators = n_oscillators

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * n_oscillators,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            device=device,
            dtype=dtype,
        )

        if bias:
            self.bias = Parameter(torch.zeros(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        """Initialize with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv.bias, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        B, _, H, W = x.shape

        x = x.view(B, self.out_channels, self.n_oscillators, H, W)
        x = torch.linalg.norm(x, dim=2)

        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)

        return x

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"n_oscillators={self.n_oscillators}"
        )


class KuramotoBlock(nn.Module):
    """Kuramoto oscillator block with connectivity and dynamics.

    Args:
        channels: Number of channels (must be divisible by n_oscillators).
        n_oscillators: Number of oscillators per group.
        connectivity: Connectivity type ('conv' or 'attn').
        kernel_size: Kernel size for convolutional connectivity.
        n_heads: Number of attention heads for attention connectivity.
        hw: Spatial dimensions (height, width) for attention.
        use_omega: Enable natural frequency term.
        omega_init: Initial omega value.
        global_omega: Use single omega (True) or per-group omega (False).
        learnable_omega: Make omega a learnable parameter.
        use_gta: Enable Group-Theoretic Attention.
        c_norm: Normalization type for stimulus ('gn' or None).
    """

    def __init__(
        self,
        channels: int,
        n_oscillators: int,
        connectivity: str = "conv",
        kernel_size: int = 7,
        n_heads: int = 8,
        hw: Optional[Tuple[int, int]] = None,
        use_omega: bool = True,
        omega_init: float = 1.0,
        global_omega: bool = True,
        learnable_omega: bool = True,
        use_gta: bool = False,
        c_norm: Optional[str] = "gn",
        **kwargs,
    ):
        super().__init__()

        self.channels = channels
        self.n_oscillators = n_oscillators
        self.use_omega = use_omega

        if connectivity == "conv":
            self.connectivity = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=True,
            )
        elif connectivity == "attn":
            self.connectivity = GTAttention(
                embed_dim=channels,
                n_heads=n_heads,
                weight_type="conv",
                kernel_size=1,
                stride=1,
                padding=0,
                use_gta=use_gta,
                use_rope=False,
                hw=hw,
                dropout=0.0,
            )
        else:
            raise ValueError(f"Unknown connectivity: {connectivity}")

        self.c_norm = (
            nn.GroupNorm(channels // n_oscillators, channels, affine=True)
            if c_norm == "gn"
            else nn.Identity()
        )

        if use_omega:
            omega_shape = 1 if global_omega else channels // n_oscillators
            omega = torch.full((omega_shape,), omega_init)
            self.omega = Parameter(omega, requires_grad=learnable_omega)
        else:
            self.register_parameter("omega", None)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        n_timesteps: int,
        gamma: float = 1.0,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Run Kuramoto dynamics for n_timesteps iterations.

        Args:
            x: Initial oscillator state [B, C, H, W].
            c: Stimulus input [B, C, H, W].
            n_timesteps: Number of integration steps.
            gamma: Step size for integration.

        Returns:
            Tuple of (states, energies) lists per timestep.
        """
        xs = []
        es = [torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)]

        c = self.c_norm(c)
        x = K.normalize_oscillators2d(x, self.n_oscillators)

        omega = (
            self.omega.squeeze()
            if self.omega is not None and self.omega.numel() == 1
            else self.omega
        )

        for _ in range(n_timesteps):
            coupling = self.connectivity(x)

            x, energy = K.kuramoto_step(
                x=x,
                coupling=coupling,
                stimulus=c,
                n_oscillators=self.n_oscillators,
                omega=omega,
                step_size=gamma,
                apply_projection=True,
                normalize=True,
                spatial_ndim=2,
            )

            xs.append(x)
            es.append(energy)

        return xs, es


class AKOrNHierarchical(nn.Module):
    """Hierarchical AKOrN for multi-scale vision tasks.

    Args:
        input_size: Input spatial size (H, W) or single int for square.
        in_channels: Number of input channels.
        base_channels: Base number of channels.
        num_layers: Number of hierarchical layers.
        channel_multiplier: Channel expansion factor (int or per-layer list).
        spatial_stages: Number of stages with spatial downsampling.
        n_oscillators: Oscillators per group (int or per-layer list).
        n_timesteps: Timesteps per layer (int or per-layer list).
        connectivity: Connectivity type per layer ('conv', 'attn', or list).
        kernel_sizes: Kernel sizes per layer.
        readout_kernel_size: Kernel size for readout conv.
        readout_n_oscillators: Oscillators for readout (int or per-layer list).
        use_omega: Enable natural frequency term.
        omega_init: Initial omega value.
        global_omega: Use single omega (True) or per-group (False).
        learnable_omega: Make omega learnable.
        gamma: Integration step size.
        norm_type: Normalization type for feedforward blocks.
        c_norm: Normalization type for stimulus.
        use_input_norm: Apply input normalization.
        out_channels: Output channels (defaults to last layer channels).
        return_all_states: Return intermediate states and energies.
        final_head: Optional final head module.
        input_norm_stats: Optional (mean, std) tuples for input normalization.
        use_positional_encoding: Add positional encoding to stem output.
        activation_order: Activation order for feedforward ('pre' or 'post').
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 32,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 3,
        channel_multiplier: Union[int, List[int]] = 2,
        spatial_stages: int = 3,
        n_oscillators: Union[int, List[int]] = 4,
        n_timesteps: Union[int, List[int]] = 3,
        connectivity: Union[str, List[str]] = "conv",
        kernel_sizes: Optional[List[int]] = None,
        readout_kernel_size: int = 3,
        readout_n_oscillators: Union[int, List[int]] = 2,
        use_omega: bool = True,
        omega_init: float = 1.0,
        global_omega: bool = True,
        learnable_omega: bool = True,
        gamma: float = 1.0,
        norm_type: str = "bn",
        c_norm: str = "gn",
        use_input_norm: bool = True,
        out_channels: Optional[int] = None,
        return_all_states: bool = False,
        final_head: Optional[nn.Module] = None,
        input_norm_stats: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
        use_positional_encoding: bool = False,
        activation_order: str = "post",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.gamma = Parameter(torch.tensor([gamma]), requires_grad=False)
        self.return_all_states = return_all_states

        n_oscillators = self._expand_param(n_oscillators, num_layers)
        n_timesteps = self._expand_param(n_timesteps, num_layers)
        connectivity = self._expand_param(connectivity, num_layers)
        readout_n_oscillators = self._expand_param(readout_n_oscillators, num_layers)

        if kernel_sizes is None:
            kernel_sizes = [max(9 - 2 * i, 5) for i in range(num_layers)]
        elif len(kernel_sizes) != num_layers:
            kernel_sizes = self._expand_param(kernel_sizes, num_layers)

        if isinstance(channel_multiplier, int):
            channels = [
                base_channels * (channel_multiplier ** min(i, spatial_stages - 1))
                for i in range(num_layers)
            ]
        else:
            channels = [base_channels * m for m in channel_multiplier]

        strides = [2 if i < spatial_stages else 1 for i in range(num_layers)]

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        hw_sizes = []
        h, w = input_size
        for stride in strides:
            h, w = h // stride, w // stride
            hw_sizes.append((h, w))

        if use_input_norm:
            if input_norm_stats is not None:
                mean, std = input_norm_stats
                self.register_buffer("norm_mean", torch.tensor(mean).view(1, -1, 1, 1))
                self.register_buffer("norm_std", torch.tensor(std).view(1, -1, 1, 1))
                self.input_norm = lambda x: (x - self.norm_mean) / self.norm_std
            else:
                self.register_buffer(
                    "norm_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)
                )
                self.register_buffer(
                    "norm_std", torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)
                )
                self.input_norm = lambda x: (x - self.norm_mean) / self.norm_std
        else:
            self.input_norm = nn.Identity()

        self.stem = nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False)

        if use_positional_encoding:
            h_stem = input_size if isinstance(input_size, int) else input_size[0]
            w_stem = input_size if isinstance(input_size, int) else input_size[1]
            pe = positional_encoding_2d(channels[0], h_stem, w_stem)
            self.register_buffer("pos_encoding", pe)
            self.use_positional_encoding = True
        else:
            self.use_positional_encoding = False

        self.transitions_x = nn.ModuleList()
        self.transitions_c = nn.ModuleList()
        self.kuramoto_blocks = nn.ModuleList()
        self.readouts = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.transitions_x.append(nn.Identity())
                self.transitions_c.append(nn.Identity())
            else:
                in_ch, out_ch, stride = channels[i - 1], channels[i], strides[i]
                pad = readout_kernel_size // 2
                self.transitions_x.append(
                    nn.Conv2d(
                        in_ch, out_ch, readout_kernel_size, stride, pad, bias=True
                    )
                )
                self.transitions_c.append(
                    nn.Conv2d(
                        in_ch, out_ch, readout_kernel_size, stride, pad, bias=True
                    )
                )

            self.kuramoto_blocks.append(
                KuramotoBlock(
                    channels=channels[i],
                    n_oscillators=n_oscillators[i],
                    connectivity=connectivity[i],
                    kernel_size=kernel_sizes[i],
                    hw=tuple(hw_sizes[i]) if hw_sizes[i] is not None else None,
                    use_omega=use_omega,
                    omega_init=omega_init,
                    global_omega=global_omega,
                    learnable_omega=learnable_omega,
                    c_norm=c_norm,
                )
            )

            ch = channels[i]
            pad = readout_kernel_size // 2
            norm_layer = (
                nn.BatchNorm2d
                if norm_type == "bn"
                else lambda c: nn.GroupNorm(c // 2, c)
            )

            readout = _ReadoutBlock(
                first=ConvReadout(
                    in_channels=ch,
                    out_channels=ch,
                    n_oscillators=readout_n_oscillators[i],
                    kernel_size=readout_kernel_size,
                    padding=pad,
                ),
                second=nn.Sequential(
                    norm_layer(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, ch, readout_kernel_size, 1, pad, bias=True),
                    norm_layer(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, ch, readout_kernel_size, 1, pad, bias=True),
                ),
                third=nn.Sequential(
                    norm_layer(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, ch, readout_kernel_size, 1, pad, bias=True),
                ),
            )

            self.readouts.append(readout)

        self.n_timesteps = n_timesteps
        self.out_channels = out_channels or channels[-1]
        self.final_head = final_head

    def _expand_param(self, param, length):
        if isinstance(param, (list, tuple)):
            if len(param) == length:
                return list(param)
            elif len(param) == 1:
                return list(param) * length
            else:
                result = list(param)
                while len(result) < length:
                    result.append(param[-1])
                return result[:length]
        return [param] * length

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, List, List]]:
        x = self.input_norm(x)
        c = self.stem(x)

        if self.use_positional_encoding:
            c = c + self.pos_encoding

        torch.manual_seed(999999)
        x = torch.randn_like(c)

        all_xs = []
        all_es = []

        for trans_x, trans_c, kuramoto, readout, n_steps in zip(
            self.transitions_x,
            self.transitions_c,
            self.kuramoto_blocks,
            self.readouts,
            self.n_timesteps,
        ):
            x = trans_x(x)
            c = trans_c(c)

            xs, es = kuramoto(x, c, n_steps, self.gamma.item())
            all_xs.append(xs)
            all_es.append(es)

            x = xs[-1]
            c = readout(x)

        output = self.final_head(c) if self.final_head is not None else c

        if self.return_all_states:
            return output, all_xs, all_es
        return output


class AKOrNDense(nn.Module):
    """Dense AKOrN for image classification.

    Args:
        image_size: Input image size (H, W) or single int for square.
        patch_size: Patch size for patchification.
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        num_layers: Number of AKOrN layers.
        n_oscillators: Number of oscillators per group.
        n_timesteps: Timesteps per layer (int or per-layer list).
        connectivity: Connectivity type ('attn' or 'conv').
        kernel_size: Kernel size for conv connectivity.
        n_heads: Number of attention heads.
        use_gta: Enable Group-Theoretic Attention.
        readout_n_oscillators: Oscillators for readout.
        use_omega: Enable natural frequency term.
        omega_init: Initial omega value.
        global_omega: Use single omega (True) or per-group (False).
        learnable_omega: Make omega learnable.
        gamma: Integration step size.
        use_pos_encoding: Add learned positional encoding.
        use_input_norm: Apply input normalization.
        maxpool: Unused (kept for compatibility).
        project: Unused (kept for compatibility).
        no_readout: Skip readout processing.
        c_norm: Normalization type for stimulus.
        return_all_states: Return intermediate states and energies.
        final_head: Optional final head module.
    """

    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 128,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 1,
        n_oscillators: int = 4,
        n_timesteps: Union[int, List[int]] = 8,
        connectivity: str = "attn",
        kernel_size: int = 1,
        n_heads: int = 8,
        use_gta: bool = True,
        readout_n_oscillators: int = 4,
        use_omega: bool = False,
        omega_init: float = 1.0,
        global_omega: bool = False,
        learnable_omega: bool = True,
        gamma: float = 1.0,
        use_pos_encoding: bool = False,
        use_input_norm: bool = True,
        maxpool: bool = True,
        project: bool = True,
        no_readout: bool = False,
        c_norm: Optional[str] = None,
        return_all_states: bool = False,
        final_head: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_layers = num_layers
        self.gamma = Parameter(torch.tensor([gamma]), requires_grad=False)
        self.no_readout = no_readout
        self.return_all_states = return_all_states

        if isinstance(n_timesteps, int):
            self.n_timesteps = [n_timesteps] * num_layers
        else:
            self.n_timesteps = n_timesteps

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        h_patches = image_size[0] // patch_size
        w_patches = image_size[1] // patch_size

        if use_input_norm:
            self.register_buffer(
                "patchify_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)
            )
            self.register_buffer(
                "patchify_std", torch.tensor([0.5, 0.5, 0.5]).view(1, -1, 1, 1)
            )
            self.patchify = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, 0)
            )
        else:
            self.patchify = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size, 0)

        if use_pos_encoding and not use_gta:
            self.pos_embed_x = Parameter(
                positional_encoding_2d(embed_dim, h_patches, w_patches)
            )
            self.pos_embed_c = Parameter(
                positional_encoding_2d(embed_dim, h_patches, w_patches)
            )
        else:
            self.register_parameter("pos_embed_x", None)
            self.register_parameter("pos_embed_c", None)

        self.kuramoto_blocks = nn.ModuleList()
        self.readouts = nn.ModuleList()

        for i in range(num_layers):
            self.kuramoto_blocks.append(
                KuramotoBlock(
                    channels=embed_dim,
                    n_oscillators=n_oscillators,
                    connectivity=connectivity,
                    kernel_size=kernel_size,
                    n_heads=n_heads,
                    hw=(h_patches, w_patches) if connectivity == "attn" else None,
                    use_omega=use_omega,
                    omega_init=omega_init,
                    global_omega=global_omega,
                    learnable_omega=learnable_omega,
                    use_gta=use_gta,
                    c_norm=c_norm,
                )
            )

            if not no_readout:
                self.readouts.append(
                    ConvReadout(
                        in_channels=embed_dim,
                        out_channels=embed_dim,
                        n_oscillators=readout_n_oscillators
                        if i == 0
                        else n_oscillators,
                        kernel_size=1,
                        padding=0,
                    )
                )
            else:
                self.readouts.append(nn.Identity())

        self.final_head = final_head

    patchify_mean: Tensor
    patchify_std: Tensor

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, List, List]]:
        if hasattr(self, "patchify_mean"):
            x = (x - self.patchify_mean) / self.patchify_std
        c = self.patchify(x)

        if self.pos_embed_c is not None:
            c = c + self.pos_embed_c

        torch.manual_seed(999999)
        x = torch.randn_like(c)
        if self.pos_embed_x is not None:
            x = x + self.pos_embed_x

        all_xs = []
        all_es = []

        for kuramoto, readout, n_steps in zip(
            self.kuramoto_blocks, self.readouts, self.n_timesteps
        ):
            xs, es = kuramoto(x, c, n_steps, self.gamma.item())
            all_xs.append(xs)
            all_es.append(es)

            x = xs[-1]
            c = readout(x) if not self.no_readout else x

        output = self.final_head(c) if self.final_head is not None else c

        if self.return_all_states:
            return output, all_xs, all_es
        return output


class AKOrNGrid(nn.Module):
    """Grid-structured AKOrN for spatial reasoning tasks.

    Args:
        grid_size: Grid spatial dimensions (H, W).
        vocab_size: Vocabulary size for discrete inputs (None for continuous).
        embed_dim: Embedding dimension.
        num_layers: Number of AKOrN layers.
        n_oscillators: Number of oscillators per group.
        n_timesteps: Timesteps per layer.
        connectivity: Connectivity type ('attn' or 'conv').
        n_heads: Number of attention heads.
        use_gta: Enable Group-Theoretic Attention.
        use_omega: Enable natural frequency term.
        omega_init: Initial omega value.
        global_omega: Use single omega (True) or per-group (False).
        learnable_omega: Make omega learnable.
        gamma: Integration step size.
        use_nonlinearity: Add nonlinear readout processing.
        out_channels: Unused (kept for compatibility).
        c_norm: Normalization type for stimulus.
        return_all_states: Return intermediate states and energies.
        final_head: Optional final head module.
        activation_order: Activation order for feedforward ('pre' or 'post').
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (9, 9),
        vocab_size: Optional[int] = None,
        embed_dim: int = 64,
        num_layers: int = 1,
        n_oscillators: int = 4,
        n_timesteps: int = 16,
        connectivity: str = "attn",
        n_heads: int = 8,
        use_gta: bool = True,
        use_omega: bool = True,
        omega_init: float = 0.1,
        global_omega: bool = True,
        learnable_omega: bool = False,
        gamma: float = 1.0,
        use_nonlinearity: bool = True,
        out_channels: Optional[int] = None,
        c_norm: Optional[str] = None,
        return_all_states: bool = False,
        final_head: Optional[nn.Module] = None,
        activation_order: str = "post",
    ):
        super().__init__()

        self.grid_size = grid_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.n_timesteps = n_timesteps
        self.gamma = Parameter(torch.tensor([gamma]), requires_grad=False)
        self.return_all_states = return_all_states

        if vocab_size is not None:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embedding = None

        self.x0 = Parameter(torch.randn(1, embed_dim, *grid_size))

        self.kuramoto_blocks = nn.ModuleList()
        self.readouts = nn.ModuleList()

        for _ in range(num_layers):
            self.kuramoto_blocks.append(
                KuramotoBlock(
                    channels=embed_dim,
                    n_oscillators=n_oscillators,
                    connectivity=connectivity,
                    kernel_size=1 if connectivity == "attn" else 3,
                    n_heads=n_heads,
                    hw=grid_size,
                    use_omega=use_omega,
                    omega_init=omega_init,
                    global_omega=global_omega,
                    learnable_omega=learnable_omega,
                    use_gta=use_gta,
                    c_norm=c_norm,
                )
            )

            readout = _ReadoutBlock(
                first=ConvReadout(
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    n_oscillators=n_oscillators,
                    kernel_size=1,
                    padding=0,
                ),
                second=nn.Sequential(
                    nn.Identity(),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=True),
                    nn.Identity(),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=True),
                )
                if use_nonlinearity
                else None,
                third=nn.Sequential(
                    nn.Identity(),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim, 1, 1, 0, bias=True),
                )
                if use_nonlinearity
                else None,
            )

            self.readouts.append(readout)

        self.final_head = final_head

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, List, List]]:
        if self.embedding is not None:
            if x.dim() == 3:
                c = self.embedding(x).permute(0, 3, 1, 2)
            else:
                raise ValueError("Expected 3D input for discrete mode")
        else:
            c = x

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1:
                mask = mask.expand(-1, c.size(1), -1, -1)

            torch.manual_seed(999999)
            noise = torch.randn_like(c)
            x = mask * c + (~mask) * noise
        else:
            x = self.x0.expand(c.size(0), -1, -1, -1)

        all_xs = []
        all_es = []

        for kuramoto, readout in zip(self.kuramoto_blocks, self.readouts):
            xs, es = kuramoto(x, c, self.n_timesteps, self.gamma.item())
            all_xs.append(xs)
            all_es.append(es)

            x = xs[-1]
            c = readout(x)

        output = self.final_head(c) if self.final_head is not None else c

        if self.return_all_states:
            return output, all_xs, all_es
        return output
