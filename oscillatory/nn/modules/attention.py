"""Group-theoretic attention with SO(2) equivariance."""

from typing import Optional, Literal, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from oscillatory.nn.functional import (
    make_2d_coord,
    make_so2_matrices,
    apply_group_action,
)


__all__ = [
    "GTAttention",
]


class GTAttention(nn.Module):
    """Group-Theoretic Attention with optional GTA or RoPE positional encoding.

    Implements multi-head attention with learnable SO(2) group actions (GTA)
    or rotary position embeddings (RoPE) for 2D spatial inputs.

    Args:
        embed_dim: Embedding dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        weight_type: Type of projection layers ('conv' or 'fc')
        kernel_size: Kernel size for convolutional projections
        stride: Stride for convolutional projections
        padding: Padding for convolutional projections
        use_gta: Enable learnable Group-Theoretic Attention
        use_rope: Enable Rotary Position Embeddings
        hw: Spatial dimensions (height, width) if using GTA/RoPE
        dropout: Dropout probability
    """

    _hw: Tensor
    _source_h: int
    _source_w: int

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 8,
        weight_type: Literal["conv", "fc"] = "conv",
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        use_gta: bool = False,
        use_rope: bool = False,
        hw: Optional[Union[Tuple[int, int], Tensor]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % n_heads == 0, (
            f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"
        )
        assert not (use_gta and use_rope), "Cannot use both GTA and RoPE"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.weight_type = weight_type
        self.stride = stride
        self.use_gta = use_gta
        self.use_rope = use_rope
        self.hw = hw

        if weight_type == "conv":
            self.W_qkv = nn.Conv2d(
                embed_dim,
                3 * embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self.W_o = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif weight_type == "fc":
            self.W_qkv = nn.Linear(embed_dim, 3 * embed_dim)
            self.W_o = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError(f"weight_type must be 'conv' or 'fc', got {weight_type}")

        self.dropout_p = dropout

        # For fc weight_type, we need spatial dimensions even without GTA/RoPE
        if weight_type == "fc" and hw is not None:
            if isinstance(hw, (tuple, list)):
                self._source_h, self._source_w = int(hw[0]), int(hw[1])
            else:
                self._source_h, self._source_w = hw.shape[0], hw.shape[1]

        if use_gta or use_rope:
            if hw is None:
                raise ValueError("hw must be provided when using GTA or RoPE")
            self._init_so2_matrices(hw)

    def _init_so2_matrices(self, hw: Union[Tuple[int, int], Tensor]) -> None:
        """Initialize SO(2) rotation matrices for GTA/RoPE."""
        n_freqs = self.head_dim // 4
        if self.head_dim % 4 != 0:
            n_freqs += 1

        if isinstance(hw, (tuple, list)):
            coords = make_2d_coord(hw[0], hw[1])
            self._source_h, self._source_w = int(hw[0]), int(hw[1])
        else:
            coords = hw
            self._source_h, self._source_w = hw.shape[0], hw.shape[1]

        # Generate SO(2) matrices: shape (H, W, n_freqs, 2, 2)
        matrices = make_so2_matrices(coords, n_freqs)
        # Reshape to (H*W, n_freqs, 2, 2)
        matrices = matrices.flatten(0, 1).flatten(1, 2)
        # Truncate to head_dim//2 frequencies
        matrices = matrices[:, : self.head_dim // 2, :, :]

        if self.use_gta:
            # Learnable parameters for GTA
            self.mat_q = nn.Parameter(matrices.clone())
            self.mat_k = nn.Parameter(matrices.clone())
            self.mat_v = nn.Parameter(matrices.clone())
            self.mat_o = nn.Parameter(matrices.transpose(-2, -1).clone())
        else:  # use_rope
            # Fixed buffers for RoPE
            self.register_buffer("mat_q", matrices)
            self.register_buffer("mat_k", matrices)

        self.register_buffer("_hw", torch.tensor([self._source_h, self._source_w]))

    def _rescale_gta_matrices(self, mat: Tensor, target_hw: Tuple[int, int]) -> Tensor:
        """Rescale GTA matrices to match target spatial dimensions via interpolation.

        Args:
            mat: Matrix of shape (H*W, n_freqs, 2, 2)
            target_hw: Target (height, width)

        Returns:
            Rescaled matrix of shape (target_H*target_W, n_freqs, 2, 2)
        """
        target_h, target_w = target_hw

        # Fast path: no rescaling needed
        if self._source_h == target_h and self._source_w == target_w:
            return mat

        n_freqs = mat.shape[1]
        # Reshape to 2D spatial: (n_freqs*4, source_h, source_w)
        mat_2d = mat.reshape(self._source_h, self._source_w, n_freqs * 4)
        mat_2d = mat_2d.permute(2, 0, 1).unsqueeze(0)

        # Bilinear interpolation
        mat_2d = F.interpolate(
            mat_2d,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to (target_H*target_W, n_freqs, 2, 2)
        mat_2d = mat_2d.squeeze(0).permute(1, 2, 0)
        return mat_2d.reshape(target_h * target_w, n_freqs, 2, 2)

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with optional group-theoretic positional encoding.

        Args:
            x: Input tensor of shape (B, C, H, W) for conv or (B, L, C) for fc
            attn_mask: Optional attention mask

        Returns:
            Output tensor with same shape as input
        """
        B = x.shape[0]
        input_is_4d = x.dim() == 4

        if self.weight_type == "conv":
            h = x.shape[2] // self.stride
            w = x.shape[3] // self.stride
        else:
            # For fc mode, input can be (B, C, H, W) - reshape to (B, L, C)
            if input_is_4d:
                h, w = x.shape[2], x.shape[3]
                x = x.permute(0, 2, 3, 1).reshape(B, h * w, -1)
            else:
                # Already in (B, L, C) format
                h, w = self._source_h, self._source_w

        qkv = self.W_qkv(x)

        if self.weight_type == "conv":
            q, k, v = qkv.chunk(3, dim=1)
            q = (
                q.view(B, self.head_dim, self.n_heads, h, w)
                .permute(0, 2, 3, 4, 1)
                .reshape(B, self.n_heads, h * w, self.head_dim)
            )
            k = (
                k.view(B, self.head_dim, self.n_heads, h, w)
                .permute(0, 2, 3, 4, 1)
                .reshape(B, self.n_heads, h * w, self.head_dim)
            )
            v = (
                v.view(B, self.head_dim, self.n_heads, h, w)
                .permute(0, 2, 3, 4, 1)
                .reshape(B, self.n_heads, h * w, self.head_dim)
            )
        else:
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_gta:
            mat_q = self._rescale_gta_matrices(self.mat_q, (h, w))
            mat_k = self._rescale_gta_matrices(self.mat_k, (h, w))
            mat_v = self._rescale_gta_matrices(self.mat_v, (h, w))

            q = apply_group_action(mat_q, q)
            k = apply_group_action(mat_k, k)
            v = apply_group_action(mat_v, v)

        elif self.use_rope:
            mat_q = self._rescale_gta_matrices(self.mat_q, (h, w))
            mat_k = self._rescale_gta_matrices(self.mat_k, (h, w))

            q = apply_group_action(mat_q, q)
            k = apply_group_action(mat_k, k)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        if self.use_gta:
            mat_o = self._rescale_gta_matrices(self.mat_o, (h, w))
            attn_output = apply_group_action(mat_o, attn_output)

        if self.weight_type == "conv":
            attn_output = attn_output.reshape(B, self.n_heads, h, w, self.head_dim)
            attn_output = attn_output.permute(0, 4, 1, 2, 3)
            attn_output = attn_output.reshape(B, self.embed_dim, h, w)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(B, -1, self.embed_dim)

        output = self.W_o(attn_output)

        # For fc mode with 4D input, reshape back to (B, C, H, W)
        if self.weight_type == "fc" and input_is_4d:
            output = output.view(B, h, w, -1).permute(0, 3, 1, 2)

        return output
