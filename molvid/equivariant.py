"""Shared channel-only vector operators."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AxisPreservingLinear(nn.Module):
    """A channel-only, bias-free linear map for vector features."""

    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        if int(input_width) < 1 or int(output_width) < 1:
            raise ValueError("AxisPreservingLinear widths must be positive")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.weight = nn.Parameter(torch.empty(self.output_width, self.input_width))
        self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, value: Tensor) -> Tensor:
        if value.shape[-1] != self.input_width:
            raise ValueError(
                f"expected final channel width {self.input_width}, got {value.shape[-1]}"
            )
        return torch.matmul(value, self.weight.t())
class SO3ChannelNorm(nn.Module):
    """Normalize vector channels with xyz-contracted, rotation-invariant norms."""

    def __init__(self, channels: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        if int(channels) < 1:
            raise ValueError("SO3ChannelNorm channels must be positive")
        self.channels = int(channels)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(self.channels))

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim < 3 or value.shape[-2] != 3 or value.shape[-1] != self.channels:
            raise ValueError("SO3ChannelNorm expects [...,3,C] vector features")
        input_dtype = value.dtype
        value_fp32 = value.float()
        denominator = value_fp32.square().mean(dim=-2, keepdim=True).add(self.eps).sqrt()
        scale = self.scale.float().reshape((1,) * (value.ndim - 1) + (self.channels,))
        return (value_fp32 / denominator * scale).to(dtype=input_dtype)
