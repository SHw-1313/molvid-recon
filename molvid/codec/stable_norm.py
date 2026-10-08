"""Overflow-safe evaluation of the unchanged affine LayerNorm formula."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class OverflowSafeLayerNorm(nn.LayerNorm):
    """Keep the original kernel except when squared reductions risk overflow.

    Bilinear equivariant features can be finite while their float32 variance
    is not. Float64 reduction on those exceptional batches preserves epsilon,
    learned affine parameters and gradients; it does not clip the features.
    """

    @classmethod
    def from_layer(cls, layer):
        if not isinstance(layer, nn.LayerNorm):
            raise TypeError('Expected the inherited scalar LayerNorm')
        result = cls(layer.normalized_shape, eps=layer.eps,
                     elementwise_affine=layer.elementwise_affine,
                     bias=layer.bias is not None)
        result.weight, result.bias = layer.weight, layer.bias
        result.high_precision_calls = 0
        return result

    def forward(self, value):
        # PyTorch accumulates fp16/bf16 LayerNorm statistics in float32.
        accumulation = torch.float64 if value.dtype == torch.float64 else torch.float32
        limit = math.sqrt(torch.finfo(accumulation).max /
                          (4 * math.prod(self.normalized_shape)))
        if value.dtype != torch.float64 and bool((value.detach().abs() > limit).any()):
            self.high_precision_calls += 1
            return F.layer_norm(value.double(), self.normalized_shape,
                None if self.weight is None else self.weight.double(),
                None if self.bias is None else self.bias.double(), self.eps).to(value.dtype)
        return super().forward(value)
