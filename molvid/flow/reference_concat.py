"""Atomwise reference/current feature fusion with identity initialization."""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceConcat(nn.Module):
    """Project concatenated channels; vector channels share all XYZ weights.

    h/ref_h are [N,D], v/ref_v [N,3,D]. There is no pooling or cross-atom
    operation. Both learned matrices start as [I,0], without a bias or gate.
    Computing x + (W-[I,0])@[x,ref] preserves the original current features
    exactly at initialization even under autocast (a direct identity GEMM
    would first quantize a float32 residual stream to bfloat16).
    """

    def __init__(self, width):
        super().__init__()
        self.scalar = nn.Linear(2*width, width, bias=False)
        self.vector = nn.Linear(2*width, width, bias=False)
        identity = torch.cat((torch.eye(width), torch.zeros(width, width)), dim=-1)
        self.register_buffer('_identity', identity, persistent=False)
        with torch.no_grad():
            self.scalar.weight.copy_(identity)
            self.vector.weight.copy_(identity)

    def forward(self, h, v, reference_h, reference_v):
        if reference_h is None or reference_v is None:
            raise ValueError('Reference concat requires cached observed-prefix features')
        if reference_h.shape != h.shape or reference_v.shape != v.shape:
            raise ValueError('Reference/current features must share atom and channel axes')
        h = h + F.linear(torch.cat((h, reference_h), -1),
                         self.scalar.weight-self._identity)
        v = v + F.linear(torch.cat((v, reference_v), -1),
                         self.vector.weight-self._identity)
        return h, v
