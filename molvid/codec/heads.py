"""Equivariant coordinate decoder head and centered-coordinate feature stem."""

from __future__ import annotations

import torch
from torch import Tensor, nn

class EquivariantCoordinateHead(nn.Module):
    """Map vector channels to coordinates using invariant scalar gates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, 1, bias=False)

    def forward(self, h: Tensor, v: Tensor) -> Tensor:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("coordinate head expects h=[T,N,C] and v=[T,N,3,C]")
        gate = torch.sigmoid(self.gate(h)).unsqueeze(2)
        return self.out(v * gate).squeeze(-1).float()


class CoordinateVectorStem(nn.Module):
    """Inject centered coordinates into generated equivariant vector features.

    This is part of the learned latent path, not coordinate metadata.  The
    map is bias-free and acts independently on the Cartesian components, so a
    translation removed by ``center_coordinates`` cannot re-enter the vector
    representation as an extra origin or time-dependent displacement.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if int(channels) < 1:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.projection = nn.Linear(1, self.channels, bias=False)
        nn.init.ones_(self.projection.weight)

    def forward(self, centered_coordinates: Tensor) -> Tensor:
        if centered_coordinates.ndim != 3 or centered_coordinates.shape[-1] != 3:
            raise ValueError("centered coordinates must have shape [T, N, 3]")
        return self.projection(centered_coordinates.unsqueeze(-1))

