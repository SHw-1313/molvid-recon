"""Equal-compute, soft task routing of atomwise O(3)-equivariant adapters.

Both branches always execute. Shared mode uses (1/2,1/2); routed mode uses
(3/4,1/4) for joint and (1/4,3/4) for actual dynamic Q1. The task label is
explicit, never inferred from query count or a static single-frame objective.
"""
import torch
from torch import nn


class EquivariantTaskAdapter(nn.Module):
    def __init__(self, width, bottleneck):
        super().__init__()
        self.norm = nn.LayerNorm(2 * width)
        self.down_h = nn.Linear(2 * width, bottleneck)
        self.down_v = nn.Linear(width, bottleneck, bias=False)
        self.up_h = nn.Linear(bottleneck, width)
        self.up_v = nn.Linear(bottleneck, width, bias=False)
        # Only the final maps are zero. Their input and scalar vector gate
        # remain active, so each up map receives gradients on update one.
        nn.init.zeros_(self.up_h.weight)
        nn.init.zeros_(self.up_h.bias)
        nn.init.zeros_(self.up_v.weight)

    def forward(self, h, v):
        invariant = torch.linalg.vector_norm(v.float(), dim=-2).to(h.dtype)
        scalar = torch.nn.functional.silu(self.down_h(self.norm(torch.cat((h, invariant), dim=-1))))
        vector = self.down_v(v) * (2. * torch.sigmoid(scalar)).unsqueeze(-2)
        return self.up_h(scalar), self.up_v(vector)


class TaskAdapters(nn.Module):
    def __init__(self, width, bottleneck, mode):
        super().__init__()
        if mode not in ('shared', 'routed'):
            raise ValueError('task adapter mode must be shared or routed')
        if isinstance(bottleneck, bool) or not isinstance(bottleneck, int) or bottleneck < 1:
            raise ValueError('task adapter bottleneck must be a positive integer')
        self.mode = mode
        self.bottleneck = bottleneck
        self.branches = nn.ModuleList([EquivariantTaskAdapter(width, bottleneck) for _ in range(2)])

    def mixing_weights(self, route_task):
        if route_task not in ('joint', 'q1'):
            raise ValueError('enabled adapters require explicit dynamic route_task=joint or q1')
        if self.mode == 'shared':
            return (.5, .5)
        return (.75, .25) if route_task == 'joint' else (.25, .75)

    def forward(self, h, v, route_task):
        weights = self.mixing_weights(route_task)
        dh0, dv0 = self.branches[0](h, v)
        dh1, dv1 = self.branches[1](h, v)
        return h + (weights[0] * dh0 + weights[1] * dh1), v + (weights[0] * dv0 + weights[1] * dv1)
