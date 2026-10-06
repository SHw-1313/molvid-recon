"""Atom-wise history/query mixing over shared geometric features.

Scalars [T,N,D], vectors [T,N,3,D], times [T,N] in physical ps. Attention
never mixes atoms/systems. Vector channel maps have no Cartesian bias.
Both output maps start at zero, preserving the loaded geometric prior.
"""
import math
import torch
from torch import nn


def physical_features(time_ps):
    scale = time_ps.new_tensor([10., 100., 1000., 10000.])
    signed_log = time_ps.sign() * torch.log1p(time_ps.abs())
    return torch.cat((time_ps[..., None] / 1000., signed_log[..., None],
                      (time_ps.sign() * torch.log1p(time_ps.abs() / 100.))[..., None],
                      time_ps[..., None] / (time_ps.abs()[..., None] + scale)), -1)


class AtomTimeAttention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('width must be divisible by attention heads')
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.time = nn.Sequential(nn.Linear(7, width), nn.SiLU(), nn.Linear(width, width))
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.value_h = nn.Linear(width, width)
        self.value_v = nn.Linear(width, width, bias=False)
        self.out_h = nn.Linear(width, width)
        self.out_v = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out_h.weight)
        nn.init.zeros_(self.out_h.bias)
        nn.init.zeros_(self.out_v.weight)

    def forward(self, query_h, memory_h, memory_v, query_time_ps, memory_time_ps):
        qlen, atoms, width = query_h.shape
        length = memory_h.shape[0]
        channels = width // self.heads
        q = self.q(self.norm(query_h) + self.time(physical_features(query_time_ps)))
        k = self.k(self.norm(memory_h) + self.time(physical_features(memory_time_ps)))
        q = q.reshape(qlen, atoms, self.heads, channels).permute(1, 2, 0, 3)
        k = k.reshape(length, atoms, self.heads, channels).permute(1, 2, 0, 3)
        weights = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(channels), -1)
        values_h = self.value_h(memory_h).reshape(length, atoms, self.heads, channels)
        values_v = self.value_v(memory_v).reshape(length, atoms, 3, self.heads, channels)
        mixed_h = torch.einsum('naqh,hnac->qnac', weights, values_h).reshape(qlen, atoms, width)
        mixed_v = torch.einsum('naqh,hnrac->qnrac', weights, values_v).reshape(qlen, atoms, 3, width)
        return self.out_h(mixed_h), self.out_v(mixed_v)


class BridgeHistoryConditioner(nn.Module):
    """Observed-prefix attention followed by joint query-state attention.

Query mixing sees generated flow states, never target future structures. It
is not autoregressive physical dynamics: the joint stochastic bridge learns
the conditional distribution of all requested future frames together.
"""
    def __init__(self, width, heads=4):
        super().__init__()
        self.observed_attention = AtomTimeAttention(width, heads)
        self.query_attention = AtomTimeAttention(width, heads)
        self.displacement_gate = nn.Linear(width, width)

    def forward(self, h, v, history_h, history_v, history_displacement_A,
                query_lag_ps, history_lag_ps):
        motion_vectors = history_displacement_A[..., None] * torch.tanh(
            self.displacement_gate(history_h)).unsqueeze(2)
        dh, dv = self.observed_attention(h, history_h, history_v + motion_vectors,
                                         query_lag_ps, history_lag_ps)
        h, v = h + dh, v + dv
        dh, dv = self.query_attention(h, h, v, query_lag_ps, query_lag_ps)
        return h + dh, v + dv
