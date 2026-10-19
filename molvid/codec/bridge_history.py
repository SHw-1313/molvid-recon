"""Atom-wise history/query mixing over shared geometric features.

Scalars [T,N,D], vectors [T,N,3,D], times [T,N] in physical ps. Attention
keeps atom-specific Values. Optional future Q/K share routing within protein
chemical units; history stays atom-wise. Vector maps have no Cartesian bias.
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


def future_query_grouping(batch):
    """Prepare (atom_group, valid_count, valid_atom, representative) once.

    Keys include packed sample, component and input residue/block identity.
    edge_mask marks ligand atoms: each gets its own routing group regardless
    of block_id (which can cover an entire flexible ligand). loss_mask only
    selects contributors to Q/K pooling; all atom Values/outputs are retained.
    """
    atoms = len(batch.abid)
    index = torch.arange(atoms, device=batch.abid.device)
    fallback = torch.where(batch.edge_mask != 0, index, -torch.ones_like(index))
    keys, group = torch.unique(torch.stack(
        (batch.abid, batch.component_id, batch.block_id, fallback), -1),
        dim=0, return_inverse=True)
    valid = batch.loss_mask.bool()
    count = torch.zeros(len(keys), device=group.device, dtype=torch.float32)
    count.index_add_(0, group, valid.float())
    representative = torch.full((len(keys),), atoms, device=group.device, dtype=torch.long)
    representative.scatter_reduce_(0, group, index, reduce='amin', include_self=True)
    return group, count, valid, representative


def _pooled_query_inputs(h, time_ps, grouping):
    """Masked scalar mean [Q,G,D] BEFORE norm/Q/K; physical time stays ps."""
    group, count, valid, representative = grouping
    sums = h.new_zeros((len(h), len(count), h.shape[-1]), dtype=torch.float32)
    sums.index_add_(1, group, torch.where(valid[None, :, None], h.float(), 0.))
    # Empty groups have a finite zero descriptor; they never contaminate others.
    return sums / count.clamp_min(1)[None, :, None], time_ps[:, representative], group


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

    def _weights(self, query_h, memory_h, query_time_ps, memory_time_ps):
        qlen, groups, width = query_h.shape
        length = memory_h.shape[0]
        channels = width // self.heads
        q = self.q(self.norm(query_h) + self.time(physical_features(query_time_ps)))
        k = self.k(self.norm(memory_h) + self.time(physical_features(memory_time_ps)))
        q = q.reshape(qlen, groups, self.heads, channels).permute(1, 2, 0, 3)
        k = k.reshape(length, groups, self.heads, channels).permute(1, 2, 0, 3)
        return torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(channels), -1)

    def forward(self, query_h, memory_h, memory_v, query_time_ps, memory_time_ps,
                *, query_grouping=None):
        qlen, atoms, width = query_h.shape
        length = memory_h.shape[0]
        channels = width // self.heads
        if query_grouping is None:
            weights = self._weights(query_h, memory_h, query_time_ps, memory_time_ps)
        else:
            # This option is exclusively for joint future self-attention.
            if query_h is not memory_h or qlen != length:
                raise ValueError('Grouped Q/K requires future self-attention')
            pooled, times, group = _pooled_query_inputs(query_h, query_time_ps, query_grouping)
            weights = self._weights(pooled, pooled, times, times)[group]

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
    def __init__(self, width, heads=4, *, query_qk_mode='atom'):
        super().__init__()
        if query_qk_mode not in ('atom', 'group'):
            raise ValueError("query_qk_mode must be 'atom' or 'group'")
        self.query_qk_mode = query_qk_mode
        self.observed_attention = AtomTimeAttention(width, heads)
        self.query_attention = AtomTimeAttention(width, heads)
        self.displacement_gate = nn.Linear(width, width)

    def forward(self, h, v, history_h, history_v, history_displacement_A,
                query_lag_ps, history_lag_ps, *, query_grouping=None):
        motion_vectors = history_displacement_A[..., None] * torch.tanh(
            self.displacement_gate(history_h)).unsqueeze(2)
        dh, dv = self.observed_attention(h, history_h, history_v + motion_vectors,
                                         query_lag_ps, history_lag_ps)
        h, v = h + dh, v + dv
        if self.query_qk_mode == 'group' and query_grouping is None:
            raise ValueError('Group Q/K mode requires prepared sample/component/block grouping')
        dh, dv = self.query_attention(h, h, v, query_lag_ps, query_lag_ps,
            query_grouping=query_grouping if self.query_qk_mode == 'group' else None)
        return h + dh, v + dv
