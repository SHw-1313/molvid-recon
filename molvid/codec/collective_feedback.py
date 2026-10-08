"""Equivariant block communication inside the all-atom denoising backbone.

Blocks are the input's chemical units (protein residues / ligand blocks), not
generated C-alpha coordinates. Pooling and lifting retain the atom stream.
Positions [N,3] are current noisy coordinates in Angstrom; no target is used.
"""
import math

import torch
from torch import nn


class CollectiveFeedback(nn.Module):
    """Global block attention with scalar and Cartesian-vector messages.

Block keys include sample and component identity. Attention crosses components
within a complex, never packed samples. Scalar attention depends only on
rotation-invariant features; vector channel maps have no Cartesian bias.
Query chunking bounds temporary pair memory without truncating interactions.
The zero output maps preserve the inherited backbone at initialization.
"""
    def __init__(self, width, channels=128, heads=4, query_chunk=64):
        super().__init__()
        if channels % heads:
            raise ValueError('Collective channels must be divisible by heads')
        self.heads, self.channels, self.query_chunk = heads, channels, query_chunk
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, channels, bias=False)
        self.key = nn.Linear(width, channels, bias=False)
        self.value_h = nn.Linear(width, channels)
        self.value_v = nn.Linear(width, channels, bias=False)
        self.pair_bias = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, heads))
        self.direction_gate = nn.Linear(width, channels)
        self.out_h = nn.Linear(channels, width)
        self.out_v = nn.Linear(channels, width, bias=False)
        self.atom_gate = nn.Linear(width, width)
        for layer in (self.out_h, self.out_v):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.out_h.bias)

    @staticmethod
    def grouping(batch):
        keys, inverse = torch.unique(torch.stack(
            (batch.abid, batch.component_id, batch.block_id), -1),
            dim=0, return_inverse=True)
        return keys, inverse, torch.bincount(inverse, minlength=len(keys))

    def forward(self, h, v, position_A, grouping):
        """h[N,D], v[N,3,D] -> updated atom features, not rigid coordinates."""
        keys, group, count = grouping
        groups = len(keys)
        # Accumulate in float32, including under CUDA autocast.
        def pool(value):
            sums = value.new_zeros((groups, *value.shape[1:]), dtype=torch.float32)
            sums.index_add_(0, group, value.float())
            return sums / count.reshape((groups,) + (1,) * (value.ndim - 1))

        pooled_h = self.norm(pool(h))
        centers = pool(position_A)
        width = self.channels // self.heads
        q = self.query(pooled_h).reshape(groups, self.heads, width).transpose(0, 1)
        k = self.key(pooled_h).reshape(groups, self.heads, width).transpose(0, 1)
        vh = self.value_h(pooled_h).reshape(groups, self.heads, width).transpose(0, 1)
        vv = self.value_v(pool(v)).reshape(groups, 3, self.heads, width).permute(2, 0, 1, 3)
        scalar_messages, vector_messages = [], []
        for start in range(0, groups, self.query_chunk):
            stop = min(groups, start + self.query_chunk)
            # Direction points from receiving to sending chemical unit.
            delta = centers[None] - centers[start:stop, None]
            distance = torch.linalg.vector_norm(delta, dim=-1)
            direction = delta / distance.clamp_min(1e-6)[..., None]
            same_sample = keys[start:stop, 0, None] == keys[None, :, 0]
            same_component = same_sample & (keys[start:stop, 1, None] == keys[None, :, 1])
            self_pair = group.new_tensor(range(start, stop))[:, None] == torch.arange(groups, device=h.device)[None]
            pair = torch.stack((distance / 10., torch.log1p(distance),
                1. / (1. + distance), same_component.float(), self_pair.float()), -1)
            logits = (q[:, start:stop].float() @ k.float().transpose(-1, -2)) / math.sqrt(width)
            logits = logits + self.pair_bias(pair).permute(2, 0, 1).float()
            weights = torch.softmax(logits.masked_fill(~same_sample[None], -torch.inf), -1)
            mh = torch.einsum('hij,hjc->ihc', weights, vh.float())
            mv = torch.einsum('hij,hjrc->irhc', weights, vv.float())
            shared_direction = torch.einsum('hij,ijr->irh', weights, direction)
            direction_channels = torch.tanh(self.direction_gate(pooled_h[start:stop])).reshape(-1, self.heads, width)
            mv = mv + shared_direction[..., None] * direction_channels[:, None]
            scalar_messages.append(mh.reshape(-1, self.channels))
            vector_messages.append(mv.reshape(-1, 3, self.channels))
        dh = self.out_h(torch.cat(scalar_messages))
        dv = self.out_v(torch.cat(vector_messages))
        gate = torch.sigmoid(self.atom_gate(self.norm(h)))
        return h + gate * dh[group], v + gate[:, None] * dv[group]
