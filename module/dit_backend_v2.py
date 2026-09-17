"""Vectorized helpers for factorized scalar/vector DiT execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


ATTENTION_BACKEND = "matmul_softmax_shared_weights_v2"


def _version(value: Tensor) -> int:
    return int(getattr(value, "_version", 0))


@dataclass(frozen=True)
class FactorizedLayout:
    atom_to_group: Tensor
    atom_slot: Tensor
    group_counts: Tensor
    group_samples: Tensor
    padded_group: Tensor
    valid_blocks: Tensor
    batch_size: int
    num_atoms: int
    num_groups: int
    max_blocks: int
    abid_ref: Tensor
    block_id_ref: Tensor
    atom_ptr_ref: Tensor
    abid_version: int
    block_id_version: int
    atom_ptr_version: int

    def matches(self, batch: Any) -> bool:
        return (
            self.batch_size == int(batch.batch_size)
            and self.num_atoms == int(batch.num_atoms)
            and self.abid_ref is batch.abid
            and self.block_id_ref is batch.block_id
            and self.atom_ptr_ref is batch.atom_ptr
            and self.abid_version == _version(batch.abid)
            and self.block_id_version == _version(batch.block_id)
            and self.atom_ptr_version == _version(batch.atom_ptr)
            and self.atom_to_group.device == batch.state_h.device
        )

    def contract(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "num_atoms": self.num_atoms,
            "num_groups": self.num_groups,
            "max_blocks": self.max_blocks,
        }


def build_factorized_layout(batch: Any) -> FactorizedLayout:
    """Materialize static sample/block mappings once per batch metadata view."""

    batch_size = int(batch.batch_size)
    num_atoms = int(batch.num_atoms)
    abid = batch.abid.detach().to(device="cpu", dtype=torch.long).tolist()
    block_id = batch.block_id.detach().to(device="cpu", dtype=torch.long).tolist()
    atom_ptr = batch.atom_ptr.detach().to(device="cpu", dtype=torch.long).tolist()
    if len(atom_ptr) != batch_size + 1 or atom_ptr[0] != 0 or atom_ptr[-1] != num_atoms:
        raise ValueError("batch atom_ptr is incompatible with factorized layout")
    if len(abid) != num_atoms or len(block_id) != num_atoms:
        raise ValueError("batch topology metadata is incompatible with factorized layout")

    atom_to_group = [-1] * num_atoms
    atom_slot = [-1] * num_atoms
    counts: list[int] = []
    samples: list[int] = []
    padded: list[list[int]] = []
    per_sample_counts: list[int] = []

    for sample in range(batch_size):
        start, stop = int(atom_ptr[sample]), int(atom_ptr[sample + 1])
        by_block: dict[int, list[int]] = {}
        for atom in range(start, stop):
            if int(abid[atom]) != sample:
                raise ValueError("abid must follow atom_ptr sample boundaries")
            by_block.setdefault(int(block_id[atom]), []).append(atom)
        sample_groups: list[int] = []
        for slot, block in enumerate(sorted(by_block)):
            group = len(counts)
            sample_groups.append(group)
            atoms = by_block[block]
            counts.append(len(atoms))
            samples.append(sample)
            for atom in atoms:
                atom_to_group[atom] = group
                atom_slot[atom] = slot
        padded.append(sample_groups)
        per_sample_counts.append(len(sample_groups))

    if any(value < 0 for value in atom_to_group + atom_slot):
        raise ValueError("every atom must map to one block group")
    num_groups = len(counts)
    max_blocks = max(per_sample_counts, default=0)
    device = batch.state_h.device
    padded_tensor = torch.full(
        (batch_size, max_blocks), -1, dtype=torch.long, device=device
    )
    for sample, groups in enumerate(padded):
        if groups:
            padded_tensor[sample, : len(groups)] = torch.as_tensor(
                groups, dtype=torch.long, device=device
            )
    return FactorizedLayout(
        atom_to_group=torch.as_tensor(atom_to_group, dtype=torch.long, device=device),
        atom_slot=torch.as_tensor(atom_slot, dtype=torch.long, device=device),
        group_counts=torch.as_tensor(counts, dtype=torch.long, device=device),
        group_samples=torch.as_tensor(samples, dtype=torch.long, device=device),
        padded_group=padded_tensor,
        valid_blocks=padded_tensor >= 0,
        batch_size=batch_size,
        num_atoms=num_atoms,
        num_groups=num_groups,
        max_blocks=max_blocks,
        abid_ref=batch.abid,
        block_id_ref=batch.block_id,
        atom_ptr_ref=batch.atom_ptr,
        abid_version=_version(batch.abid),
        block_id_version=_version(batch.block_id),
        atom_ptr_version=_version(batch.atom_ptr),
    )

def scalar_vector_attention(
    attention: Any,
    h: Tensor,
    v: Tensor,
    key_mask: Tensor,
    *,
    query_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply scalar q/k attention with one shared scalar weight matrix."""

    if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
        raise ValueError("attention expects [B,L,Dh] and [B,L,3,Dv]")
    if key_mask.shape != h.shape[:2]:
        raise ValueError("key_mask must have shape [B,L]")
    if query_mask is not None and query_mask.shape != h.shape[:2]:
        raise ValueError("query_mask must have shape [B,L]")
    batch, length = h.shape[:2]
    q = attention.q(h).reshape(
        batch, length, attention.heads, attention.scalar_head
    ).transpose(1, 2)
    k = attention.k(h).reshape(
        batch, length, attention.heads, attention.scalar_head
    ).transpose(1, 2)
    logits = torch.matmul(q, k.transpose(-1, -2)) / (attention.scalar_head ** 0.5)
    valid = key_mask.to(dtype=torch.bool).reshape(batch, 1, 1, length)
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = attention.dropout(torch.softmax(logits, dim=-1))

    scalar_value = attention.scalar_value(h).reshape(
        batch, length, attention.heads, attention.scalar_head
    ).permute(0, 2, 1, 3)
    scalar_mixed = torch.matmul(weights, scalar_value).transpose(1, 2).reshape(
        batch, length, attention.scalar_width
    )

    vector_value = attention.vector_value(v).reshape(
        batch, length, 3, attention.heads, attention.vector_head
    ).permute(0, 3, 1, 2, 4)
    vector_mixed = torch.einsum("bhij,bhjrc->bhirc", weights, vector_value)
    vector_mixed = vector_mixed.permute(0, 2, 3, 1, 4).reshape(
        batch, length, 3, attention.vector_width
    )
    scalar_output = attention.scalar_out(scalar_mixed)
    vector_output = attention.vector_out(vector_mixed)
    if query_mask is not None:
        query = query_mask.to(dtype=scalar_output.dtype).unsqueeze(-1)
        scalar_output = scalar_output * query
        vector_output = vector_output * query.unsqueeze(-1)
    return scalar_output, vector_output
def block_spatial_attention(
    h: Tensor,
    v: Tensor,
    batch: Any,
    attention: Any,
    layout: FactorizedLayout,
) -> tuple[Tensor, Tensor]:
    """Pool atoms, attend over padded local blocks, and broadcast to atoms."""

    k_count, num_atoms = h.shape[:2]
    if num_atoms != layout.num_atoms or layout.batch_size != int(batch.batch_size):
        raise ValueError("factorized layout does not match model inputs")
    if layout.num_groups == 0:
        return torch.zeros_like(h), torch.zeros_like(v)

    pooled_h = h.new_zeros((k_count, layout.num_groups, h.shape[-1]))
    pooled_h.index_add_(1, layout.atom_to_group, h)
    pooled_h = pooled_h / layout.group_counts.to(dtype=h.dtype).view(1, -1, 1)

    pooled_v = v.new_zeros((k_count, layout.num_groups, 3, v.shape[-1]))
    pooled_v.index_add_(1, layout.atom_to_group, v)
    pooled_v = pooled_v / layout.group_counts.to(dtype=v.dtype).view(1, -1, 1, 1)

    padded_index = layout.padded_group.clamp_min(0)
    valid = layout.valid_blocks
    block_h = pooled_h[:, padded_index] * valid.to(dtype=h.dtype).view(
        1, *valid.shape, 1
    )
    block_v = pooled_v[:, padded_index] * valid.to(dtype=v.dtype).view(
        1, *valid.shape, 1, 1
    )
    block_h = block_h.permute(1, 0, 2, 3).reshape(
        layout.batch_size * k_count, layout.max_blocks, h.shape[-1]
    )
    block_v = block_v.permute(1, 0, 2, 3, 4).reshape(
        layout.batch_size * k_count, layout.max_blocks, 3, v.shape[-1]
    )
    key_mask = valid.unsqueeze(1).expand(
        layout.batch_size, k_count, layout.max_blocks
    ).reshape(layout.batch_size * k_count, layout.max_blocks)
    query_mask = batch.token_mask.reshape(
        layout.batch_size * k_count, 1
    ).expand(layout.batch_size * k_count, layout.max_blocks)
    context_h, context_v = scalar_vector_attention(
        attention, block_h, block_v, key_mask, query_mask=query_mask
    )
    context_h = context_h.reshape(
        layout.batch_size, k_count, layout.max_blocks, h.shape[-1]
    ).permute(1, 0, 2, 3)
    context_v = context_v.reshape(
        layout.batch_size, k_count, layout.max_blocks, 3, v.shape[-1]
    ).permute(1, 0, 2, 3, 4)
    return (
        context_h[:, batch.abid, layout.atom_slot],
        context_v[:, batch.abid, layout.atom_slot],
    )


def atom_temporal_attention(
    h: Tensor,
    v: Tensor,
    batch: Any,
    attention: Any,
) -> tuple[Tensor, Tensor]:
    """Attend over latent time for every atom in one dense call."""

    h_atoms = h.permute(1, 0, 2)
    v_atoms = v.permute(1, 0, 2, 3)
    valid = batch.token_mask.index_select(0, batch.abid)
    output_h, output_v = scalar_vector_attention(
        attention, h_atoms, v_atoms, valid, query_mask=valid
    )
    return output_h.permute(1, 0, 2), output_v.permute(1, 0, 2, 3)


def factorized_block_forward(
    block: Any,
    h: Tensor,
    v: Tensor,
    condition: Tensor,
    batch: Any,
    layout: FactorizedLayout,
) -> tuple[Tensor, Tensor]:
    """Run one existing factorized block with the vectorized execution path."""

    h_norm, v_norm, h_gate, v_gate = block.spatial_adaln(h, v, condition)
    spatial_h, spatial_v = block_spatial_attention(
        h_norm, v_norm, batch, block.spatial, layout
    )
    h = h + torch.tanh(h_gate) * spatial_h
    v = v + torch.tanh(v_gate).unsqueeze(-2) * spatial_v

    h_norm, v_norm, h_gate, v_gate = block.temporal_adaln(h, v, condition)
    temporal_h, temporal_v = atom_temporal_attention(
        h_norm, v_norm, batch, block.temporal
    )
    h = h + torch.tanh(h_gate) * temporal_h
    v = v + torch.tanh(v_gate).unsqueeze(-2) * temporal_v

    v_pre = v
    h_norm, v_norm, h_gate, v_gate = block.ffn_adaln(h, v, condition)
    scalar_vector = v_pre if block.ffn_norm_source == "pre_adaln" else v_norm
    ffn_h, ffn_v = block.ffn(
        h_norm, v_norm, scalar_vector=scalar_vector
    )
    h = h + torch.tanh(h_gate) * ffn_h
    v = v + torch.tanh(v_gate).unsqueeze(-2) * ffn_v
    return h, v


__all__ = [
    "ATTENTION_BACKEND",
    "FactorizedLayout",
    "atom_temporal_attention",
    "block_spatial_attention",
    "build_factorized_layout",
    "factorized_block_forward",
    "scalar_vector_attention",
]
