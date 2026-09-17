from __future__ import annotations

import torch

from module.state_detail_codec_v2 import StaticTopologyMetadata, StateDetailLatent
from module.state_detail_latent_adapter import DiTLatentBatch, LatentFieldSet


def make_batch(
    ratio: int,
    *,
    width: int = 4,
    atom_counts: tuple[int, ...] = (2, 3),
    seed: int = 11,
    invalid_last_token: bool = False,
    loss_mask: torch.Tensor | None = None,
) -> DiTLatentBatch:
    ratio = int(ratio)
    k_count = 16 // ratio
    batch_size = len(atom_counts)
    generator = torch.Generator().manual_seed(seed)
    n_atoms = sum(atom_counts)
    abid = torch.cat(
        [torch.full((count,), index, dtype=torch.long) for index, count in enumerate(atom_counts)]
    )
    block_id_parts = []
    for index, count in enumerate(atom_counts):
        block_id_parts.append(torch.arange(count, dtype=torch.long) // 2 + index * 10)
    block_id = torch.cat(block_id_parts)
    token_mask = torch.ones(batch_size, k_count, dtype=torch.bool)
    block_frame_mask = torch.ones(batch_size, k_count, ratio, dtype=torch.bool)
    if invalid_last_token:
        token_mask[-1, -1] = False
        block_frame_mask[-1, -1] = False
    detail_valid = token_mask.clone()
    detail_component_mask = detail_valid.unsqueeze(-1).expand(batch_size, k_count, ratio - 1).clone()
    fields = LatentFieldSet(
        torch.randn(k_count, n_atoms, width, generator=generator),
        torch.randn(k_count, n_atoms, width, generator=generator),
        torch.randn(k_count, n_atoms, 3, width, generator=generator),
        torch.randn(k_count, n_atoms, 3, width, generator=generator),
    )
    time = torch.arange(16, dtype=torch.float32).expand(batch_size, -1).clone()
    block_time = time.reshape(batch_size, k_count, ratio)
    atom_ptr = torch.tensor([0, *torch.cumsum(torch.tensor(atom_counts), dim=0).tolist()], dtype=torch.long)
    topology = StaticTopologyMetadata(
        atom_type=torch.arange(n_atoms, dtype=torch.long) % 8,
        block_type=block_id % 5,
        abid=abid,
        block_id=block_id,
        component_id=abid,
        atom_ptr=atom_ptr,
        covalent_bond_index=torch.empty(2, 0, dtype=torch.long),
        covalent_bond_type=torch.empty(0, dtype=torch.long),
        topology_id=tuple(f"topology-{index}" for index in range(batch_size)),
        sample_id=tuple(f"sample-{index}" for index in range(batch_size)),
    )
    if loss_mask is None:
        loss_mask = torch.ones(n_atoms, dtype=torch.bool)
    else:
        loss_mask = torch.as_tensor(loss_mask, dtype=torch.bool).flatten()
        if loss_mask.numel() != n_atoms:
            raise ValueError("loss_mask fixture must have shape [N]")
    return DiTLatentBatch(
        fields=fields,
        token_mask=token_mask,
        detail_valid=detail_valid,
        detail_component_mask=detail_component_mask,
        block_frame_mask=block_frame_mask,
        block_time_ps=block_time,
        frame_time_ps=time,
        abid=abid,
        sample_origin=torch.zeros(batch_size, 3),
        topology=topology,
        ratio=ratio,
        mode=f"ratio{ratio}_state_detail",
        width=width,
        loss_mask=loss_mask,
        atom_type=topology.atom_type,
        block_type=topology.block_type,
        component_id=topology.component_id,
        block_id=topology.block_id,
        atom_ptr=atom_ptr,
    ).zero_invalid()


def make_latent(ratio: int, *, width: int = 4, seed: int = 17) -> StateDetailLatent:
    batch = make_batch(ratio, width=width, seed=seed)
    components = ratio - 1
    generator = torch.Generator().manual_seed(seed + 1)
    return StateDetailLatent(
        state_h=batch.state_h,
        state_v=batch.state_v,
        detail_h=batch.detail_h,
        detail_v=batch.detail_v,
        raw_detail_h=torch.randn(batch.tokens, batch.num_atoms, components, width, generator=generator),
        raw_detail_v=torch.randn(batch.tokens, batch.num_atoms, components, 3, width, generator=generator),
        token_mask=batch.token_mask,
        detail_valid=batch.detail_valid,
        detail_component_mask=batch.detail_component_mask,
        block_frame_mask=batch.block_frame_mask,
        frame_time_ps=batch.frame_time_ps,
        block_time_ps=batch.block_time_ps,
        abid=batch.abid,
        sample_origin=batch.sample_origin,
        topology=batch.topology,
        ratio=ratio,
        mode=f"ratio{ratio}_state_detail",
        coefficient_order=("D01",) if ratio == 2 else ("Dmid", "D01", "D23"),
        width=width,
    )
