"""TRAIN-window bond/1–3 angle normal bands for an RF endpoint prediction.

This auxiliary reads fixed batch topology, never infers edges from coordinates,
uses no random numbers, and does not call the model. Coordinates are Angstrom.
"""
from collections.abc import Mapping

import torch

from ..data.batch import ClipBatch


def _topology(batch: ClipBatch, atoms: int):
    """Unique valid bonds and incident-bond pairs for unique centered angles."""
    bonds = batch.bond_index
    if bonds.ndim != 2 or bonds.shape[0] != 2:
        raise ValueError('bond_index must be [2,E]')
    if bonds.numel() and bool(((bonds < 0) | (bonds >= atoms)).any()):
        raise ValueError('Out-of-range covalent atom index')
    if bonds.numel() and bool((batch.abid[bonds[0]] != batch.abid[bonds[1]]).any()):
        raise ValueError('Cross-system covalent edge')
    lo, hi = bonds.min(dim=0).values, bonds.max(dim=0).values
    valid = (lo != hi) & batch.loss_mask[lo] & batch.loss_mask[hi]
    codes = torch.unique(lo[valid] * atoms + hi[valid], sorted=True)
    bonds = torch.stack((codes // atoms, codes % atoms))
    count = bonds.shape[1]
    if count == 0:
        empty = bonds.new_empty((0,))
        return bonds, torch.stack((empty, empty)), torch.stack((empty, empty)), torch.stack((empty, empty, empty))

    # Sort the two endpoints of each bond into per-center adjacency lists.
    # Angular work scales with covalent degree, never with all atom pairs.
    centers = bonds.flatten()
    neighbors = bonds.flip(0).flatten()
    order = torch.argsort(centers * atoms + neighbors)
    centers, neighbors = centers[order], neighbors[order]
    edge = torch.arange(count, device=bonds.device).repeat(2)[order]
    sign = torch.cat((torch.ones_like(codes), -torch.ones_like(codes)))[order]
    degree = torch.bincount(centers, minlength=atoms)
    maximum_degree = int(degree.max())
    slots = torch.triu_indices(maximum_degree, maximum_degree, offset=1, device=bonds.device)
    occupied = slots[1][None, :] < degree[:, None]
    start = degree.cumsum(0) - degree
    left = (start[:, None] + slots[0][None, :])[occupied]
    right = (start[:, None] + slots[1][None, :])[occupied]
    angle_edges = torch.stack((edge[left], edge[right]))
    angle_signs = torch.stack((sign[left], sign[right]))
    angle_atoms = torch.stack((neighbors[left], centers[left], neighbors[right]))
    return bonds, angle_edges, angle_signs, angle_atoms


def _normal_band(target, *, floor: float, relative: float, mad_scale: float):
    # torch.median uses the lower middle entry for even Q; quantile implements
    # the conventional midpoint median used by the reference-MD diagnostics.
    median = torch.quantile(target, 0.5, dim=0)
    mad = torch.quantile((target - median).abs(), 0.5, dim=0)
    width = torch.maximum(torch.full_like(median, floor), mad_scale * mad)
    return median, torch.maximum(width, relative * median.abs())


def _component_means(feature_loss, feature_atoms, batch):
    """Return [B,2] means/counts; mixed features belong to both touched groups."""
    systems = batch.abid[feature_atoms[0]]
    ligand = batch.component_id[feature_atoms] > 0
    membership = torch.stack(((~ligand).any(dim=0), ligand.any(dim=0)), dim=1)
    group = (2 * systems[:, None] + torch.arange(2, device=systems.device)[None, :])[membership]
    values = feature_loss[:, None].expand(-1, 2)[membership]
    sums = feature_loss.new_zeros(2 * batch.batch_size).index_add(0, group, values)
    counts = feature_loss.new_zeros(2 * batch.batch_size).index_add(0, group, torch.ones_like(values))
    return (sums / counts.clamp_min(1)).reshape(-1, 2), counts.reshape(-1, 2)


def local_geometry_band_loss(
    endpoint_A: torch.Tensor,
    target_A: torch.Tensor,
    batch: ClipBatch,
    progress: torch.Tensor,
    settings: Mapping,
) -> tuple[torch.Tensor, dict]:
    """Unweighted endpoint auxiliary and detached scalar metrics.

    ``endpoint_A`` and TRAIN ``target_A`` are [Q,N_total,3], and ``progress``
    is [B]. All supplied query frames must be valid; ``batch.frame_mask``
    describes observed history and is deliberately not reused as a future mask.
    Every endpoint of a true bond/angle must pass ``batch.loss_mask``. Residue,
    block and radius filters are never applied. ``component_id == 0`` denotes
    protein, and positive IDs denote the ligand group, matching preprocessing.

    Required settings: min_progress, bond_floor_A, bond_relative, mad_scale,
    cos_floor. The caller owns ``weight`` and applies it once to the return.
    Target medians/MADs are detached. Each normalized band excess is squared;
    average frames/features, available bond/angle types, available protein/
    ligand groups, then ALL B systems. Early-flow and topology-empty systems
    contribute zero without changing that denominator. Cross-component true
    features count in both touched groups. Missing feature types are omitted
    from their component mean, rather than treated as passing observations.
    """
    if endpoint_A.ndim != 3 or endpoint_A.shape[-1] != 3 or endpoint_A.shape != target_A.shape:
        raise ValueError('endpoint_A and target_A must have identical [Q,N,3] shape')
    queries, atoms, _ = endpoint_A.shape
    if queries == 0 or atoms == 0 or batch.batch_size == 0:
        raise ValueError('Nonempty query frames, atoms and systems required')
    if progress.shape != (batch.batch_size,):
        raise ValueError('progress must be [B]')
    for name in ('abid', 'component_id', 'loss_mask'):
        value = getattr(batch, name)
        if value.shape != (atoms,) or value.device != endpoint_A.device:
            raise ValueError(f'{name} must be [N] on the coordinate device')
    if target_A.device != endpoint_A.device or progress.device != endpoint_A.device or batch.bond_index.device != endpoint_A.device:
        raise ValueError('Coordinates, progress and topology must share a device')
    if bool(((batch.abid < 0) | (batch.abid >= batch.batch_size) | (batch.component_id < 0)).any()):
        raise ValueError('Invalid packed system/component identifiers')
    if not bool((torch.isfinite(progress) & (progress >= 0) & (progress <= 1)).all()):
        raise ValueError('Flow progress must be finite and in [0,1]')

    minimum = float(settings['min_progress'])
    bond_floor = float(settings['bond_floor_A'])
    bond_relative = float(settings['bond_relative'])
    mad_scale = float(settings['mad_scale'])
    cos_floor = float(settings['cos_floor'])
    if not (0 <= minimum <= 1 and 0 < bond_floor < float('inf') and
            0 <= bond_relative < float('inf') and 0 <= mad_scale < float('inf') and
            0 < cos_floor < float('inf')):
        raise ValueError('Invalid explicit normal-band settings')
    active = progress.detach() >= minimum
    bonds, angle_edges, angle_signs, angle_atoms = _topology(batch, atoms)
    # Float32 geometry remains stable under mixed precision; retain float64 for
    # numerical derivative checks. No gradients enter the target statistics.
    dtype = torch.float64 if endpoint_A.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=endpoint_A.device.type, enabled=False):
        predicted = endpoint_A.to(dtype=dtype)
        reference = target_A.detach().to(dtype=dtype)
        if not bool(torch.isfinite(reference[:, batch.loss_mask]).all()):
            raise ValueError('Nonfinite TRAIN target coordinates on valid atoms')
        if not bool(torch.isfinite(predicted[:, batch.loss_mask & active[batch.abid]]).all()):
            raise ValueError('Nonfinite endpoint coordinates on active valid atoms')
        # Inactive endpoints cannot leak NaN/Inf into a zero-weight expression.
        predicted = torch.where(active[batch.abid][None, :, None], predicted, reference)
        # An empty reduction attaches zero gradients without reading masked NaNs.
        zero = endpoint_A.reshape(-1)[:0].sum().to(dtype=dtype)
        bond_means = zero + predicted.new_zeros((batch.batch_size, 2))
        angle_means = bond_means
        bond_counts = predicted.new_zeros((batch.batch_size, 2))
        angle_counts = bond_counts
        if bonds.shape[1]:
            predicted_vectors = predicted[:, bonds[1]] - predicted[:, bonds[0]]
            reference_vectors = reference[:, bonds[1]] - reference[:, bonds[0]]
            predicted_lengths = torch.linalg.vector_norm(predicted_vectors, dim=-1)
            reference_lengths = torch.linalg.vector_norm(reference_vectors, dim=-1)
            if bool((reference_lengths <= 1e-8).any()):
                raise ValueError('Degenerate TRAIN covalent bond; normal angle is undefined')
            median, width = _normal_band(reference_lengths, floor=bond_floor,
                                         relative=bond_relative, mad_scale=mad_scale)
            bond_error = (torch.relu((predicted_lengths - median).abs() - width) / width).square()
            bond_means, bond_counts = _component_means(bond_error.mean(dim=0), bonds, batch)
            if angle_edges.shape[1]:
                # Both bond and angle losses reuse these same vectors/norms.
                pred_unit = predicted_vectors / predicted_lengths.clamp_min(1e-8)[..., None]
                ref_unit = reference_vectors / reference_lengths[..., None]
                sign = (angle_signs[0] * angle_signs[1]).to(dtype=dtype)
                pred_cos = (pred_unit[:, angle_edges[0]] * pred_unit[:, angle_edges[1]]).sum(-1) * sign
                ref_cos = (ref_unit[:, angle_edges[0]] * ref_unit[:, angle_edges[1]]).sum(-1) * sign
                pred_cos, ref_cos = pred_cos.clamp(-1, 1), ref_cos.clamp(-1, 1)
                median, width = _normal_band(ref_cos, floor=cos_floor, relative=0., mad_scale=mad_scale)
                angle_error = (torch.relu((pred_cos - median).abs() - width) / width).square()
                angle_means, angle_counts = _component_means(angle_error.mean(dim=0), angle_atoms, batch)
        bond_present, angle_present = bond_counts > 0, angle_counts > 0
        kinds = bond_present.to(dtype) + angle_present.to(dtype)
        component_means = (bond_means + angle_means) / kinds.clamp_min(1)
        components = (kinds > 0).sum(dim=1)
        system_means = component_means.sum(dim=1) / components.clamp_min(1)
        loss = (system_means * active).mean()
        bond_metric = (bond_means.sum(dim=1) / bond_present.sum(dim=1).clamp_min(1) * active).mean()
        angle_metric = (angle_means.sum(dim=1) / angle_present.sum(dim=1).clamp_min(1) * active).mean()
        values = torch.stack((loss, bond_metric, angle_metric,
                              (component_means[:, 0] * active).mean(),
                              (component_means[:, 1] * active).mean(),
                              active.to(dtype).mean())).detach().cpu().tolist()
    metrics = dict(zip(('local_geometry_loss', 'local_geometry_bond_loss', 'local_geometry_angle_loss',
                        'local_geometry_protein_loss', 'local_geometry_ligand_loss',
                        'local_geometry_active_fraction'), values))
    metrics.update(local_geometry_bonds=bonds.shape[1], local_geometry_angles=angle_edges.shape[1],
                   local_geometry_systems=batch.batch_size)
    return loss, metrics


def ligand_geometry_band_loss(endpoint_A, target_A, batch, settings, *, band_reference_A=None):
    """Only positive-component internal bonds/angles, equal ligands then systems.

    Reuse the old S topology and detached TRAIN bands. Compute covalent vectors
    and norms once for both features. Protein atoms never enter a denominator;
    positive components are separate ligands. No torsion/chirality/plane loss.
    """
    from dataclasses import replace
    if endpoint_A.ndim!=3 or endpoint_A.shape!=target_A.shape or endpoint_A.shape[-1]!=3:
        raise ValueError('Expected equal [Q,N,3] endpoint/target in Angstrom')
    if band_reference_A is not None:
        if band_reference_A.ndim!=3 or band_reference_A.shape[1:]!=target_A.shape[1:] or len(band_reference_A)<8:
            raise ValueError('Explicit TRAIN band support must be multi-frame same-atom Angstrom coordinates')
    reference_A=target_A if band_reference_A is None else band_reference_A
    valid=batch.loss_mask & (batch.component_id>0)
    internal=(batch.component_id[batch.bond_index[0]]==batch.component_id[batch.bond_index[1]])
    lig=replace(batch,loss_mask=valid,bond_index=batch.bond_index[:,internal])
    bonds,angle_edges,signs,angle_atoms=_topology(lig,endpoint_A.shape[1])
    pairs=torch.stack((batch.abid,batch.component_id),dim=1)
    keys,groups=torch.unique(pairs,dim=0,return_inverse=True)
    groups_count=len(keys)
    def means(values, feature_atoms):
        group=groups[feature_atoms[0]]
        sums=values.new_zeros(groups_count).index_add(0,group,values)
        counts=values.new_zeros(groups_count).index_add(0,group,torch.ones_like(values))
        return sums/counts.clamp_min(1),counts
    dtype=torch.float64 if endpoint_A.dtype==torch.float64 else torch.float32
    with torch.autocast(device_type=endpoint_A.device.type,enabled=False):
        predicted=endpoint_A.to(dtype);reference=reference_A.detach().to(dtype)
        if not bool(torch.isfinite(predicted[:,valid]).all() and torch.isfinite(reference[:,valid]).all()):
            raise ValueError('Nonfinite ligand endpoint/target')
        zero=endpoint_A.reshape(-1)[:0].sum().to(dtype)
        bm=zero+predicted.new_zeros(groups_count);am=bm
        bc=predicted.new_zeros(groups_count);ac=bc
        if bonds.shape[1]:
            pv=predicted[:,bonds[1]]-predicted[:,bonds[0]]
            rv=reference[:,bonds[1]]-reference[:,bonds[0]]
            pl=torch.linalg.vector_norm(pv,dim=-1);rl=torch.linalg.vector_norm(rv,dim=-1)
            if bool((rl<=1e-8).any()): raise ValueError('Degenerate TRAIN ligand bond')
            median,width=_normal_band(rl,floor=float(settings['bond_floor_A']),
                relative=float(settings['bond_relative']),mad_scale=float(settings['mad_scale']))
            be=(torch.relu((pl-median).abs()-width)/width).square()
            bm,bc=means(be.mean(0),bonds)
            if angle_edges.shape[1]:
                pu=pv/pl.clamp_min(1e-8)[...,None];ru=rv/rl[...,None]
                sign=(signs[0]*signs[1]).to(dtype)
                pc=((pu[:,angle_edges[0]]*pu[:,angle_edges[1]]).sum(-1)*sign).clamp(-1,1)
                rc=((ru[:,angle_edges[0]]*ru[:,angle_edges[1]]).sum(-1)*sign).clamp(-1,1)
                median,width=_normal_band(rc,floor=float(settings['cos_floor']),relative=0.,
                    mad_scale=float(settings['mad_scale']))
                ae=(torch.relu((pc-median).abs()-width)/width).square()
                am,ac=means(ae.mean(0),angle_atoms)
        present=(bc>0)|(ac>0)
        kinds=(bc>0).to(dtype)+(ac>0).to(dtype)
        component=(bm+am)/kinds.clamp_min(1)
        def system_mean(values, has):
            sums=values.new_zeros(batch.batch_size).index_add(0,keys[:,0],values*has)
            count=values.new_zeros(batch.batch_size).index_add(0,keys[:,0],has.to(dtype))
            return (sums/count.clamp_min(1)).sum()/(count>0).sum().clamp_min(1)
        loss=system_mean(component,present)
        bl=system_mean(bm,bc>0);al=system_mean(am,ac>0)
        values=torch.stack((loss,bl,al)).detach().cpu().tolist()
    return loss,dict(local_geometry_loss=values[0],local_geometry_bond_loss=values[1],
        local_geometry_angle_loss=values[2],local_geometry_protein_loss=0.,
        local_geometry_ligand_loss=values[0],local_geometry_bonds=bonds.shape[1],
        local_geometry_angles=angle_edges.shape[1],ligand_components=int(present.sum()),
        ligand_systems=int(torch.unique(keys[present,0]).numel()),ligand_atoms=int(valid.sum()),
        geometry_band_frames=len(reference_A),
        topology_scope='same-positive-component internal true bonds/1-3 angles')
