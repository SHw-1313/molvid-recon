"""Observed-reference handedness margin, not a coordinate-restoration loss.

Signed volumes are Å³. The squared hinge on relative oriented volume is
dimensionless. It is zero above a small positive margin, so it does not force
generated conformations back to observed bond angles or motion amplitudes.
Current automatic topology selection covers protein Cα only, not ligand R/S.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class ChiralReference:
    indices: torch.Tensor  # [C,4], ordered center,N,C,CB for protein Cα
    system: torch.Tensor  # [C], packed system ID
    volume: torch.Tensor  # [C], detached last-observed signed volume Å³
    batch_size: int


def signed_volume(coordinates, indices):
    """Coordinates [...,N,3], ordered quadruplets [C,4], result [...,C]."""
    center, a, b, c = indices.T
    return (torch.cross(coordinates[..., a, :]-coordinates[..., center, :],
                        coordinates[..., b, :]-coordinates[..., center, :], dim=-1)
            * (coordinates[..., c, :]-coordinates[..., center, :])).sum(-1)


def observed_ca_reference(*, observed, atom_identity, block_id, abid, frame_mask,
                          loss_mask, bond_index, minimum_volume_A3=.2):
    """Build only from observed coordinates and supplied atom/bond topology.

    Exclude missing/masked centers and near-planar observations. Require all
    three Cα bonds; atom names alone never establish covalent stereochemistry.
    CPU topology enumeration is explicit; no future coordinates are accepted.
    """
    if not math.isfinite(minimum_volume_A3) or minimum_volume_A3 <= 0:
        raise ValueError('positive finite reference-volume threshold required')
    n = observed.shape[1]
    if observed.ndim != 3 or observed.shape[-1] != 3 or len(atom_identity) != n:
        raise ValueError('observed [H,N,3] and N atom identities required')
    groups = defaultdict(dict)
    samples = abid.detach().cpu().tolist()
    blocks = block_id.detach().cpu().tolist()
    mask = loss_mask.detach().cpu().bool().tolist()
    edges = {tuple(sorted(edge)) for edge in bond_index.detach().cpu().T.tolist()}
    for i, (identity, sample, block) in enumerate(zip(atom_identity, samples, blocks)):
        name = str(identity).rsplit(':', 2)[-2].strip().upper()
        key = (sample, block)
        if name in groups[key]:
            raise ValueError('ambiguous atom name in packed residue')
        groups[key][name] = i
    selected, system = [], []
    for (sample, _), atoms in sorted(groups.items()):
        names = ('CA','N','C','CB')
        if not all(name in atoms for name in names):
            continue
        ids = [atoms[name] for name in names]
        if not all(mask[i] for i in ids):
            continue
        if not all(tuple(sorted((ids[0], j))) in edges for j in ids[1:]):
            raise ValueError('named Cα center lacks supplied covalent bonds')
        selected.append(ids); system.append(sample)
    indices = torch.tensor(selected, device=observed.device, dtype=torch.long).reshape(-1,4)
    system = torch.tensor(system, device=observed.device, dtype=torch.long)
    last = frame_mask.to(observed.device).sum(1).long()-1
    if bool((last < 0).any()):
        raise ValueError('each packed system requires observed frames')
    with torch.no_grad():
        reference = observed.detach()[last[abid], torch.arange(n, device=observed.device)].float()
        volume = signed_volume(reference, indices)
        valid = volume.abs() > minimum_volume_A3
    return ChiralReference(indices[valid], system[valid], volume[valid], frame_mask.shape[0])


def handedness_margin_loss(coordinates, reference, *, frame_mask, margin=.1):
    """System-equal mean squared hinge over valid centers/query frames.

    Reference tensors are detached, including when a caller constructs its own
    ChiralReference. No penalty is applied to centers with correct volume above
    the margin. A zero gradient is possible at total geometric collapse; this
    is not a hard stereochemical constraint or a universal geometry repair.
    """
    if not math.isfinite(margin) or not 0 < margin <= 1:
        raise ValueError('margin must lie in (0,1]')
    if frame_mask.shape != (reference.batch_size, coordinates.shape[0]):
        raise ValueError('query frame mask must be [B,Q]')
    if reference.indices.numel() == 0:
        return coordinates.sum()*0.
    ref = reference.volume.detach().to(coordinates)
    if not bool(torch.isfinite(ref).all()) or bool((ref == 0).any()):
        raise ValueError('finite nonzero reference volumes required')
    valid = frame_mask.to(coordinates.device).bool()[reference.system].T
    # Invalid query frames must not contribute even through NaN derivatives.
    xyz = coordinates.float()[:, reference.indices]
    xyz = torch.where(valid[:, :, None, None], xyz, 0.)
    u, v, w = (xyz[:, :, j]-xyz[:, :, 0] for j in (1,2,3))
    ratio = (torch.cross(u, v, dim=-1)*w).sum(-1) * ref.sign() / ref.abs()
    penalty = (margin-ratio).relu().square() * valid
    sums = penalty.new_zeros(reference.batch_size).index_add_(0, reference.system, penalty.sum(0))
    counts = penalty.new_zeros(reference.batch_size).index_add_(0, reference.system, valid.sum(0).float())
    applicable = counts > 0
    return (sums[applicable]/counts[applicable]).mean() if bool(applicable.any()) else coordinates.sum()*0.
