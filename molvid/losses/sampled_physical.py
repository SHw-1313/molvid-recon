"""A-line actual-source losses in Å, with explicit packed condition reduction.

Targets enter motion losses only. Clash and bond-band use observed topology
and frozen train-derived tables. Chunked CUDA distances are the correctness
backend, not a claimed sparse implementation; there is no neighbor cap.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .distribution import _align_query_to_observed_reference, _sample_metadata, _sample_pairs


@dataclass(frozen=True)
class PhysicalAuxiliaryConfig:
    clash_weight: float = 0.0
    rmsf_weight: float = 0.0
    displacement_weight: float = 0.0
    bond_band_weight: float = 0.0
    rmsf_scale_A: float | None = None
    displacement_scale_A2: float | None = None
    # Keys are actual topology atom-type IDs; no assumed atomic-number mapping.
    vdw_radii_A: Mapping[str, float] | None = None
    allowed_overlap_A: float | None = None
    one_four_overlap_A: float | None = None
    # Keys: "min_atom_type:max_atom_type:bond_type", then element pair, then '*'.
    bond_band_widths_A: Mapping[str, float] | None = None
    table_provenance: Mapping[str, Any] | None = None
    distance_chunk_size: int = 256

    @classmethod
    def resolve(cls, value: Mapping[str, Any]) -> "PhysicalAuxiliaryConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown physical auxiliary keys: {sorted(unknown)}")
        result = cls(**value)
        for name in ("clash_weight", "rmsf_weight", "displacement_weight", "bond_band_weight"):
            weight = getattr(result, name)
            if isinstance(weight, bool) or not math.isfinite(float(weight)) or float(weight) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for weight, scale in ((result.rmsf_weight, result.rmsf_scale_A),
                              (result.displacement_weight, result.displacement_scale_A2)):
            if weight > 0 and (scale is None or not math.isfinite(float(scale)) or scale <= 0):
                raise ValueError("active motion group requires a positive fixed train-only scale")
        if isinstance(result.distance_chunk_size, bool) or not isinstance(result.distance_chunk_size, int) or result.distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be a positive integer")
        if result.clash_weight > 0:
            if not result.vdw_radii_A or result.allowed_overlap_A is None or result.one_four_overlap_A is None:
                raise ValueError("clash requires explicit frozen radii and topology-class overlap rules")
            for radius in result.vdw_radii_A.values():
                if not math.isfinite(float(radius)) or radius <= 0:
                    raise ValueError("vdW radii must be finite and positive")
            if not all(math.isfinite(float(v)) and v >= 0 for v in
                       (result.allowed_overlap_A, result.one_four_overlap_A)):
                raise ValueError("overlap tolerances must be finite and nonnegative")
            if result.one_four_overlap_A < result.allowed_overlap_A:
                raise ValueError("1–4 contacts must use the conservative (larger) overlap allowance")
        if result.bond_band_weight > 0:
            if not result.bond_band_widths_A:
                raise ValueError("bond band requires frozen training-derived widths")
            if any(not math.isfinite(float(v)) or v < 0 for v in result.bond_band_widths_A.values()):
                raise ValueError("band widths must be finite and nonnegative")
        if result.clash_weight > 0 or result.bond_band_weight > 0:
            provenance = result.table_provenance or {}
            if provenance.get("split") != "train" or not provenance.get("source_hash"):
                raise ValueError("physical tables require train split and source_hash provenance")
        return result

    def contract(self) -> dict[str, Any]:
        return asdict(self)


def motion_features(coordinates: Tensor, batch: Any, sample: int, *,
                    rmsf: bool, displacement: bool) -> dict[str, Tensor]:
    """Features for one condition, aligned to its last observed reference.

    vector_norm has the zero subgradient at zero motion; no sqrt(0) backward.
    No covalent increment/product group is calculated by this function.
    """
    nodes, alignment, reference, frames, block_id = _sample_metadata(batch, sample)
    aligned = _align_query_to_observed_reference(
        coordinates.index_select(0, frames), reference, nodes, alignment,
    )
    result = {}
    if rmsf:
        centered = aligned - aligned.mean(dim=0, keepdim=True)
        atom_rmsf = torch.linalg.vector_norm(centered.permute(1, 0, 2).flatten(1), dim=-1) / math.sqrt(len(frames))
        result["sampled_rmsf"] = torch.stack([
            atom_rmsf[block_id == block].mean() for block in torch.unique(block_id, sorted=True)
        ])
    if displacement:
        ref = reference.index_select(0, nodes).float()
        msd = (aligned - ref).square().sum(-1).mean(-1)
        sequence = torch.cat((ref.unsqueeze(0), aligned))
        increments = (sequence[1:] - sequence[:-1]).square().sum(-1).mean(-1)
        result["sampled_displacement"] = torch.cat((msd, increments))
    return result


def _topological_distances(topology: Any) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """CPU static metadata only: shortest graph distance <=2 and exactly 3."""
    adjacency = [set() for _ in range(topology.num_atoms)]
    for a, b in topology.covalent_bond_index.detach().cpu().T.tolist():
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    excluded, one_four = {}, {}
    for atom in range(topology.num_atoms):
        seen, frontier = {atom}, {atom}
        for depth in (1, 2, 3):
            frontier = set().union(*(adjacency[n] for n in frontier)) - seen if frontier else set()
            seen |= frontier
            if depth == 2:
                excluded[atom] = seen.copy()
            elif depth == 3:
                one_four[atom] = frontier.copy()
    return excluded, one_four


def _clash_condition(coordinates: Tensor, batch: Any, sample: int,
                     config: PhysicalAuxiliaryConfig, distances) -> tuple[Tensor, int]:
    nodes, _, _, frames, _ = _sample_metadata(batch, sample)
    topology = batch.observed_context.topology
    atom_ids = nodes.detach().cpu().tolist()
    atom_types = topology.atom_type.index_select(0, nodes).detach().cpu().tolist()
    radii = coordinates.new_tensor([config.vdw_radii_A[str(t)] for t in atom_types], dtype=torch.float32)
    total = coordinates.sum() * 0.0
    overlaps = 0
    excluded, one_four = distances
    for frame in frames.tolist():
        xyz = coordinates[frame].index_select(0, nodes).float()
        for start in range(0, len(atom_ids), config.distance_chunk_size):
            stop = min(start + config.distance_chunk_size, len(atom_ids))
            # Static topology exclusion includes real disulfide bonds.
            valid = torch.tensor([
                [j > i and b not in excluded[a] for j, b in enumerate(atom_ids)]
                for i, a in enumerate(atom_ids[start:stop], start)
            ], device=xyz.device, dtype=torch.bool)
            is14 = torch.tensor([
                [b in one_four[a] for b in atom_ids] for a in atom_ids[start:stop]
            ], device=xyz.device, dtype=torch.bool)
            allowance = torch.where(is14, config.one_four_overlap_A, config.allowed_overlap_A)
            minimum = (radii[start:stop, None] + radii[None, :] - allowance).clamp_min(0)
            # Candidate search is detached. Selected distances retain autograd.
            with torch.no_grad():
                delta = xyz[start:stop, None] - xyz[None, :]
                neighbor_distance = torch.linalg.vector_norm(delta, dim=-1)
                candidates = valid & (neighbor_distance < minimum)
                overlaps += int((valid & (neighbor_distance == 0)).sum())
                ii, jj = torch.where(candidates)
            distance = torch.linalg.vector_norm(xyz[start + ii] - xyz[jj], dim=-1)
            total = total + (minimum[ii, jj] - distance).relu().square().sum()
    # Each pair once; average by valid frames and atoms, never violation count.
    return total / (len(frames) * len(nodes)), overlaps


def _band_condition(coordinates: Tensor, batch: Any, sample: int,
                    config: PhysicalAuxiliaryConfig) -> tuple[Tensor, bool]:
    nodes, _, reference, frames, _ = _sample_metadata(batch, sample)
    pairs = _sample_pairs(batch, sample, nodes)
    if pairs.numel() == 0:
        return coordinates.sum() * 0.0, False
    topology = batch.observed_context.topology
    types = topology.atom_type.detach().cpu().tolist()
    bond_types = {}
    for (a, b), kind in zip(topology.covalent_bond_index.detach().cpu().T.tolist(),
                            topology.covalent_bond_type.detach().cpu().tolist()):
        key = (min(a, b), max(a, b))
        if key in bond_types and bond_types[key] != kind:
            raise ValueError("duplicate covalent edges disagree on bond type")
        bond_types[key] = kind
    widths = []
    for a, b in pairs.detach().cpu().T.tolist():
        lo, hi = sorted((types[a], types[b]))
        keys = (f"{lo}:{hi}:{bond_types[(min(a,b), max(a,b))]}", f"{lo}:{hi}", "*")
        width = next((config.bond_band_widths_A[k] for k in keys if k in config.bond_band_widths_A), None)
        if width is None:
            raise ValueError(f"missing frozen bond width for {keys}")
        widths.append(width)
    a, b = pairs
    ref_length = torch.linalg.vector_norm(reference[a].float() - reference[b].float(), dim=-1).detach()
    selected = coordinates.index_select(0, frames).float()
    length = torch.linalg.vector_norm(selected[:, a] - selected[:, b], dim=-1)
    width = length.new_tensor(widths)
    return ((length - ref_length).abs() - width).relu().square().mean(), True


def physical_auxiliary_loss(sampled_coordinates: Sequence[Tensor], batch: Any,
                            config: PhysicalAuxiliaryConfig):
    """Return raw group losses/counts/diagnostics for K2 decoded [Q,N,3] draws.

    Physical time is carried by query metadata; none of these Å/Å² features is
    labelled velocity. Targets are detached and only enter motion energy scores.
    """
    if len(sampled_coordinates) != 2:
        raise ValueError("physical auxiliary requires K=2")
    if any(x.device.type != "cuda" for x in sampled_coordinates):
        raise RuntimeError("physical auxiliary requires CUDA; no CPU fallback")
    expected_shape = (batch.query.frame_mask.shape[1], batch.observed_context.topology.num_atoms, 3)
    if any(tuple(x.shape) != expected_shape for x in sampled_coordinates):
        raise ValueError("actual-source coordinates must match packed query [Q,N,3]")
    weights = {"sampled_clash": config.clash_weight, "sampled_rmsf": config.rmsf_weight,
               "sampled_displacement": config.displacement_weight, "sampled_bond_band": config.bond_band_weight}
    values = {key: [] for key in weights}
    zero = sampled_coordinates[0].sum() * 0.0
    topology_distances = _topological_distances(batch.observed_context.topology) if config.clash_weight > 0 else None
    overlap_count = 0
    for sample in range(batch.observed_context.latent.batch_size):
        bucket = str(batch.coordinate_batch.time_bucket_id[sample])
        rmsf = config.rmsf_weight > 0 and not bucket.startswith("fixed_history_") and int(batch.query.frame_mask[sample].sum()) >= 8
        displacement = config.displacement_weight > 0
        if rmsf or displacement:
            with torch.no_grad():
                target = motion_features(batch.target_coordinates.detach(), batch, sample, rmsf=rmsf, displacement=displacement)
            draws = [motion_features(x, batch, sample, rmsf=rmsf, displacement=displacement) for x in sampled_coordinates]
            for name, truth in target.items():
                scale = config.rmsf_scale_A if name == "sampled_rmsf" else config.displacement_scale_A2
                def distance(a, b):
                    return torch.linalg.vector_norm((a - b).flatten()) / (math.sqrt(a.numel()) * scale)
                values[name].append(0.5 * distance(draws[0][name], truth) + 0.5 * distance(draws[1][name], truth)
                                    - 0.5 * distance(draws[0][name], draws[1][name]))
        if config.clash_weight > 0:
            draws = [_clash_condition(x, batch, sample, config, topology_distances) for x in sampled_coordinates]
            values["sampled_clash"].append(torch.stack([v for v, _ in draws]).mean())
            overlap_count += sum(count for _, count in draws)
        if config.bond_band_weight > 0:
            draws = [_band_condition(x, batch, sample, config) for x in sampled_coordinates]
            if draws[0][1]:
                values["sampled_bond_band"].append(torch.stack([v for v, _ in draws]).mean())
    raw = {key: torch.stack(items).mean() if items else zero for key, items in values.items()}
    counts = {key: zero.new_tensor(len(items), dtype=torch.long) for key, items in values.items()}
    return raw, counts, {"exact_overlap_pair_frames": overlap_count,
                         "exact_overlap_note": "finite zero subgradient; cannot guarantee separation at exact coincidence",
                         "clash_backend": "chunked_cuda_distance_correctness_reference_no_neighbor_cap"}
