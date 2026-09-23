"""Differentiable physical features for actual source-distribution samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


SCALE_NAMES = (
    "residue_rmsf_A",
    "displacement_squared_A2",
    "internal_distance_increment_A",
    "internal_distance_increment_product_A2",
)


@dataclass(frozen=True)
class SampledFeatureScales:
    """Train-only fixed physical scales; values never depend on one condition."""

    residue_rmsf_A: float
    displacement_squared_A2: float
    internal_distance_increment_A: float
    internal_distance_increment_product_A2: float

    @classmethod
    def resolve(cls, value: Mapping[str, Any]) -> "SampledFeatureScales":
        unknown = sorted(set(value) - set(SCALE_NAMES))
        missing = sorted(set(SCALE_NAMES) - set(value))
        if unknown or missing:
            raise ValueError(f"sampled feature scale keys differ: missing={missing}, unknown={unknown}")
        result = cls(**{name: float(value[name]) for name in SCALE_NAMES})
        for name, scale in asdict(result).items():
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"sampled feature scale {name!r} must be finite and positive")
        return result

    def contract(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureScaleMoments:
    sum_squares: Mapping[str, Tensor]
    counts: Mapping[str, Tensor]


@dataclass(frozen=True)
class SampledAuxiliaryLoss:
    energy_score: Tensor
    observed_bond: Tensor
    energy_applicable_count: Tensor
    bond_applicable_count: Tensor
    available_group_counts: Mapping[str, int]


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"value is missing {name!r}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"value is missing {name!r}")
    return getattr(value, name)


def _unique_undirected_pairs(pair_index: Tensor) -> Tensor:
    if pair_index.numel() == 0:
        return pair_index.reshape(2, 0).to(dtype=torch.long)
    pairs = torch.sort(pair_index.to(dtype=torch.long), dim=0).values
    keep = pairs[0] != pairs[1]
    pairs = pairs[:, keep]
    if pairs.numel() == 0:
        return pairs.reshape(2, 0)
    return torch.unique(pairs.transpose(0, 1), dim=0, sorted=True).transpose(0, 1)


def _detached_kabsch_rotation(source: Tensor, reference: Tensor) -> Tensor:
    """Return a stable detached row-vector rotation for observed-reference alignment."""

    if source.shape != reference.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("Kabsch inputs must have matching [N,3] shapes")
    with torch.no_grad():
        if source.shape[0] < 3:
            return torch.eye(3, device=source.device, dtype=source.dtype)
        # CUDA SVD has no BF16 implementation. Merely casting the operands is
        # insufficient under autocast because the covariance matmul is cast
        # back to BF16 before it reaches SVD, so keep the full solve in FP32.
        with torch.autocast(device_type=source.device.type, enabled=False):
            source_fp32 = source.float()
            reference_fp32 = reference.float()
            source_centered = source_fp32 - source_fp32.mean(dim=0)
            reference_centered = reference_fp32 - reference_fp32.mean(dim=0)
            left, _singular, right_transpose = torch.linalg.svd(
                source_centered.transpose(0, 1) @ reference_centered,
                full_matrices=False,
            )
            rotation = left @ right_transpose
            if bool(torch.linalg.det(rotation) < 0):
                left = left.clone()
                left[:, -1] *= -1
                rotation = left @ right_transpose
            if not torch.isfinite(rotation).all():
                raise FloatingPointError("non-finite observed-reference Kabsch rotation")
        return rotation.to(dtype=source.dtype)


def _align_query_to_observed_reference(
    query_coordinates: Tensor,
    reference_coordinates: Tensor,
    nodes: Tensor,
    alignment_nodes: Tensor,
) -> Tensor:
    """Align valid query frames using a detached rotation and differentiable centering."""

    if query_coordinates.ndim != 3 or query_coordinates.shape[-1] != 3:
        raise ValueError("query coordinates must have shape [Q,N,3]")
    reference = reference_coordinates.index_select(0, alignment_nodes).float()
    reference_center = reference.mean(dim=0)
    aligned: list[Tensor] = []
    for frame in query_coordinates:
        source = frame.index_select(0, alignment_nodes).float()
        source_center = source.mean(dim=0)
        rotation = _detached_kabsch_rotation(source, reference)
        selected = frame.index_select(0, nodes).float()
        aligned.append((selected - source_center) @ rotation + reference_center)
    return torch.stack(aligned)


def _sample_metadata(batch: Any, sample: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    context = _field(batch, "observed_context")
    topology = context.latent.topology
    abid = topology.abid
    loss_mask = context.loss_mask.to(dtype=torch.bool)
    coordinate_batch = _field(batch, "coordinate_batch")
    align_mask = coordinate_batch.align_mask.to(device=abid.device, dtype=torch.bool)
    nodes = torch.nonzero((abid == sample) & loss_mask, as_tuple=False).flatten()
    alignment_nodes = torch.nonzero((abid == sample) & align_mask, as_tuple=False).flatten()
    if nodes.numel() == 0 or alignment_nodes.numel() == 0:
        raise ValueError("each sampled condition requires loss and alignment atoms")
    observed_last = context.latent.frame_mask[sample].sum().to(dtype=torch.long) - 1
    if int(observed_last) < 0:
        raise ValueError("sampled condition has no valid observed frame")
    reference = context.coordinates[observed_last]
    query = _field(batch, "query")
    frames = torch.nonzero(query.frame_mask[sample], as_tuple=False).flatten()
    if frames.numel() == 0:
        raise ValueError("sampled condition has no valid query frame")
    block_id = topology.block_id.index_select(0, nodes)
    return nodes, alignment_nodes, reference, frames, block_id


def _sample_pairs(batch: Any, sample: int, nodes: Tensor) -> Tensor:
    context = _field(batch, "observed_context")
    topology = context.latent.topology
    pair = _unique_undirected_pairs(topology.covalent_bond_index)
    if pair.numel() == 0:
        return pair
    pair_sample = topology.abid.index_select(0, pair[0])
    if not torch.equal(pair_sample, topology.abid.index_select(0, pair[1])):
        raise ValueError("sampled feature topology contains a cross-condition pair")
    member = torch.zeros(topology.num_atoms, device=nodes.device, dtype=torch.bool)
    member[nodes] = True
    keep = (
        (pair_sample == sample)
        & member.index_select(0, pair[0])
        & member.index_select(0, pair[1])
    )
    return pair[:, keep]


def _raw_condition_features(coordinates: Tensor, batch: Any, sample: int) -> dict[str, Tensor]:
    nodes, alignment_nodes, reference, frames, block_id = _sample_metadata(batch, sample)
    selected_query = coordinates.index_select(0, frames)
    aligned = _align_query_to_observed_reference(
        selected_query,
        reference,
        nodes,
        alignment_nodes,
    )
    reference_nodes = reference.index_select(0, nodes).float()

    centered = aligned - aligned.mean(dim=0, keepdim=True)
    atom_rmsf = centered.square().sum(dim=-1).mean(dim=0).clamp_min(0.0).sqrt()
    residue_rmsf = torch.stack([
        atom_rmsf[block_id == block].mean()
        for block in torch.unique(block_id, sorted=True)
    ])

    displacement_squared = (aligned - reference_nodes).square().sum(dim=-1).mean(dim=1)
    sequence = torch.cat((reference_nodes.unsqueeze(0), aligned), dim=0)
    adjacent_squared = (sequence[1:] - sequence[:-1]).square().sum(dim=-1).mean(dim=1)

    pairs = _sample_pairs(batch, sample, nodes)
    internal_increment = coordinates.new_empty((0,), dtype=torch.float32)
    internal_product = coordinates.new_empty((0,), dtype=torch.float32)
    if pairs.numel():
        source, destination = pairs
        reference_length = torch.linalg.vector_norm(
            reference.index_select(0, source).float() - reference.index_select(0, destination).float(),
            dim=-1,
        )
        future_length = torch.linalg.vector_norm(
            selected_query.index_select(1, source).float()
            - selected_query.index_select(1, destination).float(),
            dim=-1,
        )
        length_sequence = torch.cat((reference_length.unsqueeze(0), future_length), dim=0)
        increments = length_sequence[1:] - length_sequence[:-1]
        internal_increment = increments.reshape(-1)
        if increments.shape[0] > 1:
            internal_product = (increments[1:] * increments[:-1]).reshape(-1)

    result = {
        "residue_rmsf_A": residue_rmsf.reshape(-1),
        "displacement_squared_A2": torch.cat((displacement_squared, adjacent_squared)),
        "internal_distance_increment_A": internal_increment,
        "internal_distance_increment_product_A2": internal_product,
    }
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("sampled physical feature contains NaN or Inf")
    return result


def feature_scale_moments(coordinates: Tensor, batch: Any) -> FeatureScaleMoments:
    """Accumulate target-only moments used to fit one fixed training scale per feature type."""

    context = _field(batch, "observed_context")
    sums = {name: coordinates.new_zeros((), dtype=torch.float64) for name in SCALE_NAMES}
    counts = {name: coordinates.new_zeros((), dtype=torch.long) for name in SCALE_NAMES}
    for sample in range(context.latent.batch_size):
        features = _raw_condition_features(coordinates, batch, sample)
        for name, value in features.items():
            sums[name] = sums[name] + value.double().square().sum()
            counts[name] = counts[name] + value.numel()
    return FeatureScaleMoments(sum_squares=sums, counts=counts)


def resolve_feature_scales(
    moments: Sequence[FeatureScaleMoments],
    *,
    minimum_scale: float = 1.0e-6,
) -> tuple[SampledFeatureScales, dict[str, Any]]:
    if not moments:
        raise ValueError("feature scale fitting requires training moments")
    if not math.isfinite(float(minimum_scale)) or float(minimum_scale) <= 0.0:
        raise ValueError("minimum feature scale must be finite and positive")
    values: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for name in SCALE_NAMES:
        total = sum((item.sum_squares[name].detach().cpu().double() for item in moments), torch.zeros((), dtype=torch.float64))
        count = sum(int(item.counts[name].detach().cpu()) for item in moments)
        if count == 0:
            raise ValueError(f"training calibration has no structurally eligible {name} values")
        rms = math.sqrt(float(total) / count)
        fallback = not math.isfinite(rms) or rms < float(minimum_scale)
        values[name] = 1.0 if fallback else rms
        diagnostics[name] = {
            "value_count": count,
            "raw_rms": rms,
            "resolved_scale": values[name],
            "constant_feature_unit_scale_fallback": fallback,
        }
    return SampledFeatureScales.resolve(values), diagnostics


def _scaled_groups(
    coordinates: Tensor,
    batch: Any,
    sample: int,
    scales: SampledFeatureScales,
) -> dict[str, Tensor]:
    raw = _raw_condition_features(coordinates, batch, sample)
    groups: dict[str, Tensor] = {
        "residue_rmsf": raw["residue_rmsf_A"] / scales.residue_rmsf_A,
        "displacement": raw["displacement_squared_A2"] / scales.displacement_squared_A2,
    }
    internal = []
    if raw["internal_distance_increment_A"].numel():
        internal.append(
            raw["internal_distance_increment_A"] / scales.internal_distance_increment_A
        )
    if raw["internal_distance_increment_product_A2"].numel():
        internal.append(
            raw["internal_distance_increment_product_A2"]
            / scales.internal_distance_increment_product_A2
        )
    if internal:
        groups["internal_distance"] = torch.cat(internal)
    return groups


def _feature_distance(first: Mapping[str, Tensor], second: Mapping[str, Tensor]) -> Tensor:
    if set(first) != set(second):
        raise ValueError("sampled feature groups differ within one condition")
    distances: list[Tensor] = []
    for name in sorted(first):
        if first[name].shape != second[name].shape or first[name].numel() == 0:
            raise ValueError(f"sampled feature coordinate mismatch for {name}")
        distances.append((first[name] - second[name]).square().mean().add(1.0e-12).sqrt())
    if not distances:
        raise ValueError("sampled condition has no structurally eligible feature group")
    return torch.stack(distances).mean()


def energy_score_loss(
    sampled_coordinates: Sequence[Tensor],
    target_coordinates: Tensor,
    batch: Any,
    scales: SampledFeatureScales,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Compute the unbiased K=2 energy score independently for every condition."""

    if len(sampled_coordinates) != 2:
        raise ValueError("sampled energy score is frozen at K=2")
    if any(value.shape != target_coordinates.shape for value in sampled_coordinates):
        raise ValueError("sampled and target coordinates must share [Q,N,3]")
    context = _field(batch, "observed_context")
    losses: list[Tensor] = []
    group_counts: dict[str, int] = {}
    for sample in range(context.latent.batch_size):
        target = _scaled_groups(target_coordinates, batch, sample, scales)
        generated = [
            _scaled_groups(coordinates, batch, sample, scales)
            for coordinates in sampled_coordinates
        ]
        if any(set(value) != set(target) for value in generated):
            raise ValueError("sampled feature eligibility changed with predicted coordinates")
        for name in target:
            group_counts[name] = group_counts.get(name, 0) + 1
        data_term = 0.5 * (
            _feature_distance(generated[0], target)
            + _feature_distance(generated[1], target)
        )
        diversity_term = 0.5 * _feature_distance(generated[0], generated[1])
        losses.append(data_term - diversity_term)
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("sampled energy score is non-finite")
    count = loss.new_tensor(len(losses), dtype=torch.long)
    return loss, count, group_counts


def observed_reference_bond_loss(
    sampled_coordinates: Sequence[Tensor],
    batch: Any,
) -> tuple[Tensor, Tensor]:
    """Penalize actual-sample bond lengths against the last observed frame only."""

    if not sampled_coordinates:
        raise ValueError("observed bond loss requires actual samples")
    context = _field(batch, "observed_context")
    topology = context.latent.topology
    pair = _unique_undirected_pairs(topology.covalent_bond_index)
    sample_losses: list[Tensor] = []
    for sample in range(context.latent.batch_size):
        nodes, _alignment, reference, frames, _block = _sample_metadata(batch, sample)
        member = torch.zeros(topology.num_atoms, device=nodes.device, dtype=torch.bool)
        member[nodes] = True
        if pair.numel():
            keep = (
                (topology.abid.index_select(0, pair[0]) == sample)
                & member.index_select(0, pair[0])
                & member.index_select(0, pair[1])
            )
            local = pair[:, keep]
        else:
            local = pair
        if local.numel() == 0:
            continue
        source, destination = local
        observed_length = torch.linalg.vector_norm(
            reference.index_select(0, source).float() - reference.index_select(0, destination).float(),
            dim=-1,
        )
        draw_losses = []
        for coordinates in sampled_coordinates:
            predicted = coordinates.index_select(0, frames)
            predicted_length = torch.linalg.vector_norm(
                predicted.index_select(1, source).float()
                - predicted.index_select(1, destination).float(),
                dim=-1,
            )
            draw_losses.append((predicted_length - observed_length.unsqueeze(0)).square().mean())
        sample_losses.append(torch.stack(draw_losses).mean())
    if not sample_losses:
        zero = sampled_coordinates[0].sum() * 0.0
        return zero, zero.new_zeros((), dtype=torch.long)
    loss = torch.stack(sample_losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("actual-sample observed bond loss is non-finite")
    return loss, loss.new_tensor(len(sample_losses), dtype=torch.long)


def sampled_auxiliary_loss(
    sampled_coordinates: Sequence[Tensor],
    target_coordinates: Tensor,
    batch: Any,
    scales: SampledFeatureScales,
) -> SampledAuxiliaryLoss:
    energy, energy_count, group_counts = energy_score_loss(
        sampled_coordinates,
        target_coordinates,
        batch,
        scales,
    )
    bond, bond_count = observed_reference_bond_loss(sampled_coordinates, batch)
    return SampledAuxiliaryLoss(
        energy_score=energy,
        observed_bond=bond,
        energy_applicable_count=energy_count,
        bond_applicable_count=bond_count,
        available_group_counts=group_counts,
    )
