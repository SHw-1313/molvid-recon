"""D2: static single-structure supervision through the shared RF/DiT trunk.

No optimizer lives here. Dynamic and static gradients meet in the existing
joint trainer before its single clipping/AdamW/scheduler update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..data.batch import ClipBatch
from ..flow.objective import FrameRectifiedFlowObjective, frame_endpoint_from_velocity
from ..latent.types import FrameLatentBatch, StructureCondition
from .batches import prepare_batch_then_to_device
from .static_decoder import (PreparedStaticDecoderBatch, StaticPerturbationConfig,
    StaticDecoderLossConfig, prepare_static_decoder_batch, static_decoder_loss)

if TYPE_CHECKING:
    from ..model import FrameJointModel


@dataclass(frozen=True)
class StaticBackboneConfig:
    """Weights are frozen after train-only calibration; no adaptive controller."""
    weight: float
    seed: int = 20261004
    coordinate: float = 1.0
    bond: float = 0.1
    rollout_steps: int = 4
    rollout_cadence: int = 8
    rollout_weight: float = 1.0
    gradient_scope: str = 'shared'

    def __post_init__(self):
        if self.gradient_scope not in {'shared', 'decoder_only'}:
            raise ValueError('static gradient scope must be shared or decoder_only')
        for name in ('weight', 'coordinate', 'bond', 'rollout_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid D2 {name}')
        if self.weight <= 0 or self.rollout_steps < 1 or self.rollout_cadence < 1:
            raise ValueError('D2 requires positive calibrated weight, steps and cadence')
        if any(type(getattr(self, name)) is not int for name in ('seed', 'rollout_steps', 'rollout_cadence')) or self.seed < 0:
            raise ValueError('D2 seed, steps and cadence must be nonnegative/positive integers')

    def contract(self):
        return dict(schema='molvid.static_backbone.v2', **asdict(self),
                    task_kind='static', time_modules='bypassed', optimizer_steps='one_joint_step',
                    geometry_scope='coordinate_and_bond_v1; angle/clash not yet enabled')


def backward_static_loss(loss: Tensor, model: FrameJointModel, config: StaticBackboneConfig) -> None:
    """Same forward/task in both arms; restrict only the auxiliary backward.

    Existing dynamic gradients are accumulated, never cleared or rescaled.
    `inputs` prevents even zero static grads/moments on the excluded trunk.
    The decoder-only arm's RF term is a logged constant w.r.t. its parameters.
    """
    parameters = tuple(p for n, p in model.named_parameters() if p.requires_grad
        and static_parameter_role(n) == 'shared'
        and (config.gradient_scope == 'shared' or n.startswith('decoder.')))
    if not parameters:
        raise ValueError('no parameters eligible for static gradients')
    torch.autograd.backward(loss, inputs=parameters)


@dataclass(frozen=True)
class PreparedStaticBackboneBatch:
    reconstruction: PreparedStaticDecoderBatch
    condition: StructureCondition
    normalized_target: FrameLatentBatch
    source_center: FrameLatentBatch


@torch.no_grad()
def prepare_static_backbone_batch(
    model: FrameJointModel, batch: ClipBatch, *, device: torch.device,
    generator: torch.Generator,
    perturbation: StaticPerturbationConfig | None = None,
) -> PreparedStaticBackboneBatch:
    # Repeated middle sigma realizes the predeclared 1:2:1 distribution.
    prepared = prepare_static_decoder_batch(model.target_teacher, batch, device=device,
        generator=generator, perturbation=perturbation or StaticPerturbationConfig((0.1, 0.3, 0.3, 0.8), 0.0))
    clean = prepare_batch_then_to_device(model.target_teacher, batch, device)
    target, _ = model.target_teacher(clean, reference_origin=prepared.observed_context.sample_origin)
    return PreparedStaticBackboneBatch(prepared, StructureCondition(prepared.observed_context),
        model.normalize(target), model.normalize(prepared.observed_context.latent))


def static_backbone_loss(
    model: FrameJointModel, batch: PreparedStaticBackboneBatch, config: StaticBackboneConfig,
    *, generator: torch.Generator, successful_updates: int,
) -> tuple[Tensor, dict[str, float]]:
    """RF + differentiable endpoint/actual-source geometry; target is loss-only."""
    objective = FrameRectifiedFlowObjective(model.source_config)
    draw = objective.sample(batch.normalized_target, batch.source_center, generator=generator)
    velocity = model.dit.forward_static(draw.interpolated, condition=batch.condition, flow_time=draw.flow_time)
    flow = objective.loss(velocity, draw.target_velocity)
    endpoint = frame_endpoint_from_velocity(draw.interpolated, velocity, draw.flow_time)
    decoded = model.decoder.forward_static(model.inverse(endpoint), condition=batch.condition)
    geometry_config = StaticDecoderLossConfig(coordinate=config.coordinate, bond=config.bond)
    geometry = static_decoder_loss(decoded, batch.reconstruction, geometry_config)
    total = flow.total + geometry.total
    metrics = {'static_flow': float(flow.total.detach()),
               **{f'static_{k}': float(v.detach()) for k, v in geometry.raw.items()}}
    # The same independent static source draw is reused, not a clean target
    # interpolation masquerading as generated-source training.
    if successful_updates % config.rollout_cadence == 0 and config.rollout_weight > 0:
        state = draw.source
        for step in range(config.rollout_steps):
            progress = state.h.new_full((state.batch_size,), step / config.rollout_steps)
            update = model.dit.forward_static(state, condition=batch.condition, flow_time=progress)
            state = state.with_features(state.h + update.h / config.rollout_steps,
                                        state.v + update.v / config.rollout_steps)
        sampled = model.decoder.forward_static(model.inverse(state), condition=batch.condition)
        sampled_loss = static_decoder_loss(sampled, batch.reconstruction, geometry_config)
        total = total + config.rollout_weight * sampled_loss.total
        metrics['static_rollout_geometry'] = float(sampled_loss.total.detach())
    metrics['static_unweighted_total'] = float(total.detach())
    metrics['static_weighted_total'] = float((config.weight * total).detach())
    return config.weight * total, metrics


def static_parameter_role(name: str) -> str:
    """Explicit shared/temporal/teacher partition, also used by gradient gates."""
    if name.startswith('target_teacher.'):
        return 'teacher'
    if name.startswith(('history_encoder.', 'motion_encoder.')):
        return 'temporal'
    if name.startswith('dit.query_time.') or name.startswith('decoder.time_embedding.'):
        return 'temporal'
    if name.startswith(('dit.blocks.', 'decoder.blocks.')):
        part = name.split('.')[3]
        if part.startswith(('history', 'temporal', 'motion_context')):
            return 'temporal'
    return 'shared'


def calibrated_static_weight(dynamic_norms: list[float], static_norms: list[float], *, ratio: float = 0.25) -> float:
    """Calibrate on >=16 train pairs, never on validation metrics."""
    if len(dynamic_norms) != len(static_norms) or len(dynamic_norms) < 16:
        raise ValueError('D2 calibration requires at least 16 paired training batches')
    if not 0 < ratio <= 0.5 or any(not math.isfinite(v) or v <= 0 for v in dynamic_norms + static_norms):
        raise ValueError('invalid calibration ratio or gradient norm')
    from statistics import median
    return ratio * median(dynamic_norms) / median(static_norms)
