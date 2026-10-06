"""Experimental coordinate-state RF; features are not independent ODE states.

Coordinates and velocities are [Q,N,3], in Angstrom and Angstrom per unit
flow progress. Query times are physical ps, not flow progress. Conditioning
contains only observed coordinates. Targets belong exclusively to the loss.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import math

import torch
from torch import Tensor, nn

from ..codec.heads import EquivariantCoordinateHead
from ..codec.frame import FrameGeometryEncoder
from ..data.batch import ClipBatch
from ..dit.capacity import expand_coordinate_dit, validate_coordinate_depth
from ..latent.types import FrameLatentBatch, ObservedContext, QuerySpec, StructureCondition
from ..model import FrameJointCondition, FrameJointModel


@dataclass(frozen=True)
class CoordinateCondition:
    observed: ClipBatch
    context: ObservedContext
    temporal: FrameJointCondition | None
    task: str


class CoordinateFlowModel(nn.Module):
    """Reuse a compatible trunk, retire its latent decoder, predict XYZ flow.

    This is a changed architecture/warm start, NOT an exact A1 continuation.
    The original teacher stays frozen but current-query coordinate gradients
    pass through it. Its radius graph is rebuilt at each flow evaluation.
    The initial prototype explicitly supports only the legacy history path.
    """

    def __init__(self, parent: FrameJointModel, *, sigma_angstrom: float = 1.0,
                 train_query_geometry: bool = False, dit_depth: int = 4):
        super().__init__()
        validate_coordinate_depth(dit_depth)
        self._depth_expanded = False
        if parent.history_backend != "legacy":
            raise ValueError("coordinate prototype requires legacy observed history")
        if not math.isfinite(sigma_angstrom) or sigma_angstrom <= 0:
            raise ValueError("coordinate source sigma must be positive and finite")
        self.sigma_angstrom = float(sigma_angstrom)
        # Share ownership with the disposable loaded parent; callers must not
        # subsequently train/use that parent as an independent control.
        self.target_teacher = parent.target_teacher
        self.history_encoder = parent.history_encoder
        self.motion_encoder = parent.motion_encoder
        self.dit = parent.dit
        for name in ("frame_h_mean", "frame_h_std", "frame_v_rms"):
            self.register_buffer(name, getattr(parent, name).detach().clone())
        self.statistics_hash = parent.statistics_hash
        self.readout = EquivariantCoordinateHead(self.frame_h_mean.numel()).to(
            device=self.frame_h_mean.device,
        )
        nn.init.zeros_(self.readout.out.weight)
        self.history_encoder.set_trainable(True)
        if self.motion_encoder is not None:
            self.motion_encoder.requires_grad_(True)
        self.dit.set_trainable(True)
        self.target_teacher.requires_grad_(False)
        self.target_teacher.eval()
        self.query_encoder = None
        self.warm_start_report = {
            "kind": "changed_architecture_warm_start_fresh_optimizer_required",
            "reused_state_keys": sorted(k for k in parent.state_dict() if not k.startswith("decoder.")),
            "retired_state_keys": sorted(k for k in parent.state_dict() if k.startswith("decoder.")),
            "new_state_keys": ["readout." + k for k in self.readout.state_dict()],
            "initial_readout": "zero_velocity_not_parent_forecast_parity",
        }
        if type(train_query_geometry) is not bool:
            raise ValueError("train_query_geometry must be an explicit boolean")
        if train_query_geometry:
            self.enable_query_geometry()

        if dit_depth == 8:
            self.expand_depth()

    def expand_depth(self):
        """Expand the loaded parent before constructing a fresh optimizer."""
        report = expand_coordinate_dit(self.dit)
        self._depth_expanded = True
        mapping = {'dit.' + a: 'dit.' + b for a, b in report['old_to_new_state_keys'].items()}
        self.warm_start_report.update(
            kind='coordinate_depth_model_only_fresh_optimizer', depth_expansion=report,
            reused_state_keys=[mapping.get(k, k) for k in self.warm_start_report['reused_state_keys']],
            new_state_keys=self.warm_start_report['new_state_keys'] + ['dit.' + k for k in report['new_state_keys']])
        return report

    def enable_query_geometry(self) -> None:
        """Clone the LOADED teacher; no RNG draws or observed-path changes.

        Call before optimizer creation. Eval-mode numerical behavior is kept
        even during training; only geometry parameter gradients are enabled.
        """
        if self.query_encoder is not None:
            raise ValueError("query geometry already enabled")
        self.query_encoder = FrameGeometryEncoder(
            copy.deepcopy(self.target_teacher.frame_encoder),
            copy.deepcopy(self.target_teacher.coordinate_stem),
        ).requires_grad_(True).eval()
        # load_parent sets the real teacher to one frame per encoder call.
        # This is runtime metadata, not a state_dict tensor: copy it explicitly
        # to preserve arithmetic grouping and the existing activation budget.
        self.query_encoder.frame_chunk_size = self.target_teacher.frame_chunk_size
        self.query_encoder.checkpoint_frames = self.target_teacher.checkpoint_frames

    def set_query_geometry_recompute(self, enabled: bool) -> None:
        """Recompute query frame activations; frozen observations stay no-grad.

        This execution setting must be selected before a training/resume
        contract is captured. It does not change trainable ownership.
        """
        if type(enabled) is not bool:
            raise ValueError("query geometry recomputation requires an explicit boolean")
        encoder = self.query_encoder if self.query_encoder is not None else self.target_teacher
        encoder.checkpoint_frames = enabled

    def contract(self) -> dict:
        result = dict(schema="molvid.coordinate_flow.prototype.v1",
                    state="coordinates_angstrom_Q_N_3", source_sigma_angstrom=self.sigma_angstrom,
                    statistics_hash=self.statistics_hash, history_backend="legacy",
                    query_encoder=("trainable_query_copy_frozen_observation_current_radius_graph"
                                   if self.query_encoder is not None else
                                   "frozen_weights_coordinate_grad_current_radius_graph"),
                    static="corrupted_structure_condition_shared_spatial_trunk_no_physical_time",
                    decoder="equivariant_coordinate_velocity_readout")
        encoder = self.query_encoder if self.query_encoder is not None else self.target_teacher
        if encoder.checkpoint_frames:
            result["query_geometry_execution"] = "per_chunk_nonreentrant_recompute_preserve_rng_v1"
        if self._depth_expanded:
            result.update(dit_depth=self.dit.depth,
                dit_geometry_blocks_one_based=list(self.dit.geometry_block_indices),
                depth_expansion='identity_interleave_keep_geometry_count')
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_teacher.eval()
        if self.query_encoder is not None:
            self.query_encoder.eval()
        return self

    def prepare_batch(self, observed_cpu: ClipBatch) -> None:
        """Register supplied topology on CPU before transferring observations."""
        self.target_teacher.prepare_batch(observed_cpu)
        if self.query_encoder is not None:
            self.query_encoder.prepare_batch(observed_cpu)

    def prepare_condition(self, observed: ClipBatch, *, task: str = "dynamic") -> CoordinateCondition:
        """Static input must already be corrupted; clean targets are not accepted."""
        if task not in ("dynamic", "static") or (task == "static" and observed.frames != 1):
            raise ValueError("task must be dynamic or single-structure static")
        context = self.target_teacher.observed_context(observed)
        temporal = None
        if task == "dynamic":
            temporal = FrameJointCondition(
                self.history_encoder(context),
                self.motion_encoder(context) if self.motion_encoder is not None else None,
            )
        return CoordinateCondition(observed, context, temporal, task)

    def normalize(self, latent: FrameLatentBatch) -> FrameLatentBatch:
        return latent.with_features(
            (latent.h - self.frame_h_mean.to(latent.h)) / self.frame_h_std.to(latent.h),
            latent.v / self.frame_v_rms.to(latent.v), statistics_hash=self.statistics_hash,
        )

    @staticmethod
    def _validate_query(condition: CoordinateCondition, query: QuerySpec) -> None:
        batch = condition.observed
        if query.batch_size != batch.batch_size or query.frames < 1:
            raise ValueError("query dimensions disagree with observations")
        if query.time_ps.device != batch.x.device or query.frame_mask.device != batch.x.device:
            raise ValueError("query and observations must share a device")
        if not bool(query.frame_mask[:, 0].all()):
            raise ValueError("every system must have a valid first query")
        if condition.task == "dynamic":
            last = batch.time_ps.gather(1, (batch.frame_mask.sum(1).long() - 1)[:, None])
            if bool(((query.time_ps <= last) & query.frame_mask).any()):
                raise ValueError("dynamic query times must follow observed history")

    def _encode_state(self, coordinates: Tensor, condition: CoordinateCondition, query: QuerySpec) -> FrameLatentBatch:
        self._validate_query(condition, query)
        observed = condition.observed
        if coordinates.shape != (query.frames, observed.atom_count, 3):
            raise ValueError("coordinate flow state must be [Q,N,3]")
        # Canonical block centroids depend ONLY on current generated coordinates.
        # The frame encoder presently ignores bpos; never carry target metadata.
        _, group = torch.unique(observed.block_id, return_inverse=True)
        count = torch.bincount(group).to(coordinates)
        block_sum = coordinates.new_zeros(query.frames, count.numel(), 3)
        block_sum.index_add_(1, group, coordinates)
        bpos = (block_sum / count[None, :, None])[:, group]
        current = replace(observed, x=coordinates, bpos=bpos,
                          time_ps=query.time_ps, frame_mask=query.frame_mask,
                          delta_time_ps=query.time_ps[:, 1:] - query.time_ps[:, :-1])
        encoder = self.target_teacher if self.query_encoder is None else self.query_encoder
        encoded, _ = encoder.encode_with_coordinate_grad(
            current, reference_origin=condition.context.sample_origin,
        )
        return self.normalize(encoded)

    @staticmethod
    def _flow_time(flow_time: Tensor, condition: CoordinateCondition) -> Tensor:
        t = torch.as_tensor(flow_time, device=condition.observed.x.device, dtype=torch.float32)
        if t.shape != (condition.observed.batch_size,) or not bool(torch.isfinite(t).all()):
            raise ValueError("flow_time must be finite [B]")
        if bool(((t < 0) | (t > 1)).any()):
            raise ValueError("flow_time must lie in [0,1]")
        return t

    def forward(self, coordinates: Tensor, *, condition: CoordinateCondition,
                query: QuerySpec, flow_time: Tensor, checkpoint_blocks: bool = False) -> Tensor:
        if condition.task != "dynamic" or condition.temporal is None:
            raise ValueError("dynamic forward requires observed temporal condition")
        noisy = self._encode_state(coordinates, condition, query)
        features = self.dit(noisy, context=condition.context, query=query,
                            flow_time=self._flow_time(flow_time, condition),
                            history_memory=condition.temporal.history_memory,
                            motion_context=condition.temporal.motion_context,
                            checkpoint_blocks=checkpoint_blocks)
        return self.readout(features.h, features.v) * noisy.atom_frame_mask()[..., None]

    def forward_static(self, coordinates: Tensor, *, condition: CoordinateCondition,
                       flow_time: Tensor) -> Tensor:
        if condition.task != "static" or condition.temporal is not None:
            raise ValueError("static forward requires a corrupted single structure")
        # Storage clock only: forward_static explicitly bypasses physical time.
        query = QuerySpec(torch.zeros_like(condition.observed.time_ps), condition.observed.frame_mask)
        noisy = self._encode_state(coordinates, condition, query)
        features = self.dit.forward_static(noisy, condition=StructureCondition(condition.context),
                                           flow_time=self._flow_time(flow_time, condition))
        return self.readout(features.h, features.v) * noisy.atom_frame_mask()[..., None]

    def source(self, *, condition: CoordinateCondition, query: QuerySpec,
               generator: torch.Generator) -> Tensor:
        """Observed-last + iid Cartesian Gaussian; no independent feature noise."""
        if condition.task != "dynamic":
            raise ValueError("static RF source is the supplied corrupted structure")
        self._validate_query(condition, query)
        batch = condition.observed
        last = (batch.frame_mask.sum(1).long() - 1)[batch.abid]
        atoms = torch.arange(batch.atom_count, device=batch.x.device)
        center = batch.x[last, atoms][None].expand(query.frames, -1, -1)
        noise = torch.randn(center.shape, device=center.device, dtype=center.dtype, generator=generator)
        mask = query.frame_mask[:, :].index_select(0, batch.abid).T[..., None]
        return center + self.sigma_angstrom * noise * mask

    def rollout(self, source: Tensor, *, condition: CoordinateCondition, query: QuerySpec,
                steps: int) -> Tensor:
        """Differentiable actual-source Euler integration, not no-grad inference."""
        if steps < 1:
            raise ValueError("Euler steps must be positive")
        state = source
        for index in range(steps):
            progress = source.new_full((condition.observed.batch_size,), index / steps)
            state = state + self(state, condition=condition, query=query, flow_time=progress) / steps
        return state


def coordinate_flow_loss(predicted: Tensor, target_velocity: Tensor, *,
                         batch: ClipBatch, query: QuerySpec) -> Tensor:
    """System-equal masked coordinate-velocity MSE, units Angstrom squared."""
    if predicted.shape != target_velocity.shape or predicted.shape != (query.frames, batch.atom_count, 3):
        raise ValueError("coordinate velocity loss requires matching [Q,N,3]")
    mask = query.frame_mask.index_select(0, batch.abid).T & batch.loss_mask.bool()[None]
    error = (predicted.float() - target_velocity.detach().float()).square().mean(-1)
    sums = error.new_zeros(batch.batch_size).index_add_(0, batch.abid, (error * mask).sum(0))
    counts = error.new_zeros(batch.batch_size).index_add_(0, batch.abid, mask.sum(0).float())
    if bool((counts <= 0).any()):
        raise ValueError("each system needs supervised coordinate velocities")
    return (sums / counts).mean()
