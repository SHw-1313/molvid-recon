"""Readable staged optimization for the Frame Joint v1 model."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ..checkpoints import load_training_checkpoint, restore_rng_state, save_training_checkpoint
from ..flow.objective import FrameFlowLoss, FrameRectifiedFlowObjective
from ..latent.types import FrameLatentBatch
from ..losses.geometry import FutureBondLoss, future_bond_distance_loss
from ..model import FrameJointModel, FrameJointOutput
from ..runtime import sha256_file
from .batches import PreparedFrameJointBatch
from .dit import module_state_hash


STAGES = ("decoder_warmup", "flow_start", "joint", "continuation", "frozen_decoder")


@dataclass(frozen=True)
class JointLossConfig:
    """Single resolved source for every explicit Frame Joint loss weight."""

    bond_enabled: bool
    generated_bond: float
    clean_coordinate: float = 1.0
    clean_bond: float = 0.1
    near_coordinate: float = 0.1
    near_bond: float = 0.01
    generated_bond_min_flow_time: float = 0.75
    near_min_flow_time: float = 0.9

    @classmethod
    def resolve(cls, value: Mapping[str, Any]) -> "JointLossConfig":
        allowed = {
            "bond_enabled", "generated_bond", "clean_coordinate", "clean_bond", "near_coordinate",
            "near_bond", "generated_bond_min_flow_time", "near_min_flow_time",
            "calibration_target_ratio",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unknown loss keys: {unknown}")
        enabled_value = value.get("bond_enabled", True)
        if not isinstance(enabled_value, bool):
            raise ValueError("bond_enabled must be a boolean")
        enabled = enabled_value
        generated = value.get("generated_bond")
        if generated is None:
            raise ValueError("generated_bond must be calibrated or explicitly set")
        result = cls(
            bond_enabled=enabled,
            generated_bond=float(generated) if enabled else 0.0,
            clean_coordinate=float(value.get("clean_coordinate", 1.0)),
            clean_bond=float(value.get("clean_bond", 0.1)) if enabled else 0.0,
            near_coordinate=float(value.get("near_coordinate", 0.1)),
            near_bond=float(value.get("near_bond", 0.01)) if enabled else 0.0,
            generated_bond_min_flow_time=float(value.get("generated_bond_min_flow_time", 0.75)),
            near_min_flow_time=float(value.get("near_min_flow_time", 0.9)),
        )
        for name, weight in asdict(result).items():
            if name == "bond_enabled":
                continue
            if not math.isfinite(float(weight)) or float(weight) < 0:
                raise ValueError(f"loss setting {name!r} must be finite and non-negative")
        if result.generated_bond_min_flow_time > 1 or result.near_min_flow_time > 1:
            raise ValueError("flow-time thresholds must lie in [0,1]")
        return result

    def contract(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JointStepLoss:
    total: Tensor
    raw: Mapping[str, Tensor]
    weighted: Mapping[str, Tensor]
    applicable_samples: Mapping[str, Tensor]
    flow: FrameFlowLoss
    bond_diagnostics: Mapping[str, FutureBondLoss]


@dataclass(frozen=True)
class WarmStartReport:
    """Auditable model-only initialization from a changed experiment."""

    source_path: str
    source_sha256: str
    source_step: int
    source_model_contract: Mapping[str, Any]
    target_model_contract: Mapping[str, Any]
    loaded: tuple[str, ...]
    initialized: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]
    optimizer_restored: bool = False
    scheduler_restored: bool = False
    cursor_restored: bool = False
    rng_restored: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_equal_coordinate_mse(
    prediction: Tensor,
    target: Tensor,
    batch: PreparedFrameJointBatch,
    eligible: Tensor,
) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("coordinate loss expects matching [Q,N,3] tensors")
    context = batch.observed_context
    atom_valid = context.loss_mask.unsqueeze(0)
    frame_valid = batch.query.frame_mask.index_select(0, context.latent.abid).transpose(0, 1)
    sample_valid = eligible.index_select(0, context.latent.abid).unsqueeze(0)
    mask = atom_valid & frame_valid & sample_valid
    error = (prediction.float() - target.float()).square().mean(dim=-1)
    sample = context.latent.abid.unsqueeze(0).expand(prediction.shape[0], -1)
    sums = error.new_zeros((context.latent.batch_size,))
    counts = error.new_zeros((context.latent.batch_size,))
    sums.scatter_add_(0, sample[mask], error[mask])
    counts.scatter_add_(0, sample[mask], torch.ones_like(error[mask]))
    applicable = counts > 0
    safe = sums / counts.clamp_min(1)
    return (safe * applicable).sum() / applicable.sum().clamp_min(1), applicable.sum()


def _future_bond(
    prediction: Tensor,
    batch: PreparedFrameJointBatch,
    eligible: Tensor,
) -> FutureBondLoss:
    context = batch.observed_context
    return future_bond_distance_loss(
        prediction.float(),
        batch.target_coordinates.float(),
        frame_mask=batch.query.frame_mask,
        observed_frames=torch.zeros_like(batch.query.frame_mask),
        loss_mask=context.loss_mask,
        bond_index=context.topology.covalent_bond_index,
        abid=context.latent.abid,
        eligible_samples=eligible,
    )


def frame_joint_loss(
    output: FrameJointOutput,
    flow_loss: FrameFlowLoss,
    batch: PreparedFrameJointBatch,
    flow_time: Tensor,
    *,
    stage: str,
    config: JointLossConfig,
) -> JointStepLoss:
    """Compute all raw and weighted terms from one transparent forward pass."""

    if stage not in STAGES:
        raise ValueError(f"unknown Frame Joint stage {stage!r}")
    zero = flow_loss.total * 0.0
    all_samples = torch.ones_like(flow_time, dtype=torch.bool)
    generated_eligible = flow_time >= config.generated_bond_min_flow_time
    near_eligible = flow_time >= config.near_min_flow_time
    raw: dict[str, Tensor] = {
        "flow": flow_loss.total,
        "flow_h": flow_loss.h,
        "flow_v": flow_loss.v,
        "generated_bond": zero,
        "clean_coordinate": zero,
        "clean_bond": zero,
        "near_coordinate": zero,
        "near_bond": zero,
    }
    zero_count = torch.zeros((), device=flow_time.device, dtype=torch.long)
    applicable_samples: dict[str, Tensor] = {
        "flow": flow_loss.applicable_samples.sum(),
        "generated_bond": zero_count,
        "clean_coordinate": zero_count,
        "clean_bond": zero_count,
        "near_coordinate": zero_count,
        "near_bond": zero_count,
    }
    diagnostics: dict[str, FutureBondLoss] = {}
    if output.clean is not None:
        raw["clean_coordinate"], applicable_samples["clean_coordinate"] = _sample_equal_coordinate_mse(
            output.clean.coordinates, batch.target_coordinates, batch, all_samples
        )
        diagnostics["clean_bond"] = _future_bond(output.clean.coordinates, batch, all_samples)
        raw["clean_bond"] = diagnostics["clean_bond"].loss
        applicable_samples["clean_bond"] = diagnostics["clean_bond"].applicable_count
    if output.generated is not None:
        diagnostics["generated_bond"] = _future_bond(
            output.generated.coordinates, batch, generated_eligible
        )
        raw["generated_bond"] = diagnostics["generated_bond"].loss
        applicable_samples["generated_bond"] = diagnostics["generated_bond"].applicable_count
    if output.near is not None:
        raw["near_coordinate"], applicable_samples["near_coordinate"] = _sample_equal_coordinate_mse(
            output.near.coordinates, batch.target_coordinates, batch, near_eligible
        )
        diagnostics["near_bond"] = _future_bond(output.near.coordinates, batch, near_eligible)
        raw["near_bond"] = diagnostics["near_bond"].loss
        applicable_samples["near_bond"] = diagnostics["near_bond"].applicable_count

    active_flow = stage in {"flow_start", "joint", "continuation", "frozen_decoder"}
    active_clean = stage in {"decoder_warmup", "joint", "continuation"}
    active_joint = stage in {"joint", "continuation", "frozen_decoder"}
    weights = {
        "flow": 1.0 if active_flow else 0.0,
        "generated_bond": config.generated_bond if active_joint else 0.0,
        "clean_coordinate": config.clean_coordinate if active_clean else 0.0,
        "clean_bond": config.clean_bond if active_clean else 0.0,
        "near_coordinate": config.near_coordinate if active_joint else 0.0,
        "near_bond": config.near_bond if active_joint else 0.0,
    }
    weighted = {name: raw[name] * weight for name, weight in weights.items()}
    return JointStepLoss(
        total=sum(weighted.values(), zero),
        raw=raw,
        weighted=weighted,
        applicable_samples=applicable_samples,
        flow=flow_loss,
        bond_diagnostics=diagnostics,
    )


def _gradient_norm(loss: Tensor, parameters: Sequence[nn.Parameter], *, retain_graph: bool) -> Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    if not values:
        return loss.new_zeros(())
    return torch.stack(values).sum().sqrt()


def calibrate_generated_bond_weight(
    model: FrameJointModel,
    batches: Iterable[PreparedFrameJointBatch],
    *,
    generator: torch.Generator,
    target_ratio: float = 0.1,
) -> tuple[float, dict[str, Any]]:
    """Match generated-bond and flow DiT gradient norms on 16 train batches."""

    if not math.isfinite(float(target_ratio)) or float(target_ratio) <= 0:
        raise ValueError("calibration target_ratio must be finite and positive")
    parameters = [parameter for parameter in model.dit.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("generated-bond calibration requires trainable DiT parameters")
    objective = FrameRectifiedFlowObjective()
    flow_norms: list[Tensor] = []
    bond_norms: list[Tensor] = []
    was_training = model.training
    model.train()
    batch_count = 0
    for batch in batches:
        batch_count += 1
        if batch_count > 16:
            raise ValueError("generated-bond calibration received more than 16 mini-batches")
        device = batch.normalized_target.h.device
        flow_time = 0.75 + 0.25 * torch.rand(
            (batch.normalized_target.batch_size,),
            device=device,
            dtype=batch.normalized_target.h.dtype,
            generator=generator,
        )
        sample = objective.sample(
            batch.normalized_target,
            batch.source_center,
            generator=generator,
            flow_time=flow_time,
        )
        output = model(
            sample.interpolated,
            context=batch.observed_context,
            query=batch.query,
            flow_time=flow_time,
            decode_generated=True,
        )
        flow_loss = objective.loss(output.velocity, sample.target_velocity).total
        assert output.generated is not None
        bond_loss = _future_bond(
            output.generated.coordinates,
            batch,
            torch.ones_like(flow_time, dtype=torch.bool),
        ).loss
        flow_norms.append(_gradient_norm(flow_loss, parameters, retain_graph=True))
        bond_norms.append(_gradient_norm(bond_loss, parameters, retain_graph=False))
    if batch_count != 16:
        raise ValueError("generated-bond calibration requires exactly 16 train mini-batches")
    if not was_training:
        model.eval()
    flow = torch.stack(flow_norms)
    bond = torch.stack(bond_norms)
    if not torch.isfinite(flow).all() or not torch.isfinite(bond).all():
        raise FloatingPointError("non-finite generated-bond calibration gradient norm")
    flow_mean = flow.mean()
    bond_mean = bond.mean()
    if float(flow_mean) <= 0 or float(bond_mean) <= 0:
        raise RuntimeError(
            "zero calibration gradient; inspect the flow/decoder path before choosing lambda"
        )
    coefficient = float(target_ratio) * float(flow_mean / bond_mean)
    diagnostics = {
        "batch_count": 16,
        "target_bond_to_flow_dit_gradient_norm_ratio": float(target_ratio),
        "flow_gradient_norm_mean": float(flow_mean),
        "bond_gradient_norm_mean": float(bond_mean),
        "generated_bond_weight": coefficient,
        "flow_time_range": [0.75, 1.0],
    }
    return coefficient, diagnostics


def _component_parameter_groups(
    model: FrameJointModel,
    *,
    learning_rates: Mapping[str, float],
    weight_decay: float,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    component_modules: dict[str, list[nn.Module]] = {
        "history_encoder": [
            model.history_encoder,
            *([] if model.motion_encoder is None else [model.motion_encoder]),
        ],
        "dit": [model.dit],
        "decoder": [model.decoder],
    }
    for component, modules in component_modules.items():
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for module in modules:
            for name, parameter in module.named_parameters():
                if not parameter.requires_grad:
                    continue
                (no_decay if parameter.ndim == 1 or name.endswith("bias") else decay).append(parameter)
        for suffix, parameters, decay_value in (
            ("decay", decay, float(weight_decay)),
            ("no_decay", no_decay, 0.0),
        ):
            if parameters:
                groups.append({
                    "params": parameters,
                    "lr": float(learning_rates[component]),
                    "weight_decay": decay_value,
                    "group_name": f"{component}.{suffix}",
                })
    return groups


class FrameJointTrainer:
    """Stage-aware optimizer whose model call remains the DDP boundary."""

    def __init__(
        self,
        model: nn.Module,
        *,
        loss_config: JointLossConfig,
        learning_rates: Mapping[str, float],
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        stage: str = "joint",
        amp: bool = False,
        data_hash: str = "",
        teacher_artifact_sha256: str = "",
        schedule_contract: Mapping[str, Any] | None = None,
        continuation_parent: Mapping[str, Any] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
    ) -> None:
        self.model = model
        self.base_model = model.module if hasattr(model, "module") else model
        if not isinstance(self.base_model, FrameJointModel):
            raise TypeError("FrameJointTrainer requires FrameJointModel or DDP(FrameJointModel)")
        if any(parameter.requires_grad for parameter in self.base_model.target_teacher.parameters()):
            raise RuntimeError("target teacher must be frozen")
        self.loss_config = loss_config
        self.learning_rates = {name: float(learning_rates[name]) for name in ("history_encoder", "dit", "decoder")}
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)
        self.amp = bool(amp)
        self.data_hash = str(data_hash)
        self.teacher_artifact_sha256 = str(teacher_artifact_sha256)
        self.schedule_contract = dict(schedule_contract or {})
        self.flow = FrameRectifiedFlowObjective()
        self.optimizer = optimizer or torch.optim.AdamW(
            _component_parameter_groups(
                self.base_model,
                learning_rates=self.learning_rates,
                weight_decay=self.weight_decay,
            )
        )
        self.scheduler = scheduler
        self.step = 0
        self.successful_updates = 0
        self.continuation_parent = (
            None if continuation_parent is None else dict(continuation_parent)
        )
        self.stage = ""
        self.configure_stage(stage)

    def configure_stage(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown Frame Joint stage {stage!r}")
        active = {
            "decoder_warmup": {"decoder"},
            "flow_start": {"history_encoder", "dit"},
            "joint": {"history_encoder", "dit", "decoder"},
            "continuation": {"history_encoder", "dit", "decoder"},
            "frozen_decoder": {"history_encoder", "dit"},
        }[stage]
        for name in ("history_encoder", "dit", "decoder"):
            module = getattr(self.base_model, name)
            if name == "history_encoder":
                module.set_trainable(name in active)
            elif name == "dit":
                module.set_trainable(name in active)
            else:
                module.requires_grad_(name in active)
        if self.base_model.motion_encoder is not None:
            self.base_model.motion_encoder.requires_grad_("history_encoder" in active)
        self.base_model.target_teacher.requires_grad_(False)
        self.stage = stage

    def set_learning_rates(self, values: Mapping[str, float]) -> None:
        """Set named component rates without changing optimizer moments."""

        for group in self.optimizer.param_groups:
            name = str(group.get("group_name", "")).split(".", 1)[0]
            if name not in values:
                raise ValueError(f"missing learning rate for optimizer component {name!r}")
            value = float(values[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError("learning rates must be finite and non-negative")
            group["lr"] = value

    def _autocast(self):
        parameter = next(self.base_model.dit.parameters())
        if self.amp and parameter.device.type != "cuda":
            raise RuntimeError("Frame Joint BF16 training requires CUDA")
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def _distributed_total(self, losses: JointStepLoss) -> Tensor:
        """Scale each local mean by its own global applicable-sample count."""

        if not dist.is_available() or not dist.is_initialized():
            return losses.total
        names = tuple(losses.weighted)
        local = torch.stack([
            losses.applicable_samples[name].to(
                device=losses.total.device, dtype=torch.float32
            )
            for name in names
        ])
        global_counts = local.clone()
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        scales = local * dist.get_world_size() / global_counts.clamp_min(1.0)
        return sum(
            (losses.weighted[name] * scales[index] for index, name in enumerate(names)),
            losses.total * 0.0,
        )

    def train_step(
        self,
        batch: PreparedFrameJointBatch,
        *,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        self.model.train()
        self.base_model.target_teacher.eval()
        sample = self.flow.sample(
            batch.normalized_target,
            batch.source_center,
            generator=generator,
        )
        need_clean = self.stage in {"decoder_warmup", "joint", "continuation"}
        need_joint = self.stage in {"joint", "continuation", "frozen_decoder"}
        with self._autocast():
            output = self.model(
                sample.interpolated,
                context=batch.observed_context,
                query=batch.query,
                flow_time=sample.flow_time,
                clean_future=batch.target_future,
                decode_generated=need_joint,
                decode_clean=need_clean,
                decode_near=need_joint,
            )
            flow_loss = self.flow.loss(output.velocity, sample.target_velocity)
            losses = frame_joint_loss(
                output,
                flow_loss,
                batch,
                sample.flow_time,
                stage=self.stage,
                config=self.loss_config,
            )
        if not torch.isfinite(losses.total):
            raise FloatingPointError("non-finite Frame Joint loss")
        self.optimizer.zero_grad(set_to_none=True)
        scaled = self._distributed_total(losses)
        scaled.backward()
        active = [parameter for parameter in self.base_model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(active, self.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite Frame Joint gradient norm")
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step += 1
        self.successful_updates += 1
        result: dict[str, Any] = {
            "step": self.step,
            "stage": self.stage,
            "loss": float(losses.total.detach()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "flow_time_mean": float(sample.flow_time.mean().detach()),
            "batch_size": batch.observed_context.latent.batch_size,
        }
        result.update({f"raw_{name}": float(value.detach()) for name, value in losses.raw.items()})
        result.update({f"weighted_{name}": float(value.detach()) for name, value in losses.weighted.items()})
        return result

    def contracts(self) -> dict[str, Any]:
        return {
            "trainer": "molvid.frame_joint.v1",
            "model": dict(self.base_model.contract()),
            "statistics_hash": self.base_model.statistics_hash,
            "teacher_artifact_sha256": self.teacher_artifact_sha256,
            "teacher_state_hash": module_state_hash(self.base_model.target_teacher),
            "data_hash": self.data_hash,
            "loss": self.loss_config.contract(),
            "stage": self.stage,
            "optimization": {
                "base_learning_rates": dict(self.learning_rates),
                "weight_decay": self.weight_decay,
                "grad_clip": self.grad_clip,
                "amp_bfloat16": self.amp,
                "norm_bias_weight_decay": 0.0,
            },
            "schedule": dict(self.schedule_contract),
            "continuation_parent": self.continuation_parent,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        cursor: Mapping[str, Any],
        generator: torch.Generator,
        rank_states: Mapping[str, Any] | None = None,
    ) -> Path:
        return save_training_checkpoint(
            path,
            model=self.base_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.step,
            cursor=cursor,
            contracts=self.contracts(),
            extra_state={
                "successful_optimizer_updates": self.successful_updates,
                "statistics_state": self.base_model.statistics.state_dict(),
                "training_generator_state": generator.get_state().detach().cpu(),
                "rank_states": dict(rank_states or {}),
                "continuation_parent": self.continuation_parent,
            },
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        generator: torch.Generator,
        expected_sha256: str | None = None,
        rank: int = 0,
    ) -> Mapping[str, Any]:
        stage = self.stage
        # Moment validation requires parameters owning moments to be marked
        # trainable during restoration; restore the requested stage afterward.
        for name in ("history_encoder", "dit", "decoder"):
            module = getattr(self.base_model, name)
            module.set_trainable(True) if name in {"history_encoder", "dit"} else module.requires_grad_(True)
        if self.base_model.motion_encoder is not None:
            self.base_model.motion_encoder.requires_grad_(True)
        payload = load_training_checkpoint(
            path,
            model=self.base_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_contracts=self.contracts(),
            expected_sha256=expected_sha256,
        )
        extra = payload["extra_state"]
        rank_states = extra.get("rank_states")
        rank_state = rank_states.get(str(int(rank))) if isinstance(rank_states, Mapping) else None
        state = (
            rank_state.get("training_generator_state")
            if isinstance(rank_state, Mapping)
            else extra.get("training_generator_state")
        )
        if not isinstance(state, Tensor):
            raise ValueError("Frame Joint checkpoint lacks training generator state")
        generator.set_state(state.detach().cpu())
        if isinstance(rank_state, Mapping) and isinstance(rank_state.get("rng_state"), Mapping):
            restore_rng_state(rank_state["rng_state"])
        self.step = int(payload["step"])
        self.successful_updates = int(extra["successful_optimizer_updates"])
        self.configure_stage(stage)
        return payload

    def load_warm_start(
        self,
        path: str | Path,
        *,
        expected_sha256: str,
    ) -> WarmStartReport:
        """Load compatible model tensors while resetting all training state.

        Warm start is deliberately distinct from strict resume and exact
        continuation.  It permits only newly introduced G/M tensors to be
        absent from the parent; old tensors must all exist with identical
        shapes and dtypes.  Optimizer moments, counters, cursor, and RNG are
        never read into the child experiment.
        """

        actual_sha256 = sha256_file(path)
        if actual_sha256 != str(expected_sha256):
            raise ValueError("warm-start parent SHA-256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != "molvid.training.checkpoint.v1":
            raise ValueError("unsupported warm-start checkpoint schema")
        parent_contracts = payload.get("contracts")
        source_state = payload.get("model_state")
        if not isinstance(parent_contracts, Mapping) or not isinstance(source_state, Mapping):
            raise ValueError("warm-start parent lacks contracts or model state")
        for name, expected in (
            ("statistics_hash", self.base_model.statistics_hash),
            ("teacher_artifact_sha256", self.teacher_artifact_sha256),
        ):
            if parent_contracts.get(name) != expected:
                raise ValueError(f"warm-start parent {name!r} differs")
        parent_model = parent_contracts.get("model")
        if not isinstance(parent_model, Mapping):
            raise ValueError("warm-start parent model contract is missing")
        target_state = self.base_model.state_dict()
        loaded: dict[str, Tensor] = {}
        unexpected: list[str] = []
        shape_mismatch: list[str] = []
        for name, value in source_state.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise ValueError("warm-start model state must contain named tensors")
            target = target_state.get(name)
            if target is None:
                unexpected.append(name)
            elif value.shape != target.shape or value.dtype != target.dtype:
                shape_mismatch.append(name)
            else:
                loaded[name] = value
        if unexpected or shape_mismatch:
            raise ValueError(
                "warm-start parent has incompatible old tensors: "
                f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
            )
        initialized = sorted(set(target_state) - set(loaded))
        allowed_fragments = (
            "motion_encoder.",
            ".local_geometry.",
            ".motion_context.",
            ".history_time_decay_raw",
            ".temporal_time_decay_raw",
        )
        forbidden_missing = [
            name for name in initialized
            if not any(fragment in name for fragment in allowed_fragments)
        ]
        if forbidden_missing:
            raise ValueError(
                "warm-start is missing non-G/M tensors: "
                f"{forbidden_missing[:8]}"
            )
        result = self.base_model.load_state_dict(loaded, strict=False)
        if tuple(sorted(result.missing_keys)) != tuple(initialized) or result.unexpected_keys:
            raise RuntimeError("warm-start load report disagrees with the validated tensor sets")
        restored = self.base_model.state_dict()
        for name, value in loaded.items():
            if not torch.equal(restored[name].detach().cpu(), value.detach().cpu()):
                raise RuntimeError(f"warm-start tensor changed while loading {name!r}")
        self.step = 0
        self.successful_updates = 0
        self.continuation_parent = {
            "mode": "model_only_warm_start",
            "sha256": actual_sha256,
            "step": int(payload.get("step", -1)),
            "source_data_hash": parent_contracts.get("data_hash"),
            "optimizer_restored": False,
            "cursor_restored": False,
            "rng_restored": False,
        }
        return WarmStartReport(
            source_path=str(Path(path).resolve()),
            source_sha256=actual_sha256,
            source_step=int(payload.get("step", -1)),
            source_model_contract=dict(parent_model),
            target_model_contract=dict(self.base_model.contract()),
            loaded=tuple(sorted(loaded)),
            initialized=tuple(initialized),
            unexpected=tuple(unexpected),
            shape_mismatch=tuple(shape_mismatch),
        )

    def load_continuation_parent(
        self,
        path: str | Path,
        *,
        generator: torch.Generator,
        expected_sha256: str,
        rank: int = 0,
    ) -> Mapping[str, Any]:
        """Fork an exact parent into a new objective/scheduler experiment.

        Model state, AdamW moments, data cursor, and every RNG stream are
        restored.  The child update counter and scheduler start at zero; its
        loss contract and learning-rate schedule are deliberately new.
        """

        target_stage = self.stage
        if target_stage not in {"continuation", "frozen_decoder"}:
            raise ValueError("continuation parents require continuation or frozen_decoder stage")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != str(expected_sha256):
            raise ValueError("continuation parent SHA-256 mismatch")
        preview = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(preview, Mapping) or not isinstance(preview.get("contracts"), Mapping):
            raise ValueError("continuation parent lacks training contracts")
        parent = preview["contracts"]
        current = self.contracts()
        for name in (
            "trainer",
            "model",
            "statistics_hash",
            "teacher_artifact_sha256",
            "teacher_state_hash",
            "data_hash",
        ):
            if parent.get(name) != current.get(name):
                raise ValueError(f"continuation parent {name!r} differs")
        allowed_parent_stages = {"joint", "continuation"}
        if target_stage == "frozen_decoder":
            allowed_parent_stages.add("flow_start")
        if parent.get("stage") not in allowed_parent_stages:
            raise ValueError("continuation parent stage is not valid for this child")
        parent_optimization = parent.get("optimization")
        current_optimization = current.get("optimization")
        if not isinstance(parent_optimization, Mapping) or not isinstance(current_optimization, Mapping):
            raise ValueError("continuation optimization contracts are missing")
        for name in ("weight_decay", "grad_clip", "amp_bfloat16", "norm_bias_weight_decay"):
            if parent_optimization.get(name) != current_optimization.get(name):
                raise ValueError(f"continuation optimizer setting {name!r} differs")
        JointLossConfig.resolve(parent.get("loss", {}))

        for name in ("history_encoder", "dit", "decoder"):
            module = getattr(self.base_model, name)
            module.set_trainable(True) if name in {"history_encoder", "dit"} else module.requires_grad_(True)
        if self.base_model.motion_encoder is not None:
            self.base_model.motion_encoder.requires_grad_(True)
        payload = load_training_checkpoint(
            path,
            model=self.base_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            expected_contracts=parent,
            expected_sha256=expected_sha256,
        )
        extra = payload["extra_state"]
        rank_states = extra.get("rank_states")
        rank_state = rank_states.get(str(int(rank))) if isinstance(rank_states, Mapping) else None
        state = (
            rank_state.get("training_generator_state")
            if isinstance(rank_state, Mapping)
            else extra.get("training_generator_state")
        )
        if not isinstance(state, Tensor):
            raise ValueError("continuation parent lacks training generator state")
        generator.set_state(state.detach().cpu())
        if isinstance(rank_state, Mapping) and isinstance(rank_state.get("rng_state"), Mapping):
            restore_rng_state(rank_state["rng_state"])
        self.step = 0
        self.successful_updates = 0
        self.continuation_parent = {
            "sha256": actual_sha256,
            "step": int(payload["step"]),
            "stage": str(parent["stage"]),
        }
        self.configure_stage(target_stage)
        return payload
