"""Single-frame denoising updates for the shared trajectory decoder.

Clean coordinates are retained only as explicit loss targets.  The model
condition, frame-encoder graph, origin, latent features, decoder reference
coordinates, and coordinate-derived block positions all come from the same
perturbed structure.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..codec.decoder import TrajectoryDecoderOutput
from ..codec.frame import FrozenFrameTeacher
from ..data.batch import ClipBatch, STATIC_TASK
from ..data.static import STATIC_DECODER_TASK_TYPE
from ..latent.types import ObservedContext
from ..losses.geometry import FutureBondLoss, future_bond_distance_loss
from ..losses.sampled_physical import PhysicalAuxiliaryConfig, packed_clash_loss
from ..model import FrameJointModel
from .batches import prepare_batch_then_to_device


@dataclass(frozen=True)
class StaticPerturbationConfig:
    """Reproducible isotropic coordinate noise in physical Angstrom units."""

    sigma_per_axis_angstrom: tuple[float, ...] = (0.05, 0.10, 0.20)
    clean_probability: float = 0.20

    @classmethod
    def resolve(cls, value: Mapping[str, Any] | None) -> "StaticPerturbationConfig":
        raw = dict(value or {})
        unknown = sorted(set(raw) - {"sigma_per_axis_angstrom", "clean_probability"})
        if unknown:
            raise ValueError(f"unknown static perturbation keys: {unknown}")
        sigma = raw.get("sigma_per_axis_angstrom", cls.sigma_per_axis_angstrom)
        if isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
            sigma = (float(sigma),)
        if not isinstance(sigma, (list, tuple)) or not sigma:
            raise ValueError("static sigma_per_axis_angstrom must be a non-empty sequence")
        sigma_values = tuple(float(item) for item in sigma)
        if any(not math.isfinite(item) or item <= 0.0 for item in sigma_values):
            raise ValueError("static per-axis sigma values must be finite and positive")
        clean_probability = float(raw.get("clean_probability", cls.clean_probability))
        if not math.isfinite(clean_probability) or not 0.0 <= clean_probability <= 1.0:
            raise ValueError("static clean_probability must lie in [0,1]")
        return cls(
            sigma_per_axis_angstrom=sigma_values,
            clean_probability=clean_probability,
        )

    def contract(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "coordinate_unit": "angstrom",
            "sigma_definition": "independent Gaussian standard deviation per Cartesian axis",
            "three_dimensional_rms_displacement_angstrom": [
                math.sqrt(3.0) * value for value in self.sigma_per_axis_angstrom
            ],
        }


@dataclass(frozen=True)
class PreparedStaticDecoderBatch:
    """One-frame observed condition plus a clean target kept outside the model."""

    input_batch: ClipBatch
    observed_context: ObservedContext
    target_coordinates: Tensor
    nominal_sigma_per_axis_angstrom: Tensor
    realized_rms_displacement_angstrom: Tensor
    clean_sample_mask: Tensor
    bpos_exact_anchor_count: int
    bpos_fallback_count: int
    task_type: str = STATIC_DECODER_TASK_TYPE
    motion_available: bool = False
    physical_lag_applicable: bool = False

    def __post_init__(self) -> None:
        context = self.observed_context
        if context.latent.frames != 1 or self.target_coordinates.shape != context.coordinates.shape:
            raise ValueError("static decoder batch requires matching single-frame [1,N,3] tensors")
        batch_size = context.latent.batch_size
        for name, value in (
            ("nominal sigma", self.nominal_sigma_per_axis_angstrom),
            ("realized displacement", self.realized_rms_displacement_angstrom),
            ("clean mask", self.clean_sample_mask),
        ):
            if value.shape != (batch_size,):
                raise ValueError(f"static {name} must have shape [B]")
        if self.task_type != STATIC_DECODER_TASK_TYPE:
            raise ValueError("static decoder task type differs from its contract")
        if self.motion_available or self.physical_lag_applicable:
            raise ValueError("static decoder batches cannot advertise motion or physical lag")


def _static_input_contract(batch: ClipBatch) -> None:
    if batch.frames != 1:
        raise ValueError("static decoder preparation requires T=1")
    task_ids = batch.host_task_ids or tuple(int(item) for item in batch.task.tolist())
    if len(task_ids) != batch.batch_size or any(item != STATIC_TASK for item in task_ids):
        raise ValueError("static decoder preparation requires only static task records")
    if batch.frame_mask.device.type != "cpu" or batch.time_ps.device.type != "cpu":
        raise RuntimeError("validate and register the CPU static batch before CUDA transfer")
    if not bool(torch.all(batch.frame_mask)):
        raise ValueError("the single static frame must be valid for every sample")
    if not bool(torch.all(batch.time_ps == 0)):
        raise ValueError("static time must be the coordinate origin 0, not a physical lag")
    if batch.delta_time_ps.shape != (batch.batch_size, 0):
        raise ValueError("static batches cannot contain physical time intervals")


def _block_reference_maps(batch: ClipBatch, *, exact_tolerance: float = 1.0e-4) -> tuple[Tensor, Tensor, int, int]:
    """Map each CPU bpos row to an input atom or an explicit block fallback."""

    if batch.x.device.type != "cpu":
        raise RuntimeError("block-reference mapping must be derived before CUDA transfer")
    coordinates = batch.x[0].float()
    block_positions = batch.bpos[0].float()
    anchor = torch.full((batch.atom_count,), -1, dtype=torch.long)
    group = torch.empty((batch.atom_count,), dtype=torch.long)
    group_index = 0
    for sample_index in range(batch.batch_size):
        start = int(batch.atom_ptr[sample_index])
        stop = int(batch.atom_ptr[sample_index + 1])
        sample_blocks = batch.block_id[start:stop]
        for block in torch.unique(sample_blocks, sorted=True):
            members = torch.nonzero(sample_blocks == block, as_tuple=False).flatten() + start
            group[members] = group_index
            candidates = coordinates.index_select(0, members)
            references = block_positions.index_select(0, members)
            distances = torch.cdist(references, candidates)
            minimum, local_index = distances.min(dim=1)
            exact = minimum <= float(exact_tolerance)
            if bool(torch.any(exact)):
                exact_members = members[exact]
                anchor[exact_members] = members.index_select(0, local_index[exact])
            group_index += 1
    exact_count = int((anchor >= 0).sum())
    return anchor, group, exact_count, int(batch.atom_count - exact_count)


def _perturb_block_positions(
    clean_batch: ClipBatch,
    noise_angstrom: Tensor,
    anchor_cpu: Tensor,
    group_cpu: Tensor,
) -> Tensor:
    """Apply input-coordinate displacements to every coordinate-derived bpos."""

    if noise_angstrom.shape != clean_batch.x.shape or clean_batch.frames != 1:
        raise ValueError("static bpos perturbation requires noise shaped [1,N,3]")
    device = noise_angstrom.device
    anchor = anchor_cpu.to(device=device)
    group = group_cpu.to(device=device)
    atom_noise = noise_angstrom[0]
    group_count = int(group_cpu.max()) + 1
    sums = atom_noise.new_zeros((group_count, 3))
    counts = atom_noise.new_zeros((group_count, 1))
    sums.index_add_(0, group, atom_noise)
    counts.index_add_(0, group, atom_noise.new_ones((atom_noise.shape[0], 1)))
    shifts = (sums / counts.clamp_min(1)).index_select(0, group)
    exact = anchor >= 0
    if bool(torch.any(exact)):
        shifts[exact] = atom_noise.index_select(0, anchor[exact])
    return clean_batch.bpos + shifts.unsqueeze(0)


@torch.no_grad()
def prepare_static_decoder_batch(
    teacher: FrozenFrameTeacher,
    batch: ClipBatch,
    *,
    device: torch.device,
    generator: torch.Generator,
    perturbation: StaticPerturbationConfig,
) -> PreparedStaticDecoderBatch:
    """Create one perturbed observed condition and a separate clean target."""

    _static_input_contract(batch)
    anchor_cpu, group_cpu, exact_count, fallback_count = _block_reference_maps(batch)
    clean = prepare_batch_then_to_device(teacher, batch, device)
    sigma_choices = torch.tensor(
        perturbation.sigma_per_axis_angstrom,
        device=device,
        dtype=torch.float32,
    )
    choice = torch.randint(
        sigma_choices.numel(),
        (clean.batch_size,),
        device=device,
        generator=generator,
    )
    nominal_sigma = sigma_choices.index_select(0, choice)
    clean_mask = torch.rand(
        (clean.batch_size,), device=device, generator=generator
    ) < perturbation.clean_probability
    nominal_sigma = nominal_sigma.masked_fill(clean_mask, 0.0)
    noise = torch.randn(
        clean.x.shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    ) * nominal_sigma.index_select(0, clean.abid).view(1, -1, 1)
    perturbed_x = clean.x.float() + noise
    perturbed_bpos = _perturb_block_positions(clean, noise, anchor_cpu, group_cpu)
    input_batch = replace(clean, x=perturbed_x, bpos=perturbed_bpos)
    latent, origin = teacher(input_batch)
    context = ObservedContext(
        latent=latent,
        coordinates=input_batch.x,
        sample_origin=origin,
        loss_mask=input_batch.loss_mask.to(dtype=torch.bool),
    )
    squared_displacement = noise.square().sum(dim=-1)[0]
    sums = squared_displacement.new_zeros((clean.batch_size,))
    counts = squared_displacement.new_zeros((clean.batch_size,))
    valid = input_batch.loss_mask.to(dtype=torch.bool)
    sums.scatter_add_(0, input_batch.abid[valid], squared_displacement[valid])
    counts.scatter_add_(0, input_batch.abid[valid], torch.ones_like(squared_displacement[valid]))
    realized = (sums / counts.clamp_min(1)).sqrt()
    return PreparedStaticDecoderBatch(
        input_batch=input_batch,
        observed_context=context,
        target_coordinates=clean.x.float(),
        nominal_sigma_per_axis_angstrom=nominal_sigma,
        realized_rms_displacement_angstrom=realized,
        clean_sample_mask=clean_mask,
        bpos_exact_anchor_count=exact_count,
        bpos_fallback_count=fallback_count,
    )


@dataclass(frozen=True)
class StaticDecoderLossConfig:
    """One resolved source for the complete static objective."""

    coordinate: float = 1.0
    bond: float = 0.1
    clash: float = 0.0
    gamma: float = 1.0

    @classmethod
    def resolve(cls, value: Mapping[str, Any]) -> "StaticDecoderLossConfig":
        unknown = sorted(set(value) - {"coordinate", "bond", "clash", "gamma"})
        if unknown:
            raise ValueError(f"unknown static loss keys: {unknown}")
        result = cls(**{name: float(value.get(name, getattr(cls, name))) for name in ("coordinate", "bond", "clash", "gamma")})
        for name, weight in asdict(result).items():
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"static loss weight {name!r} must be finite and non-negative")
        if result.coordinate == result.bond == result.clash == 0.0:
            raise ValueError("at least one static loss term must be active")
        if result.gamma <= 0.0:
            raise ValueError("static gamma must be positive")
        return result

    def contract(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class StaticClashLoss:
    loss: Tensor
    applicable_count: Tensor
    diagnostics: Mapping[str, Any]


StaticClashLossFunction = Callable[[Tensor, PreparedStaticDecoderBatch], StaticClashLoss]


def static_clash_loss_function(
    physical_config: PhysicalAuxiliaryConfig,
) -> StaticClashLossFunction:
    """Bind A's frozen reviewed clash rule to the static batch interface."""

    if physical_config.clash_weight <= 0.0:
        raise ValueError("static clash requires A's active released clash configuration")

    def evaluate(
        coordinates: Tensor,
        batch: PreparedStaticDecoderBatch,
    ) -> StaticClashLoss:
        result = packed_clash_loss(
            coordinates,
            topology=batch.observed_context.topology,
            loss_mask=batch.observed_context.loss_mask,
            frame_mask=batch.observed_context.latent.frame_mask,
            config=physical_config,
        )
        return StaticClashLoss(
            loss=result.loss,
            applicable_count=result.applicable_count,
            diagnostics={
                "enabled": True,
                "source": "A1 released PhysicalAuxiliaryConfig",
                "table_provenance": dict(physical_config.table_provenance or {}),
                **result.diagnostics(),
            },
        )

    return evaluate


@dataclass(frozen=True)
class StaticDecoderLoss:
    total: Tensor
    unscaled_total: Tensor
    raw: Mapping[str, Tensor]
    weighted: Mapping[str, Tensor]
    applicable_samples: Mapping[str, Tensor]
    bond_diagnostics: FutureBondLoss
    clash_diagnostics: Mapping[str, Any]


def _sample_equal_coordinate_mse(
    prediction: Tensor,
    batch: PreparedStaticDecoderBatch,
) -> tuple[Tensor, Tensor]:
    target = batch.target_coordinates
    if prediction.shape != target.shape or prediction.shape[0] != 1 or prediction.shape[-1] != 3:
        raise ValueError("static coordinate loss expects matching [1,N,3] tensors")
    context = batch.observed_context
    valid = context.loss_mask & context.latent.frame_mask[:, 0].index_select(0, context.latent.abid)
    error = (prediction[0].float() - target[0].float()).square().mean(dim=-1)
    sums = error.new_zeros((context.latent.batch_size,))
    counts = error.new_zeros((context.latent.batch_size,))
    sums.scatter_add_(0, context.latent.abid[valid], error[valid])
    counts.scatter_add_(0, context.latent.abid[valid], torch.ones_like(error[valid]))
    applicable = counts > 0
    loss = ((sums / counts.clamp_min(1)) * applicable).sum() / applicable.sum().clamp_min(1)
    return loss, applicable.sum()


def _static_bond_loss(
    prediction: Tensor,
    batch: PreparedStaticDecoderBatch,
) -> FutureBondLoss:
    context = batch.observed_context
    return future_bond_distance_loss(
        prediction.float(),
        batch.target_coordinates.float(),
        frame_mask=context.latent.frame_mask,
        observed_frames=torch.zeros_like(context.latent.frame_mask),
        loss_mask=context.loss_mask,
        bond_index=context.topology.covalent_bond_index,
        abid=context.latent.abid,
        eligible_samples=torch.ones(
            (context.latent.batch_size,), device=prediction.device, dtype=torch.bool
        ),
    )


def static_decoder_loss(
    output: TrajectoryDecoderOutput,
    batch: PreparedStaticDecoderBatch,
    config: StaticDecoderLossConfig,
    *,
    clash_loss_function: StaticClashLossFunction | None = None,
) -> StaticDecoderLoss:
    """Compute MSE, clean-target bond loss, and an optional reviewed clash term."""

    coordinate, coordinate_count = _sample_equal_coordinate_mse(output.coordinates, batch)
    bond = _static_bond_loss(output.coordinates, batch)
    zero = coordinate * 0.0
    clash = StaticClashLoss(
        loss=zero,
        applicable_count=zero.new_zeros((), dtype=torch.long),
        diagnostics={"enabled": False, "reason": "no reviewed clash implementation supplied"},
    )
    if config.clash > 0.0:
        if clash_loss_function is None:
            raise ValueError(
                "static clash weight is positive but no reviewed A clash implementation was supplied"
            )
        clash = clash_loss_function(output.coordinates, batch)
        if clash.loss.ndim != 0 or not torch.isfinite(clash.loss):
            raise FloatingPointError("static clash implementation returned a non-finite scalar")
    raw = {"coordinate_mse": coordinate, "bond_distance": bond.loss, "clash": clash.loss}
    coefficients = {"coordinate_mse": config.coordinate, "bond_distance": config.bond, "clash": config.clash}
    weighted_before_gamma = {
        name: raw[name] * coefficients[name] for name in raw
    }
    unscaled_total = sum(weighted_before_gamma.values(), zero)
    weighted = {name: value * config.gamma for name, value in weighted_before_gamma.items()}
    return StaticDecoderLoss(
        total=sum(weighted.values(), zero),
        unscaled_total=unscaled_total,
        raw=raw,
        weighted=weighted,
        applicable_samples={
            "coordinate_mse": coordinate_count,
            "bond_distance": bond.applicable_count,
            "clash": clash.applicable_count,
        },
        bond_diagnostics=bond,
        clash_diagnostics=dict(clash.diagnostics),
    )


def parameter_gradient_norm(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool = False,
) -> Tensor:
    """Return the L2 autograd norm without populating ``parameter.grad``."""

    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    values = [gradient.float().square().sum() for gradient in gradients if gradient is not None]
    return loss.new_zeros(()) if not values else torch.stack(values).sum().sqrt()


def resolve_static_gamma(
    dynamic_decoder_gradient_norms: Sequence[float],
    static_decoder_gradient_norms: Sequence[float],
    *,
    target_ratio: float = 0.075,
) -> tuple[float, dict[str, Any]]:
    """Match median active static decoder gradient to 5--10% of dynamic decoder loss."""

    if not 0.05 <= float(target_ratio) <= 0.10:
        raise ValueError("static decoder gradient target ratio must lie in [0.05,0.10]")
    if len(dynamic_decoder_gradient_norms) != len(static_decoder_gradient_norms) or not dynamic_decoder_gradient_norms:
        raise ValueError("static gamma calibration requires paired non-empty gradient norms")
    dynamic = torch.tensor(dynamic_decoder_gradient_norms, dtype=torch.float64)
    static = torch.tensor(static_decoder_gradient_norms, dtype=torch.float64)
    if not bool(torch.isfinite(dynamic).all() and torch.isfinite(static).all()):
        raise ValueError("static gamma calibration norms must be finite")
    if bool(torch.any(dynamic <= 0) or torch.any(static <= 0)):
        raise ValueError("static gamma calibration norms must be positive")
    dynamic_median = dynamic.median()
    static_median = static.median()
    gamma = float(target_ratio) * float(dynamic_median / static_median)
    return gamma, {
        "schema": "molvid.static_decoder.gamma_calibration.v1",
        "split": "train",
        "batch_count": len(dynamic_decoder_gradient_norms),
        "dynamic_reference": "sum of active dynamic losses with decoder gradients",
        "target_static_to_dynamic_decoder_gradient_ratio": float(target_ratio),
        "dynamic_decoder_gradient_norms": [float(item) for item in dynamic],
        "static_decoder_gradient_norms_before_gamma": [float(item) for item in static],
        "dynamic_decoder_gradient_norm_median": float(dynamic_median),
        "static_decoder_gradient_norm_median_before_gamma": float(static_median),
        "resolved_gamma": gamma,
        "achieved_median_ratio": float(gamma * static_median / dynamic_median),
    }


class StaticDecoderAuxiliaryTrainer:
    """Run a static optimizer update while exposing only decoder gradients."""

    def __init__(
        self,
        model: FrameJointModel,
        *,
        loss_config: StaticDecoderLossConfig,
        optimizer: torch.optim.Optimizer | None = None,
        decoder_learning_rate: float = 5.0e-5,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        amp: bool = False,
        clash_loss_function: StaticClashLossFunction | None = None,
    ) -> None:
        if not isinstance(model, FrameJointModel):
            raise TypeError("StaticDecoderAuxiliaryTrainer requires FrameJointModel")
        if any(parameter.requires_grad for parameter in model.target_teacher.parameters()):
            raise RuntimeError("static target teacher must remain frozen")
        if loss_config.clash > 0.0 and clash_loss_function is None:
            raise ValueError("a positive static clash weight requires the reviewed A implementation")
        self.model = model
        self.loss_config = loss_config
        self.grad_clip = float(grad_clip)
        self.amp = bool(amp)
        self.clash_loss_function = clash_loss_function
        self.optimizer = optimizer or torch.optim.AdamW(
            model.decoder.parameters(),
            lr=float(decoder_learning_rate),
            weight_decay=float(weight_decay),
        )
        decoder_ids = {id(parameter) for parameter in model.decoder.parameters()}
        optimizer_parameters = [
            parameter for group in self.optimizer.param_groups for parameter in group["params"]
        ]
        if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
            raise ValueError("static optimizer contains duplicate parameters")
        if not decoder_ids.issubset({id(parameter) for parameter in optimizer_parameters}):
            raise ValueError("static optimizer does not contain every decoder parameter")
        self._decoder_ids = decoder_ids
        self.successful_updates = 0

    def _autocast(self):
        parameter = next(self.model.decoder.parameters())
        if self.amp and parameter.device.type != "cuda":
            raise RuntimeError("static decoder BF16 training requires CUDA")
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def train_step(self, batch: PreparedStaticDecoderBatch) -> dict[str, Any]:
        self.model.target_teacher.eval()
        self.model.decoder.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.model.decoder(batch.observed_context)
            losses = static_decoder_loss(
                output,
                batch,
                self.loss_config,
                clash_loss_function=self.clash_loss_function,
            )
        if not torch.isfinite(losses.total):
            raise FloatingPointError("non-finite static decoder loss")
        losses.total.backward()
        inactive_with_grad = [
            name
            for name, parameter in self.model.named_parameters()
            if id(parameter) not in self._decoder_ids and parameter.grad is not None
        ]
        if inactive_with_grad:
            raise RuntimeError(
                "static update produced gradients outside the decoder: "
                f"{inactive_with_grad[:8]}"
            )
        decoder_parameters = [
            parameter for parameter in self.model.decoder.parameters() if parameter.requires_grad
        ]
        if not decoder_parameters or all(parameter.grad is None for parameter in decoder_parameters):
            raise RuntimeError("static update produced no decoder gradient")
        grad_norm = torch.nn.utils.clip_grad_norm_(decoder_parameters, self.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite static decoder gradient norm")
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in decoder_parameters):
            raise FloatingPointError("static optimizer produced non-finite decoder parameters")
        self.successful_updates += 1
        result: dict[str, Any] = {
            "static_update": self.successful_updates,
            "task_type": batch.task_type,
            "loss": float(losses.total.detach()),
            "unscaled_loss": float(losses.unscaled_total.detach()),
            "decoder_grad_norm": float(torch.as_tensor(grad_norm).detach()),
            "batch_size": batch.observed_context.latent.batch_size,
            "atom_count": batch.observed_context.latent.num_atoms,
            "motion_available": False,
            "physical_lag_applicable": False,
            "nominal_sigma_per_axis_angstrom": [
                float(item) for item in batch.nominal_sigma_per_axis_angstrom.detach().cpu()
            ],
            "realized_rms_displacement_angstrom": [
                float(item) for item in batch.realized_rms_displacement_angstrom.detach().cpu()
            ],
            "bpos_exact_anchor_count": batch.bpos_exact_anchor_count,
            "bpos_fallback_count": batch.bpos_fallback_count,
        }
        result.update({f"raw_{name}": float(value.detach()) for name, value in losses.raw.items()})
        result.update({f"weighted_{name}": float(value.detach()) for name, value in losses.weighted.items()})
        return result

    def contract(self) -> dict[str, Any]:
        groups = []
        for group in self.optimizer.param_groups:
            groups.append(
                {
                    "parameter_count": len(group["params"]),
                    "lr": float(group["lr"]),
                    "weight_decay": float(group["weight_decay"]),
                    "group_name": group.get("group_name"),
                }
            )
        return {
            "schema": "molvid.static_decoder_auxiliary.v1",
            "task_type": STATIC_DECODER_TASK_TYPE,
            "teacher_frozen_eval": True,
            "model_path": "perturbed_coordinates->frozen_teacher->shared_decoder",
            "motion_available": False,
            "physical_lag_applicable": False,
            "trainable_scope": ["decoder"],
            "inactive_scope": ["target_teacher", "history_encoder", "motion_encoder", "dit"],
            "loss": self.loss_config.contract(),
            "optimizer_groups": groups,
            "scheduler_advanced_by_static_updates": False,
            "amp_bfloat16": self.amp,
            "grad_clip": self.grad_clip,
        }
