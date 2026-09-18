"""Independent DiT optimization with frozen codec boundaries and exact resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
from torch import Tensor, nn

from ..checkpoints import capture_rng_state, load_training_checkpoint, restore_rng_state, save_training_checkpoint
from ..flow.objective import RectifiedFlowObjective
from ..latent.adapter import StateDetailLatentAdapter
from ..latent.statistics import LatentStatistics
from ..latent.types import LatentBatch, LatentFields, contract_hash
from ..runtime import seed_all, sha256_file


def module_state_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class DiTTrainConfig:
    ratio: int
    mode: str
    codec_width: int = 128
    scalar_width: int = 256
    vector_width: int = 128
    depth: int = 4
    heads: int = 8
    ffn_multiplier: int = 4
    dropout: float = 0.0
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 100
    seed: int = 0
    amp: bool = False
    output_root: str = "outputs/dit_state_detail_probe_v1"
    data_hash: str = ""
    codec_hash: str = ""
    stats_hash: str = ""
    source_mode: str = "gaussian"
    center_kind: str = "repeat_last_coordinate_encode"
    source_sigma: float = 1.0
    normalization_hash: str = ""
    init_hash: str = ""
    observation_mixture: tuple[int, ...] = (0, 4, 8)
    metadata: dict[str, Any] = field(default_factory=dict)
    history_probabilities: tuple[float, float, float] = (0.5, 0.25, 0.25)

    def validate(self) -> None:
        if self.ratio not in (2, 4) or self.mode != f"ratio{self.ratio}_state_detail":
            raise ValueError("DiT config must select ratio2_state_detail or ratio4_state_detail")
        if self.codec_width < 1 or self.scalar_width < 1 or self.vector_width < 1:
            raise ValueError("all model widths must be positive")
        if self.depth < 1 or self.heads < 1 or self.ffn_multiplier < 1:
            raise ValueError("depth, heads, and ffn_multiplier must be positive")
        if self.scalar_width % self.heads or self.vector_width % self.heads:
            raise ValueError("model widths must be divisible by heads")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.observation_mixture or any(value not in (0, 4, 8) for value in self.observation_mixture):
            raise ValueError("observation mixture must contain only H=0,4,8")
        if len(self.history_probabilities) != 3 or any(value < 0 for value in self.history_probabilities) or abs(sum(self.history_probabilities) - 1.0) > 1e-8:
            raise ValueError("history corruption probabilities must be nonnegative and sum to one")
        if self.source_mode not in ("gaussian", "conditional"):
            raise ValueError("source_mode must be gaussian or conditional")
        if not torch.isfinite(torch.tensor(float(self.source_sigma))) or float(self.source_sigma) != 1.0:
            raise ValueError("source_sigma is frozen at one")

    def contract(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        value["observation_mixture"] = list(self.observation_mixture)
        value["history_probabilities"] = list(self.history_probabilities)
        return value


class DiTTrainer:
    """Minimal trainer with explicit frozen-module and checkpoint contracts."""

    def __init__(
        self,
        model: nn.Module,
        adapter: StateDetailLatentAdapter,
        *,
        config: DiTTrainConfig,
        statistics: Optional[LatentStatistics] = None,
        codec: Optional[nn.Module] = None,
        frame_encoder: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        config.validate()
        if getattr(model, "adapter", adapter) is not adapter:
            raise ValueError("model and trainer must share the same latent adapter")
        if config.ratio != adapter.ratio or config.mode != adapter.mode:
            raise ValueError("trainer config and adapter ratio/mode disagree")
        self.model = model
        self.adapter = adapter
        self.config = config
        self.statistics = statistics
        model_parameter = next(iter(model.parameters()), None)
        if model_parameter is None:
            raise ValueError("the DiT must expose parameters")
        self.device = model_parameter.device
        if config.amp and self.device.type != "cuda":
            raise RuntimeError("DiT AMP requires a CUDA model; CPU fallback is not AMP")
        self.amp_enabled = bool(config.amp)
        if statistics is not None:
            if statistics.ratio != config.ratio or statistics.mode != config.mode:
                raise ValueError("statistics ratio/mode disagrees with trainer config")
            if config.stats_hash and config.stats_hash != statistics.hash:
                raise ValueError("configured statistics hash does not match statistics")
            config.stats_hash = statistics.hash
        self.codec = codec
        self.frame_encoder = frame_encoder
        self._freeze_module(codec, "codec")
        self._freeze_module(frame_encoder, "frame_encoder")
        self.flow = RectifiedFlowObjective()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("the DiT has no trainable parameters")
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = scheduler
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.step = 0
        self.successful_updates = 0
        self.last_activation_dtypes: dict[str, str] = {}
        self._rng_seed = int(config.seed)
        seed_all(self._rng_seed)
        self.frozen_hashes = self.frozen_state_hashes()
        self.codec_contract = (
            self.codec.model_contract() if self.codec is not None and hasattr(self.codec, "model_contract") else {}
        )

    def autocast_context(self):
        """Use the repository's explicit CUDA BF16 policy when AMP is enabled."""

        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    @staticmethod
    def _freeze_module(module: Optional[nn.Module], label: str) -> None:
        if module is None:
            return
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError(f"{label} contains a trainable parameter; freeze it before DiT training")
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    def frozen_state_hashes(self) -> dict[str, str]:
        result = {}
        if self.codec is not None:
            result["codec"] = module_state_hash(self.codec)
        if self.frame_encoder is not None:
            result["frame_encoder"] = module_state_hash(self.frame_encoder)
        return result

    def _normalise_batch(self, batch: LatentBatch) -> LatentBatch:
        if batch.ratio != self.config.ratio or batch.mode != self.config.mode:
            raise ValueError("batch ratio/mode disagrees with trainer")
        if self.statistics is None:
            return batch.zero_invalid()
        return self.statistics.normalize(batch)

    def train_step(
        self,
        batch: LatentBatch,
        *,
        generator: Optional[torch.Generator] = None,
        source_center: Optional[LatentFields] = None,
        auxiliary_objective: Optional[
            Callable[
                [LatentFields, Any, LatentBatch],
                Optional[tuple[Tensor, Mapping[str, Any]]],
            ]
        ] = None,
    ) -> dict[str, Any]:
        self.model.train()
        if self.codec is not None:
            self.codec.eval()
        if self.frame_encoder is not None:
            self.frame_encoder.eval()
        batch = self._normalise_batch(batch)
        flow_sample = self.flow.sample(
            batch,
            generator=generator,
            source_center=source_center,
            source_mode=self.config.source_mode,
        )
        noisy_batch = batch.with_fields(flow_sample.interpolated)
        auxiliary_diagnostics: Mapping[str, Any] = {}
        with self.autocast_context():
            prediction = self.model(noisy_batch, flow_sample.tau)
            self.last_activation_dtypes = {
                name: str(getattr(prediction, name).dtype)
                for name in prediction.names()
            }
            loss = self.flow.loss(prediction, flow_sample.target, batch)
            total_loss = loss.total
            if auxiliary_objective is not None:
                auxiliary = auxiliary_objective(prediction, flow_sample, batch)
                if auxiliary is not None:
                    auxiliary_loss, auxiliary_diagnostics = auxiliary
                    if auxiliary_loss.ndim != 0:
                        raise ValueError("auxiliary DiT loss must be scalar")
                    if auxiliary_loss.device != loss.total.device:
                        raise ValueError("auxiliary DiT loss must share the RF loss device")
                    total_loss = total_loss + auxiliary_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite DiT loss")
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            self.config.grad_clip,
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite DiT gradient norm")
        scale_before = float(self.scaler.get_scale()) if self.scaler.is_enabled() else None
        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if scale_before is not None and float(self.scaler.get_scale()) < scale_before:
                raise FloatingPointError("AMP scaler skipped a non-finite optimizer update")
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if any(
            not torch.isfinite(parameter.detach()).all()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ):
            raise FloatingPointError("optimizer produced non-finite trainable parameters")
        self.step += 1
        self.successful_updates += 1
        valid = batch.field_masks()
        observation = batch.observed_mask
        result = {
            "step": self.step,
            "loss": float(total_loss.detach().cpu()),
            "rf_loss": float(loss.total.detach().cpu()),
            "state_h_loss": float(loss.fields["state_h"].detach().cpu()),
            "detail_h_loss": float(loss.fields["detail_h"].detach().cpu()),
            "state_v_loss": float(loss.fields["state_v"].detach().cpu()),
            "detail_v_loss": float(loss.fields["detail_v"].detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "tau_mean": float(flow_sample.tau.mean().detach().cpu()),
            "tau_min": float(flow_sample.tau.min().detach().cpu()),
            "tau_max": float(flow_sample.tau.max().detach().cpu()),
            "observation_fraction": float(observation.float().mean().detach().cpu()),
            "valid_elements": {name: int(valid[name].sum().item()) for name in valid},
        }
        for name, value in auxiliary_diagnostics.items():
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError(f"auxiliary diagnostic {name!r} must be scalar")
                result[str(name)] = float(value.detach().cpu())
            elif isinstance(value, (bool, int, float, str)):
                result[str(name)] = value
            else:
                raise TypeError(f"unsupported auxiliary diagnostic {name!r}")
        return result

    def _contracts(self) -> dict[str, Any]:
        config = self.config.contract()
        config.pop("max_steps")
        config.pop("output_root")
        return {
            "trainer": "molvid.dit",
            "config": config,
            "model_contract": self.model.contract(),
            "model_contract_hash": contract_hash(self.model.contract()),
            "adapter_contract_hash": contract_hash(self.adapter.contract()),
            "statistics_hash": "" if self.statistics is None else self.statistics.hash,
            "codec_hash": self.config.codec_hash,
            "data_hash": self.config.data_hash,
            "frozen_hashes": self.frozen_state_hashes(),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        cursor: Mapping[str, Any] | None = None,
        generator: torch.Generator | None = None,
    ) -> Path:
        return save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.step,
            cursor=dict(cursor or {}),
            contracts=self._contracts(),
            extra_state={
                "successful_optimizer_updates": self.successful_updates,
                "statistics_state": None if self.statistics is None else self.statistics.state_dict(),
                "training_generator_state": None if generator is None else generator.get_state().detach().cpu(),
            },
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        import copy

        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise ValueError("training checkpoint SHA-256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("DiT checkpoint must be a mapping")
        extra = payload.get("extra_state")
        if not isinstance(extra, Mapping):
            raise ValueError("DiT checkpoint extra state is missing")
        if int(extra.get("successful_optimizer_updates", -1)) != int(payload.get("step", -2)):
            raise ValueError("DiT successful update count disagrees with checkpoint step")
        stats_state = extra.get("statistics_state")
        if self.statistics is None:
            if stats_state is not None:
                raise ValueError("DiT trainer requires matching statistics before loading")
        else:
            if not isinstance(stats_state, Mapping) or LatentStatistics.from_state_dict(stats_state).hash != self.statistics.hash:
                raise ValueError("DiT statistics artifact differs")
        generator_state = extra.get("training_generator_state")
        if generator is not None and not isinstance(generator_state, Tensor):
            raise ValueError("DiT checkpoint lacks training-generator state")
        model_before = copy.deepcopy(self.model.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        scheduler_before = None if self.scheduler is None else copy.deepcopy(self.scheduler.state_dict())
        scaler_before = copy.deepcopy(self.scaler.state_dict())
        rng_before = capture_rng_state()
        generator_before = None if generator is None else generator.get_state().clone()
        step_before, updates_before = self.step, self.successful_updates
        try:
            loaded = load_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                expected_contracts=self._contracts(),
                expected_sha256=expected_sha256,
            )
            if generator is not None:
                generator.set_state(generator_state.detach().to(device="cpu", dtype=torch.uint8))
            self.step = int(loaded["step"])
            self.successful_updates = int(extra["successful_optimizer_updates"])
            if self.frozen_state_hashes() != self.frozen_hashes:
                raise RuntimeError("frozen codec/frame-encoder state changed during load")
            return loaded
        except Exception:
            self.model.load_state_dict(model_before, strict=True)
            self.optimizer.load_state_dict(optimizer_before)
            if self.scheduler is not None:
                self.scheduler.load_state_dict(scheduler_before)
            self.scaler.load_state_dict(scaler_before)
            restore_rng_state(rng_before)
            if generator is not None:
                generator.set_state(generator_before.detach().to(device="cpu"))
            self.step, self.successful_updates = step_before, updates_before
            raise


def train_dit(
    trainer: DiTTrainer,
    batches,
    *,
    generator: torch.Generator | None = None,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Run only the requested short/long budget; callers own batch scheduling."""

    target = trainer.config.max_steps if max_steps is None else int(max_steps)
    if target < trainer.step:
        raise ValueError("target step precedes current checkpoint")
    rows: list[dict[str, Any]] = []
    for prepared in batches:
        if trainer.step >= target:
            break
        rows.append(
            trainer.train_step(
                prepared.observed,
                generator=generator,
                source_center=prepared.source_center,
            )
        )
    if trainer.step != target:
        raise RuntimeError("DiT batch source ended before target step")
    return rows
