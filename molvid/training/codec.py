"""One optimizer loop for the current trajectory codec and strict resumable state."""

from __future__ import annotations

import contextlib
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
from torch import Tensor, nn

from ..checkpoints import (
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from ..data.batch import TASK_NAMES
from ..losses.reconstruction import (
    BucketNormalization,
    CodecLossWeights,
    TimeBucketSpec,
    compute_codec_losses,
    fit_time_bucket_normalization,
)
from .batches import prepare_batch_then_to_device

CODEC_CONFIG_SCHEMA = "molvid.codec.config.v1"


def validate_codec_config(config: Mapping[str, Any]) -> tuple[TimeBucketSpec, ...]:
    """Validate the explicit physical-time and normalization configuration."""

    if str(config.get("schema_version", "")) != CODEC_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported codec config schema; expected {CODEC_CONFIG_SCHEMA!r}"
        )
    time_config = config.get("time")
    if not isinstance(time_config, Mapping):
        raise ValueError("codec config requires a time mapping")
    if str(time_config.get("unit", "")) != "ps":
        raise ValueError("codec time.unit must be the canonical 'ps'")
    continuous_scale = float(time_config.get("continuous_scale_ps", 0.0))
    if not math.isfinite(continuous_scale) or continuous_scale <= 0:
        raise ValueError("time.continuous_scale_ps must be finite and positive")
    buckets = time_config.get("buckets")
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("codec config requires at least one time bucket")
    specs = tuple(
        TimeBucketSpec(
            bucket_id=str(item.get("id", "")),
            center_ps=float(item.get("center_ps", 0.0)),
            tolerance_ps=float(item.get("tolerance_ps", 0.0)),
            weight=float(item.get("weight", 1.0)),
        )
        for item in buckets
        if isinstance(item, Mapping)
    )
    if len(specs) != len(buckets) or len({item.bucket_id for item in specs}) != len(specs):
        raise ValueError("time bucket entries must be mappings with unique ids")
    normalization = time_config.get("normalization", {})
    if not isinstance(normalization, Mapping):
        raise ValueError("time.normalization must be a mapping")
    min_count = int(normalization.get("min_count", 1))
    epsilon = float(normalization.get("epsilon", 0.0))
    if min_count < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("normalization min_count must be positive and epsilon must be positive")
    return specs


@dataclass
class CodecTrainConfig:
    """Validated runtime subset of ``configs/codec_train.yaml``."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1
    grad_clip: Optional[float] = 1.0
    warmup_steps: int = 0
    device: str = "cpu"
    loss_schedule: tuple[tuple[int, CodecLossWeights], ...] = ((0, CodecLossWeights()),)
    bucket_specs: tuple[TimeBucketSpec, ...] = (
        TimeBucketSpec("static", 0.0, 0.0, 1.0),
        TimeBucketSpec("dt_100ps", 100.0, 1.0, 1.0),
    )
    continuous_scale_ps: float = 100.0
    normalization_min_count: int = 32
    normalization_epsilon: float = 1e-6
    precision: str = "fp32"

    def __post_init__(self) -> None:
        self.loss_schedule = tuple(
            (int(start), weights if isinstance(weights, CodecLossWeights) else CodecLossWeights.from_mapping(weights))
            for start, weights in self.loss_schedule
        )
        if not math.isfinite(float(self.lr)) or float(self.lr) <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not math.isfinite(float(self.weight_decay)) or float(self.weight_decay) < 0:
            raise ValueError("weight decay must be finite and non-negative")
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if self.grad_clip is not None and (not math.isfinite(float(self.grad_clip)) or float(self.grad_clip) <= 0):
            raise ValueError("grad_clip must be positive when supplied")
        if int(self.warmup_steps) < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not self.loss_schedule or self.loss_schedule[0][0] != 0:
            raise ValueError("loss schedule must start at step zero")
        if any(start < 0 for start, _ in self.loss_schedule):
            raise ValueError("loss schedule steps must be non-negative")
        if any(self.loss_schedule[i][0] >= self.loss_schedule[i + 1][0] for i in range(len(self.loss_schedule) - 1)):
            raise ValueError("loss schedule steps must be strictly increasing")
        if not math.isfinite(float(self.continuous_scale_ps)) or float(self.continuous_scale_ps) <= 0:
            raise ValueError("continuous_scale_ps must be finite and positive")
        if int(self.normalization_min_count) < 1 or float(self.normalization_epsilon) <= 0:
            raise ValueError("normalization guards must be positive")
        self.precision = str(self.precision).lower()
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CodecTrainConfig":
        specs = validate_codec_config(config)
        time_config = config["time"]
        training = config.get("training", {})
        if not isinstance(training, Mapping):
            raise ValueError("training config must be a mapping")
        base_weights = CodecLossWeights.from_mapping(training.get("loss_weights"))
        stages = training.get("loss_schedule")
        if stages is None:
            schedule = ((0, base_weights),)
        else:
            if not isinstance(stages, list) or not stages:
                raise ValueError("training.loss_schedule must be a non-empty list")
            parsed = []
            for stage in stages:
                if not isinstance(stage, Mapping):
                    raise ValueError("each loss schedule stage must be a mapping")
                stage_weights = stage.get("weights", stage)
                parsed.append((int(stage.get("start_step", 0)), CodecLossWeights.from_mapping(stage_weights)))
            schedule = tuple(sorted(parsed, key=lambda item: item[0]))
        normalization = time_config.get("normalization", {})
        grad_clip = training.get("grad_clip", 1.0)
        return cls(
            lr=float(training.get("lr", 1e-4)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_steps=int(training.get("max_steps", 1)),
            grad_clip=None if grad_clip is None else float(grad_clip),
            warmup_steps=int(training.get("warmup_steps", 0)),
            device=str(training.get("device", "cpu")),
            loss_schedule=schedule,
            bucket_specs=specs,
            continuous_scale_ps=float(time_config["continuous_scale_ps"]),
            normalization_min_count=int(normalization.get("min_count", 32)),
            normalization_epsilon=float(normalization.get("epsilon", 1e-6)),
            precision=str(training.get("precision", "fp32")).lower(),
        )

    def weights_at(self, step: int) -> CodecLossWeights:
        selected = self.loss_schedule[0][1]
        for start, weights in self.loss_schedule:
            if int(step) >= start:
                selected = weights
            else:
                break
        return selected

    def bucket(self, bucket_id: str) -> TimeBucketSpec:
        for spec in self.bucket_specs:
            if spec.bucket_id == str(bucket_id):
                return spec
        raise ValueError(f"unconfigured time bucket: {bucket_id!r}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CODEC_CONFIG_SCHEMA,
            "time": {
                "unit": "ps",
                "continuous_scale_ps": float(self.continuous_scale_ps),
                "buckets": [spec.as_dict() for spec in self.bucket_specs],
                "normalization": {
                    "min_count": int(self.normalization_min_count),
                    "epsilon": float(self.normalization_epsilon),
                },
            },
            "training": {
                "lr": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "max_steps": int(self.max_steps),
                "grad_clip": self.grad_clip,
                "warmup_steps": int(self.warmup_steps),
                "device": self.device,
                "precision": self.precision,
                "loss_schedule": [
                    {"start_step": start, "weights": weights.as_dict()}
                    for start, weights in self.loss_schedule
                ],
            },
        }


class CodecTrainer:
    """Small self-contained optimizer loop with new-format checkpoint state."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable[Any],
        valid_loader: Optional[Iterable[Any]] = None,
        config: CodecTrainConfig | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
        normalization_stats: Mapping[str, BucketNormalization | Mapping[str, Any]] | None = None,
        non_blocking_transfer: bool = True,
    ) -> None:
        self.config = (
            config if isinstance(config, CodecTrainConfig)
            else CodecTrainConfig.from_mapping(config) if config is not None
            else CodecTrainConfig()
        )
        selected_device = device or self.config.device
        if str(selected_device).lower() == "auto":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "training.device='auto' requires CUDA; GPU is unavailable and "
                    "the production trainer will not fall back to CPU"
                )
            selected_device = "cuda"
        if str(selected_device).lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"requested CUDA device {selected_device!r} is unavailable; "
                "the production trainer will not fall back to CPU"
            )
        self.device = torch.device(selected_device)
        if self.config.precision == "bf16" and self.device.type != "cuda":
            raise RuntimeError(
                "BF16 codec training requires CUDA; refusing a CPU precision fallback"
            )
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        self.normalization_stats: dict[str, BucketNormalization] = {}
        for key, value in (normalization_stats or {}).items():
            self.normalization_stats[str(key)] = value if isinstance(value, BucketNormalization) else BucketNormalization.from_dict(value)
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self._train_iterator: Optional[Iterable[Any]] = None

    def fit_normalization(self, batches: Optional[Iterable[Any]] = None) -> dict[str, BucketNormalization]:
        source = self.train_loader if batches is None else batches
        self.normalization_stats = fit_time_bucket_normalization(
            source,
            min_count=self.config.normalization_min_count,
            epsilon=self.config.normalization_epsilon,
        )
        return self.normalization_stats

    def _validate_batch_clock(self, batch: Any) -> float:
        bucket_ids = tuple(str(item) for item in batch.time_bucket_id)
        if not bucket_ids:
            raise ValueError("codec batch must contain at least one time bucket")
        delta_time = batch.delta_time_ps
        frame_mask = batch.frame_mask
        if isinstance(delta_time, Tensor) and delta_time.device.type != "cpu":
            raise RuntimeError(
                "batch clock validation must run on the CPU batch before transfer"
            )
        if isinstance(frame_mask, Tensor) and frame_mask.device.type != "cpu":
            raise RuntimeError(
                "batch frame-mask validation must run on the CPU batch before transfer"
            )
        frames = int(batch.frames) if hasattr(batch, "frames") else int(batch.x.shape[0])
        weights = []
        for sample_index, bucket_id in enumerate(bucket_ids):
            spec = self.config.bucket(bucket_id)
            weights.append(spec.weight)
            if frames > 1:
                valid_delta = delta_time[sample_index][frame_mask[sample_index, 1:]]
                if valid_delta.numel() and torch.any(
                    (valid_delta - spec.center_ps).abs() > spec.tolerance_ps
                ):
                    raise ValueError(
                        f"batch delta_time_ps does not match bucket {bucket_id!r} center/tolerance"
                    )
        return float(sum(weights) / len(weights))

    def _autocast_context(self):
        if self.config.precision == "bf16":
            if self.device.type != "cuda":
                raise RuntimeError(
                    "BF16 autocast requires CUDA; CPU fallback is not permitted"
                )
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _prepare_batch(self, batch: Any) -> None:
        prepare = getattr(self.model, "prepare_batch", None)
        if callable(prepare):
            prepare(batch)

    def _loss_for_batch(self, batch: Any) -> tuple[dict[str, Tensor], float, Any]:
        # Validate immutable clock metadata while it is still on the CPU. This
        # keeps the CUDA transfer/forward path free of Python CUDA scalar reads.
        bucket_weight = self._validate_batch_clock(batch)
        batch = prepare_batch_then_to_device(
            self.model,
            batch,
            self.device,
            non_blocking=self.non_blocking_transfer,
        )
        with self._autocast_context():
            output = self.model(batch)
        losses = compute_codec_losses(
            output,
            batch,
            weights=self.config.weights_at(self.step),
            normalization=self.normalization_stats,
        )
        losses["total"] = losses["total"] * bucket_weight
        return losses, bucket_weight, batch

    @staticmethod
    def _metrics(losses: Mapping[str, Tensor], batch: Any) -> dict[str, float]:
        metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
        host_task_ids = tuple(getattr(batch, "host_task_ids", ()))
        if host_task_ids:
            task_ids = tuple(sorted(set(int(value) for value in host_task_ids)))
        else:
            task = torch.as_tensor(batch.task, dtype=torch.long).flatten()
            if task.device.type != "cpu":
                raise RuntimeError(
                    "CUDA metric task labels require host_task_ids from CPU collation"
                )
            task_ids = tuple(sorted(set(int(value) for value in task.tolist())))
        for task_id in task_ids:
            task_name = TASK_NAMES[int(task_id)]
            metrics[f"task_{task_name}_total"] = metrics["total"]
        for bucket_id in dict.fromkeys(str(item) for item in batch.time_bucket_id):
            safe_bucket = bucket_id.replace("/", "_")
            metrics[f"bucket_{safe_bucket}_velocity"] = metrics["velocity"]
            metrics[f"bucket_{safe_bucket}_velocity_raw"] = metrics["velocity_raw"]
            metrics[f"bucket_{safe_bucket}_acceleration"] = metrics["acceleration"]
            metrics[f"bucket_{safe_bucket}_acceleration_raw"] = metrics["acceleration_raw"]
        return metrics

    def optimizer_step(self, batch: Any) -> dict[str, float]:
        self.model.train()
        next_step = self.step + 1
        if self.config.warmup_steps:
            scale = min(1.0, next_step / float(self.config.warmup_steps))
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr * scale
        self.optimizer.zero_grad(set_to_none=True)
        losses, _, moved_batch = self._loss_for_batch(batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("codec loss is NaN or Inf before backward")
        total.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        for parameter in self.model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("codec gradient is NaN or Inf")
        self.optimizer.step()
        self.step = next_step
        return self._metrics(losses, moved_batch)

    @torch.no_grad()
    def evaluate_batch(self, batch: Any) -> dict[str, float]:
        self.model.eval()
        losses, _, moved_batch = self._loss_for_batch(batch)
        return self._metrics(losses, moved_batch)

    def _batch_sampler(self):
        return getattr(self.train_loader, "batch_sampler", None)

    def _new_train_iterator(self) -> Iterable[Any]:
        sampler = self._batch_sampler()
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)
        iterator = iter(self.train_loader)
        for _ in range(int(self.batch_in_epoch)):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "checkpoint batch_in_epoch exceeds the current epoch schedule"
                ) from exc
        return iterator

    def run(
        self,
        max_steps: Optional[int] = None,
        *,
        log_path: str | Path | None = None,
        log_every: int = 1,
    ) -> dict[str, float]:
        """Optimize until max_steps and optionally append step metrics as JSONL."""

        target = int(max_steps if max_steps is not None else self.config.max_steps)
        if target < self.step:
            raise ValueError("max_steps cannot be less than the current checkpoint step")
        if int(log_every) < 1:
            raise ValueError("log_every must be positive")
        if len(self.train_loader) < 1:
            raise RuntimeError("training loader contains no batches")
        log_handle = None
        if log_path is not None:
            destination = Path(log_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            log_handle = destination.open("a", encoding="utf-8")
        if self._train_iterator is None:
            self._train_iterator = self._new_train_iterator()
        last: dict[str, float] = {}
        try:
            while self.step < target:
                if self._train_iterator is None:
                    self._train_iterator = self._new_train_iterator()
                try:
                    batch = next(self._train_iterator)
                except StopIteration:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self._train_iterator = self._new_train_iterator()
                    batch = next(self._train_iterator)
                last = self.optimizer_step(batch)
                self.batch_in_epoch += 1
                finished_epoch = self.batch_in_epoch >= len(self.train_loader)
                logged_epoch = self.epoch
                if finished_epoch:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                    self._train_iterator = None
                if log_handle is not None and (
                    self.step % int(log_every) == 0 or self.step == target
                ):
                    record = {
                        "schema_version": "molvid.codec.train.v1",
                        "split": "train",
                        "step": int(self.step),
                        "epoch": int(logged_epoch),
                        "batch_in_epoch": int(self.batch_in_epoch),
                        "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                        "metrics": last,
                    }
                    log_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    log_handle.flush()
        finally:
            if log_handle is not None:
                log_handle.close()
        return last

    def _contracts(self) -> dict[str, Any]:
        config = self.config.as_dict()
        config["training"] = dict(config["training"])
        config["training"].pop("max_steps")
        parameters = [
            (name, list(value.shape), str(value.dtype), bool(value.requires_grad))
            for name, value in self.model.named_parameters()
        ]
        state = [
            (name, list(value.shape), str(value.dtype))
            for name, value in self.model.state_dict().items()
        ]
        return {
            "trainer": "molvid.codec",
            "model_type": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "model_mode": getattr(self.model, "temporal_codec_mode", None),
            "coordinate_stem": getattr(self.model, "coordinate_stem_mode", None),
            "frame_encoder_source_hash": getattr(self.model, "frame_encoder_source_hash", None),
            "distance_reference_contract": self.model.distance_reference_contract() if hasattr(self.model, "distance_reference_contract") else None,
            "bond_construction": getattr(getattr(self.model, "frame_encoder", None), "bond_construction_mode", None),
            "parameters": parameters,
            "state": state,
            "config": config,
        }

    def _sampler_state(self):
        sampler = self._batch_sampler()
        return sampler.state_dict() if sampler is not None and hasattr(sampler, "state_dict") else None

    def save_checkpoint(self, path: str | Path) -> Path:
        return save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            step=self.step,
            cursor={"epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch},
            sampler_state=self._sampler_state(),
            contracts=self._contracts(),
            extra_state={
                "config": self.config.as_dict(),
                "normalization_stats": {
                    key: value.as_dict() for key, value in self.normalization_stats.items()
                },
            },
        )

    def load_checkpoint(self, path: str | Path, *, expected_sha256: str | None = None) -> None:
        # Validate trainer-owned fields before the shared loader mutates any state.
        if expected_sha256 is not None:
            from ..runtime import sha256_file

            if sha256_file(path) != expected_sha256:
                raise ValueError("training checkpoint SHA-256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("training checkpoint must be a mapping")
        extra = payload.get("extra_state")
        cursor = payload.get("cursor")
        if not isinstance(extra, Mapping) or not isinstance(extra.get("config"), Mapping):
            raise ValueError("codec checkpoint is missing its config")
        if not isinstance(cursor, Mapping):
            raise ValueError("codec checkpoint is missing its cursor")
        saved_config = CodecTrainConfig.from_mapping(extra["config"])
        if self.config.max_steps < saved_config.max_steps:
            raise ValueError("resume max_steps cannot be shorter than checkpoint max_steps")
        try:
            epoch = int(cursor["epoch"])
            batch_in_epoch = int(cursor["batch_in_epoch"])
            step = int(payload["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("codec checkpoint cursor/step is invalid") from exc
        if min(epoch, batch_in_epoch, step) < 0 or batch_in_epoch >= len(self.train_loader):
            raise ValueError("codec checkpoint cursor is out of range")
        if not isinstance(extra.get("normalization_stats"), Mapping):
            raise ValueError("codec checkpoint normalization statistics are missing")
        normalization = {
            str(key): BucketNormalization.from_dict(value)
            for key, value in extra["normalization_stats"].items()
        }
        saved_optimizer = payload.get("optimizer_state")
        groups = saved_optimizer.get("param_groups") if isinstance(saved_optimizer, Mapping) else None
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError("codec checkpoint optimizer groups are invalid")
        expected_lr = self.config.lr
        if self.config.warmup_steps and step > 0:
            expected_lr *= min(1.0, step / float(self.config.warmup_steps))
        if (
            float(groups[0].get("lr", float("nan"))) != float(expected_lr)
            or float(groups[0].get("weight_decay", float("nan"))) != float(self.config.weight_decay)
        ):
            raise ValueError("codec checkpoint optimizer hyperparameters disagree")
        sampler = self._batch_sampler()
        saved_sampler = payload.get("sampler_state")
        if saved_sampler is not None:
            if sampler is None or not hasattr(sampler, "load_state_dict"):
                raise ValueError("codec checkpoint sampler cannot be restored")
            validator = getattr(sampler, "validate_state_dict", None)
            if callable(validator):
                validator(saved_sampler)
        old_model = copy.deepcopy(self.model.state_dict())
        old_optimizer = copy.deepcopy(self.optimizer.state_dict())
        old_sampler = copy.deepcopy(self._sampler_state())
        old_rng = capture_rng_state()
        old_cursor = (self.step, self.epoch, self.batch_in_epoch, self._train_iterator)
        old_normalization = self.normalization_stats
        try:
            load_training_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                expected_contracts=self._contracts(),
                expected_sha256=expected_sha256,
            )
            if saved_sampler is not None:
                sampler.load_state_dict(saved_sampler)
            self.step, self.epoch, self.batch_in_epoch = step, epoch, batch_in_epoch
            self.normalization_stats = normalization
            self._train_iterator = None
        except Exception:
            self.model.load_state_dict(old_model, strict=True)
            self.optimizer.load_state_dict(old_optimizer)
            if sampler is not None and old_sampler is not None:
                sampler.load_state_dict(old_sampler)
            restore_rng_state(old_rng)
            self.step, self.epoch, self.batch_in_epoch, self._train_iterator = old_cursor
            self.normalization_stats = old_normalization
            raise


def train_codec(
    model: nn.Module,
    train_loader: Iterable[Any],
    *,
    config: CodecTrainConfig,
    valid_loader: Optional[Iterable[Any]] = None,
    max_steps: Optional[int] = None,
) -> tuple[CodecTrainer, dict[str, float]]:
    trainer = CodecTrainer(model, train_loader, valid_loader, config)
    return trainer, trainer.run(max_steps=max_steps)
