"""Train the current latent DiT on a frozen train/validation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..checkpoints import load_codec_artifact
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.manifest import load_datasets
from ..data.sampling import TaskAwareClipBatchSampler, TrajectoryCappedBatchSampler
from ..dit.model import MolecularDiT
from ..latent.adapter import StateDetailLatentAdapter
from ..latent.conditioning import sample_history_sigmas
from ..latent.statistics import LatentStatistics
from ..losses.geometry import FutureBondAuxiliary
from ..runtime import canonical_hash, configure_device, seed_all, sha256_file
from ..training.batches import prepare_dit_batch
from ..training.dit import DiTTrainConfig, DiTTrainer, module_state_hash


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    return parser


def _sampler(dataset: Any, training: Mapping[str, Any], seed: int):
    budget = int(training["max_tokens"])
    capped = training.get("clips_per_trajectory")
    if capped is not None:
        return TrajectoryCappedBatchSampler(
            dataset, max_tokens=budget, clips_per_trajectory=int(capped),
            seed=seed, shuffle=False,
        )
    return TaskAwareClipBatchSampler(
        dataset, max_tokens=budget, seed=seed, shuffle=False,
        replacement=False, oversize_policy="error",
    )


def _load_statistics(config: Mapping[str, Any], *, device: torch.device) -> LatentStatistics:
    path = Path(config["path"])
    expected = str(config["sha256"])
    if sha256_file(path) != expected:
        raise ValueError("statistics file SHA-256 differs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    statistics = LatentStatistics.from_state_dict(state)
    if statistics.hash != str(config["statistics_hash"]):
        raise ValueError("statistics contract hash differs")
    return statistics.to(device=device)


def _resume_cursor(
    loaded: Mapping[str, Any],
    sampler: Any,
    corruption_generator: torch.Generator,
) -> tuple[int, int]:
    cursor = loaded.get("cursor")
    if not isinstance(cursor, Mapping):
        raise ValueError("DiT checkpoint lacks a batch cursor")
    epoch = int(cursor["epoch"])
    batch_index = int(cursor["batch_index"])
    if epoch < 0 or batch_index < 0:
        raise ValueError("DiT checkpoint cursor is negative")
    sampler.set_epoch(epoch)
    sampler.validate_state_dict(cursor["sampler_state"])
    if canonical_hash(sampler.global_batches) != cursor.get("batch_schedule_hash"):
        raise ValueError("DiT checkpoint batch schedule differs")
    if batch_index > len(sampler.global_batches):
        raise ValueError("DiT checkpoint batch cursor exceeds epoch schedule")
    corruption_state = cursor.get("corruption_generator_state")
    if not isinstance(corruption_state, torch.Tensor):
        raise ValueError("DiT checkpoint lacks corruption-generator state")
    corruption_generator.set_state(corruption_state.detach().cpu())
    return epoch, batch_index


def _cursor(epoch: int, batch_index: int, sampler: Any, corruption_generator: torch.Generator) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "sampler_state": sampler.state_dict(),
        "batch_schedule_hash": canonical_hash(sampler.global_batches),
        "corruption_generator_state": corruption_generator.get_state().detach().cpu(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume_sha256 and args.resume is None:
        raise ValueError("--resume-sha256 requires --resume")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be nonnegative")
    raw = load_config(args.config, schema="molvid.dit.train.v1")
    for name in ("codec", "statistics", "model", "source", "training"):
        if not isinstance(raw.get(name), Mapping):
            raise ValueError(f"DiT configuration requires {name} mapping")
    training = dict(raw["training"])
    if args.max_steps is not None:
        training["max_steps"] = int(args.max_steps)
    seed = int(training["seed"])
    device = configure_device(str(training.get("device", "cuda")), deterministic=bool(training.get("deterministic", True)))
    splits = load_datasets(raw["manifest_root"])
    try:
        codec_config = raw["codec"]
        artifact = load_codec_artifact(
            codec_config["checkpoint"],
            expected_sha256=str(codec_config["sha256"]),
            device=device,
        )
        codec = artifact.model.eval()
        for parameter in codec.parameters():
            parameter.requires_grad_(False)
        statistics = _load_statistics(raw["statistics"], device=device)
        provenance = statistics.provenance
        if provenance.get("data_hash") != splits.data_hash:
            raise ValueError("statistics and frozen manifest data hashes differ")
        if provenance.get("codec_checkpoint_sha256") != artifact.report.source_sha256:
            raise ValueError("statistics and codec checkpoint identities differ")
        if statistics.ratio != codec.temporal_ratio or statistics.width != codec.temporal_codec.channels:
            raise ValueError("statistics and codec latent ratio/width differ")
        model_config = dict(raw["model"])
        seed_all(seed)
        adapter = StateDetailLatentAdapter(
            codec_width=statistics.width,
            scalar_width=int(model_config["scalar_width"]),
            vector_width=int(model_config["vector_width"]),
            ratio=statistics.ratio,
        )
        model = MolecularDiT(
            adapter=adapter,
            scalar_width=int(model_config["scalar_width"]),
            vector_width=int(model_config["vector_width"]),
            depth=int(model_config["depth"]),
            heads=int(model_config["heads"]),
            ffn_multiplier=int(model_config["ffn_multiplier"]),
            dropout=float(model_config.get("dropout", 0.0)),
            execution_backend=str(model_config.get("execution_backend", "factorized_v2")),
            ffn_norm_source=str(model_config.get("ffn_norm_source", "post_adaln")),
        ).to(device)
        history_order = tuple(int(value) for value in training["history_order"])
        if not history_order or any(value not in (4, 8) for value in history_order):
            raise ValueError("DiT history_order must contain H4/H8 only")
        probabilities = tuple(float(value) for value in training["history_probabilities"])
        geometry = training.get("geometry", {})
        if not isinstance(geometry, Mapping):
            raise ValueError("training.geometry must be a mapping")
        source = raw["source"]
        codec_hash = module_state_hash(codec)
        train_config = DiTTrainConfig(
            ratio=statistics.ratio,
            mode=statistics.mode,
            codec_width=statistics.width,
            scalar_width=int(model_config["scalar_width"]),
            vector_width=int(model_config["vector_width"]),
            depth=int(model_config["depth"]),
            heads=int(model_config["heads"]),
            ffn_multiplier=int(model_config["ffn_multiplier"]),
            dropout=float(model_config.get("dropout", 0.0)),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            grad_clip=float(training["grad_clip"]),
            max_steps=int(training["max_steps"]),
            seed=seed,
            amp=bool(training.get("amp", False)),
            output_root=str(training["output_root"]),
            data_hash=splits.data_hash,
            codec_hash=codec_hash,
            stats_hash=statistics.hash,
            source_mode=str(source["mode"]),
            center_kind=str(source["center_kind"]),
            normalization_hash=statistics.hash,
            init_hash=module_state_hash(model),
            observation_mixture=history_order,
            history_probabilities=probabilities,
            metadata={
                "history_order": list(history_order),
                "max_tokens": int(training["max_tokens"]),
                "clips_per_trajectory": training.get("clips_per_trajectory"),
                "codec_checkpoint_sha256": artifact.report.source_sha256,
                "deterministic": bool(training.get("deterministic", True)),
                "geometry": dict(geometry),
            },
        )
        trainer = DiTTrainer(
            model, adapter, config=train_config, statistics=statistics,
            codec=codec, frame_encoder=codec.frame_encoder,
        )
        rf_generator = torch.Generator(device=device).manual_seed(seed + 1)
        corruption_generator = torch.Generator(device=device).manual_seed(seed + 2)
        sampler = _sampler(splits.train, training, seed)
        epoch, batch_index = 0, 0
        sampler.set_epoch(epoch)
        if args.resume:
            loaded = trainer.load_checkpoint(
                args.resume, expected_sha256=args.resume_sha256, generator=rf_generator
            )
            epoch, batch_index = _resume_cursor(loaded, sampler, corruption_generator)
        lambda_bond = float(geometry.get("lambda_bond", 0.0))
        tau_threshold = float(geometry.get("tau_threshold", 0.75))
        if args.dry_run and trainer.step >= train_config.max_steps:
            raise ValueError("dry run requires one remaining optimizer step")
        last: dict[str, Any] = {}
        while trainer.step < train_config.max_steps:
            if batch_index >= len(sampler.global_batches):
                epoch += 1
                batch_index = 0
                sampler.set_epoch(epoch)
            indices = sampler.global_batches[batch_index]
            batch = collate_clip_records([splits.train[index] for index in indices])
            if batch.frames != 16:
                raise ValueError("DiT training requires 16-frame trajectory clips")
            history = history_order[trainer.step % len(history_order)]
            sigmas = sample_history_sigmas(
                batch.batch_size,
                generator=corruption_generator, device=device, dtype=batch.x.dtype,
                probabilities=probabilities,
            )
            epsilon = torch.randn(
                batch.x.shape, device=device, dtype=batch.x.dtype,
                generator=corruption_generator,
            )
            prepared = prepare_dit_batch(
                codec, adapter, batch, device=device, statistics=statistics,
                history_frames=history, sigmas_angstrom=sigmas, epsilon=epsilon,
                source_mode=train_config.source_mode, center_kind=train_config.center_kind,
                codec_hash=codec_hash, data_hash=splits.data_hash,
            )
            if args.dry_run:
                print(json.dumps({
                    "dry_run": True, "history_frames": history,
                    "batch_size": batch.batch_size,
                    "observed_tokens": int(prepared.observed.observed_mask.sum()),
                }, sort_keys=True))
                return 0
            auxiliary = None
            if lambda_bond:
                auxiliary = FutureBondAuxiliary(
                    adapter=adapter, statistics=statistics, codec=codec,
                    coordinate_batch=prepared.target_view, history_frames=history,
                    lambda_bond=lambda_bond, tau_threshold=tau_threshold,
                )
            last = trainer.train_step(
                prepared.observed, generator=rf_generator,
                source_center=prepared.source_center,
                auxiliary_objective=auxiliary,
            )
            batch_index += 1
            if args.checkpoint_every and trainer.step % args.checkpoint_every == 0:
                trainer.save_checkpoint(
                    Path(training["output_root"]) / f"dit_step_{trainer.step:08d}.pt",
                    cursor=_cursor(epoch, batch_index, sampler, corruption_generator),
                    generator=rf_generator,
                )
        checkpoint = trainer.save_checkpoint(
            Path(training["output_root"]) / f"dit_step_{trainer.step:08d}.pt",
            cursor=_cursor(epoch, batch_index, sampler, corruption_generator),
            generator=rf_generator,
        )
        print(json.dumps({"checkpoint": str(checkpoint), "step": trainer.step, "metrics": last}, sort_keys=True))
        return 0
    finally:
        splits.close()


if __name__ == "__main__":
    raise SystemExit(main())
