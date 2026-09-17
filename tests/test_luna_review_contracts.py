from __future__ import annotations

import copy
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

import trainer.codec_trainer as codec_trainer_module
from data.clip_batching import make_clip_dataloader
from data.clip_dataset import collate_clip_records
from module.bond_sources import DistanceOnlyBondCache, build_canonical_reference_index
from scripts.check_tracked_artifacts import inventory, is_generated_checkpoint, purpose_for
from scripts.run_engineering_acceptance import _compare_fp32_evidence, _relative
from trainer.codec_trainer import (
    CodecTrainConfig,
    CodecTrainer,
    PVBCodecModel,
    TimeBucketSpec,
    prepare_batch_then_to_device,
)
from tests.test_codec_training import _record


class _OffsetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor([1.0, -1.0, 0.5]))

    def forward(self, batch):
        return batch.x + self.offset.view(1, 1, 3)


def _config(max_steps: int = 2, *, lr: float = 0.1, warmup_steps: int = 1) -> CodecTrainConfig:
    return CodecTrainConfig(
        lr=lr,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        grad_clip=1.0,
        bucket_specs=(TimeBucketSpec("dt_100ps", 100.0, 1.0),),
        loss_schedule=((0, {"coordinate": 1.0}),),
        normalization_min_count=1,
    )


def _same_state(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if is_dataclass(left) and is_dataclass(right):
        return _same_state(asdict(left), asdict(right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_state(a, b) for a, b in zip(left, right)
        )
    return left == right


def _tiny_pvb(*, cutoff_upper: float = 5.0, ratio: int = 2, time_scale_ps: float = 100.0, bond_mode: str = "topology") -> PVBCodecModel:
    return PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        temporal_layers=1,
        temporal_ratio=ratio,
        num_rbf=4,
        num_heads=2,
        cutoff_lower=0.0,
        cutoff_upper=cutoff_upper,
        max_num_neighbors=4,
        neighbor_backend="dense_test",
        bond_construction={"mode": bond_mode},
        time_scale_ps=time_scale_ps,
    )


def test_prepare_cpu_graph_inputs_precedes_transfer(monkeypatch):
    batch = collate_clip_records([_record()])
    events: list[str] = []

    class OrderingModel(nn.Module):
        def prepare_batch(self, cpu_batch):
            assert cpu_batch.atom_ptr.device.type == "cpu"
            events.append("prepare")

    model = OrderingModel()

    def transfer(value, device, *, non_blocking=False):
        assert events == ["prepare"]
        events.append("transfer")
        return value

    monkeypatch.setattr(codec_trainer_module, "_to_device", transfer)
    assert prepare_batch_then_to_device(model, batch, torch.device("cpu")) is batch
    assert events == ["prepare", "transfer"]


def test_pvb_parameter_keys_are_unchanged_across_graph_modes_and_contract_is_semantic():
    topology = _tiny_pvb(bond_mode="topology")
    distance = _tiny_pvb(bond_mode="distance_only")
    top_state = topology.state_dict()
    distance_state = distance.state_dict()
    assert list(top_state) == list(distance_state)
    assert {key: tuple(value.shape) for key, value in top_state.items()} == {
        key: tuple(value.shape) for key, value in distance_state.items()
    }
    assert sum(value.numel() for value in top_state.values()) == sum(
        value.numel() for value in distance_state.values()
    )

    contract = topology.model_contract()
    for key, value in (
        ("cutoff_upper", 4.0),
        ("temporal_ratio", 4),
        ("time_scale_ps", 50.0),
        ("max_num_neighbors", 3),
    ):
        changed = copy.deepcopy(contract)
        changed["constructor"][key] = value
        with pytest.raises(ValueError, match="contract mismatch"):
            codec_trainer_module.require_contract_equal(
                contract, changed, label="model contract"
            )
    changed = copy.deepcopy(contract)
    changed["constructor"]["bond_construction"] = {"mode": "distance_only"}
    with pytest.raises(ValueError, match="contract mismatch"):
        codec_trainer_module.require_contract_equal(
            contract, changed, label="model contract"
        )


def test_checkpoint_rejects_non_shape_graph_semantic_mismatch_before_mutation(tmp_path: Path):
    batch = collate_clip_records([_record()])
    config = _config()
    source = CodecTrainer(_tiny_pvb(), [batch], config=config, device="cpu")
    checkpoint = source.save_checkpoint(tmp_path / "source.pt")
    target = CodecTrainer(
        _tiny_pvb(cutoff_upper=4.0), [batch], config=config, device="cpu"
    )
    before_model = copy.deepcopy(target.model.state_dict())
    before_optimizer = copy.deepcopy(target.optimizer.state_dict())
    with pytest.raises(ValueError, match="model contract mismatch"):
        target.load_checkpoint(checkpoint)
    assert _same_state(before_model, target.model.state_dict())
    assert _same_state(before_optimizer, target.optimizer.state_dict())
    assert (target.step, target.epoch, target.batch_in_epoch) == (0, 0, 0)


def test_resume_matches_uninterrupted_next_step_and_failed_load_is_atomic(tmp_path: Path):
    batch = collate_clip_records([_record()])
    batches = [batch, batch]
    config = _config(max_steps=1)
    uninterrupted = CodecTrainer(_OffsetModel(), batches, config=config, device="cpu")
    uninterrupted.fit_normalization()
    uninterrupted.run(max_steps=1)
    checkpoint = uninterrupted.save_checkpoint(tmp_path / "resume.pt")
    uninterrupted.run(max_steps=2)

    resumed = CodecTrainer(_OffsetModel(), batches, config=_config(max_steps=2), device="cpu")
    resumed.fit_normalization()
    resumed.load_checkpoint(checkpoint)
    assert (resumed.step, resumed.epoch, resumed.batch_in_epoch) == (
        uninterrupted.step - 1,
        0,
        1,
    )
    resumed.run(max_steps=2)
    assert torch.equal(resumed.model.offset, uninterrupted.model.offset)
    assert _same_state(resumed.optimizer.state_dict(), uninterrupted.optimizer.state_dict())

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["config"]["training"]["lr"] = 0.2
    bad = tmp_path / "bad_config.pt"
    torch.save(payload, bad)
    failed = CodecTrainer(_OffsetModel(), batches, config=_config(max_steps=2), device="cpu")
    failed.fit_normalization()
    failed.optimizer_step(batch)
    model_before = copy.deepcopy(failed.model.state_dict())
    optimizer_before = copy.deepcopy(failed.optimizer.state_dict())
    counters_before = (failed.step, failed.epoch, failed.batch_in_epoch)
    stats_before = copy.deepcopy(failed.normalization_stats)
    with pytest.raises(ValueError, match="training config mismatch"):
        failed.load_checkpoint(bad)
    assert _same_state(model_before, failed.model.state_dict())
    assert _same_state(optimizer_before, failed.optimizer.state_dict())
    assert counters_before == (failed.step, failed.epoch, failed.batch_in_epoch)
    assert _same_state(stats_before, failed.normalization_stats)


def test_explicit_legacy_v1_resume_derives_and_validates_optimizer_contract(tmp_path: Path):
    batch = collate_clip_records([_record()])
    source = CodecTrainer(_OffsetModel(), [batch], config=_config(max_steps=1), device="cpu")
    source.fit_normalization()
    source.run(max_steps=1)
    payload = copy.deepcopy(source.checkpoint_state())
    payload["schema_version"] = codec_trainer_module.LEGACY_CODEC_CHECKPOINT_SCHEMA
    for field in ("model_contract", "distance_reference_contract", "optimizer_contract"):
        payload.pop(field)
    legacy_path = tmp_path / "legacy-v1.pt"
    torch.save(payload, legacy_path)

    target = CodecTrainer(_OffsetModel(), [batch], config=_config(max_steps=2), device="cpu")
    with pytest.raises(ValueError, match="legacy codec checkpoint v1"):
        target.load_checkpoint(legacy_path)
    target.load_checkpoint(legacy_path, allow_legacy=True)
    assert target.step == source.step
    assert torch.equal(target.model.offset, source.model.offset)
    assert target.optimizer.param_groups[0]["lr"] == source.optimizer.param_groups[0]["lr"]


def test_sampler_contract_is_validated_on_checkpoint_load(tmp_path: Path):
    records = [_record() for _ in range(2)]
    records[0]["sample_id"] = "s0"
    records[1]["sample_id"] = "s1"
    loader = make_clip_dataloader(
        records,
        max_tokens=100,
        collate_fn=collate_clip_records,
        num_workers=0,
        shuffle=False,
        replacement=False,
    )
    trainer = CodecTrainer(_OffsetModel(), loader, config=_config(), device="cpu")
    checkpoint = trainer.save_checkpoint(tmp_path / "sampler.pt")
    different_loader = make_clip_dataloader(
        records,
        max_tokens=8,
        collate_fn=collate_clip_records,
        num_workers=0,
        shuffle=False,
        replacement=False,
    )
    target = CodecTrainer(
        _OffsetModel(), different_loader, config=_config(), device="cpu"
    )
    with pytest.raises(ValueError, match="sampler max_tokens differs"):
        target.load_checkpoint(checkpoint)


def test_distance_reference_selection_is_split_isolated_and_identity_provenance_is_stable():
    train = _record()
    train.update({"sample_id": "z-train", "topology_id": "shared", "split": "train"})
    validation = copy.deepcopy(train)
    validation.update({"sample_id": "a-valid", "split": "valid"})
    validation["x"] = validation["x"].copy()
    validation["x"][0, 1, 0] += 0.9

    references = build_canonical_reference_index([train], source_split="train")
    assert references["shared"]["sample_id"] == "z-train"
    assert references["shared"]["source_split"] == "train"
    identity = references["shared"]["atom_identity_sha256"]
    assert len(identity) == 64
    before = references["shared"]["coordinates"].clone()
    validation["x"][0, 1, 0] += 9.0
    assert torch.equal(before, references["shared"]["coordinates"])
    with pytest.raises(RuntimeError, match="expected 'train'"):
        build_canonical_reference_index([train, validation], source_split="train")


def test_bf16_relative_error_detects_equal_norm_tensor_perturbation():
    left = torch.tensor([1.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    assert torch.linalg.vector_norm(left) == torch.linalg.vector_norm(right)
    assert _relative(left, right) > 1.0


def _minimal_fp32_evidence():
    reference = {
        "reference_sha256": "a" * 64,
        "encoded_h": torch.tensor([1.0, 2.0]),
        "encoded_v": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "decoded_x_coarse": torch.tensor([[1.0, 2.0, 3.0]]),
        "decoded_x_hat": torch.tensor([[1.5, 2.5, 3.5]]),
        "losses": {"coordinate": 2.0, "total": 2.0},
        "parameter_count": 2,
        "parameter_names": ["spatial.layer.weight"],
        "parameter_shapes": {"spatial.layer.weight": [2]},
        "parameter_gradients": {
            "spatial.layer.weight": torch.tensor([1.0, 2.0])
        },
        "position_gradient": torch.tensor([[1.0, 2.0, 3.0]]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "bond_type": torch.tensor([1, 1]),
    }
    return reference, copy.deepcopy(reference)


def test_fp32_gate_rejects_finite_gradient_and_output_regressions():
    reference, candidate = _minimal_fp32_evidence()
    acceptance = {
        "fp32_scalar_rtol": 1.0e-6,
        "fp32_scalar_atol": 1.0e-7,
        "fp32_vector_rtol": 1.0e-6,
        "fp32_vector_atol": 1.0e-7,
        "total_loss_relative": 1.0e-6,
        "gradient_relative": 1.0e-6,
    }
    candidate["parameter_gradients"]["spatial.layer.weight"][0] += 1.0
    with pytest.raises(RuntimeError, match="FP32 regression"):
        _compare_fp32_evidence("tiny", candidate, reference, acceptance)
    reference, candidate = _minimal_fp32_evidence()
    candidate["encoded_h"] = torch.tensor([2.0, 1.0])  # equal norm, different tensor
    with pytest.raises(RuntimeError, match="FP32 regression"):
        _compare_fp32_evidence("tiny", candidate, reference, acceptance)
    reference, candidate = _minimal_fp32_evidence()
    candidate["parameter_gradients"] = {}
    with pytest.raises(RuntimeError, match="gradient coverage"):
        _compare_fp32_evidence("tiny", candidate, reference, acceptance)


def test_tracked_artifact_guard_reports_generated_checkpoint(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    checkpoint = tmp_path / "outputs" / "experiment" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    subprocess.run(["git", "add", str(checkpoint)], cwd=tmp_path, check=True)
    result = inventory(tmp_path)
    assert result["tracked_binary_count"] == 1
    assert result["artifacts"][0]["path"] == "outputs/experiment/model.pt"
    assert "generated" in result["artifacts"][0]["purpose"]
    assert purpose_for(Path("outputs/experiment/model.pt"))
    assert is_generated_checkpoint(Path("outputs/experiment/model.pt"))
    assert not is_generated_checkpoint(Path("module/equiformer_v2/Jd.pt"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cache capacity test")
def test_distance_cache_rehydrates_evicted_reference_and_checks_identity():
    try:
        import torch_cluster  # noqa: F401
    except Exception:
        pytest.skip("torch_cluster CUDA extension is unavailable")
    device = torch.device("cuda:0")
    cache = DistanceOnlyBondCache(capacity=1, max_num_neighbors=16)
    coordinates = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    cache.register_reference(
        "a",
        coordinates,
        device=device,
        atom_identity_sha256="a" * 64,
        sample_id="train-a",
        source_split="train",
    )
    cache.register_reference(
        "b",
        coordinates + 10.0,
        device=device,
        atom_identity_sha256="b" * 64,
        sample_id="train-b",
        source_split="train",
    )
    assert cache.stats()["evictions"] == 1
    rehydrated = cache.materialize(
        "a",
        device=device,
        atom_count=3,
        atom_identity_sha256="a" * 64,
    )
    assert rehydrated.device.type == "cuda"
    with pytest.raises(RuntimeError, match="atom count mismatch"):
        cache.materialize(
            "a", device=device, atom_count=4, atom_identity_sha256="a" * 64
        )
    with pytest.raises(RuntimeError, match="atom identity mismatch"):
        cache.materialize(
            "a", device=device, atom_count=3, atom_identity_sha256="c" * 64
        )
