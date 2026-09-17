from __future__ import annotations

import pytest
import torch
from dataclasses import replace
from torch import nn

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import LatentStatistics, StateDetailLatentAdapter
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer
from dit_test_utils import make_batch


torch.set_num_threads(1)


def _trainer(tmp_path, *, data_hash: str = "fixture"):
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    model = MolecularDiT(adapter=adapter, scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2)
    batch = make_batch(2, width=4)
    stats = LatentStatistics.fit([batch], ratio=2, provenance={"split": "train"})
    codec = nn.Linear(2, 2)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    config = DiTTrainConfig(
        ratio=2,
        mode="ratio2_state_detail",
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        max_steps=2,
        output_root=str(tmp_path / "dit"),
        data_hash=data_hash,
    )
    return DiTTrainer(model, adapter, config=config, statistics=stats, codec=codec), batch, stats, config


def test_frozen_codec_logging_and_field_diagnostics(tmp_path) -> None:
    trainer, batch, _, _ = _trainer(tmp_path)
    assert not any(parameter.requires_grad for parameter in trainer.codec.parameters())
    assert not trainer.codec.training
    log = trainer.train_step(batch)
    assert trainer.step == 1
    assert log["loss"] >= 0
    assert set(log["valid_elements"]) == {"state_h", "detail_h", "state_v", "detail_v"}
    assert log["tau_min"] >= 0 and log["tau_max"] <= 1
    assert 0 <= log["observation_fraction"] <= 1


def test_checkpoint_roundtrip_and_contract_refusal(tmp_path) -> None:
    trainer, batch, stats, config = _trainer(tmp_path)
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "checkpoint.pt")
    adapter2 = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    model2 = MolecularDiT(adapter=adapter2, scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2)
    codec2 = nn.Linear(2, 2)
    codec2.load_state_dict(trainer.codec.state_dict())
    for parameter in codec2.parameters():
        parameter.requires_grad_(False)
    trainer2 = DiTTrainer(model2, adapter2, config=config, statistics=stats, codec=codec2)
    payload = trainer2.load_checkpoint(path)
    assert trainer2.step == 1
    assert payload["schema"].endswith("checkpoint.v2")
    reconstructed = LatentStatistics.from_state_dict(payload["statistics_state"])
    assert reconstructed.hash == stats.hash
    assert DiTTrainer.load_statistics_from_checkpoint(path).hash == stats.hash
    adapter_fresh = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    model_fresh = MolecularDiT(
        adapter=adapter_fresh, scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2
    )
    codec_fresh = nn.Linear(2, 2)
    codec_fresh.load_state_dict(trainer.codec.state_dict())
    for parameter in codec_fresh.parameters():
        parameter.requires_grad_(False)
    fresh = DiTTrainer(
        model_fresh,
        adapter_fresh,
        config=replace(config),
        statistics=None,
        codec=codec_fresh,
    )
    fresh.load_checkpoint(path)
    assert fresh.statistics is not None
    assert fresh.statistics.hash == stats.hash
    for name in (
        "state_h_mean",
        "state_h_std",
        "detail_h_mean",
        "detail_h_std",
        "state_v_rms",
        "detail_v_rms",
    ):
        assert torch.equal(getattr(fresh.statistics, name), getattr(stats, name))
    assert trainer2.train_step(batch)["step"] == 2
    trainer3, _, _, _ = _trainer(tmp_path, data_hash="different")
    with pytest.raises(ValueError, match="data_hash"):
        trainer3.load_checkpoint(path)


def test_statistics_schema_is_versioned(tmp_path) -> None:
    _, _, stats, _ = _trainer(tmp_path)
    state = stats.state_dict()
    state["schema_version"] = "pvb.dit.state_detail.stats.v1"
    with pytest.raises(ValueError, match="statistics schema"):
        LatentStatistics.from_state_dict(state)


def test_amp_requires_cuda_instead_of_silently_falling_back_to_cpu(tmp_path) -> None:
    trainer, _, stats, config = _trainer(tmp_path)
    with pytest.raises(RuntimeError, match="CUDA model"):
        DiTTrainer(
            trainer.model,
            trainer.adapter,
            config=replace(config, amp=True),
            statistics=stats,
            codec=trainer.codec,
        )
