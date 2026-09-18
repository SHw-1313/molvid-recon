"""Checkpoint migration keeps weights, AdamW moments, and frozen behavior."""

from __future__ import annotations


import numpy as np
import pytest
import torch

from data.clip_dataset import collate_clip_records as old_collate
from trainer.codec_trainer import PVBCodecModel
from molvid.checkpoints import _load_strict_model, _mapped_key, load_codec_artifact
from molvid.codec.model import TrajectoryCodec
from molvid.data.batch import collate_clip_records


APPROVED = (
    "/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "full_20260904_seed20260903/ratio4_state_detail/codec_best.pt"
)
APPROVED_SHA = "ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df"


def _record():
    frames, atoms = 4, 4
    t = np.arange(frames, dtype=np.float32)[:, None, None]
    x = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    ) + 0.03 * np.sin(0.4 * t + np.arange(atoms, dtype=np.float32)[None, :, None])
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": "approved-checkpoint-parity",
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": np.arange(frames, dtype=np.float32) * 100,
        "delta_time_ps": np.full(frames - 1, 100, dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.array([5, 5, 5, 5], dtype=np.int64),
        "btype": np.array([6, 6, 6, 6], dtype=np.int64),
        "block_id": np.arange(atoms, dtype=np.int64),
        "component_id": np.zeros(atoms, dtype=np.int64),
        "atom_source_index": np.arange(atoms, dtype=np.int64),
        "atom_identity": [f"a{i}" for i in range(atoms)],
        "edge_mask": np.zeros(atoms, dtype=np.int64),
        "loss_mask": np.ones(atoms, dtype=np.bool_),
        "align_mask": np.ones(atoms, dtype=np.bool_),
        "bond_index": np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64),
    }


def test_strict_key_map_rejects_collision_missing_and_bad_shape():
    old = PVBCodecModel(hidden_channels=8, spatial_layers=1, temporal_layers=1, temporal_ratio=4, temporal_codec_mode="ratio4_state_detail", num_rbf=8, num_heads=2)
    new = TrajectoryCodec(hidden_channels=8, spatial_layers=1, temporal_codec_mode="ratio4_state_detail", num_rbf=8, num_heads=2)
    source = old.state_dict()
    assert [_mapped_key(name) for name in source] == list(new.state_dict())
    _load_strict_model(new, source)
    collision = dict(source)
    collision["coordinate_head.gate.weight"] = source["state_detail_codec.coordinate_head.gate.weight"]
    with pytest.raises(ValueError, match="collision"):
        _load_strict_model(new, collision)
    missing = dict(source)
    missing.pop("state_detail_codec.coordinate_head.out.weight")
    with pytest.raises(ValueError, match="keys/order mismatch"):
        _load_strict_model(new, missing)
    bad_shape = dict(source)
    bad_shape["state_detail_codec.coordinate_head.out.weight"] = torch.empty(2, 8)
    with pytest.raises(ValueError, match="shape/dtype"):
        _load_strict_model(new, bad_shape)


def test_approved_checkpoint_weights_optimizer_and_cuda_step_parity():
    assert torch.cuda.is_available(), "P2 approved checkpoint gate requires CUDA"
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        artifact = load_codec_artifact(APPROVED, expected_sha256=APPROVED_SHA, device="cuda")
        assert artifact.report.weight_count == 57
        assert artifact.report.optimizer_parameter_count == 57
        assert artifact.report.optimizer_moment_count == 15
        assert artifact.step == 39721
        assert artifact.sampler_state is not None
        payload = torch.load(APPROVED, map_location="cpu", weights_only=False)
        old = PVBCodecModel.from_model_contract(payload["model_contract"]).cuda()
        old.load_state_dict(payload["model_state"], strict=True)
        old_optimizer = torch.optim.AdamW(old.parameters(), lr=payload["config"]["training"]["lr"], weight_decay=payload["config"]["training"]["weight_decay"])
        old_optimizer.load_state_dict(payload["optimizer_state"])
        new = artifact.model
        assert [_mapped_key(name) for name, _ in old.named_parameters()] == [name for name, _ in new.named_parameters()]
        assert old_optimizer.state_dict()["param_groups"] == artifact.optimizer.state_dict()["param_groups"]
        assert len(old_optimizer.state) == len(artifact.optimizer.state) == 15
        assert all(not parameter.requires_grad for parameter in new.frame_encoder.parameters())
        new.train()
        assert not new.frame_encoder.training
        old_cpu = old_collate([_record()])
        new_cpu = collate_clip_records([_record()])
        old.prepare_batch(old_cpu)
        new.prepare_batch(new_cpu)
        old_batch = old_cpu.to("cuda")
        new_batch = new_cpu.to("cuda")
        old_batch.x.requires_grad_(True)
        new_batch.x.requires_grad_(True)
        old_latent = old.encode(old_batch)
        new_latent = new.encode(new_batch)
        for name in ("state_h", "state_v", "detail_h", "detail_v"):
            torch.testing.assert_close(getattr(old_latent, name), getattr(new_latent, name), rtol=0, atol=0)
        old_output = old.decode(old_latent)
        new_output = new.decode(new_latent)
        torch.testing.assert_close(old_output.x_hat, new_output.x_hat, rtol=0, atol=0)
        old_loss = (old_output.x_hat - old_batch.x).square().mean()
        new_loss = (new_output.x_hat - new_batch.x).square().mean()
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        old_loss.backward()
        new_loss.backward()
        torch.testing.assert_close(old_batch.x.grad, new_batch.x.grad, rtol=0, atol=0)
        for (_, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in new.frame_encoder.parameters())
        old_optimizer.step()
        artifact.optimizer.step()
        for old_name, old_value in old.state_dict().items():
            torch.testing.assert_close(old_value, new.state_dict()[_mapped_key(old_name)], rtol=0, atol=0)
        for old_state, new_state in zip(old_optimizer.state.values(), artifact.optimizer.state.values()):
            for name in old_state:
                torch.testing.assert_close(old_state[name], new_state[name], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_wrong_approved_hash_fails_before_deserialization():
    with pytest.raises(ValueError, match="SHA-256"):
        load_codec_artifact(APPROVED, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mode", "digest", "weights", "moments"),
    [
        ("ratio1_state_detail", "d697933d467e4e176ebcf024ed69a59c518460eb486b3f32ad6ea10e84931a68", 52, 10),
        ("ratio2_state_detail", "b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9", 57, 15),
        ("ratio4_matched_pooling", "6c9f84420ae81673904fc8a1e838d1298e8d598765ece53913048fd70bb6ba8a", 56, 14),
    ],
)
def test_other_existing_codec_controls_migrate_without_lost_moments(mode, digest, weights, moments):
    root = APPROVED.rsplit("/", 2)[0]
    artifact = load_codec_artifact(f"{root}/{mode}/codec_best.pt", expected_sha256=digest)
    assert artifact.model.temporal_codec_mode == mode
    assert artifact.report.weight_count == weights
    assert artifact.report.optimizer_parameter_count == weights
    assert artifact.report.optimizer_moment_count == moments


def test_new_checkpoint_resume_preserves_rng_optimizer_scheduler_and_cursor(tmp_path):
    import random
    from molvid.checkpoints import save_training_checkpoint, load_training_checkpoint

    assert torch.cuda.is_available(), "P2 resume gate requires CUDA"
    random.seed(92)
    np.random.seed(92)
    torch.manual_seed(92)
    torch.cuda.manual_seed_all(92)
    model = torch.nn.Linear(3, 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def update(module, optim, sched, noise):
        optim.zero_grad(set_to_none=True)
        loss = (module(noise).square()).mean()
        loss.backward()
        optim.step()
        sched.step()
        return loss.detach()

    update(model, optimizer, scheduler, torch.randn(4, 3, device="cuda"))
    path = tmp_path / "step1.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        cursor={"epoch": 0, "batch_in_epoch": 1},
        sampler_state={"position": 1},
        contracts={"data_hash": "tiny-fixed"},
    )
    expected_random = (
        random.random(),
        float(np.random.rand()),
        torch.randn(2),
        torch.randn(4, 3, device="cuda"),
    )
    expected_loss = update(model, optimizer, scheduler, expected_random[3])
    expected_model = {name: value.clone() for name, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()

    resumed = torch.nn.Linear(3, 1).cuda()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=2e-4, weight_decay=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=1, gamma=0.9)
    restored = load_training_checkpoint(
        path,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        expected_contracts={"data_hash": "tiny-fixed"},
    )
    assert restored["step"] == 1
    assert restored["cursor"] == {"epoch": 0, "batch_in_epoch": 1}
    assert restored["sampler_state"] == {"position": 1}
    assert random.random() == expected_random[0]
    assert float(np.random.rand()) == expected_random[1]
    torch.testing.assert_close(torch.randn(2), expected_random[2], rtol=0, atol=0)
    next_noise = torch.randn(4, 3, device="cuda")
    torch.testing.assert_close(next_noise, expected_random[3], rtol=0, atol=0)
    next_loss = update(resumed, resumed_optimizer, resumed_scheduler, next_noise)
    torch.testing.assert_close(next_loss, expected_loss, rtol=0, atol=0)
    for name, value in expected_model.items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, state in expected_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(actual_optimizer["state"][parameter_id][name], value, rtol=0, atol=0)
