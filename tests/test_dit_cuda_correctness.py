from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import pytest
import torch

from data.clip_dataset import TRAJECTORY_TASK
from evaluation.dit_evaluation import trajectory_metrics
from module.latent_rectified_flow import RectifiedFlowObjective, generate_state_detail_latent
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import (
    LatentStatistics,
    StateDetailLatentAdapter,
    build_observation_condition,
)
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer
from dit_test_utils import make_batch


def _require_explicit_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_CUDA_CORRECTNESS") != "1":
        pytest.skip("CUDA correctness is an explicit, serialized gate")
    if not torch.cuda.is_available():
        pytest.fail("CUDA correctness gate was requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _rotate_fields(batch, rotation: torch.Tensor):
    fields = batch.fields if hasattr(batch, "fields") else batch
    return fields.map(
        lambda value: value
        if value.ndim == 3
        else torch.einsum("ab,knbc->knac", rotation, value)
    )


def _gradient_groups(model: MolecularDiT) -> dict[str, bool]:
    modules = {
        "spatial_attention": (model.blocks[0].spatial,),
        "temporal_attention": (model.blocks[0].temporal,),
        "scalar_ffn": (model.blocks[0].ffn.scalar_in, model.blocks[0].ffn.scalar_out),
        "vector_ffn": (
            model.blocks[0].ffn.vector_in,
            model.blocks[0].ffn.vector_out,
            model.blocks[0].ffn.vector_gate,
        ),
        "spatial_adaln": (model.blocks[0].spatial_adaln.modulation,),
        "temporal_adaln": (model.blocks[0].temporal_adaln.modulation,),
        "ffn_adaln": (model.blocks[0].ffn_adaln.modulation,),
        "output_head_state_h": (model.adapter.scalar_out["state_h"],),
        "output_head_detail_h": (model.adapter.scalar_out["detail_h"],),
        "output_head_state_v": (model.adapter.vector_out["state_v"],),
        "output_head_detail_v": (model.adapter.vector_out["detail_v"],),
    }
    result = {}
    for name, selected in modules.items():
        gradients = [
            parameter.grad
            for module in selected
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        result[name] = bool(
            gradients
            and all(torch.isfinite(value).all() for value in gradients)
            and any(torch.any(value != 0) for value in gradients)
        )
    return result


def _tiny_model(adapter: StateDetailLatentAdapter) -> MolecularDiT:
    return MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
    )


def test_cuda_dit_full_contract_rotation_amp_checkpoint_and_sampling(tmp_path) -> None:
    device = _require_explicit_cuda()
    torch.manual_seed(902)
    batch = make_batch(2, width=4).to(device)
    assert batch.state_h.device == device
    assert batch.detail_h.device == device
    assert batch.state_v.device == device
    assert batch.detail_v.device == device
    assert batch.token_mask.device == device
    assert batch.detail_valid.device == device
    assert batch.block_frame_mask.device == device
    assert batch.abid.device == device
    assert batch.loss_mask.device == device
    assert batch.topology.abid.device == device
    assert batch.topology.atom_ptr.device == device

    coordinates = torch.randn(16, batch.num_atoms, 3, device=device)
    frame_mask = torch.ones(batch.batch_size, 16, dtype=torch.bool, device=device)
    condition4 = build_observation_condition(
        batch,
        history_frames=4,
        coordinates=coordinates,
        frame_mask=frame_mask,
        loss_mask=batch.loss_mask,
    )
    condition0 = build_observation_condition(
        batch, history_frames=0, coordinates=None, frame_mask=frame_mask
    )
    condition8 = build_observation_condition(
        batch,
        history_frames=8,
        coordinates=coordinates,
        frame_mask=frame_mask,
        loss_mask=batch.loss_mask,
    )
    assert condition4.sample_origin.device == device
    assert torch.equal(condition0.sample_origin, torch.zeros_like(condition0.sample_origin))
    assert condition8.latent_observation_mask.sum() > condition4.latent_observation_mask.sum()
    observed_batch = batch.with_observation(
        condition4.latent_observation_mask, sample_origin=condition4.sample_origin
    )
    stats = LatentStatistics.fit(
        [observed_batch], ratio=2, provenance={"split": "T0_cuda_fixture"}
    )

    adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=2
    ).to(device)
    model = _tiny_model(adapter).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(observed_batch, torch.tensor([0.25, 0.75], device=device))
        loss = sum(value.square().mean() for value in prediction.as_dict().values())
        loss.backward()
        optimizer.step()
    assert all(_gradient_groups(model).values())
    assert all(
        torch.any(block.modulation.weight != 0)
        for block in (
            model.blocks[0].spatial_adaln,
            model.blocks[0].temporal_adaln,
            model.blocks[0].ffn_adaln,
        )
    )

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    model.eval()
    tau = torch.tensor([0.25, 0.75], device=device)
    first = model(observed_batch, tau)
    rotated = model(observed_batch.with_fields(_rotate_fields(observed_batch, rotation)), tau)
    assert torch.allclose(first.state_h, rotated.state_h, atol=3e-4, rtol=3e-4)
    assert torch.allclose(first.detail_h, rotated.detail_h, atol=3e-4, rtol=3e-4)
    assert torch.allclose(
        rotated.state_v,
        torch.einsum("ab,knbc->knac", rotation, first.state_v),
        atol=3e-4,
        rtol=3e-4,
    )
    assert torch.allclose(
        rotated.detail_v,
        torch.einsum("ab,knbc->knac", rotation, first.detail_v),
        atol=3e-4,
        rtol=3e-4,
    )

    objective = RectifiedFlowObjective()
    flow_sample = objective.sample(
        observed_batch,
        generator=torch.Generator(device=device).manual_seed(77),
    )
    flow_rotated = model(
        observed_batch.with_fields(_rotate_fields(flow_sample.interpolated, rotation)),
        flow_sample.tau,
    )
    flow_original = model(observed_batch.with_fields(flow_sample.interpolated), flow_sample.tau)
    assert torch.allclose(
        flow_rotated.state_v,
        torch.einsum("ab,knbc->knac", rotation, flow_original.state_v),
        atol=3e-4,
        rtol=3e-4,
    )

    amp_adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=2
    ).to(device)
    amp_model = _tiny_model(amp_adapter).to(device)
    amp_config = DiTTrainConfig(
        ratio=2,
        mode="ratio2_state_detail",
        codec_width=4,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        max_steps=2,
        amp=True,
        output_root=str(tmp_path / "cuda_dit"),
    )
    amp_trainer = DiTTrainer(
        amp_model, amp_adapter, config=amp_config, statistics=stats
    )
    amp_log = amp_trainer.train_step(observed_batch)
    assert amp_trainer.amp_enabled
    assert amp_log["loss"] >= 0.0
    assert set(amp_trainer.last_activation_dtypes) == {
        "state_h",
        "detail_h",
        "state_v",
        "detail_v",
    }
    assert all(value == "torch.bfloat16" for value in amp_trainer.last_activation_dtypes.values())

    checkpoint = tmp_path / "cuda_dit" / "checkpoint.pt"
    amp_trainer.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    statistics_state = payload["statistics_state"]
    assert all(
        name in statistics_state
        for name in (
            "state_h_mean",
            "state_h_std",
            "detail_h_mean",
            "detail_h_std",
            "state_v_rms",
            "detail_v_rms",
        )
    )
    fresh_adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=2
    ).to(device)
    fresh_model = _tiny_model(fresh_adapter).to(device)
    fresh_trainer = DiTTrainer(
        fresh_model,
        fresh_adapter,
        config=replace(amp_config),
        statistics=None,
    )
    fresh_trainer.load_checkpoint(checkpoint, map_location=device)
    assert fresh_trainer.statistics is not None
    assert fresh_trainer.statistics.hash == stats.hash
    for name in (
        "state_h_mean",
        "state_h_std",
        "detail_h_mean",
        "detail_h_std",
        "state_v_rms",
        "detail_v_rms",
    ):
        assert torch.equal(
            getattr(fresh_trainer.statistics, name).detach().cpu(),
            getattr(stats, name).detach().cpu(),
        )

    generated8, meta8 = generate_state_detail_latent(
        amp_model, amp_adapter, observed_batch, stats, steps=8, seed=802
    )
    generated16, meta16 = generate_state_detail_latent(
        amp_model, amp_adapter, observed_batch, stats, steps=16, seed=802
    )
    generated8_again, _ = generate_state_detail_latent(
        amp_model, amp_adapter, observed_batch, stats, steps=8, seed=802
    )
    for generated, metadata in ((generated8, meta8), (generated16, meta16)):
        assert generated.raw_detail_h is None
        assert generated.raw_detail_v is None
        assert all(
            torch.isfinite(getattr(generated, name)).all()
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        )
        assert all(
            getattr(generated, name).device == device
            for name in ("state_h", "detail_h", "state_v", "detail_v")
        )
        assert metadata["observed_clamp_exact"]
    assert all(
        torch.equal(getattr(generated8, name), getattr(generated8_again, name))
        for name in ("state_h", "detail_h", "state_v", "detail_v")
    )

    evaluation_batch = SimpleNamespace(
        x=coordinates,
        frame_mask=frame_mask,
        loss_mask=batch.loss_mask,
        align_mask=torch.ones(batch.num_atoms, dtype=torch.bool, device=device),
        abid=batch.abid,
        bond_index=torch.empty(2, 0, dtype=torch.long, device=device),
        delta_time_ps=torch.ones(batch.batch_size, 15, device=device),
        time_ps=torch.arange(16, device=device, dtype=torch.float32).expand(
            batch.batch_size, -1
        ),
        task=torch.full(
            (batch.batch_size,), TRAJECTORY_TASK, dtype=torch.long, device=device
        ),
    )
    metrics4 = trajectory_metrics(coordinates, coordinates, evaluation_batch, history_frames=4)
    metrics8 = trajectory_metrics(coordinates, coordinates, evaluation_batch, history_frames=8)
    assert metrics4["evaluation_protocol"]["future_frame_interval"] == [4, 16]
    assert metrics8["evaluation_protocol"]["future_frame_interval"] == [8, 16]
    assert metrics4["evaluation_protocol"]["boundary_frame_interval"] == [3, 4]
    assert metrics8["evaluation_protocol"]["boundary_frame_interval"] == [7, 8]
