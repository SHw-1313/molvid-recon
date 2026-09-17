from __future__ import annotations

import hashlib
import io
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dit_test_utils import make_batch
from module.dit_geometry_supervision import (
    FutureBondAuxiliary,
    future_bond_distance_loss,
)
from module.latent_flow_source import build_observed_center
from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import LatentFieldSet, LatentStatistics, StateDetailLatentAdapter
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash


torch.set_num_threads(1)


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1") != "1":
        pytest.skip("set DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1=1 for required CUDA checks")
    if not torch.cuda.is_available():
        pytest.fail("round2 CUDA checks were requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _value_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _observed(batch: object, history_frames: int) -> object:
    tokens = int(history_frames) // int(batch.ratio)
    observed = torch.arange(batch.tokens, device=batch.state_h.device).view(1, -1) < tokens
    return batch.with_observation(observed.expand(batch.batch_size, -1) & batch.token_mask)


class _FrozenToyCodec(nn.Module):
    """Small differentiable decoder used only to verify the frozen-decode path."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3, width))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def decode(self, latent: object) -> object:
        blocks = self.projection(latent.state_h.float())
        coordinates = blocks.repeat_interleave(int(latent.ratio), dim=0)
        origin = latent.sample_origin.index_select(0, latent.abid).unsqueeze(0)
        return SimpleNamespace(x_hat=coordinates + origin)


def _coordinate_batch(batch: object, device: torch.device) -> object:
    generator = torch.Generator(device=device).manual_seed(77)
    return SimpleNamespace(
        x=torch.randn((16, batch.num_atoms, 3), generator=generator, device=device),
        frame_mask=torch.ones((batch.batch_size, 16), dtype=torch.bool, device=device),
        loss_mask=batch.loss_mask,
        bond_index=torch.tensor([[0, 2, 3], [1, 3, 4]], dtype=torch.long, device=device),
        abid=batch.abid,
    )


def _model(device: torch.device) -> MolecularDiT:
    adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=4).to(device)
    return MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
        execution_backend="factorized_v2",
        ffn_norm_source="post_adaln",
    ).to(device)


def _trainer(device: torch.device, state: dict[str, torch.Tensor], statistics: LatentStatistics) -> DiTTrainer:
    model = _model(device)
    model.load_state_dict(state, strict=True)
    config = DiTTrainConfig(
        ratio=4,
        mode="ratio4_state_detail",
        codec_width=4,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
        learning_rate=2.0e-4,
        weight_decay=0.01,
        grad_clip=1.0,
        max_steps=4,
        seed=20260914,
        amp=True,
        output_root="/tmp/round2-cuda",
        data_hash="round2-data",
        codec_hash="round2-codec",
        stats_hash=statistics.hash,
        source_mode="conditional",
        center_kind="repeat_last_coordinate_encode",
        source_sigma=1.0,
        normalization_hash=statistics.hash,
        init_hash=module_state_hash(model),
        observation_mixture=(4, 8),
        metadata={"phase": "dit_architecture_sequential_v1", "ffn_norm_source": "post_adaln"},
    )
    return DiTTrainer(model, model.adapter, config=config, statistics=statistics)


def _warmed_state(device: torch.device) -> tuple[dict[str, torch.Tensor], LatentStatistics]:
    batch = make_batch(4, width=4).to(device)
    model = _model(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    tau = torch.tensor([0.25, 0.75], device=device)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(_observed(batch, 4), tau)
        sum(value.square().mean() for value in output.as_dict().values()).backward()
        optimizer.step()
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    statistics = LatentStatistics.fit([make_batch(4, width=4)], ratio=4, provenance={"split": "round2-cuda"}).to(device)
    return state, statistics


def test_cuda_round2_future_bond_loss_excludes_observed_padding_and_is_rigid_invariant() -> None:
    device = _require_cuda()
    generator = torch.Generator(device=device).manual_seed(31)
    prediction = torch.randn((16, 5, 3), generator=generator, device=device)
    target = torch.randn((16, 5, 3), generator=generator, device=device)
    frame_mask = torch.ones((2, 16), dtype=torch.bool, device=device)
    frame_mask[1, 12:] = False
    observed = torch.zeros_like(frame_mask)
    observed[:, :4] = True
    loss_mask = torch.tensor([True, True, True, True, False], device=device)
    abid = torch.tensor([0, 0, 1, 1, 1], device=device)
    bonds = torch.tensor([[0, 2, 3], [1, 3, 4]], device=device)
    eligible = torch.tensor([True, True], device=device)
    baseline = future_bond_distance_loss(
        prediction, target, frame_mask=frame_mask, observed_frames=observed,
        loss_mask=loss_mask, bond_index=bonds, abid=abid, eligible_samples=eligible,
    )
    perturbed = prediction.clone()
    perturbed[:4] += 1000.0
    perturbed[:, 4] += 1000.0
    perturbed[12:, 2:] += 1000.0
    masked = future_bond_distance_loss(
        perturbed, target, frame_mask=frame_mask, observed_frames=observed,
        loss_mask=loss_mask, bond_index=bonds, abid=abid, eligible_samples=eligible,
    )
    torch.testing.assert_close(masked.loss, baseline.loss, atol=0.0, rtol=0.0)
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), generator=generator, device=device))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    translation = torch.tensor([2.0, -3.0, 5.0], device=device)
    rotated = future_bond_distance_loss(
        prediction @ rotation.T + translation,
        target @ rotation.T + translation,
        frame_mask=frame_mask,
        observed_frames=observed,
        loss_mask=loss_mask,
        bond_index=bonds,
        abid=abid,
        eligible_samples=eligible,
    )
    torch.testing.assert_close(rotated.loss, baseline.loss, atol=2e-5, rtol=2e-5)
    assert baseline.applicable_samples == 2
    assert baseline.valid_pair_frames == 20


def test_cuda_round2_frozen_decode_bond_gradient_reaches_dit() -> None:
    device = _require_cuda()
    state, statistics = _warmed_state(device)
    trainer = _trainer(device, state, statistics)
    batch = _observed(make_batch(4, width=4).to(device), 4)
    codec = _FrozenToyCodec(4).to(device)
    coordinate = _coordinate_batch(batch, device)
    auxiliary = FutureBondAuxiliary(
        adapter=trainer.adapter,
        statistics=statistics,
        codec=codec,
        coordinate_batch=coordinate,
        history_frames=4,
        lambda_bond=0.2,
        tau_threshold=0.0,
    )
    normalized = trainer._normalise_batch(batch)
    generator = torch.Generator(device=device).manual_seed(43)
    with trainer.autocast_context():
        sample = trainer.flow.sample(normalized, generator=generator, source_center=LatentFieldSet.zeros_like(normalized.fields), source_mode="conditional")
        prediction = trainer.model(normalized.with_fields(sample.interpolated), sample.tau)
        raw = auxiliary.raw_loss(prediction, sample, normalized)
    gradients = torch.autograd.grad(raw.loss, tuple(parameter for parameter in trainer.model.parameters() if parameter.requires_grad), allow_unused=True)
    assert raw.applicable_samples == batch.batch_size
    assert any(gradient is not None and bool(torch.any(gradient.detach() != 0)) for gradient in gradients)
    codec_hash = module_state_hash(codec)
    model_before = module_state_hash(trainer.model)
    row = trainer.train_step(
        batch,
        generator=torch.Generator(device=device).manual_seed(47),
        source_center=LatentFieldSet.zeros_like(batch.fields),
        auxiliary_objective=auxiliary,
    )
    assert row["geometry_applicable_samples"] == batch.batch_size
    assert row["geometry_bond_loss"] >= 0.0
    assert module_state_hash(trainer.model) != model_before
    assert module_state_hash(codec) == codec_hash
    assert all(parameter.grad is None for parameter in codec.parameters())


def test_cuda_round2_lambda_zero_is_exact_rf_update() -> None:
    device = _require_cuda()
    state, statistics = _warmed_state(device)
    first = _trainer(device, state, statistics)
    second = _trainer(device, state, statistics)
    batch = _observed(make_batch(4, width=4).to(device), 4)
    codec = _FrozenToyCodec(4).to(device)
    zero_auxiliary = FutureBondAuxiliary(
        adapter=second.adapter,
        statistics=statistics,
        codec=codec,
        coordinate_batch=_coordinate_batch(batch, device),
        history_frames=4,
        lambda_bond=0.0,
    )
    center = LatentFieldSet.zeros_like(batch.fields)
    first_generator = torch.Generator(device=device).manual_seed(59)
    second_generator = torch.Generator(device=device).manual_seed(59)
    first_row = first.train_step(batch, generator=first_generator, source_center=center)
    second_row = second.train_step(batch, generator=second_generator, source_center=center, auxiliary_objective=zero_auxiliary)
    assert first_row["loss"] == second_row["loss"]
    assert module_state_hash(first.model) == module_state_hash(second.model)
    assert _value_hash(first.optimizer.state_dict()) == _value_hash(second.optimizer.state_dict())
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
