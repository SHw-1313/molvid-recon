from __future__ import annotations

from dataclasses import replace
import io
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch

from evaluation.dit_architecture_sequential_v1 import (
    assert_independent_trainers,
    generation_seed,
    load_sequential_checkpoint,
)
from module.molecular_dit import MolecularDiT, ScalarVectorFFN
from module.state_detail_latent_adapter import (
    LatentFieldSet,
    LatentStatistics,
    StateDetailLatentAdapter,
)
from trainer.dit_trainer import DiTTrainConfig, DiTTrainer, module_state_hash
from dit_test_utils import make_batch


torch.set_num_threads(1)


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1") != "1":
        pytest.skip("set DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1=1 for required CUDA checks")
    if not torch.cuda.is_available():
        pytest.fail("architecture sequential CUDA checks were requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _value_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _observed(batch: object, history_frames: int) -> object:
    token_history = int(history_frames) // int(batch.ratio)
    observed = torch.arange(batch.tokens, device=batch.state_h.device).view(1, -1)
    observed = observed < token_history
    return batch.with_observation(observed.expand(batch.batch_size, -1) & batch.token_mask)


def _model(
    device: torch.device,
    *,
    backend: str,
    ffn_norm_source: str,
) -> MolecularDiT:
    adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=4
    ).to(device)
    return MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        dropout=0.0,
        execution_backend=backend,
        ffn_norm_source=ffn_norm_source,
    ).to(device)


def _warm(model: MolecularDiT, batch: object, tau: torch.Tensor) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, tau)
        sum(value.square().mean() for value in output.as_dict().values()).backward()
        optimizer.step()
    assert torch.any(model.blocks[0].ffn_adaln.modulation.weight != 0)


def _trainer(
    tmp_path,
    device: torch.device,
    state: dict[str, torch.Tensor],
    *,
    ffn_norm_source: str,
) -> DiTTrainer:
    model = _model(
        device,
        backend="factorized_v2",
        ffn_norm_source=ffn_norm_source,
    )
    model.load_state_dict(state, strict=True)
    batch = make_batch(4, width=4)
    statistics = LatentStatistics.fit(
        [batch], ratio=4, provenance={"split": "architecture-sequential-test"}
    ).to(device)
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
        output_root=str(tmp_path),
        data_hash="architecture-data",
        codec_hash="architecture-codec",
        stats_hash=statistics.hash,
        source_mode="conditional",
        center_kind="repeat_last_coordinate_encode",
        source_sigma=1.0,
        normalization_hash=statistics.hash,
        init_hash=module_state_hash(model),
        observation_mixture=(4, 8),
        metadata={
            "phase": "dit_architecture_sequential_v1",
            "execution_backend": "factorized_v2",
            "ffn_norm_source": ffn_norm_source,
        },
    )
    return DiTTrainer(model, model.adapter, config=config, statistics=statistics)


def test_cuda_round1_ffn_norm_source_is_explicit_scale_sensitive_and_rotation_invariant() -> None:
    device = _require_cuda()
    generator = torch.Generator(device=device).manual_seed(101)
    ffn = ScalarVectorFFN(8, 4, 2).to(device)
    h = torch.randn((2, 5, 8), device=device, generator=generator)
    v_post = torch.randn((2, 5, 3, 4), device=device, generator=generator)
    v_pre = torch.randn((2, 5, 3, 4), device=device, generator=generator)

    legacy_h = h.detach().clone().requires_grad_(True)
    legacy_v = v_post.detach().clone().requires_grad_(True)
    legacy = ffn(legacy_h, legacy_v)
    legacy_inputs = (legacy_h, legacy_v, *tuple(ffn.parameters()))
    legacy_gradients = torch.autograd.grad(
        sum(value.square().mean() for value in legacy), legacy_inputs
    )
    explicit_h = h.detach().clone().requires_grad_(True)
    explicit_v = v_post.detach().clone().requires_grad_(True)
    explicit_post = ffn(explicit_h, explicit_v, scalar_vector=explicit_v)
    explicit_inputs = (explicit_h, explicit_v, *tuple(ffn.parameters()))
    explicit_gradients = torch.autograd.grad(
        sum(value.square().mean() for value in explicit_post), explicit_inputs
    )
    for actual, expected in zip(explicit_post, legacy):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    for actual, expected in zip(explicit_gradients, legacy_gradients):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    captured: list[torch.Tensor] = []
    handle = ffn.scalar_in.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    for scale in (0.5, 1.0, 2.0):
        ffn(h, v_post, scalar_vector=v_pre * scale)
        expected = (v_pre * scale).float().square().sum(dim=-2).add(1.0e-6).sqrt()
        torch.testing.assert_close(captured[-1][..., -4:].float(), expected)

    rotation, _ = torch.linalg.qr(torch.randn((3, 3), device=device, generator=generator))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    rotated = torch.einsum("ab,knbc->knac", rotation, v_pre)
    ffn(h, v_post, scalar_vector=rotated)
    torch.testing.assert_close(captured[-1][..., -4:], captured[1][..., -4:], atol=2e-6, rtol=2e-6)
    handle.remove()

    assert not torch.allclose(captured[0][..., -4:], captured[1][..., -4:])
    assert not torch.allclose(captured[1][..., -4:], captured[2][..., -4:])


@pytest.mark.parametrize("ffn_norm_source", ("post_adaln", "pre_adaln"))
def test_cuda_round1_reference_factorized_output_and_gradient_match(
    ffn_norm_source: str,
) -> None:
    device = _require_cuda()
    torch.manual_seed(20260914)
    batch = _observed(make_batch(4, width=4).to(device), 4)
    tau = torch.tensor([0.25, 0.75], device=device)
    reference = _model(
        device, backend="reference", ffn_norm_source=ffn_norm_source
    )
    optimized = _model(
        device, backend="factorized_v2", ffn_norm_source=ffn_norm_source
    )
    optimized.load_state_dict(reference.state_dict(), strict=True)
    _warm(reference, batch, tau)
    optimized.load_state_dict(reference.state_dict(), strict=True)

    reference.zero_grad(set_to_none=True)
    optimized.zero_grad(set_to_none=True)
    expected = reference(batch, tau)
    actual = optimized(batch, tau)
    for name in expected.names():
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=2e-5, rtol=2e-4
        )
    sum(value.square().mean() for value in expected.as_dict().values()).backward()
    sum(value.square().mean() for value in actual.as_dict().values()).backward()
    for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert expected_name == actual_name
        if expected_parameter.grad is None or actual_parameter.grad is None:
            assert expected_parameter.grad is actual_parameter.grad
            continue
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=2e-4,
            rtol=2e-3,
        )


def test_cuda_trainers_and_sequential_checkpoint_loads_are_independent(tmp_path) -> None:
    device = _require_cuda()
    torch.manual_seed(20260914)
    base = _model(device, backend="factorized_v2", ffn_norm_source="post_adaln")
    state = {
        name: value.detach().cpu().clone()
        for name, value in base.state_dict().items()
    }
    first = _trainer(
        tmp_path / "first", device, state, ffn_norm_source="post_adaln"
    )
    second = _trainer(
        tmp_path / "second", device, state, ffn_norm_source="post_adaln"
    )
    assert_independent_trainers(first, second)
    assert first.optimizer is not second.optimizer
    assert first.scaler is not second.scaler

    batch = _observed(make_batch(4, width=4).to(device), 4)
    zero_center = LatentFieldSet.zeros_like(batch.fields)
    first_generator = torch.Generator(device=device).manual_seed(11)
    second_generator = torch.Generator(device=device).manual_seed(11)
    second_model_before = module_state_hash(second.model)
    second_optimizer_before = _value_hash(second.optimizer.state_dict())
    second_rng_before = second_generator.get_state().clone()
    first.train_step(
        batch, generator=first_generator, source_center=zero_center
    )
    assert module_state_hash(second.model) == second_model_before
    assert _value_hash(second.optimizer.state_dict()) == second_optimizer_before
    assert torch.equal(second_generator.get_state(), second_rng_before)
    assert first.successful_updates == 1
    assert second.successful_updates == 0

    from scripts.run_dit_architecture_sequential_v1 import _checkpoint_payload

    first_payload = _checkpoint_payload(
        first,
        round_name="round1",
        arm="control",
        parent_path="parent.pt",
        parent_sha256="parent-sha256",
        parent_model_hash=module_state_hash(base),
        cursor={"epoch": 0, "batch_index": 1},
        generator=first_generator,
        added_tokens=32,
        added_updates=1,
        schedule_hash="schedule-hash",
        unique_variable={"ffn_norm_source": "post_adaln"},
    )
    first_path = tmp_path / "first.pt"
    torch.save(first_payload, first_path)

    resumed = _trainer(
        tmp_path / "resumed", device, state, ffn_norm_source="post_adaln"
    )
    loaded_payload = resumed.load_checkpoint(first_path, map_location=device)
    resumed_generator = torch.Generator(device=device)
    resumed_generator.set_state(
        loaded_payload["sequential"]["training_generator_state"].detach().cpu()
    )
    continuous_row = first.train_step(
        batch, generator=first_generator, source_center=zero_center
    )
    resumed_row = resumed.train_step(
        batch, generator=resumed_generator, source_center=zero_center
    )
    assert continuous_row["tau_mean"] == resumed_row["tau_mean"]
    assert continuous_row["tau_min"] == resumed_row["tau_min"]
    assert continuous_row["tau_max"] == resumed_row["tau_max"]
    assert module_state_hash(first.model) == module_state_hash(resumed.model)
    assert _value_hash(first.optimizer.state_dict()) == _value_hash(
        resumed.optimizer.state_dict()
    )
    assert first.successful_updates == resumed.successful_updates == 2

    second.train_step(
        batch, generator=second_generator, source_center=zero_center
    )
    generic_path = second.save_checkpoint(tmp_path / "generic_second.pt")
    with pytest.raises(ValueError, match="provenance"):
        load_sequential_checkpoint(
            generic_path,
            device=device,
            expected_data_hash="architecture-data",
            expected_codec_hash="architecture-codec",
            expected_statistics_hash=second.statistics.hash,
        )
    second_payload = _checkpoint_payload(
        second,
        round_name="round1",
        arm="candidate",
        parent_path="parent.pt",
        parent_sha256="parent-sha256",
        parent_model_hash=module_state_hash(base),
        cursor={"epoch": 0, "batch_index": 1},
        generator=second_generator,
        added_tokens=32,
        added_updates=1,
        schedule_hash="schedule-hash",
        unique_variable={"ffn_norm_source": "post_adaln"},
    )
    second_path = tmp_path / "second.pt"
    torch.save(second_payload, second_path)
    first_loaded = load_sequential_checkpoint(
        first_path,
        device=device,
        expected_data_hash="architecture-data",
        expected_codec_hash="architecture-codec",
        expected_statistics_hash=first.statistics.hash,
    )
    first_loaded_hash = module_state_hash(first_loaded.model)
    second_loaded = load_sequential_checkpoint(
        second_path,
        device=device,
        expected_data_hash="architecture-data",
        expected_codec_hash="architecture-codec",
        expected_statistics_hash=second.statistics.hash,
    )
    assert module_state_hash(first_loaded.model) == first_loaded_hash
    assert first_loaded.adapter is first_loaded.model.adapter
    assert second_loaded.adapter is second_loaded.model.adapter
    assert {
        parameter.data_ptr() for parameter in first_loaded.model.parameters()
    }.isdisjoint(
        {parameter.data_ptr() for parameter in second_loaded.model.parameters()}
    )
    assert first_loaded.step == first_loaded.successful_updates == 1
    assert second_loaded.step == second_loaded.successful_updates == 1


def test_generation_seed_excludes_arm_round_checkpoint_and_batch_position() -> None:
    seed = generation_seed(20260914, "system_R1_w000030", 8, 2)
    assert seed == generation_seed(20260914, "system_R1_w000030", 8, 2)
    assert seed != generation_seed(20260914, "system_R1_w000030", 4, 2)
    assert seed != generation_seed(20260914, "system_R1_w000030", 8, 3)


def test_future_token_accounting_excludes_observed_history() -> None:
    from scripts.run_dit_architecture_sequential_v1 import _batch_tokens

    specs = (
        SimpleNamespace(atoms=10, frames=16),
        SimpleNamespace(atoms=7, frames=16),
    )
    assert _batch_tokens(specs, (0, 1), 4) == 17 * 12
    assert _batch_tokens(specs, (0, 1), 8) == 17 * 8


def test_paired_resume_uses_largest_common_atomic_checkpoint_and_truncates_jsonl(
    tmp_path,
) -> None:
    from scripts.run_dit_architecture_sequential_v1 import (
        _round1_resume_paths,
        _truncate_jsonl,
    )

    root = tmp_path / "round1"
    for arm in ("control", "candidate"):
        (root / arm).mkdir(parents=True)

    def payload(arm: str, step: int) -> dict[str, object]:
        return {
            "successful_optimizer_updates": step,
            "sequential": {
                "round": "round1",
                "arm": arm,
                "added_successful_updates": step,
                "added_future_atom_frame_tokens": step * 100,
                "cursor": {"epoch": 0, "batch_index": step},
            },
        }

    for arm in ("control", "candidate"):
        torch.save(payload(arm, 10), root / arm / "checkpoint_step000010.pt")
    torch.save(payload("control", 20), root / "control" / "latest.pt")
    torch.save(payload("candidate", 15), root / "candidate" / "latest.pt")
    selected = _round1_resume_paths(SimpleNamespace(output_dir=tmp_path))
    assert selected == {
        "control": root / "control" / "checkpoint_step000010.pt",
        "candidate": root / "candidate" / "checkpoint_step000010.pt",
    }

    history = tmp_path / "train_history.jsonl"
    history.write_text(
        '{"added_successful_updates": 9}\n'
        '{"added_successful_updates": 10}\n'
        '{"added_successful_updates": 11}\n'
        '{"added_successful_updates":',
        encoding="utf-8",
    )
    assert _truncate_jsonl(
        history, step_key="added_successful_updates", maximum_step=10
    ) == 2
    assert [
        json.loads(line)["added_successful_updates"]
        for line in history.read_text(encoding="utf-8").splitlines()
    ] == [9, 10]


def test_round1_common_extension_is_frozen_and_allowed_once(tmp_path) -> None:
    from scripts.run_dit_architecture_sequential_v1 import _round1_exposure_plan

    budget = {
        "selected_round1_added_updates": 2000,
        "selected_round1_extension_added_updates": 1000,
    }
    initial = _round1_exposure_plan(tmp_path, budget, resume=False)
    assert initial["target_delta"] == 2000
    assert initial["common_extension"] is False

    root = tmp_path / "round1"
    root.mkdir()
    (root / "train_summary.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "added_successful_updates_per_arm": 2000,
                "common_extension_applied": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "decision.json").write_text(
        json.dumps({"status": "NEEDS_COMMON_EXTENSION"}), encoding="utf-8"
    )
    extended = _round1_exposure_plan(tmp_path, budget, resume=True)
    assert extended["target_delta"] == 3000
    assert extended["common_extension"] is True

    summary = json.loads((root / "train_summary.json").read_text(encoding="utf-8"))
    summary["common_extension_applied"] = True
    (root / "train_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already applied"):
        _round1_exposure_plan(tmp_path, budget, resume=True)


def test_sampler_plan_is_built_once_per_epoch() -> None:
    from scripts.run_dit_architecture_sequential_v1 import _next_batch

    class Sampler:
        def __init__(self) -> None:
            self.epoch = -1
            self.set_calls: list[int] = []

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)
            self.set_calls.append(self.epoch)

        @property
        def global_batches(self):
            return ((self.epoch * 10,), (self.epoch * 10 + 1,))

        def state_dict(self):
            return {"epoch": self.epoch}

    sampler = Sampler()
    cursor = {"epoch": 0, "batch_index": 0}
    cache: dict[str, object] = {}
    first, first_hash = _next_batch(sampler, cursor, cache)
    second, second_hash = _next_batch(sampler, cursor, cache)
    third, third_hash = _next_batch(sampler, cursor, cache)
    assert (first, second, third) == ((0,), (1,), (10,))
    assert first_hash == second_hash
    assert third_hash != first_hash
    assert sampler.set_calls == [0, 1]
