from __future__ import annotations

from dataclasses import replace
import os

import pytest
import torch

from module.molecular_dit import MolecularDiT
from module.state_detail_latent_adapter import StateDetailLatentAdapter
from reference.dit_backend_v1_reference import build_reference_model
from dit_test_utils import make_batch


def _cuda_device() -> torch.device:
    if os.environ.get("DIT_RUN_CUDA_BACKEND_V2") != "1":
        pytest.skip("backend v2 CUDA gates are explicit and serialized")
    if not torch.cuda.is_available():
        pytest.fail("backend v2 CUDA gate requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _models(ratio: int, device: torch.device) -> tuple[MolecularDiT, MolecularDiT]:
    reference_adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=ratio
    ).to(device)
    reference = build_reference_model(
        reference_adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
    ).to(device)
    optimized_adapter = StateDetailLatentAdapter(
        codec_width=4, scalar_width=8, vector_width=4, ratio=ratio
    ).to(device)
    optimized = MolecularDiT(
        adapter=optimized_adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
        execution_backend="factorized_v2",
    ).to(device)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    assert reference.semantic_contract_hash == optimized.semantic_contract_hash
    return reference, optimized


def _warm_reference(reference: MolecularDiT, batch: object, tau: torch.Tensor) -> None:
    optimizer = torch.optim.SGD(reference.parameters(), lr=1.0e-2)
    reference.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = reference(batch, tau)
        loss = sum(value.square().mean() for value in output.as_dict().values())
        loss.backward()
        optimizer.step()
    assert all(
        torch.any(module.modulation.weight != 0)
        for module in (
            reference.blocks[0].spatial_adaln,
            reference.blocks[0].temporal_adaln,
            reference.blocks[0].ffn_adaln,
        )
    )


def _field_error(first: object, second: object) -> tuple[float, float]:
    absolute = max(
        float((getattr(first, name) - getattr(second, name)).abs().max())
        for name in first.names()
    )
    numerator = sum(
        float((getattr(first, name) - getattr(second, name)).float().square().sum())
        for name in first.names()
    ) ** 0.5
    denominator = sum(
        float(getattr(first, name).float().square().sum()) for name in first.names()
    ) ** 0.5
    return absolute, numerator / max(denominator, 1.0e-12)


def _observed(batch: object, history_frames: int) -> object:
    token_history = history_frames // int(batch.ratio)
    observed = torch.arange(batch.tokens, device=batch.state_h.device).view(1, -1)
    observed = observed < token_history
    observed = observed.expand(batch.batch_size, -1) & batch.token_mask
    return batch.with_observation(observed)


@pytest.mark.parametrize("ratio", (2, 4))
def test_cuda_forward_gradient_update_and_bf16_parity(ratio: int) -> None:
    device = _cuda_device()
    torch.manual_seed(20260909 + ratio)
    batch = make_batch(ratio, width=4, invalid_last_token=True).to(device)
    reference, optimized = _models(ratio, device)
    tau = torch.tensor([0.25, 0.75], device=device)
    _warm_reference(reference, batch, tau)
    optimized.load_state_dict(reference.state_dict(), strict=True)

    reference.eval()
    optimized.eval()
    for history_frames in (4, 8):
        observed = _observed(batch, history_frames)
        with torch.no_grad():
            expected = reference(observed, tau)
            actual = optimized(observed, tau)
        max_abs, relative_l2 = _field_error(expected, actual)
        assert max_abs <= 2.0e-5, (ratio, history_frames, max_abs, relative_l2)
        assert relative_l2 <= 2.0e-4, (ratio, history_frames, max_abs, relative_l2)

    observed = _observed(batch, 4)
    reference.train()
    optimized.train()
    reference.zero_grad(set_to_none=True)
    optimized.zero_grad(set_to_none=True)
    reference_loss = sum(
        value.square().mean() for value in reference(observed, tau).as_dict().values()
    )
    optimized_loss = sum(
        value.square().mean() for value in optimized(observed, tau).as_dict().values()
    )
    reference_loss.backward()
    optimized_loss.backward()
    gradient_abs = max(
        float((first.grad - second.grad).abs().max())
        for first, second in zip(reference.parameters(), optimized.parameters())
        if first.grad is not None and second.grad is not None
    )
    assert gradient_abs <= 2.0e-4, gradient_abs

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1.0e-3)
    optimized_optimizer = torch.optim.SGD(optimized.parameters(), lr=1.0e-3)
    reference_optimizer.step()
    optimized_optimizer.step()
    update_abs = max(
        float((first - second).abs().max())
        for first, second in zip(reference.parameters(), optimized.parameters())
    )
    assert update_abs <= 2.0e-5, update_abs

    reference.eval()
    optimized.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected_bf16 = reference(observed, tau)
        actual_bf16 = optimized(observed, tau)
    bf16_abs, bf16_relative = _field_error(expected_bf16, actual_bf16)
    assert bf16_abs <= 2.0e-2, (ratio, bf16_abs, bf16_relative)
    assert bf16_relative <= 5.0e-2, (ratio, bf16_abs, bf16_relative)


def test_cuda_ragged_mask_se3_isolation_and_cache_invalidation() -> None:
    device = _cuda_device()
    torch.manual_seed(20260919)
    batch = make_batch(4, width=4, invalid_last_token=True).to(device)
    reference, optimized = _models(4, device)
    tau = torch.tensor([0.3, 0.7], device=device)
    _warm_reference(reference, batch, tau)
    optimized.load_state_dict(reference.state_dict(), strict=True)

    collision_ids = batch.block_id.clone()
    collision_ids[batch.abid == 1] -= 10
    collision_topology = replace(batch.topology, block_id=collision_ids)
    collision_batch = replace(
        batch, block_id=collision_ids, topology=collision_topology
    )
    with torch.no_grad():
        expected = reference(collision_batch, tau)
        actual = optimized(collision_batch, tau)
    max_abs, relative_l2 = _field_error(expected, actual)
    assert max_abs <= 2.0e-5
    assert relative_l2 <= 2.0e-4
    assert optimized._factorized_layout is not None
    first_layout = optimized._factorized_layout

    changed = collision_batch.fields.clone()
    sample_one = collision_batch.abid == 1
    changed.state_h[:, sample_one] += 10.0
    changed.detail_h[:, sample_one] -= 7.0
    changed.state_v[:, sample_one] *= 3.0
    changed.detail_v[:, sample_one] *= -2.0
    with torch.no_grad():
        isolated = optimized(collision_batch.with_fields(changed), tau)
    sample_zero = collision_batch.abid == 0
    assert torch.allclose(
        actual.state_h[:, sample_zero], isolated.state_h[:, sample_zero],
        atol=2.0e-5, rtol=2.0e-4
    )
    assert torch.allclose(
        actual.state_v[:, sample_zero], isolated.state_v[:, sample_zero],
        atol=2.0e-5, rtol=2.0e-4
    )

    rotation, _ = torch.linalg.qr(torch.randn(3, 3, device=device))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    rotated_fields = collision_batch.fields.map(
        lambda value: value
        if value.ndim == 3
        else torch.einsum("ab,knbc->knac", rotation, value)
    )
    with torch.no_grad():
        rotated = optimized(collision_batch.with_fields(rotated_fields), tau)
    assert torch.allclose(actual.state_h, rotated.state_h, atol=3.0e-4, rtol=3.0e-4)
    assert torch.allclose(
        rotated.state_v,
        torch.einsum("ab,knbc->knac", rotation, actual.state_v),
        atol=3.0e-4, rtol=3.0e-4
    )

    all_masked = replace(
        collision_batch,
        token_mask=torch.zeros_like(collision_batch.token_mask),
        detail_valid=torch.zeros_like(collision_batch.detail_valid),
        detail_component_mask=torch.zeros_like(collision_batch.detail_component_mask),
        block_frame_mask=torch.zeros_like(collision_batch.block_frame_mask),
        block_time_ps=torch.zeros_like(collision_batch.block_time_ps),
    ).zero_invalid()
    with torch.no_grad():
        masked = optimized(all_masked, tau)
    assert all(torch.isfinite(value).all() for value in masked.as_dict().values())
    assert all(not torch.any(value) for value in masked.as_dict().values())

    changed_ids = collision_batch.block_id.clone()
    changed_ids[0] += 101
    changed_topology = replace(collision_batch.topology, block_id=changed_ids)
    changed_batch = replace(
        collision_batch, block_id=changed_ids, topology=changed_topology
    )
    with torch.no_grad():
        optimized(changed_batch, tau)
    assert optimized._factorized_layout is not first_layout
    assert optimized._factorized_layout.block_id_ref is changed_batch.block_id


def test_cuda_reuses_layout_for_sixteen_forward_steps() -> None:
    device = _cuda_device()
    torch.manual_seed(20260929)
    batch = make_batch(4, width=4).to(device)
    _, optimized = _models(4, device)
    tau = torch.linspace(0.05, 0.95, 16, device=device)
    with torch.no_grad():
        for value in tau:
            output = optimized(batch, value)
            assert all(torch.isfinite(field).all() for field in output.as_dict().values())
    assert optimized._factorized_layout is not None
    layout = optimized._factorized_layout
    with torch.no_grad():
        optimized(batch, tau[0])
    assert optimized._factorized_layout is layout
