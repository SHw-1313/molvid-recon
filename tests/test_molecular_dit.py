from __future__ import annotations

import torch.nn.functional as F
import torch

from module.molecular_dit import MolecularDiT, SO3ChannelNorm
from module.latent_rectified_flow import RectifiedFlowObjective
from module.state_detail_latent_adapter import StateDetailLatentAdapter
from dit_test_utils import make_batch


def _model(ratio: int, width: int = 4) -> tuple[MolecularDiT, object]:
    adapter = StateDetailLatentAdapter(
        codec_width=width, scalar_width=8, vector_width=4, ratio=ratio
    )
    model = MolecularDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=2,
        ffn_multiplier=2,
    )
    return model, adapter


def _warm_up(model: MolecularDiT, batch) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    model.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(batch, torch.tensor([0.25, 0.75]))
        loss = sum(value.square().mean() for value in output.as_dict().values())
        loss.backward()
        optimizer.step()


def _rotate_fields(batch, rotation: torch.Tensor):
    return batch.fields.map(
        lambda value: value
        if value.ndim == 3
        else torch.einsum("ab,knbc->knac", rotation, value)
    )


def test_shared_size_policy_and_factorized_contract() -> None:
    model2, _ = _model(2)
    model4, _ = _model(4)
    assert model2.parameter_count == model4.parameter_count
    assert "O((K*N)^2)" not in model2.contract()["attention_complexity"]
    assert model2.contract()["backend"].startswith("dense_block_attention")
    assert model2.contract()["temporal_attention"] == "bidirectional"
    assert model2.contract()["vector_maps"] == "bias_free_channel_only"


def test_scalar_invariance_vector_equivariance_and_time_conditioning() -> None:
    torch.manual_seed(41)
    model, _ = _model(2)
    batch = make_batch(2, width=4)
    _warm_up(model, batch)
    model.eval()
    tau = torch.tensor([0.25, 0.75])
    first = model(batch, tau)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    rotated_fields = _rotate_fields(batch, q)
    rotated = model(batch.with_fields(rotated_fields), tau)
    assert torch.allclose(first.state_h, rotated.state_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(first.detail_h, rotated.detail_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(
        rotated.state_v,
        torch.einsum("ab,knbc->knac", q, first.state_v),
        atol=2e-5,
        rtol=2e-5,
    )
    later_time = batch.with_fields(batch.fields.clone())
    later_time.block_time_ps = later_time.block_time_ps + 5.0
    later = model(later_time, tau)
    assert not torch.allclose(first.state_h, later.state_h)


def test_same_noise_flow_fixture_uses_nonzero_gated_model() -> None:
    torch.manual_seed(52)
    model, _ = _model(2)
    batch = make_batch(2, width=4)
    _warm_up(model, batch)
    model.eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    rotated_batch = batch.with_fields(_rotate_fields(batch, q))
    objective = RectifiedFlowObjective()
    first = objective.sample(batch, generator=torch.Generator().manual_seed(71))
    rotated_interpolated = first.interpolated.map(
        lambda value: value
        if value.ndim == 3
        else torch.einsum("ab,knbc->knac", q, value)
    )
    output = model(batch.with_fields(first.interpolated), first.tau)
    rotated_output = model(
        rotated_batch.with_fields(rotated_interpolated), first.tau
    )
    assert torch.allclose(output.state_h, rotated_output.state_h, atol=3e-4, rtol=3e-4)
    assert torch.allclose(output.detail_h, rotated_output.detail_h, atol=3e-4, rtol=3e-4)
    assert torch.allclose(
        rotated_output.state_v,
        torch.einsum("ab,knbc->knac", q, output.state_v),
        atol=3e-4,
        rtol=3e-4,
    )
    assert torch.allclose(
        rotated_output.detail_v,
        torch.einsum("ab,knbc->knac", q, output.detail_v),
        atol=3e-4,
        rtol=3e-4,
    )


def test_ragged_sample_isolation_and_all_vector_maps_are_bias_free() -> None:
    model, _ = _model(4)
    batch = make_batch(4, width=4, invalid_last_token=True)
    first = model(batch, torch.tensor([0.3, 0.6]))
    changed = batch.fields.clone()
    mask = batch.abid == 1
    changed.state_h[:, mask] += 10.0
    changed.detail_h[:, mask] -= 7.0
    changed.state_v[:, mask] *= 3.0
    changed.detail_v[:, mask] *= -2.0
    second = model(batch.with_fields(changed), torch.tensor([0.3, 0.6]))
    assert torch.allclose(first.state_h[:, ~mask], second.state_h[:, ~mask], atol=2e-5, rtol=2e-5)
    assert torch.equal(first.state_h[-1, -1], torch.zeros_like(first.state_h[-1, -1]))
    assert all(module.bias is None for module in model.modules() if module.__class__.__name__ == "AxisPreservingLinear")
    assert all(
        parameter.ndim != 2 or parameter.shape[0] != 3
        for name, parameter in model.named_parameters()
        if "vector" in name.lower()
    )


def test_factorized_trunk_and_output_receive_gradients_after_adaln_warmup() -> None:
    model, _ = _model(2)
    batch = make_batch(2, width=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        output = model(batch, torch.tensor([0.2, 0.8]))
        loss = sum(value.square().mean() for value in output.as_dict().values())
        loss.backward()
        optimizer.step()
    gated = (
        model.blocks[0].spatial_adaln,
        model.blocks[0].temporal_adaln,
        model.blocks[0].ffn_adaln,
    )
    assert all(torch.any(module.modulation.weight != 0) for module in gated)
    exact_modules = {
        "spatial_attention": model.blocks[0].spatial,
        "temporal_attention": model.blocks[0].temporal,
        "scalar_ffn": (model.blocks[0].ffn.scalar_in, model.blocks[0].ffn.scalar_out),
        "vector_ffn": (
            model.blocks[0].ffn.vector_in,
            model.blocks[0].ffn.vector_out,
            model.blocks[0].ffn.vector_gate,
        ),
        "spatial_adaln": model.blocks[0].spatial_adaln.modulation,
        "temporal_adaln": model.blocks[0].temporal_adaln.modulation,
        "ffn_adaln": model.blocks[0].ffn_adaln.modulation,
        "head_state_h": model.adapter.scalar_out["state_h"],
        "head_detail_h": model.adapter.scalar_out["detail_h"],
        "head_state_v": model.adapter.vector_out["state_v"],
        "head_detail_v": model.adapter.vector_out["detail_v"],
    }
    for label, modules in exact_modules.items():
        if not isinstance(modules, tuple):
            modules = (modules,)
        gradients = [
            parameter.grad
            for module in modules
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients, label
        assert all(torch.isfinite(value).all() for value in gradients), label
        assert any(torch.any(value != 0) for value in gradients), label


def test_vector_norm_preserves_dtype_and_bad_component_ops_fail_equivariance() -> None:
    torch.manual_seed(73)
    vector = torch.randn(2, 3, 4)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    rotated = torch.einsum("ab,nbc->nac", q, vector)
    assert torch.allclose(
        SO3ChannelNorm(4)(rotated),
        torch.einsum("ab,nbc->nac", q, SO3ChannelNorm(4)(vector)),
        atol=2e-5,
        rtol=2e-5,
    )
    assert SO3ChannelNorm(4)(vector.to(torch.bfloat16)).dtype == torch.bfloat16
    assert not torch.allclose(
        F.silu(rotated),
        torch.einsum("ab,nbc->nac", q, F.silu(vector)),
    )
    bad_norm = torch.nn.LayerNorm(4, elementwise_affine=False)
    assert not torch.allclose(
        bad_norm(rotated),
        torch.einsum("ab,nbc->nac", q, bad_norm(vector)),
    )
