"""New DiT reference and factorized-backend numerical migration tests."""

from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from dit_test_utils import make_batch
from module.molecular_dit import MolecularDiT as OldDiT
from module.state_detail_latent_adapter import StateDetailLatentAdapter as OldAdapter
from molvid.dit.model import MolecularDiT
from molvid.geometry.types import StaticTopologyMetadata
from molvid.latent.adapter import StateDetailLatentAdapter
from molvid.latent.types import LatentBatch, LatentFields


def _new_batch(old):
    values = {field.name: getattr(old, field.name) for field in fields(old)}
    values["fields"] = LatentFields(**old.fields.as_dict())
    values["topology"] = StaticTopologyMetadata(
        **{field.name: getattr(old.topology, field.name) for field in fields(old.topology)}
    )
    return LatentBatch(**values)


def _old_new(ratio, backend):
    kwargs = dict(scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2, execution_backend=backend)
    torch.manual_seed(902 + ratio)
    old = OldDiT(adapter=OldAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio), **kwargs).cuda()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(902 + ratio)
    new = MolecularDiT(adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio), **kwargs).cuda()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert old.model_hash == new.model_hash
    return old, new


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("backend", ["reference", "factorized_v2"])
def test_same_seed_weight_tau_velocity_gradient_and_update(ratio, backend):
    assert torch.cuda.is_available(), "P3 DiT numerical gate requires CUDA"
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(backend == "factorized_v2")
    try:
        old, new = _old_new(ratio, backend)
        old_batch = make_batch(ratio, width=4, invalid_last_token=True).to("cuda")
        new_batch = _new_batch(old_batch)
        tau = torch.tensor([0.20616373419761658, 0.6913903951644897], device="cuda")
        old_velocity = old(old_batch, tau)
        new_velocity = new(new_batch, tau)
        for name in old_velocity.names():
            torch.testing.assert_close(getattr(old_velocity, name), getattr(new_velocity, name), rtol=0, atol=0)
        old_loss = sum(value.square().mean() for value in old_velocity.as_dict().values())
        new_loss = sum(value.square().mean() for value in new_velocity.as_dict().values())
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        old_loss.backward()
        new_loss.backward()
        for (_, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        old_optim = torch.optim.AdamW(old.parameters(), lr=2e-4)
        new_optim = torch.optim.AdamW(new.parameters(), lr=2e-4)
        old_optim.step()
        new_optim.step()
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("ratio", [2, 4])
def test_new_reference_and_optimized_backend_keep_baseline_tolerance(ratio):
    assert torch.cuda.is_available(), "P3 backend gate requires CUDA"
    batch = _new_batch(make_batch(ratio, width=4, invalid_last_token=True).to("cuda"))
    kwargs = dict(scalar_width=8, vector_width=4, depth=1, heads=2, ffn_multiplier=2)
    torch.manual_seed(20260909 + ratio)
    reference = MolecularDiT(
        adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio),
        execution_backend="reference", **kwargs
    ).cuda()
    optimizer = torch.optim.SGD(reference.parameters(), lr=1e-2)
    tau = torch.tensor([0.25, 0.75], device="cuda")
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(value.square().mean() for value in reference(batch, tau).as_dict().values())
        loss.backward()
        optimizer.step()
    optimized = MolecularDiT(
        adapter=StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=ratio),
        execution_backend="factorized_v2", **kwargs
    ).cuda()
    optimized.load_state_dict(reference.state_dict(), strict=True)
    assert optimized.semantic_contract_hash == reference.semantic_contract_hash
    reference.eval()
    optimized.eval()
    expected = reference(batch, tau)
    actual = optimized(batch, tau)
    max_abs = max(float((getattr(expected, name) - getattr(actual, name)).abs().max()) for name in expected.names())
    assert max_abs <= 2.0e-5
    reference.zero_grad(set_to_none=True)
    optimized.zero_grad(set_to_none=True)
    expected_loss = sum(value.square().mean() for value in expected.as_dict().values())
    actual_loss = sum(value.square().mean() for value in actual.as_dict().values())
    expected_loss.backward()
    actual_loss.backward()
    grad_abs = max(float((a.grad - b.grad).abs().max()) for a, b in zip(reference.parameters(), optimized.parameters()) if a.grad is not None and b.grad is not None)
    assert grad_abs <= 2.0e-4
