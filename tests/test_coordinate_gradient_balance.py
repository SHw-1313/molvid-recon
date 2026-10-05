import math

import pytest
import torch

from molvid.codec.frame import slice_clip_frames
from molvid.evaluation.gradient_balance import gradient_snapshot, gradient_comparison, require_static_boundaries
from tools.probe_coordinate_static_balance import corrupted_structure


def test_gradient_union_and_static_active_subspace_are_distinct():
    d = {"dit.a": torch.tensor([3.]), "dit.b": torch.tensor([4.]), "readout.a": torch.tensor([2.])}
    s = {"dit.a": torch.tensor([-2.]), "readout.a": torch.tensor([1.])}
    value = gradient_comparison(d, s)
    assert value["dit"]["dynamic_norm"] == 5
    assert value["dit"]["cosine"] == -.6
    assert value["dit"]["static_active_cosine"] == -1
    assert value["readout"]["cosine"] == 1
    assert value["dit"]["static_to_dynamic_norm"] == .4


@pytest.mark.parametrize("name", ["history_encoder.x", "motion_encoder.x", "target_teacher.x",
    "dit.query_time.x", "dit.blocks.0.history_time_bias.weight", "dit.blocks.0.temporal_time_decay_raw",
    "dit.blocks.0.history.x", "dit.blocks.0.temporal_adaln.x"])
def test_static_boundary_rejects_even_zero_temporal_gradients(name):
    with pytest.raises(RuntimeError, match="static gradient"):
        require_static_boundaries({name: torch.zeros(1)})


def test_snapshot_detaches_copies_and_rejects_nonfinite():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    value = gradient_snapshot(model)
    model.weight.grad.zero_()
    assert torch.equal(value["weight"], torch.ones(1, 2))
    model.weight.grad.fill_(math.nan)
    with pytest.raises(FloatingPointError):
        gradient_snapshot(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_module_backward_is_read_only_and_static_context_is_corrupted():
    from test_coordinate_flow import fixture
    model, observed, _ = fixture()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    clean = slice_clip_frames(observed, 0, 1)
    noise = torch.randn_like(clean.x)
    corrupted = corrupted_structure(clean, noise, .25)
    assert torch.equal(corrupted.x, clean.x + .25 * noise)
    for block in torch.unique(clean.block_id):
        mask = clean.block_id == block
        expected = corrupted.x[:, mask].mean(1, keepdim=True).expand(-1, int(mask.sum()), -1)
        torch.testing.assert_close(corrupted.bpos[:, mask], expected)
    condition = model.prepare_condition(corrupted, task="static")
    assert torch.equal(condition.context.coordinates, corrupted.x)
    state = torch.lerp(corrupted.x, clean.x, .25)
    predicted = model.forward_static(state, condition=condition, flow_time=torch.full((2,), .25, device="cuda"))
    (predicted - (clean.x - corrupted.x)).square().mean().backward()
    gradients = gradient_snapshot(model)
    require_static_boundaries(gradients)
    assert any(n.startswith("dit.") and g.norm() > 0 for n, g in gradients.items())
    assert any(n.startswith("readout.") and g.norm() > 0 for n, g in gradients.items())
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
