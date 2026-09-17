from __future__ import annotations

import os

import pytest
import torch

from module.trajectory_temporal_refiner import (
    TrajectoryTemporalRefiner,
    temporal_refiner_loss,
)


torch.set_num_threads(1)


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1") != "1":
        pytest.skip("set DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1=1 for required CUDA checks")
    if not torch.cuda.is_available():
        pytest.fail("round3 CUDA checks were requested but CUDA is unavailable")
    return torch.device("cuda:0")


def _inputs(device: torch.device, *, channels: int = 5) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(301)
    frames, atoms, batch = 16, 6, 2
    frame_mask = torch.ones((batch, frames), device=device, dtype=torch.bool)
    frame_mask[1, 12:] = False
    observed = torch.zeros_like(frame_mask)
    observed[:, :4] = True
    return {
        "x_hat": torch.randn((frames, atoms, 3), generator=generator, device=device),
        "h": torch.randn((frames, atoms, channels), generator=generator, device=device),
        "v": torch.randn((frames, atoms, 3, channels), generator=generator, device=device),
        "frame_mask": frame_mask,
        "time_ps": torch.arange(frames, device=device, dtype=torch.float32).mul(100).expand(batch, -1).clone(),
        "atom_mask": torch.tensor([True, True, True, True, True, False], device=device),
        "observed_frames": observed,
        "abid": torch.tensor([0, 0, 0, 1, 1, 1], device=device),
    }


def _refiner(device: torch.device, *, cross_block: bool) -> TrajectoryTemporalRefiner:
    return TrajectoryTemporalRefiner(5, hidden=64, block_frames=4, allow_cross_block=cross_block).to(device)


def _set_nonzero_projection(refiner: TrajectoryTemporalRefiner) -> None:
    with torch.no_grad():
        refiner.vector_projection.weight.fill_(0.2)
        for parameter in refiner.gate.parameters():
            parameter.zero_()


def test_cuda_round3_identity_zero_motion_and_observed_clamp() -> None:
    device = _require_cuda()
    values = _inputs(device)
    refiner = _refiner(device, cross_block=True)
    identity = refiner(**values)
    torch.testing.assert_close(identity.x_refined, values["x_hat"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(identity.delta_x, torch.zeros_like(identity.delta_x), atol=0.0, rtol=0.0)
    _set_nonzero_projection(refiner)
    zero_vector = dict(values)
    zero_vector["v"] = torch.zeros_like(values["v"])
    zero = refiner(**zero_vector)
    torch.testing.assert_close(zero.x_refined, values["x_hat"], atol=0.0, rtol=0.0)
    active = values["frame_mask"].index_select(0, values["abid"]).transpose(0, 1)
    observed = values["observed_frames"].index_select(0, values["abid"]).transpose(0, 1)
    corrected = refiner(**values)
    assert torch.equal(corrected.x_refined[active & observed], values["x_hat"][active & observed])
    assert torch.equal(corrected.x_refined[~active], values["x_hat"][~active])
    excluded_atom = ~values["atom_mask"].unsqueeze(0).expand_as(active)
    assert torch.equal(corrected.x_refined[excluded_atom], values["x_hat"][excluded_atom])


def test_cuda_round3_refiner_is_se3_equivariant_and_sample_isolated() -> None:
    device = _require_cuda()
    values = _inputs(device)
    refiner = _refiner(device, cross_block=True)
    _set_nonzero_projection(refiner)
    baseline = refiner(**values)
    generator = torch.Generator(device=device).manual_seed(307)
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), generator=generator, device=device))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    translation = torch.tensor([1.5, -2.0, 0.25], device=device)
    transformed = dict(values)
    transformed["x_hat"] = values["x_hat"] @ rotation.T + translation
    transformed["v"] = torch.einsum("tnac,ba->tnbc", values["v"], rotation)
    rotated = refiner(**transformed)
    torch.testing.assert_close(
        rotated.x_refined,
        baseline.x_refined @ rotation.T + translation,
        atol=3e-5,
        rtol=3e-5,
    )
    perturbed = dict(values)
    perturbed["x_hat"] = values["x_hat"].clone()
    perturbed["h"] = values["h"].clone()
    perturbed["v"] = values["v"].clone()
    # These entries belong only to sample 1 or to its padded frames.
    perturbed["x_hat"][:, 3:] += 1000.0
    perturbed["h"][:, 3:] += 1000.0
    perturbed["v"][:, 3:] += 1000.0
    perturbed["x_hat"][12:, 3:] += 1000.0
    isolated = refiner(**perturbed)
    torch.testing.assert_close(isolated.x_refined[:, :3], baseline.x_refined[:, :3], atol=0.0, rtol=0.0)


def test_cuda_round3_local_and_cross_block_masks_are_the_only_connectivity_difference() -> None:
    device = _require_cuda()
    values = _inputs(device)
    local = _refiner(device, cross_block=False)
    cross = _refiner(device, cross_block=True)
    cross.load_state_dict(local.state_dict(), strict=True)
    _set_nonzero_projection(local)
    cross.load_state_dict(local.state_dict(), strict=True)
    local_output = local(**values)
    cross_output = cross(**values)
    for left in (3, 7, 11):
        assert not bool(local_output.edge_mask[left, 0, 1]) and not bool(local_output.edge_mask[left + 1, 0, 0])
        assert bool(cross_output.edge_mask[left, 0, 1]) and bool(cross_output.edge_mask[left + 1, 0, 0])
    assert torch.all(local_output.edge_mask[2, :, 1] == cross_output.edge_mask[2, :, 1])
    assert not torch.equal(local_output.delta_x, cross_output.delta_x)
    local_contract = local.contract()
    cross_contract = cross.contract()
    for key in ("schema", "channels", "hidden", "block_frames", "edges", "vector_projection", "coordinate_bias", "output"):
        assert local_contract[key] == cross_contract[key]
    assert local_contract["allow_cross_block"] is False
    assert cross_contract["allow_cross_block"] is True
    assert local_contract["cross_block_edges"] == "disabled" and cross_contract["cross_block_edges"] == "enabled"


def test_cuda_round3_loss_masks_and_parameter_gradient_isolation() -> None:
    device = _require_cuda()
    values = _inputs(device)
    first = _refiner(device, cross_block=True)
    second = _refiner(device, cross_block=True)
    second.load_state_dict(first.state_dict(), strict=True)
    assert first is not second
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(first.parameters(), second.parameters()))
    _set_nonzero_projection(first)
    output = first(**values)
    target = values["x_hat"].detach().clone()
    bonds = torch.tensor([[0, 1, 3], [1, 2, 4]], device=device)
    loss = temporal_refiner_loss(
        output.x_refined,
        target,
        frame_mask=values["frame_mask"],
        observed_frames=values["observed_frames"],
        time_ps=values["time_ps"],
        atom_mask=values["atom_mask"],
        bond_index=bonds,
        abid=values["abid"],
    )
    assert loss.coordinate_applicable_samples == 2
    assert loss.motion_applicable_samples == 2
    assert loss.motion_valid_pairs > 0
    loss.total.backward()
    assert first.vector_projection.weight.grad is not None
    assert bool(torch.any(first.vector_projection.weight.grad != 0))
    assert all(parameter.grad is None for parameter in second.parameters())
    observed_atom = values["observed_frames"].index_select(0, values["abid"]).transpose(0, 1)
    assert torch.equal(output.x_refined[observed_atom], values["x_hat"][observed_atom])
