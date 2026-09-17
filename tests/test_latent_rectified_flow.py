from __future__ import annotations

import torch

from module.latent_rectified_flow import (
    RectifiedFlowObjective,
    apply_observation_clamp,
    euler_sample,
    four_field_loss,
    rectified_flow_interpolate,
    rectified_flow_velocity,
)
from module.state_detail_latent_adapter import LatentFieldSet
from dit_test_utils import make_batch


def test_arbitrary_rank_interpolant_and_velocity() -> None:
    data = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    noise = torch.full_like(data, -1.0)
    tau = torch.tensor([0.0, 1.0])
    result = rectified_flow_interpolate(data, noise, tau)
    assert torch.equal(result[0], noise[0])
    assert torch.equal(result[1], data[1])
    assert torch.equal(rectified_flow_velocity(data, noise), data - noise)
    atom_data = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 4)
    atom_noise = torch.zeros_like(atom_data)
    sample_ids = torch.tensor([0, 1, 1])
    atom_result = rectified_flow_interpolate(
        atom_data, atom_noise, torch.tensor([0.25, 0.75]), sample_ids=sample_ids
    )
    assert torch.allclose(atom_result[:, 0], atom_data[:, 0] * 0.25)
    assert torch.allclose(atom_result[:, 2], atom_data[:, 2] * 0.75)


def test_same_noise_rotation_fixture_preserves_vector_flow_equivariance() -> None:
    torch.manual_seed(9)
    data = torch.randn(2, 3, 3, 4)
    noise = torch.randn_like(data)
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] *= -1
    tau = torch.tensor([0.2, 0.8])
    rotated_data = torch.einsum("ab,knbc->knac", q, data)
    rotated_noise = torch.einsum("ab,knbc->knac", q, noise)
    original = rectified_flow_interpolate(data, noise, tau)
    rotated = rectified_flow_interpolate(rotated_data, rotated_noise, tau)
    assert torch.allclose(rotated, torch.einsum("ab,knbc->knac", q, original))
    assert torch.allclose(
        rectified_flow_velocity(rotated_data, rotated_noise),
        torch.einsum("ab,knbc->knac", q, rectified_flow_velocity(data, noise)),
    )


def test_four_field_losses_are_independently_normalized_and_equal() -> None:
    batch = make_batch(2, width=2)
    prediction = LatentFieldSet(
        torch.ones_like(batch.state_h),
        torch.ones_like(batch.detail_h),
        torch.ones_like(batch.state_v),
        torch.ones_like(batch.detail_v),
    )
    target = LatentFieldSet.zeros_like(prediction)
    result = four_field_loss(prediction, target, batch)
    assert all(torch.allclose(value, torch.ones_like(value)) for value in result.fields.values())
    assert torch.allclose(result.total, torch.ones_like(result.total))
    assert result.valid_elements["state_v"] == int(batch.state_h.shape[0] * batch.state_h.shape[1] * 3 * 2)


def test_observed_clamp_and_invalid_zeroing() -> None:
    batch = make_batch(2, width=2, invalid_last_token=True)
    observed = torch.zeros_like(batch.token_mask)
    observed[:, 0] = True
    batch = batch.with_observation(observed)
    clean = batch.fields
    noisy = clean.map(torch.ones_like)
    clamped = apply_observation_clamp(noisy, clean, batch)
    assert torch.equal(clamped.state_h[0], clean.state_h[0])
    assert torch.equal(clamped.state_h[-1, -1], torch.zeros_like(clamped.state_h[-1, -1]))
    assert torch.equal(clamped.detail_v[-1, -1], torch.zeros_like(clamped.detail_v[-1, -1]))


def test_seeded_flow_sample_is_reproducible() -> None:
    batch = make_batch(4, width=2).with_observation(torch.zeros(2, 4, dtype=torch.bool))
    objective = RectifiedFlowObjective()
    first = objective.sample(batch, generator=torch.Generator().manual_seed(8))
    second = objective.sample(batch, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first.tau, second.tau)
    for name in first.noise.names():
        assert torch.equal(getattr(first.noise, name), getattr(second.noise, name))
        assert torch.equal(getattr(first.interpolated, name), getattr(second.interpolated, name))


def test_observed_fields_are_clean_and_excluded_from_flow_loss() -> None:
    batch = make_batch(2, width=2).with_observation(
        torch.tensor([[True] + [False] * 7, [False] * 8])
    )
    sample = RectifiedFlowObjective().sample(
        batch, generator=torch.Generator().manual_seed(31)
    )
    for name in sample.interpolated.names():
        value = getattr(sample.interpolated, name)
        clean = getattr(batch.fields, name)
        observed_atoms = batch.abid == 0
        assert torch.equal(value[0, observed_atoms], clean[0, observed_atoms])
        assert torch.equal(
            getattr(sample.target, name)[0, observed_atoms],
            torch.zeros_like(getattr(sample.target, name)[0, observed_atoms]),
        )
    prediction = LatentFieldSet.zeros_like(sample.target)
    result = RectifiedFlowObjective().loss(prediction, sample.target, batch)
    observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1)
    expected = int((batch.field_masks()["state_h"] & ~observed).sum())
    assert result.valid_elements["state_h"] == expected * batch.width


def test_euler_8_and_16_are_seeded_and_clamp_observations() -> None:
    batch = make_batch(4, width=2).with_observation(
        torch.tensor([[True, False, False, False], [False, False, False, False]])
    )

    class ZeroModel(torch.nn.Module):
        def forward(self, value, tau):
            return LatentFieldSet.zeros_like(value.fields)

    model = ZeroModel()
    first, meta8 = euler_sample(model, batch, steps=8, seed=12)
    second, _ = euler_sample(model, batch, steps=8, seed=12)
    third, meta16 = euler_sample(model, batch, steps=16, seed=12)
    assert meta8["deterministic"] and meta16["steps"] == 16
    for name in first.fields.names():
        assert torch.equal(getattr(first, name), getattr(second, name))
        assert torch.equal(getattr(first, name)[0, batch.abid == 0], getattr(batch.fields, name)[0, batch.abid == 0])
        assert torch.isfinite(getattr(third, name)).all()
