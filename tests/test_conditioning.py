"""Observed-prefix and online-corruption parity with no future-coordinate condition."""

from __future__ import annotations

from dataclasses import dataclass, fields

import pytest
import torch

from dit_test_utils import make_batch
from module.dit_history_corruption import (
    combine_clean_target_with_condition as old_combine,
    make_history_corruption_views as old_views,
    sample_history_sigmas as old_sigmas,
)
from module.state_detail_latent_adapter import build_observation_condition as old_condition
from molvid.geometry.types import StaticTopologyMetadata
from molvid.latent.conditioning import (
    build_observation_condition,
    combine_clean_target_with_condition,
    make_history_corruption_views,
    sample_history_sigmas,
)
from molvid.latent.types import LatentBatch, LatentFields


def _new_batch(old):
    values = {field.name: getattr(old, field.name) for field in fields(old)}
    values["fields"] = LatentFields(**old.fields.as_dict())
    values["topology"] = StaticTopologyMetadata(
        **{field.name: getattr(old.topology, field.name) for field in fields(old.topology)}
    )
    return LatentBatch(**values)


@dataclass
class _ClipView:
    x: torch.Tensor
    frame_mask: torch.Tensor
    abid: torch.Tensor
    loss_mask: torch.Tensor
    batch_size: int


@pytest.mark.parametrize("history", [0, 4, 8])
@pytest.mark.parametrize("ratio", [2, 4])
def test_observation_condition_matches_reference_and_ignores_future(history, ratio):
    old = make_batch(ratio, width=4)
    new = _new_batch(old)
    coordinates = torch.randn(16, old.num_atoms, 3, generator=torch.Generator().manual_seed(161))
    frame_mask = torch.ones(old.batch_size, 16, dtype=torch.bool)
    kwargs = dict(history_frames=history, coordinates=coordinates, frame_mask=frame_mask)
    a = old_condition(old, **kwargs)
    b = build_observation_condition(new, **kwargs)
    for name in ("frame_observation_mask", "latent_observation_mask", "sample_origin"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    changed_future = coordinates.clone()
    changed_future[max(history, 1):] += 10000
    isolated = build_observation_condition(
        new, history_frames=history, coordinates=changed_future, frame_mask=frame_mask
    )
    for name in ("frame_observation_mask", "latent_observation_mask", "sample_origin"):
        torch.testing.assert_close(getattr(b, name), getattr(isolated, name), rtol=0, atol=0)
    assert b.contract()["origin_source"] == ("observed_frame0_centroid" if history else "fixed_zero")


def test_history_noise_views_and_clean_target_gauge_parity():
    old = make_batch(4, width=4)
    new = _new_batch(old)
    generator_a = torch.Generator().manual_seed(711)
    generator_b = torch.Generator().manual_seed(711)
    a_sigma = old_sigmas(2, generator=generator_a, device="cpu", dtype=torch.float32)
    b_sigma = sample_history_sigmas(2, generator=generator_b, device="cpu", dtype=torch.float32)
    torch.testing.assert_close(a_sigma, b_sigma, rtol=0, atol=0)
    x = torch.randn(16, old.num_atoms, 3, generator=torch.Generator().manual_seed(89))
    epsilon = torch.randn(x.shape, generator=torch.Generator().manual_seed(90))
    clip = _ClipView(
        x=x,
        frame_mask=torch.ones(old.batch_size, 16, dtype=torch.bool),
        abid=old.abid,
        loss_mask=old.loss_mask,
        batch_size=old.batch_size,
    )
    sigmas = torch.tensor([0.02, 0.05])
    a_view = old_views(clip, history_frames=8, sigma_per_sample_angstrom=sigmas, epsilon=epsilon)
    b_view = make_history_corruption_views(clip, history_frames=8, sigma_per_sample_angstrom=sigmas, epsilon=epsilon)
    for name in ("condition_origin", "clean_origin", "target_gauge_translation", "applied_noise", "observed_frames"):
        torch.testing.assert_close(getattr(a_view, name), getattr(b_view, name), rtol=0, atol=0)
    torch.testing.assert_close(b_view.target_view.x, x, rtol=0, atol=0)
    torch.testing.assert_close(b_view.condition_view.x[8:], x[8:], rtol=0, atol=0)
    noisy_old = old.with_fields(old.fields.map(lambda value: value + 0.125))
    noisy_new = new.with_fields(new.fields.map(lambda value: value + 0.125))
    a = old_combine(old, noisy_old, views=a_view, history_frames=8)
    b = combine_clean_target_with_condition(new, noisy_new, views=b_view, history_frames=8)
    for name in old.fields.names():
        torch.testing.assert_close(getattr(a.target, name), getattr(b.target, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(a.observed, name), getattr(b.observed, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(b.target, name), getattr(new, name), rtol=0, atol=0)
    torch.testing.assert_close(b.observation.sample_origin, b_view.condition_origin, rtol=0, atol=0)
    torch.testing.assert_close(b.views.target_view.x[8:], x[8:], rtol=0, atol=0)


def test_history_probability_is_configurable_without_changing_sigma_choices():
    generator = torch.Generator().manual_seed(202)
    selected = sample_history_sigmas(
        4,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
        probabilities=(1.0, 0.0, 0.0),
    )
    assert torch.equal(selected, torch.zeros_like(selected))
    with pytest.raises(ValueError, match="nonnegative"):
        sample_history_sigmas(
            1,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
            probabilities=(1.1, -0.1, 0.0),
        )
