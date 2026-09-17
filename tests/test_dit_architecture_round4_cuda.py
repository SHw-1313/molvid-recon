from __future__ import annotations

from dataclasses import dataclass, replace
import os

import pytest
import torch

from dit_test_utils import make_batch
from module.dit_history_corruption import (
    combine_clean_target_with_condition,
    make_history_corruption_views,
)
from module.latent_flow_source import repeat_last_coordinate_template
from module.latent_rectified_flow import apply_observation_clamp
from scripts.run_dit_architecture_round4 import _block_positions_cuda
from module.state_detail_latent_adapter import LatentFieldSet


torch.set_num_threads(1)


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1") != "1":
        pytest.skip("set DIT_RUN_ARCHITECTURE_SEQUENTIAL_V1=1 for required CUDA checks")
    if not torch.cuda.is_available():
        pytest.fail("round4 CUDA checks were requested but CUDA is unavailable")
    return torch.device("cuda:0")


@dataclass
class _CoordinateBatch:
    x: torch.Tensor
    frame_mask: torch.Tensor
    abid: torch.Tensor
    atom_ptr: torch.Tensor
    loss_mask: torch.Tensor
    batch_size: int
    frames: int


def _coordinate_batch(device: torch.device) -> _CoordinateBatch:
    latent = make_batch(4, width=4).to(device)
    generator = torch.Generator(device=device).manual_seed(401)
    return _CoordinateBatch(
        x=torch.randn((16, latent.num_atoms, 3), device=device, generator=generator),
        frame_mask=torch.ones((latent.batch_size, 16), device=device, dtype=torch.bool),
        abid=latent.abid,
        atom_ptr=latent.atom_ptr,
        loss_mask=latent.loss_mask,
        batch_size=latent.batch_size,
        frames=16,
    )


def _atom_observed(frames: torch.Tensor, abid: torch.Tensor) -> torch.Tensor:
    return frames.index_select(0, abid).transpose(0, 1)


def _changed_fields(batch: object, amount: float) -> object:
    return batch.with_fields(batch.fields.map(lambda value: value + amount))


def test_cuda_round4_sigma_zero_is_exact_clean_control_and_target_is_independent() -> None:
    device = _require_cuda()
    coordinate = _coordinate_batch(device)
    clean = make_batch(4, width=4).to(device)
    views = make_history_corruption_views(
        coordinate,
        history_frames=8,
        sigma_per_sample_angstrom=torch.zeros((coordinate.batch_size,), device=device),
        epsilon=torch.randn_like(coordinate.x),
    )
    result = combine_clean_target_with_condition(
        clean,
        clean,
        views=views,
        history_frames=8,
    )
    baseline = clean.with_observation(result.observation.latent_observation_mask, sample_origin=result.observation.sample_origin)
    torch.testing.assert_close(views.condition_view.x, coordinate.x, atol=0.0, rtol=0.0)
    torch.testing.assert_close(views.target_view.x, coordinate.x, atol=0.0, rtol=0.0)
    assert views.condition_view.x.data_ptr() != coordinate.x.data_ptr()
    assert views.target_view.x.data_ptr() != coordinate.x.data_ptr()
    for name in clean.fields.names():
        torch.testing.assert_close(getattr(result.target.fields, name), getattr(clean.fields, name), atol=0.0, rtol=0.0)
        torch.testing.assert_close(getattr(result.observed.fields, name), getattr(baseline.fields, name), atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.observed.sample_origin, baseline.sample_origin, atol=0.0, rtol=0.0)


def test_cuda_round4_only_observed_condition_changes_and_clean_future_target_stays_clean() -> None:
    device = _require_cuda()
    coordinate = _coordinate_batch(device)
    clean = make_batch(4, width=4).to(device)
    generator = torch.Generator(device=device).manual_seed(409)
    views = make_history_corruption_views(
        coordinate,
        history_frames=4,
        sigma_per_sample_angstrom=torch.tensor((0.02, 0.05), device=device),
        epsilon=torch.randn(coordinate.x.shape, device=device, generator=generator),
    )
    condition_latent = _changed_fields(clean, 7.0)
    result = combine_clean_target_with_condition(
        clean,
        condition_latent,
        views=views,
        history_frames=4,
    )
    atom_observed = _atom_observed(views.observed_frames, coordinate.abid)
    torch.testing.assert_close(views.condition_view.x[~atom_observed], coordinate.x[~atom_observed], atol=0.0, rtol=0.0)
    torch.testing.assert_close(views.target_view.x, coordinate.x, atol=0.0, rtol=0.0)
    assert bool(torch.any((views.condition_view.x - coordinate.x)[atom_observed] != 0.0))
    token_observed = result.observed.observed_mask.index_select(0, result.observed.abid).transpose(0, 1)
    for name in clean.fields.names():
        clean_value = getattr(clean.fields, name)
        condition_value = getattr(condition_latent.fields, name)
        value = getattr(result.observed.fields, name)
        expanded = token_observed.reshape(token_observed.shape + (1,) * (value.ndim - 2))
        torch.testing.assert_close(value.masked_select(expanded), condition_value.masked_select(expanded), atol=0.0, rtol=0.0)
        torch.testing.assert_close(value.masked_select(~expanded), clean_value.masked_select(~expanded), atol=0.0, rtol=0.0)
    # The target fields are the clean RF target, re-expressed only through the
    # condition origin used by both target decode and observation clamp.
    for name in clean.fields.names():
        torch.testing.assert_close(getattr(result.target.fields, name), getattr(clean.fields, name), atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.target.sample_origin, views.condition_origin, atol=0.0, rtol=0.0)
    assert result.views.cache_policy == "disabled_online_corruption"


def test_cuda_round4_condition_source_and_observation_clamp_use_the_same_noisy_history() -> None:
    device = _require_cuda()
    coordinate = _coordinate_batch(device)
    clean = make_batch(4, width=4).to(device)
    views = make_history_corruption_views(
        coordinate,
        history_frames=8,
        sigma_per_sample_angstrom=torch.tensor((0.02, 0.05), device=device),
        epsilon=torch.ones_like(coordinate.x),
    )
    result = combine_clean_target_with_condition(
        clean,
        _changed_fields(clean, 3.0),
        views=views,
        history_frames=8,
    )
    template = repeat_last_coordinate_template(views.condition_view, 8)
    for sample in range(coordinate.batch_size):
        start, stop = int(coordinate.atom_ptr[sample]), int(coordinate.atom_ptr[sample + 1])
        torch.testing.assert_close(
            template[8:, start:stop],
            views.condition_view.x[7:8, start:stop].expand_as(template[8:, start:stop]),
            atol=0.0,
            rtol=0.0,
        )
    clamped = apply_observation_clamp(
        LatentFieldSet.zeros_like(result.observed.fields), result.observed.fields, result.observed
    )
    observed = result.observed.observed_mask.index_select(0, result.observed.abid).transpose(0, 1)
    for name in clamped.names():
        value = getattr(clamped, name)
        expected = getattr(result.observed.fields, name)
        expanded = observed.reshape(observed.shape + (1,) * (value.ndim - 2))
        torch.testing.assert_close(value.masked_select(expanded), expected.masked_select(expanded), atol=0.0, rtol=0.0)
    torch.testing.assert_close(result.observation.sample_origin, views.condition_origin, atol=0.0, rtol=0.0)


def test_cuda_round4_noise_is_se3_equivariant_when_the_same_epsilon_is_rotated() -> None:
    device = _require_cuda()
    coordinate = _coordinate_batch(device)
    generator = torch.Generator(device=device).manual_seed(419)
    epsilon = torch.randn(coordinate.x.shape, device=device, generator=generator)
    sigmas = torch.tensor((0.02, 0.05), device=device)
    baseline = make_history_corruption_views(
        coordinate, history_frames=4, sigma_per_sample_angstrom=sigmas, epsilon=epsilon
    )
    rotation, _ = torch.linalg.qr(torch.randn((3, 3), device=device, generator=generator))
    if torch.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    translation = torch.tensor((1.25, -0.5, 3.0), device=device)
    rotated_coordinate = replace(coordinate, x=coordinate.x @ rotation.T + translation)
    rotated = make_history_corruption_views(
        rotated_coordinate,
        history_frames=4,
        sigma_per_sample_angstrom=sigmas,
        epsilon=epsilon @ rotation.T,
    )
    torch.testing.assert_close(
        rotated.condition_view.x,
        baseline.condition_view.x @ rotation.T + translation,
        atol=3.0e-6,
        rtol=3.0e-6,
    )
    torch.testing.assert_close(rotated.target_view.x, baseline.target_view.x @ rotation.T + translation, atol=3.0e-6, rtol=3.0e-6)
    torch.testing.assert_close(rotated.condition_origin, baseline.condition_origin @ rotation.T + translation, atol=3.0e-6, rtol=3.0e-6)


def test_cuda_round4_rollout_block_positions_repeat_each_block_center() -> None:
    device = _require_cuda()
    generator = torch.Generator(device=device).manual_seed(431)
    coordinates = torch.randn((16, 7, 3), device=device, generator=generator)
    block_id = torch.tensor((11, 11, 19, 19, 19, 23, 23), device=device)
    positions = _block_positions_cuda(coordinates, block_id)
    for identifier in torch.unique(block_id):
        members = block_id == identifier
        expected = coordinates[:, members].mean(dim=1, keepdim=True).expand_as(coordinates[:, members])
        torch.testing.assert_close(positions[:, members], expected, atol=0.0, rtol=0.0)
