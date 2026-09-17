"""Focused source-contract and CUDA checks for source A/B v1."""

from __future__ import annotations

from types import SimpleNamespace
import os

import pytest
import torch

from module.dit_latent_cache import LatentFieldCache
from module.latent_flow_source import (
    block_state_zero_detail_center,
    combine_source_fields,
    repeat_last_coordinate_template,
    source_contract,
)
from module.latent_rectified_flow import LatentFieldSet, RectifiedFlowObjective
from module.state_detail_latent_adapter import DiTLatentBatch, LatentStatistics


def _require_cuda() -> torch.device:
    if os.environ.get("DIT_RUN_SOURCE_AB_V1") != "1":
        pytest.skip("set DIT_RUN_SOURCE_AB_V1=1 for the required CUDA source checks")
    if not torch.cuda.is_available():
        pytest.fail("source A/B CUDA checks require CUDA; a skip is not evidence")
    return torch.device("cuda:0")


def _fields(device: torch.device) -> LatentFieldSet:
    generator = torch.Generator(device=device).manual_seed(7)
    return LatentFieldSet(
        torch.randn((4, 8, 5), device=device, generator=generator),
        torch.randn((4, 8, 5), device=device, generator=generator),
        torch.randn((4, 8, 3, 5), device=device, generator=generator),
        torch.randn((4, 8, 3, 5), device=device, generator=generator),
    )


def _batch(device: torch.device) -> DiTLatentBatch:
    fields = _fields(device)
    return DiTLatentBatch(
        fields=fields,
        token_mask=torch.ones((2, 4), device=device, dtype=torch.bool),
        detail_valid=torch.ones((2, 4), device=device, dtype=torch.bool),
        detail_component_mask=torch.ones((2, 4, 3), device=device, dtype=torch.bool),
        block_frame_mask=torch.ones((2, 4, 4), device=device, dtype=torch.bool),
        block_time_ps=torch.arange(16, device=device, dtype=torch.float32).reshape(1, 4, 4).expand(2, -1, -1),
        frame_time_ps=torch.arange(16, device=device, dtype=torch.float32).reshape(1, 16).expand(2, -1),
        abid=torch.tensor([0] * 4 + [1] * 4, device=device),
        sample_origin=torch.zeros((2, 3), device=device),
        topology=None,
        ratio=4,
        mode="ratio4_state_detail",
        width=5,
        atom_ptr=torch.tensor([0, 4, 8], device=device),
        loss_mask=torch.ones(8, device=device, dtype=torch.bool),
    ).with_observation(torch.tensor([[True, False, False, False], [True, False, False, False]], device=device))


def _statistics() -> LatentStatistics:
    zeros = torch.zeros(5)
    ones = torch.ones(5)
    return LatentStatistics(
        ratio=4,
        mode="ratio4_state_detail",
        width=5,
        state_h_mean=zeros,
        state_h_std=ones,
        detail_h_mean=zeros,
        detail_h_std=ones,
        state_v_rms=ones,
        detail_v_rms=ones,
        provenance={"test": True},
    )


def test_source_contract_and_cache_metadata() -> None:
    stats = _statistics()
    contract = source_contract(
        source_mode="conditional",
        center_kind="repeat_last_coordinate_encode",
        statistics=stats,
    )
    assert contract["future_source"] == "m_plus_eps"
    cache = LatentFieldCache(mode="ram", max_bytes=1024 * 1024)
    fields = _fields(torch.device("cpu"))
    cache.put("key", fields)
    restored = cache.get("key", device=torch.device("cpu"), dtype=torch.float32)
    assert restored is not None
    assert all(torch.equal(getattr(fields, name), getattr(restored, name)) for name in fields.names())
    assert cache.stats().hits == 1


def test_repeat_template_does_not_read_future_coordinates() -> None:
    batch = SimpleNamespace(
        frames=16,
        batch_size=1,
        atom_ptr=torch.tensor([0, 3]),
        x=torch.arange(16 * 3 * 3, dtype=torch.float32).reshape(16, 3, 3),
    )
    first = repeat_last_coordinate_template(batch, 4)
    mutated = SimpleNamespace(**batch.__dict__)
    mutated.x = batch.x.clone()
    mutated.x[4:] += 10000.0
    second = repeat_last_coordinate_template(mutated, 4)
    assert torch.equal(first, second)
    assert torch.equal(first[4:], batch.x[3:4].expand(12, -1, -1))


def test_conditional_source_math_on_cuda() -> None:
    device = _require_cuda()
    batch = _batch(device)
    center = block_state_zero_detail_center(batch, statistics=_statistics(), history_frames=4)
    flow = RectifiedFlowObjective()
    generator_a = torch.Generator(device=device).manual_seed(19)
    generator_b = torch.Generator(device=device).manual_seed(19)
    gaussian = flow.sample(batch, generator=generator_a)
    conditional = flow.sample(
        batch,
        generator=generator_b,
        source_center=center,
        source_mode="conditional",
    )
    for name in center.names():
        torch.testing.assert_close(
            getattr(conditional.source, name) - getattr(gaussian.source, name),
            getattr(center, name),
        )
        torch.testing.assert_close(
            getattr(conditional.target, name) - getattr(gaussian.target, name),
            -getattr(center, name),
        )
    assert all(torch.isfinite(value).all() for value in conditional.interpolated.as_dict().values())
