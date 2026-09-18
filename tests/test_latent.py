"""Old/new latent adapter and train-only statistics parity."""

from __future__ import annotations

from dataclasses import fields

import torch

from dit_test_utils import make_latent
from module.state_detail_latent_adapter import (
    LatentStatistics as OldStatistics,
    StateDetailLatentAdapter as OldAdapter,
)
from molvid.codec.types import StateDetailLatent
from molvid.geometry.types import StaticTopologyMetadata
from molvid.latent.adapter import StateDetailLatentAdapter
from molvid.latent.statistics import LatentStatistics


def _new_latent(old):
    topology = StaticTopologyMetadata(
        **{field.name: getattr(old.topology, field.name) for field in fields(old.topology)}
    )
    values = {field.name: getattr(old, field.name) for field in fields(old)}
    values["topology"] = topology
    return StateDetailLatent(**values)


def test_latent_adapter_init_pack_projection_and_gradient_parity():
    old_latent = make_latent(4, width=4)
    new_latent = _new_latent(old_latent)
    torch.manual_seed(929)
    old = OldAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=4)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(929)
    new = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=4)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    a = old.pack(old_latent, codec_hash="frozen", data_hash="train")
    b = new.from_codec_latent(new_latent, codec_hash="frozen", data_hash="train")
    assert a.contract() == b.contract()
    for name in a.fields.names():
        torch.testing.assert_close(getattr(a, name), getattr(b, name), rtol=0, atol=0)
    ah, av = old.project_inputs(a)
    bh, bv = new.project_inputs(b)
    torch.testing.assert_close(ah, bh, rtol=0, atol=0)
    torch.testing.assert_close(av, bv, rtol=0, atol=0)
    ax = old.project_outputs(ah, av)
    bx = new.project_outputs(bh, bv)
    for name in ax.names():
        torch.testing.assert_close(getattr(ax, name), getattr(bx, name), rtol=0, atol=0)
    old_loss = sum(value.square().mean() for value in ax.as_dict().values())
    new_loss = sum(value.square().mean() for value in bx.as_dict().values())
    old_loss.backward()
    new_loss.backward()
    for (_, left), (_, right) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    generated = new.make_generated_latent(b, bx)
    assert generated.raw_detail_h is None and generated.raw_detail_v is None
    assert generated.topology.contract()["contains_target_coordinates"] is False


def test_statistics_fit_normalize_artifact_hash_parity():
    old_latent = make_latent(2, width=4)
    new_latent = _new_latent(old_latent)
    old_adapter = OldAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    new_adapter = StateDetailLatentAdapter(codec_width=4, scalar_width=8, vector_width=4, ratio=2)
    a = old_adapter.pack(old_latent)
    b = new_adapter.from_codec_latent(new_latent)
    old_stats = OldStatistics.fit([a, a.with_fields(a.fields.map(lambda x: x + 0.25))], ratio=2, provenance={"split": "train"})
    new_stats = LatentStatistics.fit([b, b.with_fields(b.fields.map(lambda x: x + 0.25))], ratio=2, provenance={"split": "train"})
    assert old_stats.hash == new_stats.hash
    for name in ("state_h_mean", "state_h_std", "detail_h_mean", "detail_h_std", "state_v_rms", "detail_v_rms"):
        torch.testing.assert_close(getattr(old_stats, name), getattr(new_stats, name), rtol=0, atol=0)
    old_norm = old_stats.normalize(a)
    new_norm = new_stats.normalize(b)
    for name in a.fields.names():
        torch.testing.assert_close(getattr(old_norm, name), getattr(new_norm, name), rtol=0, atol=0)
    restored = LatentStatistics.from_state_dict(new_stats.state_dict())
    assert restored.hash == new_stats.hash
    assert restored.contract()["vector_mean_subtraction"] is False
