from __future__ import annotations

import pytest
import torch

from module.temporal_codec import (
    CausalEquivariantTemporalBlock,
    CausalTemporalDecoder,
    CausalTemporalEncoder,
    CausalTemporalUpsample,
    ContinuousTimeBias,
)


def _features(frames: int = 8, atoms: int = 4, scalar: int = 8, vector: int = 6):
    torch.manual_seed(13)
    h = torch.randn(frames, atoms, scalar)
    v = torch.randn(frames, atoms, 3, vector)
    abid = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    time = torch.stack(
        [torch.arange(frames, dtype=torch.float32) * 100.0,
         torch.arange(frames, dtype=torch.float32) * 100.0],
        dim=0,
    )
    return h, v, time, abid


def test_prefix_causality_and_masked_keys():
    h, v, time, abid = _features()
    block = CausalEquivariantTemporalBlock(8, 6, num_heads=2, window_size=None)
    block.eval()
    full_h, full_v = block(h, v, time, abid=abid)
    prefix_h, prefix_v = block(h[:5], v[:5], time[:, :5], abid=abid)
    assert torch.allclose(full_h[:5], prefix_h, atol=1e-6, rtol=1e-6)
    assert torch.allclose(full_v[:5], prefix_v, atol=1e-6, rtol=1e-6)

    mask = torch.ones(2, 8, dtype=torch.bool)
    mask[:, 5:] = False
    altered_h = h.clone()
    altered_v = v.clone()
    altered_h[5:] = torch.randn_like(altered_h[5:]) * 100.0
    altered_v[5:] = torch.randn_like(altered_v[5:]) * 100.0
    masked_h, masked_v = block(altered_h, altered_v, time, mask, abid)
    base_h, base_v = block(h, v, time, mask, abid)
    assert torch.allclose(masked_h[:5], base_h[:5], atol=1e-6, rtol=1e-6)
    assert torch.allclose(masked_v[:5], base_v[:5], atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(masked_h[5:]) == 0
    assert torch.count_nonzero(masked_v[5:]) == 0


def test_all_masked_and_t1_are_handled():
    h, v, time, abid = _features(frames=1)
    block = CausalEquivariantTemporalBlock(8, 6, num_heads=2)
    out_h, out_v = block(h, v, time[:, :1], abid=abid)
    assert out_h.shape == h.shape
    assert out_v.shape == v.shape
    assert torch.isfinite(out_h).all() and torch.isfinite(out_v).all()
    with pytest.raises(ValueError, match="at least one valid"):
        block(h, v, time[:, :1], torch.zeros(2, 1, dtype=torch.bool), abid)


def test_time_bias_distinguishes_physical_intervals():
    bias = ContinuousTimeBias(2, num_rbf=8, time_scale_ps=100.0)
    a = bias(torch.tensor([0.0, 100.0, 1000.0]))
    assert a.shape == (3, 2)
    assert not torch.allclose(a[1], a[2])

    h, v, _, abid = _features(frames=4)
    block = CausalEquivariantTemporalBlock(8, 6, num_heads=2)
    block.eval()
    time_100 = torch.arange(4, dtype=torch.float32).view(1, 4).repeat(2, 1) * 100.0
    time_1ns = torch.arange(4, dtype=torch.float32).view(1, 4).repeat(2, 1) * 1000.0
    out_100, _ = block(h, v, time_100, abid=abid)
    out_1ns, _ = block(h, v, time_1ns, abid=abid)
    assert not torch.allclose(out_100, out_1ns)


def test_rotation_equivariance_and_gradients():
    h, v, time, abid = _features()
    block = CausalEquivariantTemporalBlock(8, 6, num_heads=2)
    block.eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    out_h, out_v = block(h, v, time, abid=abid)
    rotated_v = torch.einsum("ij,tnjc->tnic", q, v)
    rotated_h, rotated_out_v = block(h, rotated_v, time, abid=abid)
    expected_v = torch.einsum("ij,tnjc->tnic", q, out_v)
    assert torch.allclose(out_h, rotated_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(rotated_out_v, expected_v, atol=2e-5, rtol=2e-5)

    h = h.requires_grad_()
    v = v.requires_grad_()
    loss = block(h, v, time, abid=abid)[0].square().mean()
    loss.backward()
    assert h.grad is not None and v.grad is not None
    assert torch.isfinite(h.grad).all() and torch.isfinite(v.grad).all()


def test_encoder_ratios_and_right_edge_times():
    h, v, time, abid = _features(frames=16)
    for ratio, expected in ((1, 16), (2, 8), (4, 4)):
        encoder = CausalTemporalEncoder(8, 6, ratio=ratio, num_layers=1, num_heads=2)
        state = encoder(h, v, time, abid=abid)
        assert state.h.shape == (expected, 4, 8)
        assert state.v.shape == (expected, 4, 3, 6)
        assert state.frame_mask.shape == (2, expected)
        assert state.time_ps.shape == (2, expected)
        assert torch.all(state.time_ps[:, 1:] > state.time_ps[:, :-1])
    encoder = CausalTemporalEncoder(8, 6, ratio=4, num_layers=1, num_heads=2)
    state = encoder(h[:1], v[:1], time[:, :1], abid=abid)
    assert state.h.shape[0] == 1


def test_target_time_query_and_decoder():
    h, v, time, abid = _features(frames=8)
    encoder = CausalTemporalEncoder(8, 6, ratio=4, num_layers=1, num_heads=2)
    latent = encoder(h, v, time, abid=abid)
    upsample = CausalTemporalUpsample(8)
    target = time[:, :6]
    shifted = target + 50.0
    first = upsample(latent.h, latent.v, latent.time_ps, latent.frame_mask, target, abid=abid)
    second = upsample(latent.h, latent.v, latent.time_ps, latent.frame_mask, shifted, abid=abid)
    assert first.h.shape == (6, 4, 8)
    assert first.v.shape == (6, 4, 3, 6)
    assert not torch.allclose(first.h, second.h)

    decoder = CausalTemporalDecoder(8, 6, num_heads=2)
    decoded = decoder(latent, target_time_ps=target)
    assert decoded.h.shape == first.h.shape
    assert decoded.v.shape == first.v.shape
    assert torch.isfinite(decoded.h).all() and torch.isfinite(decoded.v).all()


def test_irregular_clock_rescaling_preserves_se3_structure():
    h, v, _, abid = _features(frames=5)
    irregular = torch.tensor([[0.0, 80.0, 240.0, 500.0, 900.0]]).repeat(2, 1)
    block = CausalEquivariantTemporalBlock(8, 6, num_heads=2)
    block.eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    scaled = irregular * 3.0
    base_h, base_v = block(h, v, scaled, abid=abid)
    rotated_h, rotated_v = block(
        h, torch.einsum("ij,tnjc->tnic", q, v), scaled, abid=abid
    )
    expected_v = torch.einsum("ij,tnjc->tnic", q, base_v)
    assert torch.isfinite(base_h).all() and torch.isfinite(base_v).all()
    assert torch.allclose(base_h, rotated_h, atol=2e-5, rtol=2e-5)
    assert torch.allclose(expected_v, rotated_v, atol=2e-5, rtol=2e-5)
