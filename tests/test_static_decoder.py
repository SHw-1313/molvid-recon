from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from molvid.codec.decoder import TrajectoryDecoder
from molvid.codec.frame import FrozenFrameTeacher
from molvid.codec.history import HistoryEncoder
from molvid.data.batch import collate_clip_records
from molvid.dit.frame import FrameDiT
from molvid.latent.adapter import FrameLatentAdapter
from molvid.latent.statistics import FrameLatentStatistics
from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
from molvid.model import FrameJointModel
from molvid.training.batches import prepare_frame_joint_batch
from molvid.training.static_decoder import (
    StaticDecoderAuxiliaryTrainer,
    StaticDecoderLossConfig,
    StaticPerturbationConfig,
    prepare_static_decoder_batch,
    resolve_static_gamma,
    static_clash_loss_function,
    static_decoder_loss,
)
from molvid.training.static_schedule import StaticRNGStreams


def _static_record(sample_id: str, offset: float = 0.0) -> dict:
    x = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.3, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    x[:, 0] += float(offset)
    bpos = np.asarray([x[0], x[0], x[2], x[2]], dtype=np.float32)
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "source": "test",
        "split": "train",
        "task": "static",
        "coordinate_unit": "angstrom",
        "time_bucket_id": "static",
        "time_ps": np.asarray([0.0], dtype=np.float32),
        "delta_time_ps": np.empty((0,), dtype=np.float32),
        "frame_mask": np.asarray([True]),
        "x": x[None],
        "bpos": bpos[None],
        "atype": np.asarray([5, 5, 6, 6], dtype=np.int64),
        "btype": np.asarray([118, 118, 119, 119], dtype=np.int64),
        "block_id": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "component_id": np.zeros(4, dtype=np.int64),
        "atom_source_index": np.arange(4, dtype=np.int64),
        "atom_identity": [f"{sample_id}:a{index}" for index in range(4)],
        "edge_mask": np.zeros(4, dtype=np.int64),
        "loss_mask": np.ones(4, dtype=np.bool_),
        "align_mask": np.asarray([True, True, True, False]),
        "bond_index": np.asarray(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64
        ),
    }


class _Encoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(()))

    def prepare_batch(self, _batch) -> None:
        return None

    def forward(self, batch):
        h = batch.x.float().square().sum(dim=-1, keepdim=True).sqrt()
        h = h.expand(-1, -1, self.channels) * self.scale
        v = batch.x.float().unsqueeze(-1).expand(-1, -1, -1, self.channels) * self.scale
        return SimpleNamespace(h=h, v=v)


class _Stem(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.channels = channels

    def forward(self, coordinates):
        return coordinates.float().unsqueeze(-1).expand(
            -1, -1, -1, self.channels
        ) * self.scale


def _model(device: torch.device) -> FrameJointModel:
    channels = 4
    teacher = FrozenFrameTeacher(_Encoder(channels), _Stem(channels))
    history = HistoryEncoder(channels=channels, scalar_width=8, vector_width=4, heads=1)
    adapter = FrameLatentAdapter(codec_width=channels, scalar_width=8, vector_width=4)
    dit = FrameDiT(
        adapter=adapter,
        scalar_width=8,
        vector_width=4,
        depth=1,
        heads=1,
    )
    decoder = TrajectoryDecoder(channels=channels, heads=1)
    statistics = FrameLatentStatistics(
        h_mean=torch.zeros(channels),
        h_std=torch.ones(channels),
        v_rms=torch.ones(channels),
        provenance={"split": "train", "fixture": True},
    )
    return FrameJointModel(
        target_teacher=teacher,
        history_encoder=history,
        dit=dit,
        decoder=decoder,
        statistics=statistics,
    ).to(device)


def test_static_configs_keep_units_and_calibration_bounds_explicit():
    config = StaticPerturbationConfig.resolve(
        {"sigma_per_axis_angstrom": [0.05, 0.1, 0.2], "clean_probability": 0.2}
    )
    assert config.contract()["coordinate_unit"] == "angstrom"
    assert config.contract()["three_dimensional_rms_displacement_angstrom"][1] == pytest.approx(
        np.sqrt(3.0) * 0.1
    )
    gamma, report = resolve_static_gamma([2.0, 4.0, 8.0], [10.0, 20.0, 40.0])
    assert gamma == pytest.approx(0.075 * 4.0 / 20.0)
    assert report["achieved_median_ratio"] == pytest.approx(0.075)
    with pytest.raises(ValueError, match=r"\[0.05,0.10\]"):
        resolve_static_gamma([1.0], [1.0], target_ratio=0.2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("amp", [False, True])
def test_static_batch_is_target_isolated_and_updates_only_decoder_on_cuda(amp: bool):
    device = torch.device("cuda")
    model = _model(device)
    cpu_batch = collate_clip_records(
        [_static_record("a"), _static_record("b", offset=10.0)]
    )
    generator = torch.Generator(device=device).manual_seed(20260929)
    prepared = prepare_static_decoder_batch(
        model.target_teacher,
        cpu_batch,
        device=device,
        generator=generator,
        perturbation=StaticPerturbationConfig(
            sigma_per_axis_angstrom=(0.1,), clean_probability=0.0
        ),
    )
    assert prepared.task_type == "static_decoder_aux"
    assert prepared.motion_available is False
    assert prepared.physical_lag_applicable is False
    assert tuple(prepared.observed_context.coordinates.shape) == (1, 8, 3)
    assert not torch.equal(prepared.observed_context.coordinates, prepared.target_coordinates)
    assert prepared.bpos_exact_anchor_count == 8
    assert prepared.bpos_fallback_count == 0
    # bpos follows the perturbation of the actual per-block reference atom.
    assert torch.equal(prepared.input_batch.bpos[0, 0], prepared.input_batch.x[0, 0])
    assert torch.equal(prepared.input_batch.bpos[0, 1], prepared.input_batch.x[0, 0])
    assert torch.equal(prepared.input_batch.bpos[0, 4], prepared.input_batch.x[0, 4])

    model.decoder.eval()
    with torch.no_grad():
        prediction = model.decoder(prepared.observed_context).coordinates
        mutated = replace(
            prepared, target_coordinates=prepared.target_coordinates + 1000.0
        )
        repeated = model.decoder(mutated.observed_context).coordinates
    assert torch.equal(prediction, repeated)

    loss_config = StaticDecoderLossConfig.resolve(
        {"coordinate": 1.0, "bond": 0.1, "clash": 0.0, "gamma": 0.075}
    )
    loss = static_decoder_loss(
        model.decoder(prepared.observed_context), prepared, loss_config
    )
    assert torch.isfinite(loss.total)
    assert int(loss.applicable_samples["coordinate_mse"]) == 2
    assert int(loss.applicable_samples["bond_distance"]) == 2
    with pytest.raises(ValueError, match="reviewed A clash"):
        static_decoder_loss(
            model.decoder(prepared.observed_context),
            prepared,
            StaticDecoderLossConfig(clash=1.0),
        )
    physical = PhysicalAuxiliaryConfig.resolve(
        {
            "clash_weight": 0.01,
            "vdw_radii_A": {"5": 2.0, "6": 2.0},
            "allowed_overlap_A": 0.8,
            "one_four_overlap_A": 1.2,
            "table_provenance": {"split": "train", "source_hash": "fixture"},
            "distance_chunk_size": 2,
        }
    )
    reviewed_clash = static_decoder_loss(
        model.decoder(prepared.observed_context),
        prepared,
        StaticDecoderLossConfig(clash=physical.clash_weight),
        clash_loss_function=static_clash_loss_function(physical),
    )
    assert torch.isfinite(reviewed_clash.raw["clash"])
    assert int(reviewed_clash.applicable_samples["clash"]) == 2
    assert reviewed_clash.clash_diagnostics["source"] == (
        "A1 released PhysicalAuxiliaryConfig"
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1.0e-3, weight_decay=0.1)
    # Prime realistic AdamW state for every dynamic component.  The following
    # static step must leave both inactive weights and these existing moments
    # bitwise unchanged.
    for parameter in trainable:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trainer = StaticDecoderAuxiliaryTrainer(
        model,
        loss_config=loss_config,
        optimizer=optimizer,
        amp=amp,
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer_before = {
        name: {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in optimizer.state[parameter].items()
        }
        for name, parameter in model.named_parameters()
        if not name.startswith("decoder.")
    }
    metrics = trainer.train_step(prepared)
    assert np.isfinite(metrics["loss"])
    assert metrics["decoder_grad_norm"] > 0.0
    changed = []
    for name, parameter in model.named_parameters():
        is_decoder = name.startswith("decoder.")
        if is_decoder and not torch.equal(parameter.detach(), before[name]):
            changed.append(name)
        if not is_decoder:
            assert torch.equal(parameter.detach(), before[name]), name
            assert parameter.grad is None, name
    assert changed
    for name, expected in optimizer_before.items():
        state = optimizer.state[dict(model.named_parameters())[name]]
        assert state.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(state[key], value, rtol=0, atol=0)
            else:
                assert state[key] == value

    # The dynamic preparation contract remains strict; no fake future is made.
    with pytest.raises(ValueError, match="leave at least one future"):
        prepare_frame_joint_batch(
            model.target_teacher,
            cpu_batch,
            device=device,
            normalizer=model,
            history_frames=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_static_rng_streams_do_not_advance_global_cuda_rng() -> None:
    device = torch.device("cuda")
    torch.cuda.manual_seed(9001)
    global_before = torch.cuda.get_rng_state(device).clone()
    streams = StaticRNGStreams(device=device, seed=77)
    assert torch.equal(torch.cuda.get_rng_state(device), global_before)

    with streams.model_random_context():
        first = torch.rand((8,), device=device)
    assert torch.equal(torch.cuda.get_rng_state(device), global_before)
    saved = streams.state_dict()
    with streams.model_random_context():
        second = torch.rand((8,), device=device)
    assert not torch.equal(first, second)
    assert torch.equal(torch.cuda.get_rng_state(device), global_before)

    restored = StaticRNGStreams(device=device, seed=77)
    restored.load_state_dict(saved)
    with restored.model_random_context():
        repeated_second = torch.rand((8,), device=device)
    torch.testing.assert_close(repeated_second, second, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(device), global_before)
