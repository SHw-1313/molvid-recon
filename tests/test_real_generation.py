"""Full approved-weight generation parity on one frozen validation clip."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from data.clip_dataset import collate_clip_records as old_collate
from evaluation.dit_reassessment import generate_fixed_noise_latent, make_fixed_noise
from module.latent_flow_source import build_observed_center as old_observed_center
from module.molecular_dit import MolecularDiT as OldDiT
from module.state_detail_codec_v2 import haar_lift as old_haar_lift
from module.state_detail_latent_adapter import (
    LatentStatistics as OldStatistics,
    StateDetailLatentAdapter as OldAdapter,
)
from scripts.run_dit_architecture_round4 import _block_positions_cuda
from scripts.run_state_detail_dit_pilot import _observed_batch
from trainer.codec_trainer import PVBCodecModel
from scripts.run_state_detail_codec_v2_t0 import _state_hash as old_codec_state_hash

from molvid.checkpoints import load_codec_artifact, load_dit_artifact
from molvid.codec.haar import haar_lift as new_haar_lift
from molvid.data.batch import collate_clip_records
from molvid.data.manifest import load_datasets
from molvid.geometry.coordinates import center_coordinates
from molvid.geometry.types import StaticTopologyMetadata
from molvid.generation import _block_positions, _observed_scaffold, sample_clip
from test_migration import APPROVED, APPROVED_SHA, HISTORICAL_DIT, HISTORICAL_DIT_SHA


MANIFEST = Path(
    "/workspace/PVB/outputs/state_detail_codec_v2/t1/"
    "manifest_20260904_token80000"
)
CUDA_LINEAR_ATOL = 1e-5
CUDA_LATENT_ATOL = 1e-4
COORDINATE_ATOL_ANGSTROM = 1e-5


def test_full_historical_dit_and_codec_match_new_generation_on_real_valid_clip():
    assert torch.cuda.is_available(), "full-weight generation parity requires CUDA"
    assert MANIFEST.is_dir(), "frozen real validation manifest is required"
    previous = torch.are_deterministic_algorithms_enabled()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    splits = load_datasets(MANIFEST)
    try:
        dit = load_dit_artifact(
            HISTORICAL_DIT, expected_sha256=HISTORICAL_DIT_SHA, device="cuda"
        )
        payload = dit["payload"]
        assert splits.data_hash == payload["data_hash"]
        codec = load_codec_artifact(APPROVED, expected_sha256=APPROVED_SHA, device="cuda")
        assert codec.report.source_sha256 == APPROVED_SHA
        assert dit["statistics"].provenance["codec_checkpoint_sha256"] == APPROVED_SHA
        assert dit["statistics"].provenance["codec_state_hash"] == payload["codec_hash"]
        index = min(
            range(len(splits.valid._clip_index_fields)),
            key=lambda value: splits.valid._clip_index_fields[value][3],
        )
        record = splits.valid[index]
        assert record["x"].shape[0] == 16
        original = old_collate([record])
        current = collate_clip_records([record])
        torch.testing.assert_close(original.x, current.x, rtol=0, atol=0)

        old_codec_payload = torch.load(APPROVED, map_location="cpu", weights_only=False)
        old_codec = PVBCodecModel.from_model_contract(old_codec_payload["model_contract"]).cuda().eval()
        old_codec.load_state_dict(old_codec_payload["model_state"], strict=True)
        assert payload["codec_hash"] == old_codec_state_hash(old_codec)
        for parameter in old_codec.parameters():
            parameter.requires_grad_(False)
        old_adapter = OldAdapter(
            codec_width=payload["config"]["codec_width"],
            scalar_width=payload["config"]["scalar_width"],
            vector_width=payload["config"]["vector_width"],
            ratio=payload["config"]["ratio"],
        ).cuda()
        model_options = payload["config"]
        old_model = OldDiT(
            adapter=old_adapter,
            scalar_width=model_options["scalar_width"],
            vector_width=model_options["vector_width"],
            depth=model_options["depth"],
            heads=model_options["heads"],
            ffn_multiplier=model_options["ffn_multiplier"],
            dropout=model_options["dropout"],
            execution_backend=model_options["metadata"]["execution_backend"],
            ffn_norm_source=model_options["metadata"]["ffn_norm_source"],
        ).cuda().eval()
        old_model.load_state_dict(payload["model_state"], strict=True)
        old_stats = OldStatistics.from_state_dict(payload["statistics_state"]).to(device="cuda")
        new_model = dit["model"].eval()
        new_codec = codec.model.eval()
        new_stats = dit["statistics"]
        seed = 20260918
        old_codec.prepare_batch(original)
        old_coordinate = original.to("cuda")
        prefix = old_coordinate.x[:8]
        scaffold = torch.cat((prefix, prefix[-1:].expand(8, -1, -1)), dim=0)
        old_coordinate = replace(
            old_coordinate, x=scaffold,
            bpos=_block_positions_cuda(scaffold, old_coordinate.block_id),
        )
        new_scaffold = _observed_scaffold(current, current.x[:8], 8).to("cuda")
        torch.testing.assert_close(new_scaffold.x, old_coordinate.x, rtol=0, atol=0)
        new_scaffold = replace(
            new_scaffold, bpos=_block_positions(new_scaffold.x, new_scaffold.block_id)
        )
        torch.testing.assert_close(new_scaffold.bpos, old_coordinate.bpos, rtol=0, atol=0)
        new_codec.prepare_batch(current)
        old_centered, old_origin = old_codec._centered_batch(old_coordinate)
        new_centered, new_origin = center_coordinates(
            new_scaffold.x, frame_mask=new_scaffold.frame_mask,
            abid=new_scaffold.abid, atom_mask=new_scaffold.loss_mask,
        )
        torch.testing.assert_close(old_origin, new_origin, rtol=0, atol=0)
        torch.testing.assert_close(old_centered.x, new_centered, rtol=0, atol=0)
        with torch.no_grad():
            old_frame = old_codec.frame_encoder(old_centered)
            new_frame = new_codec.frame_encoder(replace(new_scaffold, x=new_centered))
            torch.testing.assert_close(old_frame.graph.edge_index, new_frame.graph.edge_index, rtol=0, atol=0)
            torch.testing.assert_close(old_frame.graph.edge_weight, new_frame.graph.edge_weight, rtol=0, atol=0)
            torch.testing.assert_close(old_frame.h, new_frame.h, rtol=0, atol=0)
            torch.testing.assert_close(old_frame.v, new_frame.v, rtol=0, atol=0)
            old_lift = old_haar_lift(
                old_frame.h, 4, frame_mask=old_coordinate.frame_mask,
                abid=old_coordinate.abid,
            )
            new_lift = new_haar_lift(
                new_frame.h, 4, frame_mask=new_scaffold.frame_mask,
                abid=new_scaffold.abid,
            )
            torch.testing.assert_close(old_lift.state, new_lift.state, rtol=0, atol=0)
            torch.testing.assert_close(
                old_codec.state_detail_codec.state_encode_h.weight,
                new_codec.temporal_codec.state_encode_h.weight, rtol=0, atol=0,
            )
            torch.testing.assert_close(
                old_codec.state_detail_codec.state_encode_h.bias,
                new_codec.temporal_codec.state_encode_h.bias, rtol=0, atol=0,
            )
            assert old_lift.state.stride() == new_lift.state.stride()
            torch.testing.assert_close(
                old_codec.state_detail_codec.state_encode_h(old_lift.state),
                old_codec.state_detail_codec.state_encode_h(old_lift.state), rtol=0, atol=0,
            )
            torch.testing.assert_close(
                old_codec.state_detail_codec.state_encode_h(old_lift.state),
                new_codec.temporal_codec.state_encode_h(old_lift.state),
                rtol=0, atol=CUDA_LINEAR_ATOL,
            )
            torch.testing.assert_close(
                old_codec.state_detail_codec.state_encode_h(old_lift.state),
                new_codec.temporal_codec.state_encode_h(new_lift.state),
                rtol=0, atol=CUDA_LINEAR_ATOL,
            )
            old_v = old_frame.v + old_codec.coordinate_vector_stem(old_centered.x).to(old_frame.v.dtype)
            new_v = new_frame.v + new_codec.coordinate_stem(new_centered).to(new_frame.v.dtype)
            torch.testing.assert_close(old_v, new_v, rtol=0, atol=0)
            old_temporal = old_codec.state_detail_codec.encode(
                old_frame.h, old_v, time_ps=old_coordinate.time_ps,
                frame_mask=old_coordinate.frame_mask, abid=old_coordinate.abid,
                sample_origin=old_origin, topology=old_codec._topology_metadata(old_coordinate),
            )
            new_temporal = new_codec.temporal_codec.encode(
                new_frame.h, new_v, time_ps=new_scaffold.time_ps,
                frame_mask=new_scaffold.frame_mask, abid=new_scaffold.abid,
                sample_origin=new_origin, topology=StaticTopologyMetadata.from_batch(new_scaffold),
            )
            for name in ("state_h", "state_v", "detail_h", "detail_v"):
                torch.testing.assert_close(
                    getattr(old_temporal, name), getattr(new_temporal, name),
                    rtol=0, atol=CUDA_LATENT_ATOL,
                )
            latent = old_codec.encode(old_coordinate)
            new_latent = new_codec.encode(new_scaffold)
            for name in ("state_h", "state_v", "detail_h", "detail_v"):
                torch.testing.assert_close(
                    getattr(latent, name), getattr(new_latent, name),
                    rtol=0, atol=CUDA_LATENT_ATOL,
                )
            packed = old_adapter.pack(
                latent, codec_hash=payload["codec_hash"],
                data_hash=splits.data_hash, origin_from_latent=True,
                loss_mask=old_coordinate.loss_mask,
            )
            observed = _observed_batch(
                packed, old_coordinate, adapter=old_adapter, history_frames=8
            )
            center, _ = old_observed_center(
                "repeat_last_coordinate_encode",
                codec_model=old_codec,
                coordinate_batch=old_coordinate,
                target_batch=observed,
                adapter=old_adapter,
                statistics=old_stats,
                history_frames=8,
                codec_hash=payload["codec_hash"],
                data_hash=splits.data_hash,
            )
            noise, _ = make_fixed_noise(old_stats.normalize(observed), seed=seed)
            generated, old_metadata = generate_fixed_noise_latent(
                old_model, old_adapter, observed, old_stats, noise=noise,
                steps=8, source_center=center, source_mode="conditional",
            )
            decoded = old_codec.decode(generated).x_hat.float()
            old_prediction = torch.cat((prefix.float(), decoded[8:]), dim=0)
            prediction, metadata = sample_clip(
                new_codec, new_model, new_model.adapter, new_stats,
                template=current,
                prefix_coordinates=current.x[:8],
                history_frames=8, steps=8, seed=seed,
                codec_hash=payload["codec_hash"], data_hash=splits.data_hash,
                source_mode="conditional",
                center_kind="repeat_last_coordinate_encode",
            )
        assert old_metadata["observed_clamp_exact"] and metadata["observed_clamp_exact"]
        torch.testing.assert_close(
            prediction, old_prediction, rtol=0, atol=COORDINATE_ATOL_ANGSTROM,
        )
        assert torch.equal(prediction[:8].cpu(), current.x[:8].float())
    finally:
        splits.close()
        torch.use_deterministic_algorithms(previous)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
