"""One-clip CUDA stage profile; no optimizer update or training loop."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import torch

from molvid.checkpoints import load_codec_artifact, load_dit_artifact
from molvid.data.batch import collate_clip_records
from molvid.data.manifest import load_datasets
from molvid.generation import _block_positions, _observed_scaffold
from molvid.geometry.coordinates import center_coordinates
from molvid.geometry.frames import build_frame_graph, pack_frame_nodes, unpack_frame_features
from molvid.latent.conditioning import build_observation_condition
from molvid.runtime import configure_device
from molvid.training.batches import prepare_batch_then_to_device


def measure_cuda(fn: Callable[[], Any], device: torch.device) -> tuple[Any, dict[str, float]]:
    """Synchronize around a single cold stage and record incremental peak memory."""

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak = torch.cuda.max_memory_allocated(device)
    return result, {
        "elapsed_ms": elapsed_ms,
        "peak_additional_mib": max(0, peak - before) / (1024.0 ** 2),
    }


def profile_real_clip(
    *, codec, model, adapter, template, device: torch.device,
    codec_hash: str, data_hash: str, history_frames: int = 8,
) -> dict[str, Any]:
    """Profile new modules with the real topology and observed-only condition."""

    if device.type != "cuda":
        raise RuntimeError("stage profile requires CUDA")
    if model.adapter is not adapter:
        raise ValueError("model and adapter must be the same instance")
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.eval()
    model.eval()
    scaffold = _observed_scaffold(template, template.x[:history_frames], history_frames)
    coordinate = prepare_batch_then_to_device(codec, scaffold, device)
    coordinate = replace(
        coordinate, bpos=_block_positions(coordinate.x, coordinate.block_id)
    )
    centered, _ = center_coordinates(
        coordinate.x, frame_mask=coordinate.frame_mask,
        abid=coordinate.abid, atom_mask=coordinate.loss_mask,
    )
    centered_batch = replace(coordinate, x=centered)
    encoder = codec.frame_encoder
    stages: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        def graph_stage():
            nodes = pack_frame_nodes(centered_batch)
            graph = build_frame_graph(
                nodes, centered_batch, neighbor_builder=encoder.neighbor_builder,
                topology_cache=encoder.topology_cache,
                distance_bond_cache=encoder.distance_bond_cache,
                bond_construction_mode=encoder.bond_construction_mode,
                spatial_backbone=encoder.spatial_backbone,
            )
            return nodes, graph

        (nodes, graph), stages["graph"] = measure_cuda(graph_stage, device)
        def encoder_stage():
            encoded = encoder.backbone(
                z=graph.z, b=graph.b, pos=graph.pos, batch=graph.batch,
                edge_index=graph.edge_index, edge_weight_t=graph.edge_weight,
                edge_vec_t=graph.edge_vec, bond_type=graph.bond_type,
            )
            return unpack_frame_features(encoded, nodes)

        _features, stages["encoder"] = measure_cuda(encoder_stage, device)
        latent, stages["codec_encode_inclusive"] = measure_cuda(
            lambda: codec.encode(coordinate), device
        )
        _decoded, stages["decoder"] = measure_cuda(
            lambda: codec.decode(latent), device
        )
        packed = adapter.from_codec_latent(
            latent, codec_hash=codec_hash, data_hash=data_hash,
            origin_from_latent=True, loss_mask=coordinate.loss_mask,
        )
        observation = build_observation_condition(
            packed, history_frames=history_frames, coordinates=coordinate.x,
            frame_mask=coordinate.frame_mask, loss_mask=coordinate.loss_mask,
        )
        observed = packed.with_observation(
            observation.latent_observation_mask, sample_origin=observation.sample_origin
        )
        tau = torch.full((observed.batch_size,), 0.5, device=device)
        _velocity, stages["dit_forward"] = measure_cuda(
            lambda: model(observed, tau), device
        )
    model.zero_grad(set_to_none=True)
    model.train()
    def backward_stage():
        velocity = model(observed, tau)
        loss = sum(value.square().mean() for value in velocity.as_dict().values())
        loss.backward()
        return loss

    loss, stages["dit_forward_backward"] = measure_cuda(backward_stage, device)
    if not bool(torch.isfinite(loss)) or not any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.parameters()
    ):
        raise RuntimeError("DiT backward did not produce finite nonzero gradients")
    if any(parameter.grad is not None for parameter in codec.parameters()):
        raise RuntimeError("frozen codec acquired gradients during DiT backward")
    model.eval()
    return {
        "stages": stages,
        "graph_edges": int(graph.edge_index.shape[1]),
        "atoms": int(template.atype.numel()),
        "history_frames": int(history_frames),
        "optimizer_updated": False,
        "future_condition": "observed_prefix_only",
        "cold_single_run": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--codec-sha256", required=True)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--dit-sha256", required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--valid-index", type=int, required=True)
    parser.add_argument("--history", type=int, choices=(4, 8), default=8)
    args = parser.parse_args(argv)
    device = configure_device("cuda", deterministic=True)
    splits = load_datasets(args.manifest_root)
    try:
        if args.valid_index < 0 or args.valid_index >= len(splits.valid):
            raise IndexError("validation clip index is out of range")
        codec_artifact = load_codec_artifact(
            args.codec, expected_sha256=args.codec_sha256, device=device
        )
        dit_artifact = load_dit_artifact(
            args.dit, expected_sha256=args.dit_sha256, device=device
        )
        payload = dit_artifact["payload"]
        if payload["data_hash"] != splits.data_hash:
            raise ValueError("profile manifest differs from DiT data contract")
        if payload["statistics_state"]["provenance"]["codec_checkpoint_sha256"] != args.codec_sha256:
            raise ValueError("profile codec differs from DiT statistics provenance")
        template = collate_clip_records([splits.valid[args.valid_index]])
        result = profile_real_clip(
            codec=codec_artifact.model, model=dit_artifact["model"],
            adapter=dit_artifact["model"].adapter, template=template,
            device=device, codec_hash=payload["codec_hash"],
            data_hash=splits.data_hash, history_frames=args.history,
        )
        print(json.dumps({
            "schema_version": "molvid.cuda.profile.v1",
            "sample_id": template.sample_id[0],
            "codec_sha256": args.codec_sha256,
            "dit_sha256": args.dit_sha256,
            **result,
        }, sort_keys=True))
        return 0
    finally:
        splits.close()


if __name__ == "__main__":
    raise SystemExit(main())
