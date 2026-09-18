from __future__ import annotations

import numpy as np
import torch

from molvid.equivariant import AxisPreservingLinear, SO3ChannelNorm
from molvid.geometry.coordinates import align_and_center, center_coordinates, restore_origin
from molvid.data.batch import collate_clip_records
from molvid.geometry.frames import (
    build_frame_graph,
    pack_frame_nodes,
    unpack_frame_features,
)
from molvid.geometry.neighbors import CudaRadiusNeighborList
from molvid.geometry.topology import BoundedTopologyCache, DistanceOnlyBondCache, build_canonical_reference_index
from molvid.geometry.types import StaticTopologyMetadata


def _record(sample_id: str, offset: float) -> dict:
    x = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    x[:, :, 0] += offset
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "system_id": sample_id,
        "task": "trajectory",
        "time_bucket_id": "dt_100ps",
        "time_ps": np.array([0.0, 100.0], dtype=np.float32),
        "delta_time_ps": np.array([100.0], dtype=np.float32),
        "x": x,
        "bpos": x.copy(),
        "atype": np.array([5, 5, 5], dtype=np.int64),
        "btype": np.array([6, 6, 6], dtype=np.int64),
        "block_id": np.array([0, 0, 0], dtype=np.int64),
        "component_id": np.array([0, 0, 0], dtype=np.int64),
        "atom_source_index": np.array([0, 1, 2], dtype=np.int64),
        "atom_identity": [f"{sample_id}:{i}" for i in range(3)],
        "edge_mask": np.zeros(3, dtype=np.int64),
        "loss_mask": np.ones(3, dtype=np.bool_),
        "align_mask": np.ones(3, dtype=np.bool_),
        "bond_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
    }


def test_cuda_graph_isolates_samples_and_frames_and_retains_coordinate_gradients():
    assert torch.cuda.is_available(), "P1 CUDA graph acceptance requires CUDA"
    cpu_batch = collate_clip_records([_record("a", 0.0), _record("b", 20.0)])
    topology = BoundedTopologyCache()
    topology.register_packed_batch(cpu_batch)
    batch = cpu_batch.to("cuda")
    batch.x.requires_grad_(True)
    nodes = pack_frame_nodes(batch)
    assert nodes.graph_id.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    graph = build_frame_graph(
        nodes,
        batch,
        neighbor_builder=CudaRadiusNeighborList(
            cutoff_lower=0.0, cutoff_upper=2.5, max_num_neighbors=8
        ),
        topology_cache=topology,
    )
    assert graph.edge_index.shape == (2, 36)
    assert graph.bond_index.shape == (2, 8)
    assert graph.bond_type.sum().item() == 8
    assert torch.equal(
        nodes.graph_id[graph.edge_index[0]],
        nodes.graph_id[graph.edge_index[1]],
    )
    graph.edge_weight.sum().backward()
    assert batch.x.grad is not None
    assert torch.isfinite(batch.x.grad).all()
    assert batch.x.grad.abs().sum() > 0
    h, v = unpack_frame_features(
        (nodes.z.float().unsqueeze(-1), nodes.pos.unsqueeze(-1)), nodes
    )
    assert h.shape == (2, 6, 1)
    assert v.shape == (2, 6, 3, 1)


def test_static_topology_excludes_coordinates_and_radius_edges():
    batch = collate_clip_records([_record("a", 0.0), _record("b", 20.0)])
    metadata = StaticTopologyMetadata.from_batch(batch)
    contract = metadata.contract()
    assert metadata.num_atoms == 6
    assert metadata.batch_size == 2
    assert metadata.covalent_bond_index.shape == (2, 4)
    assert contract["coordinate_independent"] is True
    assert contract["contains_radius_edges"] is False
    assert contract["contains_target_coordinates"] is False

def test_center_restore_and_alignment_keep_atom_order_and_units():
    batch = collate_clip_records([_record("a", 0.0), _record("b", 20.0)])
    centered, origin = center_coordinates(
        batch.x,
        frame_mask=batch.frame_mask,
        abid=batch.abid,
        atom_mask=batch.loss_mask,
    )
    torch.testing.assert_close(restore_origin(centered, origin, batch.abid), batch.x, rtol=0, atol=1e-6)
    assert origin.tolist() == [[1.0, 0.0, 0.0], [21.0, 0.0, 0.0]]
    x = np.asarray(_record("a", 0.0)["x"])
    rotated = x.copy()
    rotated[1] = x[1] @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    aligned, bpos, metadata = align_and_center(
        rotated, rotated.copy(), np.array([True, True, True])
    )
    np.testing.assert_allclose(aligned, bpos, rtol=0, atol=0)
    np.testing.assert_allclose(aligned[0], x[0] - x[0].mean(axis=0), atol=1e-6)
    assert metadata["alignment_atom_count"] == 3


def test_shared_vector_operators_preserve_initialization_and_rotation():
    torch.manual_seed(211)
    expected_weight = torch.empty(4, 3)
    torch.nn.init.xavier_uniform_(expected_weight)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(211)
    linear = AxisPreservingLinear(3, 4)
    assert torch.equal(linear.weight, expected_weight)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    norm = SO3ChannelNorm(4)
    value = torch.randn(2, 3, 3)
    quarter_turn = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = torch.einsum("ij,bjc->bic", quarter_turn, value)
    linear_before = linear(value)
    linear_after = linear(rotated)
    torch.testing.assert_close(
        linear_after, torch.einsum("ij,bjc->bic", quarter_turn, linear_before),
        rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        norm(linear_after),
        torch.einsum("ij,bjc->bic", quarter_turn, norm(linear_before)),
        rtol=1e-6, atol=1e-6,
    )

def test_distance_only_bonds_use_train_frame_zero_reference():
    assert torch.cuda.is_available(), "distance-only CUDA acceptance requires CUDA"
    record = _record("a", 0.0)
    record["split"] = "train"
    references = build_canonical_reference_index([record], source_split="train")
    changed_future = _record("a", 0.0)
    changed_future["split"] = "train"
    changed_future["x"][1] += 500.0
    future_references = build_canonical_reference_index(
        [changed_future], source_split="train"
    )
    assert references["a"]["coordinate_sha256"] == future_references["a"]["coordinate_sha256"]
    cache = DistanceOnlyBondCache(
        min_distance_angstrom=0.5,
        max_distance_angstrom=1.5,
        max_num_neighbors=8,
    )
    item = references["a"]
    metadata = cache.register_reference(
        "a",
        item["coordinates"],
        device=torch.device("cuda"),
        atom_identity_sha256=item["atom_identity_sha256"],
        sample_id=item["sample_id"],
        source_split=item["source_split"],
    )
    assert metadata["source_split"] == "train"
    bonds = cache.materialize(
        "a",
        device=torch.device("cuda"),
        atom_count=3,
        atom_identity_sha256=item["atom_identity_sha256"],
    )
    assert set(map(tuple, bonds.t().cpu().tolist())) == {
        (0, 1), (1, 0), (1, 2), (2, 1)
    }
