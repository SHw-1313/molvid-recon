"""Strict device-resident neighbor-list backends for the codec graph path.

The production backend deliberately keeps only the discrete neighbor search
under ``no_grad``.  Distances and direction vectors are computed afterwards
from the original FP32 CUDA coordinates so coordinate gradients remain
available to the model and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

try:
    from torch_cluster import radius_graph as _radius_graph
except (ImportError, OSError):  # pragma: no cover - exercised by strict-error tests
    _radius_graph = None


def _assert_device_condition(condition: Tensor, message: str) -> None:
    condition = condition if condition.ndim == 0 else condition.all()
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise RuntimeError(message)


@dataclass(frozen=True)
class NeighborList:
    edge_index: Tensor
    edge_weight: Tensor
    edge_vec: Tensor
    backend_used: str


class CudaRadiusNeighborList:
    """CUDA-only radius graph construction with explicit failure semantics."""

    backend_used = "cuda_radius"

    def __init__(
        self,
        *,
        cutoff_lower: float,
        cutoff_upper: float,
        max_num_neighbors: int,
        loop: bool = True,
        strict_cap: bool = False,
    ) -> None:
        if float(cutoff_lower) < 0 or float(cutoff_upper) <= float(cutoff_lower):
            raise ValueError("invalid neighbor cutoffs")
        if int(max_num_neighbors) < 1:
            raise ValueError("max_num_neighbors must be positive")
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.loop = bool(loop)
        self.strict_cap = bool(strict_cap)
        # The topology path delegates the configured cap directly to the
        # CUDA kernel.  Distance-only inference uses a bounded over-query so
        # it can prove that its stricter cap was not saturated, then applies
        # the same source-ordered cap on-device.
        self.candidate_max_num_neighbors = (
            max(256, self.max_num_neighbors * 8)
            if self.strict_cap
            else self.max_num_neighbors
        )

    def __call__(
        self,
        positions: Tensor,
        graph_id: Tensor,
        *,
        batch_size: int | None = None,
    ) -> NeighborList:
        if positions.device.type != "cuda":
            raise RuntimeError(
                "cuda_radius neighbor construction requires CUDA coordinates; "
                "CPU/dense construction is test-only and must be selected explicitly"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; refusing to fall back to a CPU graph backend"
            )
        if _radius_graph is None:
            raise RuntimeError(
                "torch_cluster.radius_graph is unavailable for the CUDA production "
                "graph path; install/repair the CUDA extension instead of using a fallback"
            )
        if positions.dtype != torch.float32:
            raise RuntimeError(
                "cuda_radius requires original FP32 coordinates for geometry"
            )
        if graph_id.device != positions.device:
            raise RuntimeError("graph_id and coordinates must be on the same CUDA device")
        if graph_id.dtype != torch.long:
            raise RuntimeError("graph_id must be torch.long")
        if positions.ndim != 2 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape [N, 3]")
        if graph_id.ndim != 1 or graph_id.shape[0] != positions.shape[0]:
            raise ValueError("graph_id must have shape [N]")
        if batch_size is None or int(batch_size) < 1:
            raise ValueError(
                "cuda_radius requires an explicit positive batch_size; derive "
                "graph count from host batch metadata"
            )

        # The discrete search is intentionally non-differentiable.  It must
        # nevertheless receive the original CUDA positions, never a CPU copy.
        with torch.no_grad():
            edge_index = _radius_graph(
                positions,
                r=self.cutoff_upper,
                batch=graph_id,
                loop=self.loop,
                max_num_neighbors=self.candidate_max_num_neighbors,
                flow="source_to_target",
                batch_size=int(batch_size),
            )
        if not isinstance(edge_index, Tensor):
            raise RuntimeError("torch_cluster.radius_graph returned a non-tensor edge index")
        edge_index = edge_index.to(dtype=torch.long)
        if edge_index.device != positions.device:
            raise RuntimeError("CUDA radius_graph returned an edge index on the wrong device")
        if edge_index.numel():
            src, dst = edge_index
            _assert_device_condition(
                graph_id[src] == graph_id[dst],
                "CUDA radius_graph produced a cross-graph edge",
            )
            candidate_counts = torch.bincount(
                dst, minlength=positions.shape[0]
            )
            if self.strict_cap:
                _assert_device_condition(
                    candidate_counts < self.candidate_max_num_neighbors,
                    "CUDA radius candidate cap is saturated; refusing to change "
                    "distance-only bond semantics",
                )
            _assert_device_condition(
                dst[1:] >= dst[:-1],
                "CUDA radius_graph returned non-contiguous target ordering; "
                "cannot apply deterministic legacy cap",
            )
            same_target = dst[1:] == dst[:-1]
            _assert_device_condition(
                src[1:][same_target] >= src[:-1][same_target],
                "CUDA radius_graph returned non-source-ordered candidates; "
                "cannot apply deterministic legacy cap",
            )
            starts = torch.cumsum(candidate_counts, dim=0) - candidate_counts
            ranks = torch.arange(
                dst.shape[0], device=dst.device, dtype=torch.long
            ) - torch.repeat_interleave(starts, candidate_counts)
            keep_candidates = ranks < self.max_num_neighbors
            edge_index = edge_index[:, keep_candidates]
        edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
        edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
        keep = (edge_weight >= self.cutoff_lower) & (edge_weight < self.cutoff_upper)
        edge_index = edge_index[:, keep]
        edge_vec = edge_vec[keep]
        edge_weight = edge_weight[keep]
        if edge_index.numel():
            counts = torch.bincount(
                edge_index[1], minlength=positions.shape[0]
            )
            _assert_device_condition(
                counts <= self.max_num_neighbors,
                "CUDA radius_graph exceeded max_num_neighbors after deterministic "
                "cap/filtering",
            )
            if self.strict_cap:
                _assert_device_condition(
                    counts < self.max_num_neighbors,
                    f"CUDA radius neighbor cap {self.max_num_neighbors} is saturated",
                )
        return NeighborList(
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
            backend_used=self.backend_used,
        )


class DenseTestNeighborList:
    """Small CPU-only backend used by unit tests and synthetic fixtures."""

    backend_used = "dense_test"

    def __init__(
        self,
        *,
        cutoff_lower: float,
        cutoff_upper: float,
        max_num_neighbors: int,
        loop: bool = True,
    ) -> None:
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.loop = bool(loop)

    def __call__(
        self,
        positions: Tensor,
        graph_id: Tensor,
        *,
        batch_size: int | None = None,
    ) -> NeighborList:
        if positions.device.type != "cpu" or graph_id.device.type != "cpu":
            raise RuntimeError("dense_test neighbor construction is CPU-only")
        parts: list[Tensor] = []
        weights: list[Tensor] = []
        vectors: list[Tensor] = []
        for graph in torch.unique(graph_id, sorted=True).tolist():
            nodes = torch.nonzero(graph_id == graph, as_tuple=False).flatten()
            local_pos = positions.index_select(0, nodes)
            distances = torch.cdist(local_pos, local_pos)
            valid = (distances >= self.cutoff_lower) & (
                distances < self.cutoff_upper
            )
            if self.loop:
                valid.fill_diagonal_(True)
            else:
                valid.fill_diagonal_(False)
            k = min(self.max_num_neighbors, int(nodes.numel()))
            scores = distances.masked_fill(~valid, float("inf"))
            values, neighbors = torch.topk(scores, k=k, dim=1, largest=False)
            keep = torch.isfinite(values)
            target = nodes[:, None].expand_as(neighbors)[keep]
            source = nodes[neighbors][keep]
            edge_index = torch.stack([source, target], dim=0)
            edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
            parts.append(edge_index)
            weights.append(values[keep].to(dtype=positions.dtype))
            vectors.append(edge_vec)
        if not parts:
            empty_index = torch.empty((2, 0), dtype=torch.long)
            empty_weight = positions.new_empty((0,))
            empty_vec = positions.new_empty((0, 3))
            return NeighborList(empty_index, empty_weight, empty_vec, self.backend_used)
        return NeighborList(
            edge_index=torch.cat(parts, dim=1),
            edge_weight=torch.cat(weights, dim=0),
            edge_vec=torch.cat(vectors, dim=0),
            backend_used=self.backend_used,
        )


def make_neighbor_list(
    backend: str,
    *,
    cutoff_lower: float,
    cutoff_upper: float,
    max_num_neighbors: int,
    loop: bool = True,
    strict_cap: bool = False,
) -> CudaRadiusNeighborList | DenseTestNeighborList:
    """Construct an explicitly selected backend; there is no ``auto`` mode."""

    if backend == "cuda_radius":
        return CudaRadiusNeighborList(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            loop=loop,
            strict_cap=strict_cap,
        )
    if backend == "dense_test":
        return DenseTestNeighborList(
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            loop=loop,
        )
    raise ValueError(
        "neighbor_backend must be 'cuda_radius' for production or 'dense_test' "
        "for explicit CPU tests"
    )


__all__ = [
    "CudaRadiusNeighborList",
    "DenseTestNeighborList",
    "NeighborList",
    "make_neighbor_list",
]
