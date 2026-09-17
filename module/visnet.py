"""Legacy v1 representation-only ViSNet spatial backbones.

This file preserves the v1 MolViD scalar/vector implementation and its
checkpoint semantics. It is intentionally not the reference-faithful ViSNet
implementation; that implementation lives in module.visnet_v2 and is
selected only by the new visnet_v2_* backend names.

The v1 code is a MolViD project implementation. It must not be attributed to
TorchMD-Net's unrelated torchmdnet/models/visnet.py path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

from utils.bio_utils import NUM_ATOM_TYPE, NUM_BLOCK_TYPE
from utils.torchmd_utils import CosineCutoff, GaussianSmearing

from .neighbor_graph import NeighborList, make_neighbor_list
from .torchmd_et import TorchMD_VQ_ET


SUPPORTED_SPATIAL_BACKBONES = (
    "torchmd_et",
    "visnet_radius",
    "visnet_bonded",
    "visnet_v2_radius",
    "visnet_v2_bonded",
)


@dataclass
class SpatialEncoderOutput:
    """Common atom-level output and graph metadata contract."""

    h: Tensor
    v: Tensor
    edge_index: Optional[Tensor] = None
    edge_weight: Optional[Tensor] = None
    edge_vec: Optional[Tensor] = None
    edge_type: Optional[Tensor] = None
    backend_used: str = "unknown"
    graph_mode: str = "external"
    # Optional full internal lmax=2 state. v1 never populates this field.
    v_full: Optional[Tensor] = None

    @property
    def scalar(self) -> Tensor:
        return self.h

    @property
    def vector(self) -> Tensor:
        return self.v

    @property
    def bond_type(self) -> Optional[Tensor]:
        return self.edge_type

    def __iter__(self):
        yield self.h
        yield self.v


def _reset_linear_parameters(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)


def _segment_softmax(logits: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Softmax edge scores over incoming edges without host scalar reads."""

    if logits.numel() == 0:
        return logits
    if logits.ndim != 2:
        raise ValueError("attention logits must have shape [E, heads]")
    expanded_index = index.unsqueeze(-1).expand(-1, logits.shape[-1])
    maxima = torch.full(
        (int(num_nodes), int(logits.shape[-1])),
        float("-inf"),
        dtype=logits.dtype,
        device=logits.device,
    )
    maxima.scatter_reduce_(
        0, expanded_index, logits, reduce="amax", include_self=True
    )
    weights = torch.exp(logits - maxima.index_select(0, index))
    denominator = torch.zeros_like(maxima)
    denominator.scatter_add_(0, expanded_index, weights)
    return weights / denominator.index_select(0, index).clamp_min(1.0e-12)


def _aggregate(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    output = torch.zeros(
        (int(num_nodes), *messages.shape[1:]),
        dtype=messages.dtype,
        device=messages.device,
    )
    if messages.numel():
        output.index_add_(0, index, messages)
    return output


class _MolViSNetLayer(nn.Module):
    """One l=1 scalar/vector ViSNet message-passing layer."""

    def __init__(
        self,
        hidden_channels: int,
        edge_channels: int,
        num_heads: int,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        self.hidden_channels = int(hidden_channels)
        self.num_heads = int(num_heads)
        self.head_channels = int(hidden_channels // num_heads)
        self.norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.vector_value = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.edge_bias = nn.Linear(edge_channels, num_heads, dtype=dtype)
        self.edge_gate = nn.Linear(edge_channels, hidden_channels, dtype=dtype)
        self.edge_vector = nn.Linear(edge_channels, hidden_channels, dtype=dtype)
        self.scalar_update = nn.Linear(hidden_channels, hidden_channels, dtype=dtype)
        self.vector_update = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.feedforward_norm = nn.LayerNorm(hidden_channels, dtype=dtype)
        self.feedforward = nn.Sequential(
            nn.Linear(2 * hidden_channels, 4 * hidden_channels, dtype=dtype),
            nn.SiLU(),
            nn.Linear(4 * hidden_channels, hidden_channels, dtype=dtype),
        )
        self.vector_gate = nn.Linear(2 * hidden_channels, hidden_channels, dtype=dtype)
        _reset_linear_parameters(self)

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec: Tensor,
        edge_attr: Tensor,
        cutoff: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nodes = int(h.shape[0])
        source, target = edge_index[0], edge_index[1]
        h_norm = self.norm(h)
        q = self.q_proj(h_norm).view(nodes, self.num_heads, self.head_channels)
        k = self.k_proj(h_norm).view(nodes, self.num_heads, self.head_channels)
        values = self.v_proj(h_norm)
        if edge_index.numel():
            logits = (
                q.index_select(0, target) * k.index_select(0, source)
            ).sum(dim=-1) / (self.head_channels ** 0.5)
            logits = logits + self.edge_bias(edge_attr)
            attention = _segment_softmax(logits, target, nodes)
            attention_scalar = attention.mean(dim=-1) * cutoff
            scalar_messages = values.index_select(0, source) * attention_scalar.unsqueeze(
                -1
            )
            scalar_messages = scalar_messages * torch.sigmoid(
                self.edge_gate(edge_attr)
            )
            scalar_update = _aggregate(scalar_messages, target, nodes)

            direction = edge_vec / edge_weight.clamp_min(1.0e-8).unsqueeze(-1)
            direction = direction.to(dtype=v.dtype)
            vector_values = self.vector_value(v.index_select(0, source))
            directional_values = self.edge_vector(edge_attr).unsqueeze(1) * direction.unsqueeze(
                -1
            )
            vector_messages = (vector_values + directional_values) * attention_scalar[
                :, None, None
            ]
            vector_update = _aggregate(vector_messages, target, nodes)
        else:
            scalar_update = h.new_zeros((nodes, self.hidden_channels))
            vector_update = v.new_zeros((nodes, 3, self.hidden_channels))

        h = h + self.scalar_update(scalar_update)
        v = v + self.vector_update(vector_update)
        v_norm = torch.sqrt(v.square().sum(dim=1) + 1.0e-8)
        ff_input = torch.cat([self.feedforward_norm(h), v_norm], dim=-1)
        h = h + self.feedforward(ff_input)
        gate_input = torch.cat([h, v_norm], dim=-1)
        v = v + v * torch.sigmoid(self.vector_gate(gate_input)).unsqueeze(1)
        return h, v


class MolViSNetEncoder(nn.Module):
    """Representation-only MolViSNet with Cartesian lmax=1 vectors.

    forward_native owns exactly one radius-neighbor construction. The
    external graph path is supplied through forward_external and never
    constructs another graph. Both paths use the same embeddings, radial
    basis, and message-passing parameters.
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        num_rbf: int = 32,
        lmax: int = 1,
        vertex: bool = True,
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        trainable_rbf: bool = False,
        vecnorm_type: str | None = None,
        *,
        max_z: int = NUM_ATOM_TYPE,
        max_b: int = NUM_BLOCK_TYPE,
        neighbor_backend: str = "cuda_radius",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if int(lmax) != 1:
            raise ValueError(
                f"MolViSNetEncoder supports only lmax=1; got lmax={lmax}. "
                "The temporal codec requires Cartesian [M, 3, C] vectors."
            )
        if not bool(vertex):
            raise ValueError(
                "MolViSNetEncoder requires vertex=True to produce Cartesian "
                "[M, 3, C] vector features"
            )
        if int(hidden_channels) < 1 or int(num_layers) < 1:
            raise ValueError("hidden_channels and num_layers must be positive")
        if int(num_heads) < 1 or int(hidden_channels) % int(num_heads) != 0:
            raise ValueError(
                "hidden_channels must be divisible by a positive num_heads"
            )
        if int(num_rbf) < 2:
            raise ValueError("num_rbf must be at least 2")
        if vecnorm_type not in (None, "none", "l2"):
            raise ValueError("vecnorm_type must be None, 'none', or 'l2'")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if float(cutoff_lower) < 0 or float(cutoff_upper) <= float(cutoff_lower):
            raise ValueError("invalid ViSNet cutoffs")
        if int(max_num_neighbors) < 1:
            raise ValueError("max_num_neighbors must be positive")
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_rbf = int(num_rbf)
        self.lmax = 1
        self.vertex = True
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.trainable_rbf = bool(trainable_rbf)
        self.vecnorm_type = vecnorm_type
        self.max_z = int(max_z)
        self.max_b = int(max_b)
        self.dtype = dtype
        self.neighbor_backend = str(neighbor_backend)

        self.atom_embedding = nn.Embedding(
            self.max_z, self.hidden_channels, dtype=dtype
        )
        self.block_embedding = nn.Embedding(
            self.max_b, self.hidden_channels, dtype=dtype
        )
        self.bond_embedding = nn.Embedding(2, self.num_rbf, dtype=dtype)
        self.distance_expansion = GaussianSmearing(
            self.cutoff_lower,
            self.cutoff_upper,
            self.num_rbf,
            trainable=self.trainable_rbf,
            dtype=dtype,
        )
        self.cutoff = CosineCutoff(self.cutoff_lower, self.cutoff_upper)
        self.layers = nn.ModuleList(
            _MolViSNetLayer(
                self.hidden_channels,
                self.num_rbf,
                self.num_heads,
                dtype=dtype,
            )
            for _ in range(self.num_layers)
        )
        self.out_norm = nn.LayerNorm(self.hidden_channels, dtype=dtype)
        self.native_neighbor_builder = make_neighbor_list(
            self.neighbor_backend,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            max_num_neighbors=self.max_num_neighbors,
            loop=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.atom_embedding.reset_parameters()
        self.block_embedding.reset_parameters()
        self.bond_embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.out_norm.reset_parameters()
        for layer in self.layers:
            layer.norm.reset_parameters()
            layer.feedforward_norm.reset_parameters()
            _reset_linear_parameters(layer)

    @staticmethod
    def _node_fields(nodes: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        try:
            z, b, pos, graph_id = nodes.z, nodes.b, nodes.pos, nodes.graph_id
            graph_count = int(nodes.graph_count)
        except AttributeError as exc:
            raise TypeError(
                "native/external ViSNet calls require a FrameNodeBatch-like object"
            ) from exc
        return z, b, pos, graph_id, graph_count

    def _edge_features(
        self,
        edge_weight: Tensor,
        edge_type: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        distance = edge_weight.to(dtype=self.dtype)
        radial = self.distance_expansion(distance)
        if edge_type is not None:
            bond_type = edge_type.to(
                device=radial.device, dtype=torch.long
            ).flatten()
            if bond_type.numel() != radial.shape[0]:
                raise ValueError("edge type must contain one entry per edge")
            invalid = torch.any((bond_type < 0) | (bond_type > 1))
            if invalid.device.type == "cuda":
                torch._assert_async(
                    ~invalid,
                    "binary edge type must contain only 0 or 1",
                )
            elif bool(invalid):
                raise ValueError("binary edge type must contain only 0 or 1")
            radial = radial + self.bond_embedding(bond_type)
        cutoff = self.cutoff(distance)
        return radial, cutoff

    def _represent(
        self,
        *,
        z: Tensor,
        b: Tensor,
        pos: Tensor,
        graph_id: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_vec: Tensor,
        edge_type: Optional[Tensor],
        backend_used: str,
        graph_mode: str,
    ) -> SpatialEncoderOutput:
        del graph_id
        if pos.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA coordinates require an available CUDA runtime")
        if pos.dtype != torch.float32:
            raise RuntimeError(
                "ViSNet geometry requires original FP32 coordinates; "
                "use autocast for feature precision instead"
            )
        if z.ndim != 1 or b.ndim != 1 or pos.ndim != 2 or pos.shape[-1] != 3:
            raise ValueError("ViSNet node fields must be [M], [M], and [M, 3]")
        if z.shape[0] != pos.shape[0] or b.shape[0] != pos.shape[0]:
            raise ValueError("ViSNet node fields must share the atom axis")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("ViSNet edge_index must have shape [2, E]")
        if edge_weight.ndim != 1 or edge_vec.ndim != 2 or edge_vec.shape[-1] != 3:
            raise ValueError("ViSNet edge geometry must have shapes [E] and [E, 3]")
        if (
            edge_index.shape[1] != edge_weight.shape[0]
            or edge_index.shape[1] != edge_vec.shape[0]
        ):
            raise ValueError("ViSNet edge fields must share the edge axis")
        h = self.atom_embedding(z) + self.block_embedding(b)
        edge_attr, cutoff = self._edge_features(edge_weight, edge_type)
        v = h.new_zeros((pos.shape[0], 3, self.hidden_channels))
        for layer in self.layers:
            h, v = layer(
                h, v, edge_index, edge_weight, edge_vec, edge_attr, cutoff
            )
        h = self.out_norm(h)
        return SpatialEncoderOutput(
            h=h,
            v=v,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
            edge_type=edge_type,
            backend_used=str(backend_used),
            graph_mode=str(graph_mode),
        )

    def forward_native(self, nodes: Any) -> SpatialEncoderOutput:
        """Build one native radius graph and encode the packed frame nodes."""

        z, b, pos, graph_id, graph_count = self._node_fields(nodes)
        neighbors: NeighborList = self.native_neighbor_builder(
            pos, graph_id, batch_size=graph_count
        )
        return self._represent(
            z=z,
            b=b,
            pos=pos,
            graph_id=graph_id,
            edge_index=neighbors.edge_index,
            edge_weight=neighbors.edge_weight,
            edge_vec=neighbors.edge_vec,
            edge_type=None,
            backend_used=neighbors.backend_used,
            graph_mode="native_radius",
        )

    def forward_external(self, nodes: Any, graph: Any) -> SpatialEncoderOutput:
        """Encode a caller-supplied external graph without rebuilding it."""

        z, b, pos, graph_id, _graph_count = self._node_fields(nodes)
        return self._represent(
            z=z,
            b=b,
            pos=pos,
            graph_id=graph_id,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            edge_vec=graph.edge_vec,
            edge_type=graph.bond_type,
            backend_used=graph.backend_used,
            graph_mode="external",
        )

    def forward(
        self,
        z: Tensor,
        b: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        edge_vec: Optional[Tensor] = None,
        bond_type: Optional[Tensor] = None,
        *,
        edge_weight_t: Optional[Tensor] = None,
        edge_vec_t: Optional[Tensor] = None,
    ) -> SpatialEncoderOutput:
        """Direct external-graph interface for small tests and adapters."""

        if edge_weight is None:
            edge_weight = edge_weight_t
        if edge_vec is None:
            edge_vec = edge_vec_t
        if edge_index is None or edge_weight is None or edge_vec is None:
            raise ValueError(
                "MolViSNetEncoder.forward requires an explicit external graph; "
                "use forward_native for native radius construction"
            )
        graph = type(
            "Graph",
            (),
            {
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "edge_vec": edge_vec,
                "bond_type": bond_type,
                "backend_used": "external",
                "batch": batch,
            },
        )()
        nodes = type(
            "Nodes",
            (),
            {
                "z": z,
                "b": b,
                "pos": pos,
                "graph_id": batch,
                "graph_count": 1,
            },
        )()
        return self.forward_external(nodes, graph)


def make_spatial_backbone(
    name: str,
    *,
    hidden_channels: int = 128,
    num_layers: int = 2,
    num_rbf: int = 32,
    num_heads: int = 8,
    cutoff_lower: float = 0.0,
    cutoff_upper: float = 5.0,
    max_num_neighbors: int = 32,
    max_z: int = NUM_ATOM_TYPE,
    max_b: int = NUM_BLOCK_TYPE,
    neighbor_backend: str = "cuda_radius",
    dtype: torch.dtype = torch.float32,
    lmax: int = 1,
    vertex: bool = True,
    trainable_rbf: bool = False,
    vecnorm_type: str | None = None,
    vertex_type: str | None = None,
    rbf_type: str | None = None,
    trainable_vecnorm: bool = False,
    **_: Any,
) -> nn.Module:
    """Build one of the legacy or reference-faithful spatial backbones."""

    selected = str(name).lower()
    if selected == "torchmd_et":
        return TorchMD_VQ_ET(
            hidden_channels=hidden_channels,
            extra_channels=0,
            num_layers=num_layers,
            num_rbf=num_rbf,
            num_heads=num_heads,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_z=max_z,
            max_b=max_b,
            max_num_neighbors=max_num_neighbors,
            cross_attn=False,
            dtype=dtype,
        )
    if selected in {"visnet_radius", "visnet_bonded"}:
        return MolViSNetEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            lmax=lmax,
            vertex=vertex,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            trainable_rbf=trainable_rbf,
            vecnorm_type=vecnorm_type,
            max_z=max_z,
            max_b=max_b,
            neighbor_backend=neighbor_backend,
            dtype=dtype,
        )
    if selected in {"visnet_v2_radius", "visnet_v2_bonded"}:
        from .visnet_v2 import V2SpatialEncoder

        resolved_vertex_type = (
            str(vertex_type).lower()
            if vertex_type is not None
            else ("edge" if bool(vertex) else "none")
        )
        resolved_rbf_type = "expnorm" if rbf_type is None else str(rbf_type).lower()
        return V2SpatialEncoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_rbf=num_rbf,
            lmax=lmax,
            vertex_type=resolved_vertex_type,
            cutoff_lower=cutoff_lower,
            cutoff_upper=cutoff_upper,
            max_num_neighbors=max_num_neighbors,
            rbf_type=resolved_rbf_type,
            trainable_rbf=trainable_rbf,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            max_z=max_z,
            max_b=max_b,
            neighbor_backend=neighbor_backend,
            dtype=dtype,
            use_block_embedding=True,
            use_bond_embedding=selected == "visnet_v2_bonded",
        )
    raise ValueError(
        f"unsupported spatial_backbone {name!r}; expected one of "
        f"{SUPPORTED_SPATIAL_BACKBONES}"
    )


build_spatial_backbone = make_spatial_backbone


__all__ = [
    "MolViSNetEncoder",
    "SpatialEncoderOutput",
    "SUPPORTED_SPATIAL_BACKBONES",
    "make_spatial_backbone",
    "build_spatial_backbone",
]
