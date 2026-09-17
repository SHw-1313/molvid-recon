"""Reference-faithful, representation-only ViSNet spatial backbones.

The geometry in this module is adapted from the pinned AI2BMD ViSNet
representation implementation and cross-checked against the pinned PyG
implementation. It deliberately stops at scalar/vector representations:
there are no energy, force, atom-reference, or graph-reduction heads.

The project-specific additions are isolated atom/block-type embeddings, an
optional binary covalent embedding, external/native graph adapters, and the
public lmax-1 codec adapter.

Reference provenance
-------------------
AI2BMD: https://github.com/microsoft/AI2BMD.git
commit 497efaa190ee6f6cbc6030710c44208a01ece52d, branch ViSNet.
PyG cross-check: https://github.com/pyg-team/pytorch_geometric.git
commit 79d33965a40b7fa83616a9f598a0f8619f25d939.
Both sources are MIT licensed. The local implementation is a reviewed
adaptation and does not import either checkout at runtime.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import Tensor, nn

from utils.bio_utils import NUM_ATOM_TYPE, NUM_BLOCK_TYPE

from .neighbor_graph import NeighborList, make_neighbor_list
from .visnet import SpatialEncoderOutput


V2_REFERENCE_PROVENANCE = {
    "primary_repository": "https://github.com/microsoft/AI2BMD.git",
    "primary_commit": "497efaa190ee6f6cbc6030710c44208a01ece52d",
    "primary_branch": "ViSNet",
    "cross_check_repository": "https://github.com/pyg-team/pytorch_geometric.git",
    "cross_check_commit": "79d33965a40b7fa83616a9f598a0f8619f25d939",
    "license": "MIT",
    "runtime_reference_import": False,
}


def _aggregate(messages: Tensor, index: Tensor, num_nodes: int) -> Tensor:
    """Add edge messages into target nodes without changing edge order."""

    output = messages.new_zeros((int(num_nodes), *messages.shape[1:]))
    if messages.numel():
        output.index_add_(0, index, messages)
    return output


class V2CosineCutoff(nn.Module):
    """Canonical ViSNet cosine cutoff."""

    def __init__(self, cutoff: float, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if float(cutoff) <= 0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.dtype = dtype

    def forward(self, distances: Tensor) -> Tensor:
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        return cutoffs * (distances < self.cutoff).to(dtype=distances.dtype)


class V2ExpNormalSmearing(nn.Module):
    """Canonical ViSNet exponential-normal radial basis."""

    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if float(cutoff) <= 0 or int(num_rbf) < 2:
            raise ValueError("cutoff must be positive and num_rbf must be at least 2")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        self.cutoff_fn = V2CosineCutoff(cutoff, dtype=dtype)
        self.alpha = 5.0 / float(cutoff)
        means, betas = self._initial_params(dtype=dtype)
        if self.trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self, *, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        if dtype is not None:
            value_dtype = dtype
        elif hasattr(self, "means"):
            value_dtype = self.means.dtype
        else:
            value_dtype = torch.float32
        start_value = torch.exp(torch.tensor(-self.cutoff, dtype=value_dtype))
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=value_dtype)
        beta = (2 / self.num_rbf * (1 - start_value)) ** -2
        betas = torch.full((self.num_rbf,), beta, dtype=value_dtype)
        return means, betas

    def reset_parameters(self) -> None:
        means, betas = self._initial_params(dtype=self.means.dtype)
        with torch.no_grad():
            self.means.copy_(means)
            self.betas.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


class V2GaussianSmearing(nn.Module):
    """Reference-compatible Gaussian radial basis compatibility mode."""

    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 50,
        trainable: bool = True,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if float(cutoff) <= 0 or int(num_rbf) < 2:
            raise ValueError("cutoff must be positive and num_rbf must be at least 2")
        self.cutoff = float(cutoff)
        self.num_rbf = int(num_rbf)
        self.trainable = bool(trainable)
        offset, coeff = self._initial_params(dtype=dtype)
        if self.trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self, *, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        if dtype is not None:
            value_dtype = dtype
        elif hasattr(self, "offset"):
            value_dtype = self.offset.dtype
        else:
            value_dtype = torch.float32
        offset = torch.linspace(0, self.cutoff, self.num_rbf, dtype=value_dtype)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self) -> None:
        offset, coeff = self._initial_params(dtype=self.offset.dtype)
        with torch.no_grad():
            self.offset.copy_(offset)
            self.coeff.copy_(coeff)

    def forward(self, dist: Tensor) -> Tensor:
        shifted = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * shifted.pow(2))


class V2Sphere(nn.Module):
    """Reference real spherical-harmonic basis for lmax one or two."""

    def __init__(self, lmax: int = 2) -> None:
        super().__init__()
        if int(lmax) not in {1, 2}:
            raise ValueError(f"lmax must be 1 or 2, got {lmax}")
        self.lmax = int(lmax)

    def forward(self, edge_vec: Tensor) -> Tensor:
        if edge_vec.shape[-1] != 3:
            raise ValueError("edge_vec must have a final Cartesian dimension of 3")
        x, y, z = edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2]
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        if self.lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)
        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))
        return torch.stack(
            [sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4],
            dim=-1,
        )


class V2VecLayerNorm(nn.Module):
    """Reference vector normalization, including the max-min operation."""

    def __init__(
        self,
        hidden_channels: int,
        trainable: bool = False,
        norm_type: str | None = "max_min",
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if int(hidden_channels) < 1:
            raise ValueError("hidden_channels must be positive")
        resolved = None if norm_type in {None, "none"} else str(norm_type).lower()
        if resolved not in {None, "rms", "max_min"}:
            raise ValueError("norm_type must be None, 'none', 'rms', or 'max_min'")
        self.hidden_channels = int(hidden_channels)
        self.norm_type = resolved
        self.eps = 1.0e-12
        weight = torch.ones(self.hidden_channels, dtype=dtype)
        if bool(trainable):
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.fill_(1.0)

    def none_norm(self, vec: Tensor) -> Tensor:
        return vec

    def rms_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1)
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / dist.unsqueeze(-1).unsqueeze(-1).clamp_min(self.eps)

    def max_min_norm(self, vec: Tensor) -> Tensor:
        dist = torch.norm(vec, dim=1, keepdim=True)
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).reshape(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.reshape(-1, 1, 1)) / delta.reshape(-1, 1, 1)
        return torch.relu(dist) * direct

    def _apply_one(self, vec: Tensor) -> Tensor:
        if self.norm_type == "rms":
            return self.rms_norm(vec)
        if self.norm_type == "max_min":
            return self.max_min_norm(vec)
        return self.none_norm(vec)

    def forward(self, vec: Tensor) -> Tensor:
        if vec.ndim != 3 or vec.shape[1] not in {3, 8}:
            raise ValueError(
                f"V2VecLayerNorm only supports [M, 3, C] or [M, 8, C], got {tuple(vec.shape)}"
            )
        if vec.shape[-1] != self.hidden_channels:
            raise ValueError("vector channel width disagrees with hidden_channels")
        if vec.shape[1] == 3:
            normalized = self._apply_one(vec)
        else:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            normalized = torch.cat([self._apply_one(vec1), self._apply_one(vec2)], dim=1)
        return normalized * self.weight.unsqueeze(0).unsqueeze(0)


class V2NeighborEmbedding(nn.Module):
    """Reference neighbor embedding with self-loop removal."""

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(max_z), int(hidden_channels), dtype=dtype)
        self.distance_proj = nn.Linear(int(num_rbf), int(hidden_channels), dtype=dtype)
        self.combine = nn.Linear(2 * int(hidden_channels), int(hidden_channels), dtype=dtype)
        self.cutoff = V2CosineCutoff(cutoff, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        with torch.no_grad():
            self.distance_proj.bias.zero_()
            self.combine.bias.zero_()

    def forward(
        self,
        z: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        source, target = edge_index[0], edge_index[1]
        mask = source != target
        source = source[mask]
        target = target[mask]
        distances = edge_weight[mask]
        attributes = edge_attr[mask]
        c = self.cutoff(distances)
        weights = self.distance_proj(attributes) * c.unsqueeze(-1)
        neighbors = self.embedding(z)
        messages = neighbors.index_select(0, source) * weights
        aggregated = _aggregate(messages, target, int(x.shape[0]))
        return self.combine(torch.cat([x, aggregated], dim=1))


class V2EdgeEmbedding(nn.Module):
    """Reference node-conditioned edge embedding without edge aggregation."""

    def __init__(
        self,
        num_rbf: int,
        hidden_channels: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.edge_proj = nn.Linear(int(num_rbf), int(hidden_channels), dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.edge_proj.weight)
        with torch.no_grad():
            self.edge_proj.bias.zero_()

    def forward(self, edge_index: Tensor, edge_attr: Tensor, x: Tensor) -> Tensor:
        source, target = edge_index[0], edge_index[1]
        return (
            x.index_select(0, target) + x.index_select(0, source)
        ) * self.edge_proj(edge_attr)


class V2ViSMP(nn.Module):
    """Reference ViS-MP without vertex geometric features."""

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str | None,
        trainable_vecnorm: bool,
        last_layer: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if int(num_heads) < 1 or int(hidden_channels) % int(num_heads) != 0:
            raise ValueError("hidden_channels must be divisible by a positive num_heads")
        self.num_heads = int(num_heads)
        self.hidden_channels = int(hidden_channels)
        self.head_dim = self.hidden_channels // self.num_heads
        self.last_layer = bool(last_layer)
        self.layernorm = nn.LayerNorm(self.hidden_channels, dtype=dtype)
        self.vec_layernorm = V2VecLayerNorm(
            self.hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
            dtype=dtype,
        )
        self.act = nn.SiLU()
        self.attn_activation = nn.SiLU()
        self.cutoff = V2CosineCutoff(cutoff, dtype=dtype)
        self.vec_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels * 3, bias=False, dtype=dtype
        )
        self.q_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels, dtype=dtype
        )
        self.k_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels, dtype=dtype
        )
        self.v_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels, dtype=dtype
        )
        self.dk_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels, dtype=dtype
        )
        self.dv_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels, dtype=dtype
        )
        self.s_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels * 2, dtype=dtype
        )
        if not self.last_layer:
            self.f_proj = nn.Linear(
                self.hidden_channels, self.hidden_channels, dtype=dtype
            )
            self.w_src_proj = nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
            )
            self.w_trg_proj = nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
            )
        self.o_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels * 3, dtype=dtype
        )
        self.reset_parameters()

    @staticmethod
    def vector_rejection(vec: Tensor, d_ij: Tensor) -> Tensor:
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self) -> None:
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "s_proj"):
            nn.init.xavier_uniform_(getattr(self, name).weight)
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "s_proj",
            "dk_proj",
            "dv_proj",
        ):
            with torch.no_grad():
                getattr(self, name).bias.zero_()
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight)
            nn.init.xavier_uniform_(self.w_src_proj.weight)
            nn.init.xavier_uniform_(self.w_trg_proj.weight)
            with torch.no_grad():
                self.f_proj.bias.zero_()
        for name in ("vec_proj", "dk_proj", "dv_proj"):
            nn.init.xavier_uniform_(getattr(self, name).weight)
        for name in ("dk_proj", "dv_proj"):
            with torch.no_grad():
                getattr(self, name).bias.zero_()

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        value = (v_j * dv * attn.unsqueeze(2)).reshape(-1, self.hidden_channels)
        s1, s2 = torch.split(
            self.act(self.s_proj(value)), self.hidden_channels, dim=1
        )
        vector = (
            vec_j * s1.unsqueeze(1)
            + s2.unsqueeze(1) * d_ij.unsqueeze(2)
        )
        return value, vector

    def aggregate(
        self,
        features: tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        del ptr
        x, vec = features
        size = (
            int(dim_size)
            if dim_size is not None
            else int(index.max().item()) + 1
            if index.numel()
            else 0
        )
        return _aggregate(x, index, size), _aggregate(vec, index, size)

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        return self.act(self.f_proj(f_ij)) * w_dot

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x_norm = self.layernorm(x)
        vec_norm = self.vec_layernorm(vec)
        q = self.q_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        value = self.v_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec_norm), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)
        source, target = edge_index[0], edge_index[1]
        if source.numel():
            messages = self.message(
                q.index_select(0, target),
                k.index_select(0, source),
                value.index_select(0, source),
                vec_norm.index_select(0, source),
                dk,
                dv,
                r_ij,
                d_ij,
            )
            x_out, vec_out = self.aggregate(
                messages, target, dim_size=x.shape[0]
            )
        else:
            x_out = x.new_zeros((x.shape[0], self.hidden_channels))
            vec_out = vec.new_zeros(vec.shape)
        o1, o2, o3 = torch.split(
            self.o_proj(x_out), self.hidden_channels, dim=1
        )
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if self.last_layer:
            return dx, dvec, None
        df_ij = self.edge_update(
            vec_norm.index_select(0, target),
            vec_norm.index_select(0, source),
            d_ij,
            f_ij,
        )
        return dx, dvec, df_ij


class V2ViSMPVertexEdge(V2ViSMP):
    """Reference vertex-edge ViS-MP variant."""

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str | None,
        trainable_vecnorm: bool,
        last_layer: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_heads,
            hidden_channels,
            cutoff,
            vecnorm_type,
            trainable_vecnorm,
            last_layer,
            dtype=dtype,
        )
        if not self.last_layer:
            # The primary AI2BMD source replaces these modules after the
            # base reset call. The block reset repeats the base initialization.
            self.f_proj = nn.Linear(
                self.hidden_channels, self.hidden_channels * 2, dtype=dtype
            )
            self.t_src_proj = nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
            )
            self.t_trg_proj = nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
                bias=False,
                dtype=dtype,
            )

    def edge_update(
        self,
        vec_i: Tensor,
        vec_j: Tensor,
        d_ij: Tensor,
        f_ij: Tensor,
    ) -> Tensor:
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        f1, f2 = torch.split(
            self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1
        )
        return f1 * w_dot + f2 * t_dot


class V2ViSMPVertexNode(V2ViSMP):
    """Reference vertex-node ViS-MP variant."""

    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: str | None,
        trainable_vecnorm: bool,
        last_layer: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(
            num_heads,
            hidden_channels,
            cutoff,
            vecnorm_type,
            trainable_vecnorm,
            last_layer,
            dtype=dtype,
        )
        self.t_src_proj = nn.Linear(
            self.hidden_channels,
            self.hidden_channels,
            bias=False,
            dtype=dtype,
        )
        self.t_trg_proj = nn.Linear(
            self.hidden_channels,
            self.hidden_channels,
            bias=False,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.hidden_channels, self.hidden_channels * 4, dtype=dtype
        )

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_i: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        value = (v_j * dv * attn.unsqueeze(2)).reshape(-1, self.hidden_channels)
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        s1, s2 = torch.split(
            self.act(self.s_proj(value)), self.hidden_channels, dim=1
        )
        vector = (
            vec_j * s1.unsqueeze(1)
            + s2.unsqueeze(1) * d_ij.unsqueeze(2)
        )
        return value, vector, t_dot

    def aggregate(
        self,
        features: tuple[Tensor, Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del ptr
        x, vec, t_dot = features
        size = (
            int(dim_size)
            if dim_size is not None
            else int(index.max().item()) + 1
            if index.numel()
            else 0
        )
        return (
            _aggregate(x, index, size),
            _aggregate(vec, index, size),
            _aggregate(t_dot, index, size),
        )

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        x_norm = self.layernorm(x)
        vec_norm = self.vec_layernorm(vec)
        q = self.q_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        value = self.v_proj(x_norm).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        dv = self.act(self.dv_proj(f_ij)).reshape(
            -1, self.num_heads, self.head_dim
        )
        vec1, vec2, vec3 = torch.split(
            self.vec_proj(vec_norm), self.hidden_channels, dim=-1
        )
        vec_dot = (vec1 * vec2).sum(dim=1)
        source, target = edge_index[0], edge_index[1]
        if source.numel():
            messages = self.message(
                q.index_select(0, target),
                k.index_select(0, source),
                value.index_select(0, source),
                vec_norm.index_select(0, target),
                vec_norm.index_select(0, source),
                dk,
                dv,
                r_ij,
                d_ij,
            )
            x_out, vec_out, t_dot = self.aggregate(
                messages, target, dim_size=x.shape[0]
            )
        else:
            x_out = x.new_zeros((x.shape[0], self.hidden_channels))
            vec_out = vec.new_zeros(vec.shape)
            t_dot = x.new_zeros((x.shape[0], self.hidden_channels))
        o1, o2, o3, o4 = torch.split(
            self.o_proj(x_out), self.hidden_channels, dim=1
        )
        dx = vec_dot * o2 + t_dot * o3 + o4
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if self.last_layer:
            return dx, dvec, None
        df_ij = self.edge_update(
            vec_norm.index_select(0, target),
            vec_norm.index_select(0, source),
            d_ij,
            f_ij,
        )
        return dx, dvec, df_ij


_V2_MP_CLASSES = {
    "none": V2ViSMP,
    "edge": V2ViSMPVertexEdge,
    "node": V2ViSMPVertexNode,
}


class V2ViSNetBlock(nn.Module):
    """Reference ViSNet representation block with MolViD extensions isolated."""

    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: str | None = "max_min",
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        max_z: int = 100,
        max_b: int = NUM_BLOCK_TYPE,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex_type: str = "edge",
        *,
        use_block_embedding: bool = True,
        use_bond_embedding: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if int(lmax) not in {1, 2}:
            raise ValueError("lmax must be 1 or 2")
        if int(num_layers) < 1 or int(hidden_channels) < 1:
            raise ValueError("num_layers and hidden_channels must be positive")
        if int(num_heads) < 1 or int(hidden_channels) % int(num_heads) != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        resolved_rbf = str(rbf_type).lower()
        if resolved_rbf not in {"expnorm", "gauss"}:
            raise ValueError("rbf_type must be 'expnorm' or 'gauss'")
        resolved_vertex = str(vertex_type).lower()
        if resolved_vertex not in _V2_MP_CLASSES:
            raise ValueError("vertex_type must be one of 'none', 'edge', or 'node'")
        resolved_vecnorm = (
            None if vecnorm_type in {None, "none"} else str(vecnorm_type).lower()
        )
        if resolved_vecnorm not in {None, "rms", "max_min"}:
            raise ValueError(
                "vecnorm_type must be None, 'none', 'rms', or 'max_min'"
            )
        self.lmax = int(lmax)
        self.vecnorm_type = resolved_vecnorm
        self.trainable_vecnorm = bool(trainable_vecnorm)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.hidden_channels = int(hidden_channels)
        self.num_rbf = int(num_rbf)
        self.rbf_type = resolved_rbf
        self.trainable_rbf = bool(trainable_rbf)
        self.max_z = int(max_z)
        self.max_b = int(max_b)
        self.cutoff = float(cutoff)
        self.max_num_neighbors = int(max_num_neighbors)
        self.vertex_type = resolved_vertex
        self.use_block_embedding = bool(use_block_embedding)
        self.use_bond_embedding = bool(use_bond_embedding)
        self.dtype = dtype
        self.sphere = V2Sphere(self.lmax)
        self.atom_embedding = nn.Embedding(
            self.max_z, self.hidden_channels, dtype=dtype
        )
        self.block_embedding = nn.Embedding(
            self.max_b, self.hidden_channels, dtype=dtype
        )
        if self.use_bond_embedding:
            self.bond_embedding = nn.Embedding(
                2, self.num_rbf, dtype=dtype
            )
        else:
            self.bond_embedding = None
        rbf_cls = (
            V2ExpNormalSmearing
            if self.rbf_type == "expnorm"
            else V2GaussianSmearing
        )
        self.distance_expansion = rbf_cls(
            self.cutoff, self.num_rbf, self.trainable_rbf, dtype=dtype
        )
        self.neighbor_embedding = V2NeighborEmbedding(
            self.hidden_channels,
            self.num_rbf,
            self.cutoff,
            self.max_z,
            dtype=dtype,
        )
        self.edge_embedding = V2EdgeEmbedding(
            self.num_rbf, self.hidden_channels, dtype=dtype
        )
        mp_class = _V2_MP_CLASSES[self.vertex_type]
        self.vis_mp_layers = nn.ModuleList(
            mp_class(
                self.num_heads,
                self.hidden_channels,
                self.cutoff,
                self.vecnorm_type,
                self.trainable_vecnorm,
                last_layer=layer_index == self.num_layers - 1,
                dtype=dtype,
            )
            for layer_index in range(self.num_layers)
        )
        self.out_norm = nn.LayerNorm(self.hidden_channels, dtype=dtype)
        self.vec_out_norm = V2VecLayerNorm(
            self.hidden_channels,
            trainable=self.trainable_vecnorm,
            norm_type=self.vecnorm_type,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.atom_embedding.reset_parameters()
        self.block_embedding.reset_parameters()
        if self.bond_embedding is not None:
            self.bond_embedding.reset_parameters()
        self.distance_expansion.reset_parameters()
        self.neighbor_embedding.reset_parameters()
        self.edge_embedding.reset_parameters()
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def _initial_nodes(self, z: Tensor, b: Optional[Tensor]) -> Tensor:
        x = self.atom_embedding(z)
        if self.use_block_embedding:
            if b is None:
                raise ValueError(
                    "block type tensor b is required when block embedding is enabled"
                )
            x = x + self.block_embedding(b)
        return x

    def _radial_features(
        self, edge_weight: Tensor, bond_type: Optional[Tensor]
    ) -> Tensor:
        radial = self.distance_expansion(edge_weight.to(dtype=self.dtype))
        if self.use_bond_embedding:
            if self.bond_embedding is None:
                raise RuntimeError("bond embedding was not initialized")
            if bond_type is None:
                bond_type = torch.zeros(
                    edge_weight.shape,
                    dtype=torch.long,
                    device=edge_weight.device,
                )
            bond_type = bond_type.to(
                device=radial.device, dtype=torch.long
            ).flatten()
            if bond_type.numel() != radial.shape[0]:
                raise ValueError("bond_type must contain one entry per edge")
            invalid = (bond_type < 0) | (bond_type > 1)
            if invalid.device.type == "cuda":
                torch._assert_async(
                    ~invalid.any(),
                    "bond_type must contain only binary values",
                )
            elif bool(invalid.any()):
                raise ValueError("bond_type must contain only binary values")
            radial = radial + self.bond_embedding(bond_type)
        return radial

    @staticmethod
    def _normalise_directions(edge_vec: Tensor) -> Tensor:
        lengths = torch.linalg.vector_norm(edge_vec, dim=-1)
        non_self = lengths > 0
        safe = edge_vec / lengths.clamp_min(1.0e-12).unsqueeze(-1)
        return torch.where(
            non_self.unsqueeze(-1), safe, torch.zeros_like(safe)
        )

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        *,
        b: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        edge_vec: Optional[Tensor] = None,
        bond_type: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        del pos, batch
        if edge_index is None or edge_weight is None or edge_vec is None:
            raise ValueError("V2ViSNetBlock requires an explicit graph adapter")
        x = self._initial_nodes(z, b)
        edge_attr = self._radial_features(edge_weight, bond_type)
        edge_direction = self._normalise_directions(edge_vec)
        spherical = self.sphere(edge_direction)
        vec = x.new_zeros(
            (
                x.shape[0],
                ((self.lmax + 1) ** 2) - 1,
                self.hidden_channels,
            )
        )
        x = self.neighbor_embedding(
            z, x, edge_index, edge_weight, edge_attr
        )
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)
        for layer in self.vis_mp_layers:
            dx, dvec, dedge = layer(
                x, vec, edge_index, edge_weight, edge_attr, spherical
            )
            x = x + dx
            vec = vec + dvec
            if dedge is not None:
                edge_attr = edge_attr + dedge
        return self.out_norm(x), self.vec_out_norm(vec)


class V2PublicL1Adapter(nn.Module):
    """Export only the first three l=1 components after final normalization."""

    def forward(self, v_full: Tensor) -> Tensor:
        if v_full.ndim != 3 or v_full.shape[1] not in {3, 8}:
            raise ValueError(
                "internal vector state must have shape [M, 3, C] or [M, 8, C]"
            )
        return v_full[:, :3]


class V2SpatialEncoder(nn.Module):
    """MolViD adapter for the faithful representation block."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        num_rbf: int = 32,
        lmax: int = 1,
        vertex_type: str = "edge",
        cutoff_lower: float = 0.0,
        cutoff_upper: float = 5.0,
        max_num_neighbors: int = 32,
        rbf_type: str = "expnorm",
        trainable_rbf: bool = False,
        vecnorm_type: str | None = "max_min",
        trainable_vecnorm: bool = False,
        *,
        max_z: int = NUM_ATOM_TYPE,
        max_b: int = NUM_BLOCK_TYPE,
        neighbor_backend: str = "cuda_radius",
        dtype: torch.dtype = torch.float32,
        use_block_embedding: bool = True,
        use_bond_embedding: bool = False,
    ) -> None:
        super().__init__()
        if float(cutoff_lower) < 0 or float(cutoff_upper) <= float(cutoff_lower):
            raise ValueError("invalid ViSNet cutoffs")
        if int(max_num_neighbors) < 1:
            raise ValueError("max_num_neighbors must be positive")
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_rbf = int(num_rbf)
        self.lmax = int(lmax)
        self.vertex_type = str(vertex_type).lower()
        self.cutoff_lower = float(cutoff_lower)
        self.cutoff_upper = float(cutoff_upper)
        self.max_num_neighbors = int(max_num_neighbors)
        self.rbf_type = str(rbf_type).lower()
        self.trainable_rbf = bool(trainable_rbf)
        self.vecnorm_type = (
            None if vecnorm_type in {None, "none"} else str(vecnorm_type).lower()
        )
        self.trainable_vecnorm = bool(trainable_vecnorm)
        self.max_z = int(max_z)
        self.max_b = int(max_b)
        self.neighbor_backend = str(neighbor_backend)
        self.dtype = dtype
        self.use_block_embedding = bool(use_block_embedding)
        self.use_bond_embedding = bool(use_bond_embedding)
        self.representation = V2ViSNetBlock(
            lmax=self.lmax,
            vecnorm_type=self.vecnorm_type,
            trainable_vecnorm=self.trainable_vecnorm,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            hidden_channels=self.hidden_channels,
            num_rbf=self.num_rbf,
            rbf_type=self.rbf_type,
            trainable_rbf=self.trainable_rbf,
            max_z=self.max_z,
            max_b=self.max_b,
            cutoff=self.cutoff_upper,
            max_num_neighbors=self.max_num_neighbors,
            vertex_type=self.vertex_type,
            use_block_embedding=self.use_block_embedding,
            use_bond_embedding=self.use_bond_embedding,
            dtype=dtype,
        )
        self.public_l1 = V2PublicL1Adapter()
        self.native_neighbor_builder = make_neighbor_list(
            self.neighbor_backend,
            cutoff_lower=self.cutoff_lower,
            cutoff_upper=self.cutoff_upper,
            max_num_neighbors=self.max_num_neighbors,
            loop=True,
        )

    @staticmethod
    def _node_fields(nodes: Any) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        try:
            z, b, pos, graph_id = nodes.z, nodes.b, nodes.pos, nodes.graph_id
            graph_count = int(nodes.graph_count)
        except AttributeError as exc:
            raise TypeError(
                "native/external v2 calls require a FrameNodeBatch-like object"
            ) from exc
        return z, b, pos, graph_id, graph_count

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
        bond_type: Optional[Tensor],
        backend_used: str,
        graph_mode: str,
    ) -> SpatialEncoderOutput:
        del graph_id
        if pos.dtype != torch.float32:
            raise RuntimeError(
                "ViSNet geometry requires original FP32 coordinates"
            )
        if (
            pos.ndim != 2
            or pos.shape[-1] != 3
            or z.ndim != 1
            or b.ndim != 1
        ):
            raise ValueError(
                "v2 node fields must have shapes [M], [M], and [M, 3]"
            )
        if z.shape[0] != pos.shape[0] or b.shape[0] != pos.shape[0]:
            raise ValueError("v2 node fields must share the atom axis")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("v2 edge_index must have shape [2, E]")
        if (
            edge_weight.ndim != 1
            or edge_vec.ndim != 2
            or edge_vec.shape[-1] != 3
        ):
            raise ValueError(
                "v2 edge geometry must have shapes [E] and [E, 3]"
            )
        if (
            edge_index.shape[1] != edge_weight.shape[0]
            or edge_index.shape[1] != edge_vec.shape[0]
        ):
            raise ValueError("v2 edge fields must share the edge axis")
        h, v_full = self.representation(
            z,
            pos,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
            b=b,
            bond_type=bond_type,
        )
        v = self.public_l1(v_full)
        return SpatialEncoderOutput(
            h=h,
            v=v,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_vec=edge_vec,
            edge_type=bond_type,
            backend_used=str(backend_used),
            graph_mode=str(graph_mode),
            v_full=v_full,
        )

    def forward_native(self, nodes: Any) -> SpatialEncoderOutput:
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
            bond_type=None,
            backend_used=neighbors.backend_used,
            graph_mode="native_radius",
        )

    def forward_external(self, nodes: Any, graph: Any) -> SpatialEncoderOutput:
        z, b, pos, graph_id, _ = self._node_fields(nodes)
        return self._represent(
            z=z,
            b=b,
            pos=pos,
            graph_id=graph_id,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            edge_vec=graph.edge_vec,
            bond_type=graph.bond_type if self.use_bond_embedding else None,
            backend_used=graph.backend_used,
            graph_mode="external",
        )

    def forward(
        self,
        z: Tensor,
        b: Tensor,
        pos: Tensor,
        batch: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        edge_vec: Optional[Tensor] = None,
        bond_type: Optional[Tensor] = None,
        *,
        edge_weight_t: Optional[Tensor] = None,
        edge_vec_t: Optional[Tensor] = None,
    ) -> SpatialEncoderOutput:
        if edge_weight is None:
            edge_weight = edge_weight_t
        if edge_vec is None:
            edge_vec = edge_vec_t
        if edge_weight is None or edge_vec is None:
            raise ValueError(
                "v2 direct forward requires an explicit external graph"
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
                "graph_count": int(batch.max().item()) + 1
                if batch.numel()
                else 1,
            },
        )()
        return self.forward_external(nodes, graph)


__all__ = [
    "V2_REFERENCE_PROVENANCE",
    "V2CosineCutoff",
    "V2ExpNormalSmearing",
    "V2GaussianSmearing",
    "V2Sphere",
    "V2VecLayerNorm",
    "V2NeighborEmbedding",
    "V2EdgeEmbedding",
    "V2ViSMP",
    "V2ViSMPVertexEdge",
    "V2ViSMPVertexNode",
    "V2ViSNetBlock",
    "V2PublicL1Adapter",
    "V2SpatialEncoder",
]
