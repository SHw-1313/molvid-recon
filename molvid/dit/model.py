"""Current molecular DiT over structured state/detail latent batches."""

from __future__ import annotations

from typing import Mapping, Optional

import torch
from torch import Tensor, nn

from ..latent.adapter import StateDetailLatentAdapter
from ..latent.types import DIT_MODEL_SCHEMA, LatentBatch, LatentFields, contract_hash
from .backend import FactorizedLayout, _build_block_groups, build_factorized_layout, factorized_block_forward
from .blocks import FactorizedDiTBlock

def _safe_ids(value: Tensor, size: int) -> Tensor:
    return value.to(dtype=torch.long).remainder(int(size))


def _sample_token_values(value: Tensor, abid: Tensor) -> Tensor:
    return value.index_select(0, abid).transpose(0, 1)


class MolecularDiT(nn.Module):
    """One shared R2/R4 factorized scalar/vector DiT implementation."""

    def __init__(
        self,
        *,
        adapter: StateDetailLatentAdapter,
        scalar_width: int = 256,
        vector_width: int = 128,
        depth: int = 4,
        heads: int = 8,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        max_atom_type: int = 128,
        max_block_type: int = 256,
        max_component_type: int = 128,
        execution_backend: str = "reference",
        ffn_norm_source: str = "post_adaln",
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.depth = int(depth)
        self.heads = int(heads)
        self.ffn_multiplier = int(ffn_multiplier)
        self.dropout = float(dropout)
        if ffn_norm_source not in ("post_adaln", "pre_adaln"):
            raise ValueError("ffn_norm_source must be 'post_adaln' or 'pre_adaln'")
        self.ffn_norm_source = str(ffn_norm_source)
        if execution_backend not in ("reference", "factorized_v2"):
            raise ValueError("execution_backend must be 'reference' or 'factorized_v2'")
        if execution_backend == "factorized_v2" and self.dropout != 0.0:
            raise ValueError("factorized_v2 requires dropout=0 for shared-mask parity")
        self.execution_backend = str(execution_backend)
        self._factorized_layout: Optional[FactorizedLayout] = None
        if min(self.scalar_width, self.vector_width, self.depth, self.heads) < 1:
            raise ValueError("model widths, depth, and heads must be positive")
        if self.scalar_width % self.heads or self.vector_width % self.heads:
            raise ValueError("both model widths must divide heads")
        self.flow_time = nn.Sequential(
            nn.Linear(3, self.scalar_width),
            nn.SiLU(),
            nn.Linear(self.scalar_width, self.scalar_width),
        )
        self.ratio_embedding = nn.Embedding(2, self.scalar_width)
        self.observed_embedding = nn.Embedding(2, self.scalar_width)
        self.atom_embedding = nn.Embedding(int(max_atom_type), self.scalar_width)
        self.block_embedding = nn.Embedding(int(max_block_type), self.scalar_width)
        self.component_embedding = nn.Embedding(int(max_component_type), self.scalar_width)
        self.blocks = nn.ModuleList(
            [
                FactorizedDiTBlock(
                    self.scalar_width,
                    self.vector_width,
                    self.heads,
                    self.ffn_multiplier,
                    self.dropout,
                    self.ffn_norm_source,
                )
                for _ in range(self.depth)
            ]
        )
        self.backend = "dense_block_attention_factorized_atom_temporal_v1"

    def _condition(self, batch: LatentBatch, tau: Tensor) -> Tensor:
        valid_frames = batch.block_frame_mask
        count = valid_frames.sum(dim=-1).clamp_min(1).to(batch.block_time_ps.dtype)
        block_time = (batch.block_time_ps * valid_frames.to(batch.block_time_ps.dtype)).sum(dim=-1) / count
        min_time = batch.block_time_ps.masked_fill(~valid_frames, float("inf")).amin(dim=-1)
        max_time = batch.block_time_ps.masked_fill(~valid_frames, float("-inf")).amax(dim=-1)
        span = torch.where(valid_frames.any(dim=-1), max_time - min_time, torch.zeros_like(max_time))
        time = _sample_token_values(block_time, batch.abid)
        span = _sample_token_values(span, batch.abid)
        flow = _sample_token_values(
            torch.as_tensor(tau, device=batch.state_h.device, dtype=batch.state_h.dtype).reshape(-1, 1).expand(
                batch.batch_size, batch.tokens
            ),
            batch.abid,
        )
        numeric = torch.stack((flow, time * 0.01, span * 0.01), dim=-1)
        condition = self.flow_time(numeric)
        condition = condition + self.ratio_embedding(
            torch.full_like(batch.abid, 0 if batch.ratio == 2 else 1)
        ).unsqueeze(0)
        observed = batch.observed_mask.index_select(0, batch.abid).transpose(0, 1).long()
        condition = condition + self.observed_embedding(observed)
        condition = condition + self.atom_embedding(_safe_ids(batch.atom_type, self.atom_embedding.num_embeddings)).unsqueeze(0)
        condition = condition + self.block_embedding(_safe_ids(batch.block_type, self.block_embedding.num_embeddings)).unsqueeze(0)
        condition = condition + self.component_embedding(_safe_ids(batch.component_id, self.component_embedding.num_embeddings)).unsqueeze(0)
        return condition
    def _layout_for(self, batch: LatentBatch) -> FactorizedLayout:
        layout = self._factorized_layout
        if layout is None or not layout.matches(batch):
            layout = build_factorized_layout(batch)
            self._factorized_layout = layout
        return layout


    def forward(self, batch: LatentBatch, tau: Tensor | float) -> LatentFields:
        if batch.ratio != self.adapter.ratio or batch.mode != self.adapter.mode:
            raise ValueError("model and batch ratio/mode disagree")
        h, v = self.adapter.project_inputs(batch)
        condition = self._condition(batch, torch.as_tensor(tau, device=h.device, dtype=h.dtype))
        h = h + condition
        if self.execution_backend == "reference":
            groups, group_samples = _build_block_groups(batch)
            for block in self.blocks:
                h, v = block(h, v, condition, batch, groups, group_samples)
        else:
            layout = self._layout_for(batch)
            for block in self.blocks:
                h, v = factorized_block_forward(
                    block, h, v, condition, batch, layout
                )
        output = self.adapter.project_outputs(h, v)
        return _masked_output(output, batch)

    def contract(self) -> dict[str, object]:
        return {
            "schema_version": DIT_MODEL_SCHEMA,
            "adapter_schema": self.adapter_contract_hash,
            "ratio": self.adapter.ratio,
            "mode": self.adapter.mode,
            "scalar_width": self.scalar_width,
            "vector_width": self.vector_width,
            "depth": self.depth,
            "heads": self.heads,
            "ffn_multiplier": self.ffn_multiplier,
            "ffn_norm_source": self.ffn_norm_source,
            "dropout": self.dropout,
            "backend": self.backend,
            "attention_complexity": "sum_s K*M_s^2 + sum_n K^2",
            "dense_all_atom_time_attention": False,
            "temporal_attention": "bidirectional",
            "vector_maps": "bias_free_channel_only",
            "vector_adaln": "scale_and_gate_only",
            "vector_normalization": "so3_xyz_contracted_rms_fp32",
            "ffn_interaction": "vector_norms_to_scalar_and_scalar_gated_vector",
            "ffn_scalar_norm_source": self.ffn_norm_source,
        }

    @property
    def adapter_contract_hash(self) -> str:
        return contract_hash(self.adapter.contract())

    @property
    def model_hash(self) -> str:
        return contract_hash(self.contract())
    @property
    def semantic_contract_hash(self) -> str:
        return self.model_hash

    def execution_contract(self) -> dict[str, str]:
        return {
            "execution_backend": self.execution_backend,
            "semantic_contract_hash": self.semantic_contract_hash,
        }


    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_dit(
    model_config: Mapping[str, object],
    adapter: StateDetailLatentAdapter,
    *,
    ffn_norm_source: str,
) -> MolecularDiT:
    """Build the configured DiT without changing its constructor or RNG order."""
    return MolecularDiT(
        adapter=adapter,
        scalar_width=int(model_config["scalar_width"]),
        vector_width=int(model_config["vector_width"]),
        depth=int(model_config["depth"]),
        heads=int(model_config["heads"]),
        ffn_multiplier=int(model_config["ffn_multiplier"]),
        dropout=float(model_config["dropout"]),
        execution_backend=str(model_config["execution_backend"]),
        ffn_norm_source=ffn_norm_source,
    )


def _masked_output(fields: LatentFields, batch: LatentBatch) -> LatentFields:
    masks = batch.field_masks()
    return LatentFields(
        fields.state_h * masks["state_h"].to(fields.state_h.dtype).unsqueeze(-1),
        fields.detail_h * masks["detail_h"].to(fields.detail_h.dtype).unsqueeze(-1),
        fields.state_v * masks["state_v"].to(fields.state_v.dtype).unsqueeze(-1).unsqueeze(-1),
        fields.detail_v * masks["detail_v"].to(fields.detail_v.dtype).unsqueeze(-1).unsqueeze(-1),
    )
