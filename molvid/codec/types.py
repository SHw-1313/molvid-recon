"""Versioned coordinate-independent codec latent and output contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from torch import Tensor

from ..geometry.types import StaticTopologyMetadata


STATE_DETAIL_CODEC_SCHEMA = "pvb.codec.state_detail.latent.v2"
STATE_DETAIL_MODES = (
    "ratio1_state_detail",
    "ratio2_state_detail",
    "ratio4_state_detail",
    "ratio4_matched_pooling",
)
ORIGIN_RULE = "frame0_loss_masked_centroid"
RATIO_FOR_MODE = {
    "ratio1_state_detail": 1,
    "ratio2_state_detail": 2,
    "ratio4_state_detail": 4,
    "ratio4_matched_pooling": 4,
}

@dataclass
class StateDetailLatent:
    """Versioned structured latent for the R1/R2/R4 state/detail controls."""

    state_h: Tensor
    state_v: Tensor
    detail_h: Optional[Tensor]
    detail_v: Optional[Tensor]
    raw_detail_h: Optional[Tensor]
    raw_detail_v: Optional[Tensor]
    token_mask: Tensor
    detail_valid: Tensor
    detail_component_mask: Tensor
    block_frame_mask: Tensor
    frame_time_ps: Tensor
    block_time_ps: Tensor
    abid: Tensor
    sample_origin: Tensor
    topology: Any
    ratio: int
    mode: str
    coefficient_order: tuple[str, ...]
    width: int
    origin_rule: str = ORIGIN_RULE
    schema_version: str = STATE_DETAIL_CODEC_SCHEMA

    @property
    def frames(self) -> int:
        return int(self.frame_time_ps.shape[1])

    @property
    def tokens(self) -> int:
        return int(self.state_h.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.block_frame_mask.shape[0])

    @property
    def active_feature_elements_per_atom(self) -> int:
        return int(self.tokens * self.width * (1 if self.detail_h is None else 2))

    @property
    def active_feature_volume(self) -> int:
        return int(self.active_feature_elements_per_atom)

    @property
    def bank_widths(self) -> dict[str, int]:
        return {"state": self.width} | ({"detail": self.width} if self.detail_h is not None else {})

    def contract(self) -> dict[str, Any]:
        topology_contract = (
            self.topology.contract()
            if isinstance(self.topology, StaticTopologyMetadata)
            else None
        )
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "ratio": int(self.ratio),
            "width": int(self.width),
            "coefficient_order": list(self.coefficient_order),
            "state_shape": list(self.state_h.shape),
            "state_vector_shape": list(self.state_v.shape),
            "detail_shape": None if self.detail_h is None else list(self.detail_h.shape),
            "detail_vector_shape": None if self.detail_v is None else list(self.detail_v.shape),
            "token_mask_shape": list(self.token_mask.shape),
            "detail_valid_shape": list(self.detail_valid.shape),
            "detail_component_mask_shape": list(self.detail_component_mask.shape),
            "block_frame_mask_shape": list(self.block_frame_mask.shape),
            "frame_time_ps_shape": list(self.frame_time_ps.shape),
            "block_time_ps_shape": list(self.block_time_ps.shape),
            "abid_shape": list(self.abid.shape),
            "sample_origin_shape": list(self.sample_origin.shape),
            "bank_widths": self.bank_widths,
            "active_feature_volume_per_atom": self.active_feature_volume,
            "origin_rule": self.origin_rule,
            "has_per_atom_anchor": False,
            "has_target_coordinates": False,
            "topology": topology_contract,
        }


@dataclass
class MatchedPoolingLatent:
    """Unstructured two-bank latent for the capacity-matched R4 control."""

    bank_a_h: Tensor
    bank_a_v: Tensor
    bank_b_h: Tensor
    bank_b_v: Tensor
    token_mask: Tensor
    block_frame_mask: Tensor
    frame_time_ps: Tensor
    block_time_ps: Tensor
    abid: Tensor
    sample_origin: Tensor
    topology: Any
    ratio: int = 4
    mode: str = "ratio4_matched_pooling"
    width: int = 0
    origin_rule: str = ORIGIN_RULE
    schema_version: str = STATE_DETAIL_CODEC_SCHEMA

    @property
    def tokens(self) -> int:
        return int(self.bank_a_h.shape[0])

    @property
    def active_feature_volume(self) -> int:
        return int(self.tokens * self.width * 2)

    def contract(self) -> dict[str, Any]:
        topology_contract = (
            self.topology.contract()
            if isinstance(self.topology, StaticTopologyMetadata)
            else None
        )
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "ratio": self.ratio,
            "width": int(self.width),
            "bank_shapes": {
                "a_h": list(self.bank_a_h.shape),
                "a_v": list(self.bank_a_v.shape),
                "b_h": list(self.bank_b_h.shape),
                "b_v": list(self.bank_b_v.shape),
            },
            "token_mask_shape": list(self.token_mask.shape),
            "block_frame_mask_shape": list(self.block_frame_mask.shape),
            "frame_time_ps_shape": list(self.frame_time_ps.shape),
            "block_time_ps_shape": list(self.block_time_ps.shape),
            "abid_shape": list(self.abid.shape),
            "sample_origin_shape": list(self.sample_origin.shape),
            "bank_widths": {"bank_a": self.width, "bank_b": self.width},
            "active_feature_volume_per_atom": self.active_feature_volume,
            "origin_rule": self.origin_rule,
            "has_per_atom_anchor": False,
            "has_target_coordinates": False,
            "state_detail_semantics": False,
            "pooling_semantics": "linear_two_bank_pooling",
            "topology": topology_contract,
        }


@dataclass
class CodecOutput:
    x_coarse: Tensor
    x_hat: Tensor
    h: Tensor
    v: Tensor
    frame_mask: Tensor
    time_ps: Tensor
    latent: StateDetailLatent | MatchedPoolingLatent
    decoded_detail_h: Optional[Tensor] = None
    decoded_detail_v: Optional[Tensor] = None
    diagnostics: Mapping[str, Tensor] | None = None

    @property
    def coordinates(self) -> Tensor:
        return self.x_hat

