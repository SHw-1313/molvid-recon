"""Project frozen codec latent banks into and out of the DiT feature widths."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn

from ..codec.types import RATIO_FOR_MODE, STATE_DETAIL_CODEC_SCHEMA, StateDetailLatent
from ..equivariant import AxisPreservingLinear
from .types import (
    DIT_ADAPTER_SCHEMA,
    FRAME_COUNT,
    SUPPORTED_DIT_MODES,
    SUPPORTED_RATIOS,
    LatentBatch,
    LatentFields,
)

class StateDetailLatentAdapter(nn.Module):
    """Pack frozen codec latents and project them into DiT scalar/vector widths."""

    def __init__(
        self,
        codec_width: int = 128,
        scalar_width: int = 256,
        vector_width: int = 128,
        *,
        ratio: int,
        mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.codec_width = int(codec_width)
        self.scalar_width = int(scalar_width)
        self.vector_width = int(vector_width)
        self.ratio = int(ratio)
        self.mode = mode or f"ratio{self.ratio}_state_detail"
        if self.ratio not in SUPPORTED_RATIOS or self.mode not in SUPPORTED_DIT_MODES:
            raise ValueError("the DiT adapter supports only ratio2_state_detail and ratio4_state_detail")
        if RATIO_FOR_MODE.get(self.mode) != self.ratio:
            raise ValueError("ratio and mode disagree")
        if min(self.codec_width, self.scalar_width, self.vector_width) < 1:
            raise ValueError("all adapter widths must be positive")
        packed_width = 2 * self.codec_width
        self.scalar_in = nn.Linear(packed_width, self.scalar_width)
        self.vector_in = AxisPreservingLinear(packed_width, self.vector_width)
        self.scalar_out = nn.ModuleDict({
            "state_h": nn.Linear(self.scalar_width, self.codec_width),
            "detail_h": nn.Linear(self.scalar_width, self.codec_width),
        })
        self.vector_out = nn.ModuleDict({
            "state_v": AxisPreservingLinear(self.vector_width, self.codec_width),
            "detail_v": AxisPreservingLinear(self.vector_width, self.codec_width),
        })

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": DIT_ADAPTER_SCHEMA,
            "codec_schema": STATE_DETAIL_CODEC_SCHEMA,
            "ratio": self.ratio,
            "mode": self.mode,
            "codec_width": self.codec_width,
            "scalar_width": self.scalar_width,
            "vector_width": self.vector_width,
            "input_packing": "concat_state_detail_channel",
            "state_detail_token_count": FRAME_COUNT // self.ratio,
            "vector_maps": "bias_free_channel_only",
            "contains_raw_detail": False,
            "contains_target_coordinates": False,
        }

    def validate_latent(self, latent: StateDetailLatent) -> None:
        if not isinstance(latent, StateDetailLatent):
            raise TypeError("expected a StateDetailLatent from the frozen state/detail codec")
        if latent.mode != self.mode or int(latent.ratio) != self.ratio:
            raise ValueError(f"latent mode/ratio must be {self.mode}/{self.ratio}")
        if latent.frames != FRAME_COUNT:
            raise ValueError(f"DiT v1 requires T={FRAME_COUNT}")
        k_expected = FRAME_COUNT // self.ratio
        if latent.state_h.shape[0] != k_expected or latent.state_v.shape[0] != k_expected:
            raise ValueError("latent K does not match T/ratio")
        if latent.detail_h is None or latent.detail_v is None:
            raise ValueError("R2/R4 generated latents require state and detail fields")
        if latent.state_h.shape[-1] != self.codec_width or latent.detail_h.shape[-1] != self.codec_width:
            raise ValueError("latent codec width does not match the adapter")

    def from_codec_latent(
        self,
        latent: StateDetailLatent,
        *,
        codec_hash: str = "",
        data_hash: str = "",
        origin_from_latent: bool = False,
        loss_mask: Optional[Tensor] = None,
    ) -> LatentBatch:
        self.validate_latent(latent)
        batch_size = int(latent.block_frame_mask.shape[0])
        origin = latent.sample_origin if origin_from_latent else latent.sample_origin.new_zeros((batch_size, 3))
        fields = LatentFields(latent.state_h, latent.detail_h, latent.state_v, latent.detail_v)
        return LatentBatch(
            fields=fields,
            token_mask=latent.token_mask,
            detail_valid=latent.detail_valid,
            detail_component_mask=latent.detail_component_mask,
            block_frame_mask=latent.block_frame_mask,
            block_time_ps=latent.block_time_ps,
            frame_time_ps=latent.frame_time_ps,
            abid=latent.abid,
            sample_origin=origin,
            topology=latent.topology,
            ratio=self.ratio,
            mode=self.mode,
            width=self.codec_width,
            loss_mask=loss_mask,
            codec_hash=str(codec_hash),
            data_hash=str(data_hash),
        ).zero_invalid()


    def project_inputs(self, batch: LatentBatch) -> tuple[Tensor, Tensor]:
        if batch.ratio != self.ratio or batch.mode != self.mode:
            raise ValueError("batch ratio/mode does not match the adapter")
        return (
            self.scalar_in(torch.cat((batch.state_h, batch.detail_h), dim=-1)),
            self.vector_in(torch.cat((batch.state_v, batch.detail_v), dim=-1)),
        )


    def project_outputs(self, h: Tensor, v: Tensor) -> LatentFields:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("model features must have shapes [K,N,Dh] and [K,N,3,Dv]")
        return LatentFields(
            self.scalar_out["state_h"](h),
            self.scalar_out["detail_h"](h),
            self.vector_out["state_v"](v),
            self.vector_out["detail_v"](v),
        )

    def make_generated_latent(self, batch: LatentBatch, fields: LatentFields) -> StateDetailLatent:
        """Reconstruct a decoder-ready latent with raw diagnostics absent."""

        if fields.state_h.shape[-1] != self.codec_width:
            raise ValueError("generated fields must be inverse-projected to codec width")
        return StateDetailLatent(
            state_h=fields.state_h,
            state_v=fields.state_v,
            detail_h=fields.detail_h,
            detail_v=fields.detail_v,
            raw_detail_h=None,
            raw_detail_v=None,
            token_mask=batch.token_mask,
            detail_valid=batch.detail_valid,
            detail_component_mask=batch.detail_component_mask,
            block_frame_mask=batch.block_frame_mask,
            frame_time_ps=batch.frame_time_ps,
            block_time_ps=batch.block_time_ps,
            abid=batch.abid,
            sample_origin=batch.sample_origin,
            topology=batch.topology,
            ratio=batch.ratio,
            mode=batch.mode,
            coefficient_order=("D01",) if batch.ratio == 2 else ("Dmid", "D01", "D23"),
            width=batch.width,
        )
