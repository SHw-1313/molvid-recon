"""Block-local state/detail and capacity-matched temporal codecs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch
from torch import Tensor, nn

from .haar import haar_lift, haar_inverse, _prepare_blocks
from .types import (
    CodecOutput,
    MatchedPoolingLatent,
    RATIO_FOR_MODE,
    StateDetailLatent,
)

def _validate_features(h: Tensor, v: Tensor) -> tuple[int, int, int]:
    if h.ndim != 3:
        raise ValueError("h must have shape [T, N, C]")
    if v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
        raise ValueError("v must have shape [T, N, 3, C]")
    if v.shape[-1] != h.shape[-1]:
        raise ValueError("h and v must have the same channel width")
    if not torch.isfinite(h).all() or not torch.isfinite(v).all():
        raise ValueError("temporal features must be finite")
    return int(h.shape[0]), int(h.shape[1]), int(h.shape[-1])


def _normalise_context(
    h: Tensor,
    v: Tensor,
    *,
    frame_mask: Optional[Tensor],
    abid: Optional[Tensor],
    time_ps: Optional[Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    frames, atoms, _ = _validate_features(h, v)
    if frame_mask is None:
        mask = torch.ones((1, frames), device=h.device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(frame_mask, device=h.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or mask.shape[1] != frames:
            raise ValueError("frame_mask must have shape [B, T]")
    batch_size = int(mask.shape[0])
    if abid is None:
        sample_id = torch.zeros(atoms, device=h.device, dtype=torch.long)
    else:
        sample_id = torch.as_tensor(abid, device=h.device, dtype=torch.long).flatten()
        if sample_id.numel() != atoms:
            raise ValueError("abid must have shape [N]")
    if sample_id.numel() and (
        torch.any(sample_id < 0) or torch.any(sample_id >= batch_size)
    ):
        raise ValueError("abid contains an invalid sample id")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every sample must contain at least one valid frame")

    if time_ps is None:
        clock = torch.arange(frames, device=h.device, dtype=h.dtype).view(1, -1)
        clock = clock.expand(batch_size, -1)
    else:
        clock = torch.as_tensor(time_ps, device=h.device, dtype=h.dtype)
        if clock.ndim == 1:
            clock = clock.unsqueeze(0)
        if tuple(clock.shape) != tuple(mask.shape):
            raise ValueError("time_ps must have shape [B, T]")
        if not torch.isfinite(clock).all():
            raise ValueError("time_ps contains NaN or Inf")
        for sample in range(batch_size):
            indices = torch.nonzero(mask[sample], as_tuple=False).flatten()
            values = clock[sample].index_select(0, indices)
            if values.numel() > 1 and not torch.all(values[1:] > values[:-1]):
                raise ValueError("valid time_ps values must be strictly increasing")
    return mask, sample_id, clock, torch.arange(frames, device=h.device)


class StateDetailCodec(nn.Module):
    """R1/R2/R4 deterministic state/detail codec with a no-anchor decoder."""

    def __init__(
        self,
        channels: int,
        *,
        mode: str,
        identity_bottleneck: bool = False,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in {
            "ratio1_state_detail",
            "ratio2_state_detail",
            "ratio4_state_detail",
        }:
            raise ValueError(
                "StateDetailCodec mode must be ratio1_state_detail, "
                "ratio2_state_detail, or ratio4_state_detail"
            )
        self.channels = int(channels)
        self.mode = mode
        self.ratio = RATIO_FOR_MODE[mode]
        self.identity_bottleneck = bool(identity_bottleneck)
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if self.identity_bottleneck:
            self.state_encode_h = nn.Identity()
            self.state_encode_v = nn.Identity()
            self.state_decode_h = nn.Identity()
            self.state_decode_v = nn.Identity()
        else:
            self.state_encode_h = nn.Linear(self.channels, self.channels)
            self.state_encode_v = nn.Linear(self.channels, self.channels, bias=False)
            self.state_decode_h = nn.Linear(self.channels, self.channels)
            self.state_decode_v = nn.Linear(self.channels, self.channels, bias=False)
        self.detail_components = 0 if self.ratio == 1 else (1 if self.ratio == 2 else 3)
        if self.detail_components:
            if self.identity_bottleneck:
                self.detail_encode_h = nn.Identity()
                self.detail_encode_v = nn.Identity()
                self.detail_decode_h = nn.Identity()
                self.detail_decode_v = nn.Identity()
                self.detail_gate = nn.Identity()
            else:
                detail_width = self.channels * self.detail_components
                self.detail_encode_h = nn.Linear(detail_width, self.channels, bias=False)
                self.detail_encode_v = nn.Linear(detail_width, self.channels, bias=False)
                self.detail_decode_h = nn.Linear(self.channels, detail_width, bias=False)
                self.detail_decode_v = nn.Linear(self.channels, detail_width, bias=False)
                self.detail_gate = nn.Linear(self.channels, self.channels, bias=False)

    @property
    def coefficient_order(self) -> tuple[str, ...]:
        if self.ratio == 1:
            return ()
        if self.ratio == 2:
            return ("D01",)
        return ("Dmid", "D01", "D23")

    def _encode_detail(
        self,
        raw_h: Tensor,
        raw_v: Tensor,
        state_h: Tensor,
        valid: Tensor,
        sample_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        atom_valid = valid.index_select(0, sample_id).transpose(0, 1)
        if self.identity_bottleneck:
            bank_h = raw_h[:, :, 0]
            bank_v = raw_v[:, :, 0]
        else:
            flat_h = raw_h.reshape(raw_h.shape[0], raw_h.shape[1], -1)
            flat_v = raw_v.permute(0, 1, 3, 2, 4).reshape(
                raw_v.shape[0], raw_v.shape[1], 3, -1
            )
            bank_h = self.detail_encode_h(flat_h)
            bank_v = self.detail_encode_v(flat_v)
            gate = torch.sigmoid(self.detail_gate(state_h))
            bank_h = bank_h * gate
            bank_v = bank_v * gate.unsqueeze(2)
        atom_valid = atom_valid.unsqueeze(-1)
        return bank_h * atom_valid, bank_v * atom_valid.unsqueeze(2)

    def _decode_detail(
        self,
        bank_h: Tensor,
        bank_v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.identity_bottleneck:
            return bank_h.unsqueeze(2), bank_v.unsqueeze(2)
        raw_h = self.detail_decode_h(bank_h).reshape(
            bank_h.shape[0], bank_h.shape[1], self.detail_components, self.channels
        )
        raw_v = self.detail_decode_v(bank_v).reshape(
            bank_v.shape[0], bank_v.shape[1], 3, self.detail_components, self.channels
        ).permute(0, 1, 3, 2, 4)
        return raw_h, raw_v

    def encode(
        self,
        h: Tensor,
        v: Tensor,
        *,
        time_ps: Tensor,
        frame_mask: Tensor,
        abid: Tensor,
        sample_origin: Tensor,
        topology: Any = None,
    ) -> StateDetailLatent:
        mask, sample_id, clock, _ = _normalise_context(
            h, v, frame_mask=frame_mask, abid=abid, time_ps=time_ps
        )
        if sample_origin.ndim != 2 or sample_origin.shape != (mask.shape[0], 3):
            raise ValueError("sample_origin must have shape [B, 3]")
        lifted_h = haar_lift(h, self.ratio, frame_mask=mask, abid=sample_id)
        lifted_v = haar_lift(v, self.ratio, frame_mask=mask, abid=sample_id)
        if lifted_h.detail_component_mask.shape != lifted_v.detail_component_mask.shape:
            raise RuntimeError("scalar/vector lifting masks disagree")
        state_h = self.state_encode_h(lifted_h.state)
        state_v = self.state_encode_v(lifted_v.state)
        token_atom_mask = lifted_h.token_mask.index_select(0, sample_id).transpose(0, 1)
        state_h = state_h * token_atom_mask.unsqueeze(-1)
        state_v = state_v * token_atom_mask.unsqueeze(-1).unsqueeze(-1)
        detail_h = detail_v = raw_detail_h = raw_detail_v = None
        if self.detail_components:
            assert lifted_h.detail is not None and lifted_v.detail is not None
            raw_detail_h, raw_detail_v = lifted_h.detail, lifted_v.detail
            detail_h, detail_v = self._encode_detail(
                raw_detail_h,
                raw_detail_v,
                state_h,
                lifted_h.detail_valid,
                sample_id,
            )
        block_time = clock.new_zeros(
            (clock.shape[0], lifted_h.block_frame_mask.shape[1] * self.ratio)
        )
        block_time[:, : clock.shape[1]] = clock
        block_time = block_time.reshape(clock.shape[0], lifted_h.block_frame_mask.shape[1], self.ratio)
        return StateDetailLatent(
            state_h=state_h,
            state_v=state_v,
            detail_h=detail_h,
            detail_v=detail_v,
            raw_detail_h=raw_detail_h,
            raw_detail_v=raw_detail_v,
            token_mask=lifted_h.token_mask,
            detail_valid=lifted_h.detail_valid,
            detail_component_mask=lifted_h.detail_component_mask,
            block_frame_mask=lifted_h.block_frame_mask,
            frame_time_ps=clock,
            block_time_ps=block_time,
            abid=sample_id,
            sample_origin=sample_origin,
            topology=topology,
            ratio=self.ratio,
            mode=self.mode,
            coefficient_order=self.coefficient_order,
            width=self.channels,
        )

    def decode(self, latent: StateDetailLatent, *, coordinate_head: nn.Module) -> CodecOutput:
        if latent.mode != self.mode or latent.ratio != self.ratio:
            raise ValueError("latent mode/ratio does not match this codec")
        state_h = self.state_decode_h(latent.state_h)
        state_v = self.state_decode_v(latent.state_v)
        decoded_detail_h = decoded_detail_v = None
        if self.detail_components:
            if latent.detail_h is None or latent.detail_v is None:
                raise ValueError("state/detail latent is missing its detail bank")
            decoded_detail_h, decoded_detail_v = self._decode_detail(
                latent.detail_h, latent.detail_v
            )
        decoded_h = haar_inverse(
            state_h,
            decoded_detail_h,
            self.ratio,
            block_frame_mask=latent.block_frame_mask,
            detail_component_mask=latent.detail_component_mask,
            abid=latent.abid,
            output_frames=latent.frames,
        )
        decoded_v = haar_inverse(
            state_v,
            decoded_detail_v,
            self.ratio,
            block_frame_mask=latent.block_frame_mask,
            detail_component_mask=latent.detail_component_mask,
            abid=latent.abid,
            output_frames=latent.frames,
        )
        valid = latent.frame_time_ps.new_zeros(latent.frame_time_ps.shape, dtype=torch.bool)
        flat_mask = latent.block_frame_mask.reshape(latent.block_frame_mask.shape[0], -1)
        valid[:, : latent.frames] = flat_mask[:, : latent.frames]
        atom_valid = valid.index_select(0, latent.abid).transpose(0, 1)
        decoded_h = decoded_h * atom_valid.unsqueeze(-1)
        decoded_v = decoded_v * atom_valid.unsqueeze(-1).unsqueeze(-1)
        centered = coordinate_head(decoded_h, decoded_v)
        origin = latent.sample_origin.to(device=centered.device, dtype=torch.float32)
        atom_origin = origin.index_select(0, latent.abid)
        x_centered = centered * atom_valid.unsqueeze(-1).to(dtype=centered.dtype)
        x_hat = x_centered + atom_origin.unsqueeze(0)
        x_hat = torch.where(atom_valid.unsqueeze(-1), x_hat, atom_origin.unsqueeze(0))
        diagnostics: dict[str, Tensor] = {}
        if latent.raw_detail_h is not None and latent.detail_h is not None:
            diagnostics["raw_detail_h_norm"] = latent.raw_detail_h.detach().float().square().mean().sqrt()
            diagnostics["encoded_detail_h_norm"] = latent.detail_h.detach().float().square().mean().sqrt()
            assert decoded_detail_h is not None
            diagnostics["decoded_detail_h_norm"] = decoded_detail_h.detach().float().square().mean().sqrt()
            diagnostics["detail_valid_fraction"] = latent.detail_valid.detach().float().mean()
        diagnostics["state_h_norm"] = latent.state_h.detach().float().square().mean().sqrt()
        diagnostics["decoded_feature_motion_h"] = (
            decoded_h[1:] - decoded_h[:-1]
        ).detach().float().square().mean().sqrt() if latent.frames > 1 else decoded_h.new_zeros(())
        return CodecOutput(
            x_coarse=x_hat,
            x_hat=x_hat,
            h=decoded_h,
            v=decoded_v,
            frame_mask=latent.block_frame_mask.reshape(latent.block_frame_mask.shape[0], -1)[:, : latent.frames],
            time_ps=latent.frame_time_ps,
            latent=latent,
            decoded_detail_h=decoded_detail_h,
            decoded_detail_v=decoded_detail_v,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        *,
        time_ps: Tensor,
        frame_mask: Tensor,
        abid: Tensor,
        sample_origin: Tensor,
        topology: Any = None,
        coordinate_head: nn.Module,
    ) -> CodecOutput:
        latent = self.encode(
            h,
            v,
            time_ps=time_ps,
            frame_mask=frame_mask,
            abid=abid,
            sample_origin=sample_origin,
            topology=topology,
        )
        return self.decode(latent, coordinate_head=coordinate_head)

    def decode_detail_zero(self, latent: StateDetailLatent, *, coordinate_head: nn.Module) -> CodecOutput:
        if latent.detail_h is None or latent.detail_v is None:
            return self.decode(latent, coordinate_head=coordinate_head)
        zero = replace(
            latent,
            detail_h=torch.zeros_like(latent.detail_h),
            detail_v=torch.zeros_like(latent.detail_v),
        )
        return self.decode(zero, coordinate_head=coordinate_head)


class MatchedPoolingCodec(nn.Module):
    """R4 two-bank unstructured linear pooling capacity control."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        if self.channels < 1:
            raise ValueError("channels must be positive")
        input_width = self.channels * 4
        self.pool_a_h = nn.Linear(input_width, self.channels, bias=False)
        self.pool_b_h = nn.Linear(input_width, self.channels, bias=False)
        self.pool_a_v = nn.Linear(input_width, self.channels, bias=False)
        self.pool_b_v = nn.Linear(input_width, self.channels, bias=False)
        self.expand_a_h = nn.Linear(self.channels, input_width)
        self.expand_b_h = nn.Linear(self.channels, input_width)
        self.expand_a_v = nn.Linear(self.channels, input_width, bias=False)
        self.expand_b_v = nn.Linear(self.channels, input_width, bias=False)

    def encode(
        self,
        h: Tensor,
        v: Tensor,
        *,
        time_ps: Tensor,
        frame_mask: Tensor,
        abid: Tensor,
        sample_origin: Tensor,
        topology: Any = None,
    ) -> MatchedPoolingLatent:
        mask, sample_id, clock, _ = _normalise_context(
            h, v, frame_mask=frame_mask, abid=abid, time_ps=time_ps
        )
        if sample_origin.shape != (mask.shape[0], 3):
            raise ValueError("sample_origin must have shape [B, 3]")
        packed_h, block_mask, frames = _prepare_blocks(
            h, 4, frame_mask=mask, abid=sample_id
        )
        packed_v, block_mask_v, _ = _prepare_blocks(
            v, 4, frame_mask=mask, abid=sample_id
        )
        if not torch.equal(block_mask, block_mask_v):
            raise RuntimeError("matched pooling masks disagree for scalar/vector features")
        flat_h = packed_h.permute(0, 2, 1, 3).reshape(
            packed_h.shape[0], packed_h.shape[2], -1
        )
        flat_v = packed_v.permute(0, 2, 3, 1, 4).reshape(
            packed_v.shape[0], packed_v.shape[2], 3, -1
        )
        block_time = clock.new_zeros((clock.shape[0], block_mask.shape[1] * 4))
        block_time[:, :frames] = clock
        block_time = block_time.reshape(clock.shape[0], block_mask.shape[1], 4)
        return MatchedPoolingLatent(
            bank_a_h=self.pool_a_h(flat_h),
            bank_a_v=self.pool_a_v(flat_v),
            bank_b_h=self.pool_b_h(flat_h),
            bank_b_v=self.pool_b_v(flat_v),
            token_mask=block_mask.any(dim=-1),
            block_frame_mask=block_mask,
            frame_time_ps=clock,
            block_time_ps=block_time,
            abid=sample_id,
            sample_origin=sample_origin,
            topology=topology,
            width=self.channels,
        )

    def decode(self, latent: MatchedPoolingLatent, *, coordinate_head: nn.Module) -> CodecOutput:
        if latent.mode != "ratio4_matched_pooling":
            raise ValueError("latent is not a matched-pooling latent")
        h_a = self.expand_a_h(latent.bank_a_h).reshape(
            latent.bank_a_h.shape[0], latent.bank_a_h.shape[1], 4, self.channels
        )
        h_b = self.expand_b_h(latent.bank_b_h).reshape(
            latent.bank_b_h.shape[0], latent.bank_b_h.shape[1], 4, self.channels
        )
        v_a = self.expand_a_v(latent.bank_a_v).reshape(
            latent.bank_a_v.shape[0], latent.bank_a_v.shape[1], 3, 4, self.channels
        ).permute(0, 1, 3, 2, 4)
        v_b = self.expand_b_v(latent.bank_b_v).reshape(
            latent.bank_b_v.shape[0], latent.bank_b_v.shape[1], 3, 4, self.channels
        ).permute(0, 1, 3, 2, 4)
        decoded_h = (h_a + h_b).permute(0, 2, 1, 3).reshape(
            latent.bank_a_h.shape[0] * 4, latent.bank_a_h.shape[1], self.channels
        )[: latent.frame_time_ps.shape[1]]
        decoded_v = (v_a + v_b).permute(0, 2, 1, 3, 4).reshape(
            latent.bank_a_v.shape[0] * 4, latent.bank_a_v.shape[1], 3, self.channels
        )[: latent.frame_time_ps.shape[1]]
        valid = latent.block_frame_mask.reshape(latent.block_frame_mask.shape[0], -1)[:, : latent.frame_time_ps.shape[1]]
        atom_valid = valid.index_select(0, latent.abid).transpose(0, 1)
        decoded_h = decoded_h * atom_valid.unsqueeze(-1)
        decoded_v = decoded_v * atom_valid.unsqueeze(-1).unsqueeze(-1)
        centered = coordinate_head(decoded_h, decoded_v)
        origin = latent.sample_origin.to(device=centered.device, dtype=torch.float32)
        atom_origin = origin.index_select(0, latent.abid)
        x_centered = centered * atom_valid.unsqueeze(-1).to(dtype=centered.dtype)
        x_hat = x_centered + atom_origin.unsqueeze(0)
        x_hat = torch.where(atom_valid.unsqueeze(-1), x_hat, atom_origin.unsqueeze(0))
        diagnostics = {
            "bank_a_h_norm": latent.bank_a_h.detach().float().square().mean().sqrt(),
            "bank_b_h_norm": latent.bank_b_h.detach().float().square().mean().sqrt(),
        }
        return CodecOutput(
            x_coarse=x_hat,
            x_hat=x_hat,
            h=decoded_h,
            v=decoded_v,
            frame_mask=valid,
            time_ps=latent.frame_time_ps,
            latent=latent,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        h: Tensor,
        v: Tensor,
        *,
        time_ps: Tensor,
        frame_mask: Tensor,
        abid: Tensor,
        sample_origin: Tensor,
        topology: Any = None,
        coordinate_head: nn.Module,
    ) -> CodecOutput:
        return self.decode(
            self.encode(
                h,
                v,
                time_ps=time_ps,
                frame_mask=frame_mask,
                abid=abid,
                sample_origin=sample_origin,
                topology=topology,
            ),
            coordinate_head=coordinate_head,
        )
