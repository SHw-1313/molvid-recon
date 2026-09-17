"""Deterministic zero-preserving state/detail temporal codec.

The legacy codec in :mod:`module.temporal_codec` is intentionally left alone.
This module contains the versioned block-local tokenizer used by the state/detail
phase.  It operates on scalar features ``[T, N, C]`` and vector features
``[T, N, 3, C]``.  The Cartesian axis is never passed through a learned linear
map; all learned vector maps act only on the final channel axis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn


STATE_DETAIL_CODEC_SCHEMA = "pvb.codec.state_detail.latent.v2"
STATIC_TOPOLOGY_SCHEMA = "pvb.codec.state_detail.static_topology.v1"
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


def _as_bool_mask(value: Tensor, *, shape: tuple[int, ...], name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.bool)
    if tuple(result.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}, got {tuple(result.shape)}")
    return result


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


def _prepare_blocks(
    features: Tensor,
    ratio: int,
    *,
    frame_mask: Tensor,
    abid: Tensor,
) -> tuple[Tensor, Tensor, int]:
    """Pad and pack a time-major feature tensor into ``[L, ratio, N, ...]``."""

    frames, atoms = int(features.shape[0]), int(features.shape[1])
    blocks = (frames + ratio - 1) // ratio
    padded_frames = blocks * ratio
    frame_valid = frame_mask.index_select(0, abid).transpose(0, 1)
    valid_features = features * frame_valid.reshape(frames, atoms, *([1] * (features.ndim - 2)))
    padded = features.new_zeros((padded_frames, *features.shape[1:]))
    padded[:frames] = valid_features
    packed = padded.reshape(blocks, ratio, *features.shape[1:])
    mask_padded = frame_mask.new_zeros((frame_mask.shape[0], padded_frames))
    mask_padded[:, :frames] = frame_mask
    block_mask = mask_padded.reshape(frame_mask.shape[0], blocks, ratio)
    return packed, block_mask, frames


def _atom_block_mask(block_mask: Tensor, abid: Tensor) -> Tensor:
    """Return ``[L, ratio, N]`` frame validity for packed atoms."""

    return block_mask.index_select(0, abid).permute(1, 2, 0)


def _sample_pair_mask(block_mask: Tensor, first: int, second: int) -> Tensor:
    return block_mask[..., first] & block_mask[..., second]


def _reject_ambiguous_partial_blocks(block_mask: Tensor, ratio: int) -> None:
    if ratio != 4:
        return
    any_01 = block_mask[..., 0] | block_mask[..., 1]
    any_23 = block_mask[..., 2] | block_mask[..., 3]
    full_01 = _sample_pair_mask(block_mask, 0, 1)
    full_23 = _sample_pair_mask(block_mask, 2, 3)
    ambiguous = any_01 & any_23 & ~(full_01 & full_23)
    if bool(torch.any(ambiguous)):
        raise ValueError(
            "ratio-4 partial blocks must contain one complete pair or all four "
            "frames; ambiguous two-pair partial blocks cannot be encoded safely"
        )


@dataclass(frozen=True)
class HaarLift:
    """Fixed Haar coefficients and validity masks for one temporal ratio."""

    state: Tensor
    detail: Optional[Tensor]
    token_mask: Tensor
    detail_component_mask: Tensor
    block_frame_mask: Tensor
    ratio: int

    @property
    def detail_valid(self) -> Tensor:
        if self.detail_component_mask.shape[-1] == 0:
            return torch.zeros_like(self.token_mask)
        return self.detail_component_mask.all(dim=-1)


def haar_lift(
    features: Tensor,
    ratio: int,
    *,
    frame_mask: Optional[Tensor] = None,
    abid: Optional[Tensor] = None,
) -> HaarLift:
    """Apply the fixed orthonormal R1/R2/R4 lifting transform.

    ``features`` is ``[T, N, ...]``.  For partial blocks the supported rule is
    explicit: a one-frame pair keeps that frame as its state and marks detail
    invalid; ratio-4 blocks may contain one complete pair or all four frames.
    """

    ratio = int(ratio)
    if ratio not in (1, 2, 4):
        raise ValueError("ratio must be one of 1, 2, or 4")
    if features.ndim < 2:
        raise ValueError("features must have shape [T, N, ...]")
    frames, atoms = int(features.shape[0]), int(features.shape[1])
    if frame_mask is None:
        mask = torch.ones((1, frames), device=features.device, dtype=torch.bool)
    else:
        mask = torch.as_tensor(frame_mask, device=features.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim != 2 or mask.shape[1] != frames:
            raise ValueError("frame_mask must have shape [B, T]")
    if abid is None:
        sample_id = torch.zeros(atoms, device=features.device, dtype=torch.long)
    else:
        sample_id = torch.as_tensor(abid, device=features.device, dtype=torch.long).flatten()
        if sample_id.numel() != atoms:
            raise ValueError("abid must have shape [N]")
    if sample_id.numel() and (
        torch.any(sample_id < 0) or torch.any(sample_id >= mask.shape[0])
    ):
        raise ValueError("abid contains an invalid sample id")
    packed, block_mask, _ = _prepare_blocks(
        features, ratio, frame_mask=mask, abid=sample_id
    )
    atom_mask = _atom_block_mask(block_mask, sample_id)
    token_mask = block_mask.any(dim=-1)
    inv_sqrt2 = 2.0**-0.5

    if ratio == 1:
        return HaarLift(
            state=packed[:, 0],
            detail=None,
            token_mask=token_mask,
            detail_component_mask=torch.zeros(
                (*token_mask.shape, 0), device=features.device, dtype=torch.bool
            ),
            block_frame_mask=block_mask,
            ratio=ratio,
        )

    if ratio == 2:
        first, second = packed[:, 0], packed[:, 1]
        full = _sample_pair_mask(block_mask, 0, 1)
        only_first = block_mask[..., 0] & ~block_mask[..., 1]
        only_second = ~block_mask[..., 0] & block_mask[..., 1]
        full_atom = full.index_select(0, sample_id).transpose(0, 1)
        only_first_atom = only_first.index_select(0, sample_id).transpose(0, 1)
        only_second_atom = only_second.index_select(0, sample_id).transpose(0, 1)
        state_full = first * inv_sqrt2 + second * inv_sqrt2
        state = torch.where(
            full_atom.reshape(*full_atom.shape, *([1] * (features.ndim - 2))),
            state_full,
            torch.where(
                only_first_atom.reshape(*only_first_atom.shape, *([1] * (features.ndim - 2))),
                first,
                torch.where(
                    only_second_atom.reshape(*only_second_atom.shape, *([1] * (features.ndim - 2))),
                    second,
                    torch.zeros_like(first),
                ),
            ),
        )
        detail = second * inv_sqrt2 - first * inv_sqrt2
        detail = detail * full_atom.reshape(*full_atom.shape, *([1] * (features.ndim - 2)))
        return HaarLift(
            state=state,
            detail=detail.unsqueeze(2),
            token_mask=token_mask,
            detail_component_mask=full.unsqueeze(-1),
            block_frame_mask=block_mask,
            ratio=ratio,
        )

    _reject_ambiguous_partial_blocks(block_mask, ratio)
    first, second, third, fourth = (packed[:, index] for index in range(4))
    full_01 = _sample_pair_mask(block_mask, 0, 1)
    full_23 = _sample_pair_mask(block_mask, 2, 3)
    any_01 = block_mask[..., 0] | block_mask[..., 1]
    any_23 = block_mask[..., 2] | block_mask[..., 3]
    full_01_atom = full_01.index_select(0, sample_id).transpose(0, 1)
    full_23_atom = full_23.index_select(0, sample_id).transpose(0, 1)
    any_01_atom = any_01.index_select(0, sample_id).transpose(0, 1)
    any_23_atom = any_23.index_select(0, sample_id).transpose(0, 1)
    broadcast = lambda value: value.reshape(*value.shape, *([1] * (features.ndim - 2)))

    state_01_full = first * inv_sqrt2 + second * inv_sqrt2
    state_23_full = third * inv_sqrt2 + fourth * inv_sqrt2
    state_01 = torch.where(
        full_01_atom.reshape(*full_01_atom.shape, *([1] * (features.ndim - 2))),
        state_01_full,
        torch.where(
            block_mask[..., 0].index_select(0, sample_id).transpose(0, 1).reshape(*full_01_atom.shape, *([1] * (features.ndim - 2))),
            first,
            torch.where(
                block_mask[..., 1].index_select(0, sample_id).transpose(0, 1).reshape(*full_01_atom.shape, *([1] * (features.ndim - 2))),
                second,
                torch.zeros_like(first),
            ),
        ),
    )
    state_23 = torch.where(
        full_23_atom.reshape(*full_23_atom.shape, *([1] * (features.ndim - 2))),
        state_23_full,
        torch.where(
            block_mask[..., 2].index_select(0, sample_id).transpose(0, 1).reshape(*full_23_atom.shape, *([1] * (features.ndim - 2))),
            third,
            torch.where(
                block_mask[..., 3].index_select(0, sample_id).transpose(0, 1).reshape(*full_23_atom.shape, *([1] * (features.ndim - 2))),
                fourth,
                torch.zeros_like(third),
            ),
        ),
    )
    both_full = full_01 & full_23
    both_full_atom = both_full.index_select(0, sample_id).transpose(0, 1)
    state = torch.where(
        both_full_atom.reshape(*both_full_atom.shape, *([1] * (features.ndim - 2))),
        state_01 * inv_sqrt2 + state_23 * inv_sqrt2,
        torch.where(
            any_01_atom.reshape(*any_01_atom.shape, *([1] * (features.ndim - 2))),
            state_01,
            torch.where(
                any_23_atom.reshape(*any_23_atom.shape, *([1] * (features.ndim - 2))),
                state_23,
                torch.zeros_like(state_01),
            ),
        ),
    )
    detail_mid = state_23 * inv_sqrt2 - state_01 * inv_sqrt2
    detail_01 = second * inv_sqrt2 - first * inv_sqrt2
    detail_23 = fourth * inv_sqrt2 - third * inv_sqrt2
    detail_mid = detail_mid * both_full_atom.reshape(*both_full_atom.shape, *([1] * (features.ndim - 2)))
    detail_01 = detail_01 * full_01_atom.reshape(*full_01_atom.shape, *([1] * (features.ndim - 2)))
    detail_23 = detail_23 * full_23_atom.reshape(*full_23_atom.shape, *([1] * (features.ndim - 2)))
    detail = torch.stack((detail_mid, detail_01, detail_23), dim=2)
    components = torch.stack((both_full, full_01, full_23), dim=-1)
    return HaarLift(
        state=state,
        detail=detail,
        token_mask=token_mask,
        detail_component_mask=components,
        block_frame_mask=block_mask,
        ratio=ratio,
    )


def haar_inverse(
    state: Tensor,
    detail: Optional[Tensor],
    ratio: int,
    *,
    block_frame_mask: Tensor,
    detail_component_mask: Optional[Tensor] = None,
    abid: Optional[Tensor] = None,
    output_frames: Optional[int] = None,
) -> Tensor:
    """Invert :func:`haar_lift`, including the explicit partial-block rule."""

    ratio = int(ratio)
    if ratio not in (1, 2, 4):
        raise ValueError("ratio must be one of 1, 2, or 4")
    if state.ndim < 2:
        raise ValueError("state must have shape [L, N, ...]")
    blocks, atoms = int(state.shape[0]), int(state.shape[1])
    block_mask = torch.as_tensor(block_frame_mask, device=state.device, dtype=torch.bool)
    if block_mask.ndim != 3 or block_mask.shape[1:] != (blocks, ratio):
        raise ValueError("block_frame_mask must have shape [B, L, ratio]")
    if abid is None:
        sample_id = torch.zeros(atoms, device=state.device, dtype=torch.long)
    else:
        sample_id = torch.as_tensor(abid, device=state.device, dtype=torch.long).flatten()
        if sample_id.numel() != atoms:
            raise ValueError("abid must have shape [N]")
    if sample_id.numel() and torch.any(sample_id >= block_mask.shape[0]):
        raise ValueError("abid contains an invalid sample id")
    _reject_ambiguous_partial_blocks(block_mask, ratio)
    atom_mask = _atom_block_mask(block_mask, sample_id)
    broadcast = lambda value: value.reshape(*value.shape, *([1] * (state.ndim - 2)))
    inv_sqrt2 = 2.0**-0.5

    def component_broadcast(component_count: int) -> Tensor:
        if detail_component_mask is None:
            return state.new_ones(
                (blocks, atoms, component_count, *([1] * (state.ndim - 2))),
                dtype=state.dtype,
            )
        components = torch.as_tensor(
            detail_component_mask, device=state.device, dtype=torch.bool
        )
        if components.shape != (block_mask.shape[0], blocks, component_count):
            raise ValueError(
                "detail_component_mask has an incompatible shape for this ratio"
            )
        atom_components = components.index_select(0, sample_id).permute(1, 0, 2)
        return atom_components.reshape(
            blocks,
            atoms,
            component_count,
            *([1] * (state.ndim - 2)),
        ).to(dtype=state.dtype)

    if ratio == 1:
        packed = state.unsqueeze(1)
    elif ratio == 2:
        if detail is None:
            detail_value = torch.zeros_like(state)
        else:
            if detail.ndim != state.ndim + 1 or detail.shape[2] != 1:
                raise ValueError("ratio-2 detail must have shape [L, N, 1, ...]")
            detail_value = detail[:, :, 0]
            detail_value = detail_value * component_broadcast(1)[:, :, 0]
        full = _sample_pair_mask(block_mask, 0, 1)
        only_first = block_mask[..., 0] & ~block_mask[..., 1]
        only_second = ~block_mask[..., 0] & block_mask[..., 1]
        full_atom = full.index_select(0, sample_id).transpose(0, 1)
        first_atom = only_first.index_select(0, sample_id).transpose(0, 1)
        second_atom = only_second.index_select(0, sample_id).transpose(0, 1)
        frame0 = torch.where(
            broadcast(full_atom),
            state * inv_sqrt2 - detail_value * inv_sqrt2,
            torch.where(broadcast(first_atom), state, torch.zeros_like(state)),
        )
        frame1 = torch.where(
            broadcast(full_atom),
            state * inv_sqrt2 + detail_value * inv_sqrt2,
            torch.where(broadcast(second_atom), state, torch.zeros_like(state)),
        )
        packed = torch.stack((frame0, frame1), dim=1)
    else:
        if detail is None:
            detail_value = state.new_zeros((*state.shape[:2], 3, *state.shape[2:]))
        else:
            if detail.ndim != state.ndim + 1 or detail.shape[2] != 3:
                raise ValueError("ratio-4 detail must have shape [L, N, 3, ...]")
            detail_value = detail
            detail_value = detail_value * component_broadcast(3)
        full_01 = _sample_pair_mask(block_mask, 0, 1)
        full_23 = _sample_pair_mask(block_mask, 2, 3)
        any_01 = block_mask[..., 0] | block_mask[..., 1]
        any_23 = block_mask[..., 2] | block_mask[..., 3]
        both_full = full_01 & full_23
        both_atom = both_full.index_select(0, sample_id).transpose(0, 1)
        pair01_atom = any_01.index_select(0, sample_id).transpose(0, 1)
        pair23_atom = any_23.index_select(0, sample_id).transpose(0, 1)
        state01 = torch.where(
            broadcast(both_atom),
            state * inv_sqrt2 - detail_value[:, :, 0] * inv_sqrt2,
            torch.where(broadcast(pair01_atom), state, torch.zeros_like(state)),
        )
        state23 = torch.where(
            broadcast(both_atom),
            state * inv_sqrt2 + detail_value[:, :, 0] * inv_sqrt2,
            torch.where(broadcast(pair23_atom), state, torch.zeros_like(state)),
        )
        full01_atom = full_01.index_select(0, sample_id).transpose(0, 1)
        full23_atom = full_23.index_select(0, sample_id).transpose(0, 1)
        first_valid = block_mask[..., 0].index_select(0, sample_id).transpose(0, 1)
        second_valid = block_mask[..., 1].index_select(0, sample_id).transpose(0, 1)
        third_valid = block_mask[..., 2].index_select(0, sample_id).transpose(0, 1)
        fourth_valid = block_mask[..., 3].index_select(0, sample_id).transpose(0, 1)
        frame0 = torch.where(
            broadcast(full01_atom),
            state01 * inv_sqrt2 - detail_value[:, :, 1] * inv_sqrt2,
            torch.where(broadcast(first_valid), state01, torch.zeros_like(state01)),
        )
        frame1 = torch.where(
            broadcast(full01_atom),
            state01 * inv_sqrt2 + detail_value[:, :, 1] * inv_sqrt2,
            torch.where(broadcast(second_valid), state01, torch.zeros_like(state01)),
        )
        frame2 = torch.where(
            broadcast(full23_atom),
            state23 * inv_sqrt2 - detail_value[:, :, 2] * inv_sqrt2,
            torch.where(broadcast(third_valid), state23, torch.zeros_like(state23)),
        )
        frame3 = torch.where(
            broadcast(full23_atom),
            state23 * inv_sqrt2 + detail_value[:, :, 2] * inv_sqrt2,
            torch.where(broadcast(fourth_valid), state23, torch.zeros_like(state23)),
        )
        packed = torch.stack((frame0, frame1, frame2, frame3), dim=1)
    valid = atom_mask
    packed = packed * valid.reshape(*valid.shape, *([1] * (state.ndim - 2)))
    result = packed.reshape(blocks * ratio, *state.shape[1:])
    if output_frames is not None:
        result = result[: int(output_frames)]
    return result


def compute_masked_centroid_origin(
    x: Tensor,
    *,
    frame_mask: Tensor,
    abid: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Compute one frame-zero centroid per sample, with no atomwise bypass."""

    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must have shape [T, N, 3]")
    frame_mask = torch.as_tensor(frame_mask, device=x.device, dtype=torch.bool)
    abid = torch.as_tensor(abid, device=x.device, dtype=torch.long).flatten()
    atom_mask = torch.as_tensor(atom_mask, device=x.device, dtype=torch.bool).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != x.shape[0]:
        raise ValueError("frame_mask must have shape [B, T]")
    if abid.numel() != x.shape[1] or atom_mask.numel() != x.shape[1]:
        raise ValueError("abid and atom_mask must have shape [N]")
    if not torch.all(frame_mask[:, 0]):
        raise ValueError("frame zero must be valid for every sample to compute origin")
    batch_size = int(frame_mask.shape[0])
    selected = atom_mask
    counts = x.new_zeros((batch_size,))
    counts.index_add_(0, abid, selected.to(dtype=x.dtype))
    if bool(torch.any(counts <= 0)):
        raise ValueError("each sample needs at least one valid atom for the frame-zero origin")
    origin = x.new_zeros((batch_size, 3))
    origin.index_add_(0, abid, x[0] * selected.to(dtype=x.dtype).unsqueeze(-1))
    return origin / counts.unsqueeze(-1)


def center_coordinates(
    x: Tensor,
    *,
    frame_mask: Tensor,
    abid: Tensor,
    atom_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    origin = compute_masked_centroid_origin(
        x, frame_mask=frame_mask, abid=abid, atom_mask=atom_mask
    )
    atom_origin = origin.index_select(0, torch.as_tensor(abid, device=x.device))
    return x - atom_origin.unsqueeze(0), origin


@dataclass(frozen=True)
class StaticTopologyMetadata:
    """Coordinate-independent topology aligned to the latent ``[N, ...]`` atom axis.

    This deliberately does not mirror :class:`FrameGraphBatch`: radius edges, edge vectors,
    distances, positions, and frame-expanded indices are spatial-runtime data and are not valid
    latent metadata.  ``covalent_bond_type`` is currently a binary indicator because the clip
    schema stores covalent connectivity without a bond-order field.
    """

    atom_type: Tensor
    block_type: Tensor
    abid: Tensor
    block_id: Tensor
    component_id: Tensor
    atom_ptr: Tensor
    covalent_bond_index: Tensor
    covalent_bond_type: Tensor
    topology_id: tuple[str, ...] = ()
    sample_id: tuple[str, ...] = ()
    schema_version: str = STATIC_TOPOLOGY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATIC_TOPOLOGY_SCHEMA:
            raise ValueError(f"unsupported static topology schema {self.schema_version!r}")
        vectors = {
            "atom_type": self.atom_type,
            "block_type": self.block_type,
            "abid": self.abid,
            "block_id": self.block_id,
            "component_id": self.component_id,
        }
        for name, value in vectors.items():
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must have shape [N]")
        atom_count = int(self.atom_type.numel())
        if any(int(value.numel()) != atom_count for value in vectors.values()):
            raise ValueError("static topology atom fields must share the latent N axis")
        if not isinstance(self.atom_ptr, Tensor) or self.atom_ptr.ndim != 1:
            raise ValueError("atom_ptr must have shape [B+1]")
        if self.atom_ptr.numel() < 2:
            raise ValueError("atom_ptr must describe at least one sample")
        if self.atom_ptr.device != self.abid.device:
            raise ValueError("static topology tensors must share a device")
        atom_ptr = self.atom_ptr.to(dtype=torch.long)
        if int(atom_ptr[0]) != 0 or int(atom_ptr[-1]) != atom_count:
            raise ValueError("atom_ptr must start at zero and end at N")
        if torch.any(atom_ptr[1:] < atom_ptr[:-1]):
            raise ValueError("atom_ptr must be nondecreasing")
        batch_size = int(atom_ptr.numel() - 1)
        if self.abid.numel() and (
            torch.any(self.abid < 0) or torch.any(self.abid >= batch_size)
        ):
            raise ValueError("static topology abid contains an invalid sample id")
        if not isinstance(self.covalent_bond_index, Tensor):
            raise ValueError("covalent_bond_index must be a tensor")
        if self.covalent_bond_index.ndim != 2 or self.covalent_bond_index.shape[0] != 2:
            raise ValueError("covalent_bond_index must have shape [2, E_static]")
        if self.covalent_bond_index.device != self.abid.device:
            raise ValueError("covalent bond metadata must share the latent device")
        edge_count = int(self.covalent_bond_index.shape[1])
        if not isinstance(self.covalent_bond_type, Tensor) or self.covalent_bond_type.shape != (edge_count,):
            raise ValueError("covalent_bond_type must have shape [E_static]")
        if self.covalent_bond_type.device != self.abid.device:
            raise ValueError("covalent bond types must share the latent device")
        if edge_count:
            edge_index = self.covalent_bond_index.to(dtype=torch.long)
            if torch.any(edge_index < 0) or torch.any(edge_index >= atom_count):
                raise ValueError("covalent bond index exceeds the latent N atom axis")
            if torch.any(self.abid.index_select(0, edge_index[0]) != self.abid.index_select(0, edge_index[1])):
                raise ValueError("static topology contains a cross-sample covalent bond")
        if self.topology_id and len(self.topology_id) != batch_size:
            raise ValueError("topology_id must contain one entry per latent sample")
        if self.sample_id and len(self.sample_id) != batch_size:
            raise ValueError("sample_id must contain one entry per latent sample")

    @property
    def num_atoms(self) -> int:
        return int(self.atom_type.numel())

    @property
    def batch_size(self) -> int:
        return int(self.atom_ptr.numel() - 1)

    @property
    def z(self) -> Tensor:
        return self.atom_type

    @property
    def b(self) -> Tensor:
        return self.block_type

    @property
    def bond_index(self) -> Tensor:
        return self.covalent_bond_index

    @classmethod
    def from_batch(cls, batch: Any) -> "StaticTopologyMetadata":
        """Extract only static chemical fields from a ``ClipBatch``-like object."""

        def field(name: str) -> Any:
            if isinstance(batch, Mapping):
                if name not in batch:
                    raise ValueError(f"batch is missing static topology field {name!r}")
                return batch[name]
            if not hasattr(batch, name):
                raise ValueError(f"batch is missing static topology field {name!r}")
            return getattr(batch, name)

        atom_type = torch.as_tensor(field("atype"), dtype=torch.long)
        block_type = torch.as_tensor(field("btype"), device=atom_type.device, dtype=torch.long)
        abid = torch.as_tensor(field("abid"), device=atom_type.device, dtype=torch.long).flatten()
        block_id = torch.as_tensor(field("block_id"), device=atom_type.device, dtype=torch.long).flatten()
        component_id = torch.as_tensor(field("component_id"), device=atom_type.device, dtype=torch.long).flatten()
        atom_ptr = torch.as_tensor(field("atom_ptr"), device=atom_type.device, dtype=torch.long).flatten()
        bond_index = torch.as_tensor(field("bond_index"), device=atom_type.device, dtype=torch.long)
        if bond_index.numel() == 0:
            bond_index = torch.empty((2, 0), device=atom_type.device, dtype=torch.long)
        elif bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise ValueError("bond_index must have shape [2, E_static]")
        bond_type = torch.ones(
            (int(bond_index.shape[1]),), device=atom_type.device, dtype=torch.long
        )
        if isinstance(batch, Mapping):
            topology_values = batch.get("topology_id", ())
            sample_values = batch.get("sample_id", ())
        else:
            topology_values = getattr(batch, "topology_id", ())
            sample_values = getattr(batch, "sample_id", ())
        topology_id = tuple(str(value) for value in topology_values)
        sample_id = tuple(str(value) for value in sample_values)
        return cls(
            atom_type=atom_type,
            block_type=block_type,
            abid=abid,
            block_id=block_id,
            component_id=component_id,
            atom_ptr=atom_ptr,
            covalent_bond_index=bond_index,
            covalent_bond_type=bond_type,
            topology_id=topology_id,
            sample_id=sample_id,
        )

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> "StaticTopologyMetadata":
        return replace(
            self,
            atom_type=self.atom_type.to(device, non_blocking=non_blocking),
            block_type=self.block_type.to(device, non_blocking=non_blocking),
            abid=self.abid.to(device, non_blocking=non_blocking),
            block_id=self.block_id.to(device, non_blocking=non_blocking),
            component_id=self.component_id.to(device, non_blocking=non_blocking),
            atom_ptr=self.atom_ptr.to(device, non_blocking=non_blocking),
            covalent_bond_index=self.covalent_bond_index.to(device, non_blocking=non_blocking),
            covalent_bond_type=self.covalent_bond_type.to(device, non_blocking=non_blocking),
        )

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "atom_count": self.num_atoms,
            "batch_size": self.batch_size,
            "atom_type_shape": list(self.atom_type.shape),
            "block_type_shape": list(self.block_type.shape),
            "abid_shape": list(self.abid.shape),
            "block_id_shape": list(self.block_id.shape),
            "component_id_shape": list(self.component_id.shape),
            "atom_ptr_shape": list(self.atom_ptr.shape),
            "covalent_bond_index_shape": list(self.covalent_bond_index.shape),
            "covalent_bond_type_shape": list(self.covalent_bond_type.shape),
            "bond_index_space": "latent_atom_axis_N",
            "bond_encoding": "binary_covalent_connectivity",
            "coordinate_independent": True,
            "frame_invariant": True,
            "contains_radius_edges": False,
            "contains_distance_or_edge_vectors": False,
            "contains_target_coordinates": False,
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
class StateDetailDecoderOutput:
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


class _EquivariantCoordinateHead(nn.Module):
    """Map vector channels to coordinates using invariant scalar gates."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, 1, bias=False)

    def forward(self, h: Tensor, v: Tensor) -> Tensor:
        if h.ndim != 3 or v.ndim != 4 or v.shape[:2] != h.shape[:2] or v.shape[2] != 3:
            raise ValueError("coordinate head expects h=[T,N,C] and v=[T,N,3,C]")
        gate = torch.sigmoid(self.gate(h)).unsqueeze(2)
        return self.out(v * gate).squeeze(-1).float()


class CenteredCoordinateVectorStem(nn.Module):
    """Inject centered coordinates into generated equivariant vector features.

    This is part of the learned latent path, not coordinate metadata.  The
    map is bias-free and acts independently on the Cartesian components, so a
    translation removed by ``center_coordinates`` cannot re-enter the vector
    representation as an extra origin or time-dependent displacement.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        if int(channels) < 1:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.projection = nn.Linear(1, self.channels, bias=False)
        nn.init.ones_(self.projection.weight)

    def forward(self, centered_coordinates: Tensor) -> Tensor:
        if centered_coordinates.ndim != 3 or centered_coordinates.shape[-1] != 3:
            raise ValueError("centered coordinates must have shape [T, N, 3]")
        return self.projection(centered_coordinates.unsqueeze(-1))


def _frame_mask_for_atoms(latent: StateDetailLatent | MatchedPoolingLatent) -> Tensor:
    mask = latent.frame_time_ps.new_zeros(latent.frame_time_ps.shape, dtype=torch.bool)
    block = latent.block_frame_mask.reshape(latent.block_frame_mask.shape[0], -1)
    mask[:, : block.shape[1]] = block
    return mask.index_select(0, latent.abid).transpose(0, 1)


class StateDetailCodecV2(nn.Module):
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
                "StateDetailCodecV2 mode must be ratio1_state_detail, "
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
        self.coordinate_head = _EquivariantCoordinateHead(self.channels)

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

    def decode(self, latent: StateDetailLatent) -> StateDetailDecoderOutput:
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
        centered = self.coordinate_head(decoded_h, decoded_v)
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
        return StateDetailDecoderOutput(
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
    ) -> StateDetailDecoderOutput:
        latent = self.encode(
            h,
            v,
            time_ps=time_ps,
            frame_mask=frame_mask,
            abid=abid,
            sample_origin=sample_origin,
            topology=topology,
        )
        return self.decode(latent)

    def decode_detail_zero(self, latent: StateDetailLatent) -> StateDetailDecoderOutput:
        if latent.detail_h is None or latent.detail_v is None:
            return self.decode(latent)
        zero = replace(
            latent,
            detail_h=torch.zeros_like(latent.detail_h),
            detail_v=torch.zeros_like(latent.detail_v),
        )
        return self.decode(zero)


class MatchedPoolingCodecV2(nn.Module):
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
        self.coordinate_head = _EquivariantCoordinateHead(self.channels)

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

    def decode(self, latent: MatchedPoolingLatent) -> StateDetailDecoderOutput:
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
        centered = self.coordinate_head(decoded_h, decoded_v)
        origin = latent.sample_origin.to(device=centered.device, dtype=torch.float32)
        atom_origin = origin.index_select(0, latent.abid)
        x_centered = centered * atom_valid.unsqueeze(-1).to(dtype=centered.dtype)
        x_hat = x_centered + atom_origin.unsqueeze(0)
        x_hat = torch.where(atom_valid.unsqueeze(-1), x_hat, atom_origin.unsqueeze(0))
        diagnostics = {
            "bank_a_h_norm": latent.bank_a_h.detach().float().square().mean().sqrt(),
            "bank_b_h_norm": latent.bank_b_h.detach().float().square().mean().sqrt(),
        }
        return StateDetailDecoderOutput(
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
    ) -> StateDetailDecoderOutput:
        return self.decode(
            self.encode(
                h,
                v,
                time_ps=time_ps,
                frame_mask=frame_mask,
                abid=abid,
                sample_origin=sample_origin,
                topology=topology,
            )
        )


# Short names make the phase-specific module easy to discover in tests and tools.
StateDetailCodec = StateDetailCodecV2
MatchedPoolingCodec = MatchedPoolingCodecV2
StateDetailLatentV2 = StateDetailLatent


__all__ = [
    "CenteredCoordinateVectorStem",
    "HaarLift",
    "MatchedPoolingCodec",
    "MatchedPoolingCodecV2",
    "MatchedPoolingLatent",
    "ORIGIN_RULE",
    "RATIO_FOR_MODE",
    "STATE_DETAIL_CODEC_SCHEMA",
    "STATE_DETAIL_MODES",
    "StateDetailCodec",
    "StateDetailCodecV2",
    "StateDetailDecoderOutput",
    "StateDetailLatent",
    "StateDetailLatentV2",
    "STATIC_TOPOLOGY_SCHEMA",
    "StaticTopologyMetadata",
    "center_coordinates",
    "compute_masked_centroid_origin",
    "haar_inverse",
    "haar_lift",
]
