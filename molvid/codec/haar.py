"""Fixed ratio-1/2/4 Haar lifting and inverse with explicit partial masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

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
class HaarCoefficients:
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
) -> HaarCoefficients:
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
        return HaarCoefficients(
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
        return HaarCoefficients(
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
    return HaarCoefficients(
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
