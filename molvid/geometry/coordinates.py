"""Coordinate centering, restoration and trajectory alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


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


def restore_origin(centered: Tensor, origin: Tensor, abid: Tensor) -> Tensor:
    """Undo per-sample centering without changing the atom or frame order."""

    if centered.ndim != 3 or centered.shape[-1] != 3:
        raise ValueError("centered coordinates must have shape [T, N, 3]")
    atom_index = torch.as_tensor(abid, device=centered.device, dtype=torch.long)
    if atom_index.numel() != centered.shape[1]:
        raise ValueError("abid must have one entry per packed atom")
    return centered + origin.index_select(0, atom_index).unsqueeze(0)
def kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching [M, 3] shapes")
    if mobile.shape[0] == 0:
        raise ValueError("Kabsch requires at least one alignment atom")
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    covariance = mobile_centered.T @ reference_centered
    u, _, vh = np.linalg.svd(covariance, full_matrices=False)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    translation = reference_center - mobile_center @ rotation
    return rotation.astype(np.float64), translation.astype(np.float64)


def align_and_center(
    x: np.ndarray,
    bpos: np.ndarray,
    align_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Align every frame to its clip's first frame and center once."""

    if x.ndim != 3 or x.shape[-1] != 3 or bpos.shape != x.shape:
        raise ValueError("coordinates must have matching [T, N, 3] shapes")
    reference = x[0, align_mask].astype(np.float64)
    aligned_x = np.empty_like(x, dtype=np.float32)
    aligned_bpos = np.empty_like(bpos, dtype=np.float32)
    rotations: list[list[list[float]]] = []
    translations: list[list[float]] = []
    for frame_idx in range(x.shape[0]):
        if frame_idx == 0:
            rotation = np.eye(3, dtype=np.float64)
            translation = np.zeros(3, dtype=np.float64)
        else:
            rotation, translation = kabsch_align(
                x[frame_idx, align_mask].astype(np.float64), reference
            )
        aligned_x[frame_idx] = x[frame_idx] @ rotation + translation
        aligned_bpos[frame_idx] = bpos[frame_idx] @ rotation + translation
        rotations.append(rotation.tolist())
        translations.append(translation.tolist())
    center = aligned_x[0].mean(axis=0).astype(np.float32)
    aligned_x -= center
    aligned_bpos -= center
    return aligned_x, aligned_bpos, {
        "rotation": rotations,
        "translation": translations,
        "center_angstrom": center.tolist(),
        "alignment_atom_count": int(align_mask.sum()),
    }
