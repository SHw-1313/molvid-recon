"""Small post-decoder temporal refiner for the sequential R4 DiT study.

The refiner is deliberately separate from the frozen state/detail codec.  It
only mixes temporal neighbours of the *same atom*, using scalar-invariant
gates and a bias-free channel projection of vector-feature differences.  Its
output is therefore translation equivariant and rotation equivariant whenever
the decoder vector features are.  It contains no coordinate bias, topology
message passing, or raw-coordinate shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .dit_geometry_supervision import FutureBondLoss, future_bond_distance_loss


@dataclass(frozen=True)
class TemporalRefinerOutput:
    """Refined coordinates plus the applied future-only correction."""

    x_refined: Tensor
    delta_x: Tensor
    edge_mask: Tensor
    future_atom_mask: Tensor


@dataclass(frozen=True)
class RefinerLoss:
    """Coordinate-domain post-decoder objective in physical Angstrom units."""

    total: Tensor
    coordinate: Tensor
    bond: FutureBondLoss
    motion: Tensor
    coordinate_applicable_samples: int
    motion_applicable_samples: int
    motion_valid_pairs: int

    def diagnostics(self) -> dict[str, Any]:
        result = {
            "refiner_total_loss": self.total,
            "refiner_coordinate_loss": self.coordinate,
            "refiner_bond_loss": self.bond.loss,
            "refiner_motion_loss": self.motion,
            "refiner_coordinate_applicable_samples": self.coordinate_applicable_samples,
            "refiner_motion_applicable_samples": self.motion_applicable_samples,
            "refiner_motion_valid_pairs": self.motion_valid_pairs,
        }
        result.update({f"refiner_{key}": value for key, value in self.bond.diagnostics().items()})
        return result


def _traceable_zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _atom_frame_mask(frame_mask: Tensor, abid: Tensor) -> Tensor:
    """Expand a ``[B,T]`` frame mask to the packed ``[T,N]`` atom layout."""

    return frame_mask.index_select(0, abid).transpose(0, 1)


def _validate_inputs(
    x_hat: Tensor,
    h: Tensor,
    v: Tensor,
    frame_mask: Tensor,
    time_ps: Tensor,
    atom_mask: Tensor,
    observed_frames: Tensor,
    abid: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    if x_hat.ndim != 3 or x_hat.shape[-1] != 3:
        raise ValueError("x_hat must have shape [T,N,3]")
    if h.ndim != 3 or h.shape[:2] != x_hat.shape[:2]:
        raise ValueError("h must have shape [T,N,C] aligned to x_hat")
    if v.ndim != 4 or v.shape[:2] != x_hat.shape[:2] or v.shape[2] != 3:
        raise ValueError("v must have shape [T,N,3,C] aligned to x_hat")
    if v.shape[-1] != h.shape[-1]:
        raise ValueError("h and v channel widths must agree")
    frames, atoms = x_hat.shape[:2]
    device = x_hat.device
    frame_mask = torch.as_tensor(frame_mask, device=device, dtype=torch.bool)
    observed_frames = torch.as_tensor(observed_frames, device=device, dtype=torch.bool)
    time_ps = torch.as_tensor(time_ps, device=device, dtype=x_hat.dtype)
    atom_mask = torch.as_tensor(atom_mask, device=device, dtype=torch.bool).flatten()
    abid = torch.as_tensor(abid, device=device, dtype=torch.long).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != frames:
        raise ValueError("frame_mask must have shape [B,T]")
    if observed_frames.shape != frame_mask.shape:
        raise ValueError("observed_frames must match frame_mask")
    if time_ps.shape != frame_mask.shape:
        raise ValueError("time_ps must match frame_mask")
    if atom_mask.numel() != atoms or abid.numel() != atoms:
        raise ValueError("atom_mask and abid must have one entry per packed atom")
    if abid.numel() and (torch.any(abid < 0) or torch.any(abid >= frame_mask.shape[0])):
        raise ValueError("abid contains an invalid sample index")
    return x_hat, h, v, frame_mask, time_ps, atom_mask, observed_frames, abid


class TrajectoryTemporalRefiner(nn.Module):
    """One-hop same-atom temporal correction after frozen coordinate decoding.

    ``allow_cross_block`` is the sole architectural switch between the Round 3
    L and T arms.  The channel projection is zero-initialized, making a fresh
    refiner an exact identity independent of all scalar-gate parameters.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden: int = 64,
        block_frames: int = 4,
        allow_cross_block: bool,
    ) -> None:
        super().__init__()
        if int(channels) < 1 or int(hidden) < 1 or int(block_frames) < 1:
            raise ValueError("channels, hidden, and block_frames must be positive")
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.block_frames = int(block_frames)
        self.allow_cross_block = bool(allow_cross_block)
        # current h, h_neighbour-h_current, ||v_neighbour-v_current||, dt,
        # current-observed and neighbour-observed flags.
        self.gate = nn.Sequential(
            nn.Linear(3 * self.channels + 3, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 1),
        )
        self.vector_projection = nn.Linear(self.channels, 1, bias=False)
        nn.init.zeros_(self.vector_projection.weight)

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "pvb.dit.trajectory_temporal_refiner.v1",
            "channels": self.channels,
            "hidden": self.hidden,
            "block_frames": self.block_frames,
            "allow_cross_block": self.allow_cross_block,
            "edges": "same_atom_adjacent_valid_frames_only",
            "cross_block_edges": "enabled" if self.allow_cross_block else "disabled",
            "vector_projection": "bias_free_channel_only_zero_initialized",
            "coordinate_bias": False,
            "output": "x_hat_plus_future_only_delta",
        }

    def _edge_mask(
        self,
        frame_mask: Tensor,
        time_ps: Tensor,
        observed_frames: Tensor,
        abid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        atom_valid = _atom_frame_mask(frame_mask, abid)
        atom_observed = _atom_frame_mask(observed_frames, abid)
        atom_time = _atom_frame_mask(time_ps, abid)
        frames, atoms = atom_valid.shape
        mask = torch.zeros((frames, atoms, 2), device=atom_valid.device, dtype=torch.bool)
        dt = atom_time.new_zeros((frames, atoms, 2))
        if frames < 2:
            return mask, dt, atom_valid & ~atom_observed
        forward = atom_valid[:-1] & atom_valid[1:]
        valid_dt = (atom_time[1:] - atom_time[:-1]).abs() > 0
        forward = forward & valid_dt
        if not self.allow_cross_block:
            transition = torch.arange(frames - 1, device=atom_valid.device)
            same_block = (transition // self.block_frames) == ((transition + 1) // self.block_frames)
            forward = forward & same_block[:, None]
        mask[:-1, :, 1] = forward
        mask[1:, :, 0] = forward
        forward_dt = (atom_time[1:] - atom_time[:-1]).abs()
        dt[:-1, :, 1] = forward_dt
        dt[1:, :, 0] = forward_dt
        return mask, dt, atom_valid & ~atom_observed

    def forward(
        self,
        x_hat: Tensor,
        h: Tensor,
        v: Tensor,
        *,
        frame_mask: Tensor,
        time_ps: Tensor,
        atom_mask: Tensor,
        observed_frames: Tensor,
        abid: Tensor,
    ) -> TemporalRefinerOutput:
        x_hat, h, v, frame_mask, time_ps, atom_mask, observed_frames, abid = _validate_inputs(
            x_hat, h, v, frame_mask, time_ps, atom_mask, observed_frames, abid
        )
        if h.shape[-1] != self.channels:
            raise ValueError("refiner channel width does not match decoder features")
        edge_mask, edge_dt, future_atom = self._edge_mask(
            frame_mask, time_ps, observed_frames, abid
        )
        frames, atoms, _ = x_hat.shape
        observed_atom = _atom_frame_mask(observed_frames, abid)
        scalar_dtype = self.gate[0].weight.dtype
        aggregate = x_hat.float().new_zeros((frames, atoms, 3))
        counts = x_hat.float().new_zeros((frames, atoms, 1))
        for direction, offset in enumerate((-1, 1)):
            neighbour_h = torch.zeros_like(h)
            neighbour_v = torch.zeros_like(v)
            neighbour_observed = torch.zeros_like(observed_atom)
            if offset < 0:
                neighbour_h[1:] = h[:-1]
                neighbour_v[1:] = v[:-1]
                neighbour_observed[1:] = observed_atom[:-1]
            else:
                neighbour_h[:-1] = h[1:]
                neighbour_v[:-1] = v[1:]
                neighbour_observed[:-1] = observed_atom[1:]
            vector_difference = (neighbour_v - v).to(dtype=scalar_dtype)
            vector_norm = torch.linalg.vector_norm(vector_difference, dim=2)
            features = torch.cat(
                (
                    h.to(dtype=scalar_dtype),
                    (neighbour_h - h).to(dtype=scalar_dtype),
                    vector_norm,
                    (edge_dt[..., direction : direction + 1] / 100.0).to(dtype=scalar_dtype),
                    observed_atom.unsqueeze(-1).to(dtype=scalar_dtype),
                    neighbour_observed.unsqueeze(-1).to(dtype=scalar_dtype),
                ),
                dim=-1,
            )
            gate = torch.sigmoid(self.gate(features))
            projected = self.vector_projection(vector_difference).squeeze(-1)
            active = edge_mask[..., direction].unsqueeze(-1).to(dtype=projected.dtype)
            aggregate = aggregate + gate * projected * active
            counts = counts + active.to(dtype=counts.dtype)
        delta = aggregate / counts.clamp_min(1.0)
        future_atom = future_atom & atom_mask.unsqueeze(0)
        delta = delta * future_atom.unsqueeze(-1).to(dtype=delta.dtype)
        x_refined = x_hat + delta.to(dtype=x_hat.dtype)
        # This explicit select protects exact observed/padding output values even under AMP.
        x_refined = torch.where(future_atom.unsqueeze(-1), x_refined, x_hat)
        return TemporalRefinerOutput(
            x_refined=x_refined,
            delta_x=delta,
            edge_mask=edge_mask,
            future_atom_mask=future_atom,
        )


def _per_sample_coordinate_mse(
    prediction: Tensor,
    target: Tensor,
    *,
    frame_mask: Tensor,
    observed_frames: Tensor,
    atom_mask: Tensor,
    abid: Tensor,
) -> tuple[Tensor, int]:
    future = _atom_frame_mask(frame_mask, abid) & ~_atom_frame_mask(observed_frames, abid)
    valid = future & atom_mask.unsqueeze(0)
    error = ((prediction.float() - target.float()) / 1.0).square().mean(dim=-1)
    batch_size = int(frame_mask.shape[0])
    sample = abid.unsqueeze(0).expand_as(valid)
    sums = prediction.new_zeros((batch_size,), dtype=torch.float32)
    counts = prediction.new_zeros((batch_size,), dtype=torch.float32)
    sums.scatter_add_(0, sample[valid], error[valid])
    counts.scatter_add_(0, sample[valid], torch.ones_like(error[valid], dtype=torch.float32))
    applicable = counts > 0
    if not bool(torch.any(applicable)):
        return _traceable_zero(prediction), 0
    return (sums[applicable] / counts[applicable]).mean(), int(applicable.sum().item())


def future_motion_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    frame_mask: Tensor,
    observed_frames: Tensor,
    time_ps: Tensor,
    atom_mask: Tensor,
    abid: Tensor,
) -> tuple[Tensor, int, int]:
    """Per-sample-equal adjacent displacement mismatch at the physical 100 ps scale."""

    frames = int(prediction.shape[0])
    if frames < 2:
        return _traceable_zero(prediction), 0, 0
    pair_valid = frame_mask[:, :-1] & frame_mask[:, 1:]
    pair_future = (~observed_frames[:, :-1]) | (~observed_frames[:, 1:])
    dt = (time_ps[:, 1:] - time_ps[:, :-1]).abs()
    pair_valid = pair_valid & pair_future & (dt > 0)
    atom_pair = pair_valid.index_select(0, abid).transpose(0, 1)
    atom_dt = dt.index_select(0, abid).transpose(0, 1)
    valid = atom_pair & atom_mask.unsqueeze(0)
    pred_motion = (prediction[1:].float() - prediction[:-1].float()) * (100.0 / atom_dt.unsqueeze(-1).clamp_min(1e-12))
    target_motion = (target[1:].float() - target[:-1].float()) * (100.0 / atom_dt.unsqueeze(-1).clamp_min(1e-12))
    error = ((pred_motion - target_motion) / 1.0).square().mean(dim=-1)
    batch_size = int(frame_mask.shape[0])
    sample = abid.unsqueeze(0).expand_as(valid)
    sums = prediction.new_zeros((batch_size,), dtype=torch.float32)
    counts = prediction.new_zeros((batch_size,), dtype=torch.float32)
    sums.scatter_add_(0, sample[valid], error[valid])
    counts.scatter_add_(0, sample[valid], torch.ones_like(error[valid], dtype=torch.float32))
    applicable = counts > 0
    if not bool(torch.any(applicable)):
        return _traceable_zero(prediction), 0, int(valid.sum().item())
    return (
        (sums[applicable] / counts[applicable]).mean(),
        int(applicable.sum().item()),
        int(valid.sum().item()),
    )


def temporal_refiner_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    frame_mask: Tensor,
    observed_frames: Tensor,
    time_ps: Tensor,
    atom_mask: Tensor,
    bond_index: Tensor,
    abid: Tensor,
    bond_weight: float = 0.1,
    motion_weight: float = 0.1,
) -> RefinerLoss:
    """Return ``L_coord + 0.1 L_bond + 0.1 L_motion`` for a refiner arm."""

    if float(bond_weight) < 0.0 or float(motion_weight) < 0.0:
        raise ValueError("refiner loss weights must be non-negative")
    (
        prediction,
        _h,
        _v,
        frame_mask,
        time_ps,
        atom_mask,
        observed_frames,
        abid,
    ) = _validate_inputs(
        prediction,
        prediction.new_zeros((*prediction.shape[:2], 1)),
        prediction.new_zeros((*prediction.shape[:2], 3, 1)),
        frame_mask,
        time_ps,
        atom_mask,
        observed_frames,
        abid,
    )
    target = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
    if target.shape != prediction.shape:
        raise ValueError("target must match refiner prediction shape")
    coordinate, coordinate_samples = _per_sample_coordinate_mse(
        prediction,
        target,
        frame_mask=frame_mask,
        observed_frames=observed_frames,
        atom_mask=atom_mask,
        abid=abid,
    )
    bond = future_bond_distance_loss(
        prediction,
        target,
        frame_mask=frame_mask,
        observed_frames=observed_frames,
        loss_mask=atom_mask,
        bond_index=bond_index,
        abid=abid,
        eligible_samples=torch.ones(frame_mask.shape[0], device=prediction.device, dtype=torch.bool),
    )
    motion, motion_samples, motion_pairs = future_motion_loss(
        prediction,
        target,
        frame_mask=frame_mask,
        observed_frames=observed_frames,
        time_ps=time_ps,
        atom_mask=atom_mask,
        abid=abid,
    )
    total = coordinate + float(bond_weight) * bond.loss + float(motion_weight) * motion
    return RefinerLoss(
        total=total,
        coordinate=coordinate,
        bond=bond,
        motion=motion,
        coordinate_applicable_samples=coordinate_samples,
        motion_applicable_samples=motion_samples,
        motion_valid_pairs=motion_pairs,
    )


__all__ = [
    "RefinerLoss",
    "TemporalRefinerOutput",
    "TrajectoryTemporalRefiner",
    "future_motion_loss",
    "temporal_refiner_loss",
]
