"""Differentiable future-bond supervision through a frozen codec decoder."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn

from ..flow.objective import FlowSample, endpoint_from_velocity
from ..flow.sampling import apply_observation_clamp
from ..latent.adapter import StateDetailLatentAdapter
from ..latent.statistics import LatentStatistics
from ..latent.types import LatentBatch, LatentFields


@dataclass(frozen=True)
class FutureBondLoss:
    """One physical-unit loss and its applicability accounting."""

    loss: Tensor
    eligible_samples: int
    applicable_samples: int
    valid_pair_frames: int
    registered_bonds: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "geometry_eligible_samples": int(self.eligible_samples),
            "geometry_applicable_samples": int(self.applicable_samples),
            "geometry_valid_pair_frames": int(self.valid_pair_frames),
            "geometry_registered_bonds": int(self.registered_bonds),
        }


def observed_frame_mask(batch: LatentBatch) -> Tensor:
    """Expand the block-aligned observation contract to physical frames."""

    observed_blocks = batch.observed_mask & batch.token_mask
    expanded = observed_blocks.repeat_interleave(batch.ratio, dim=1)
    frames = int(batch.frame_time_ps.shape[1])
    if expanded.shape[1] < frames:
        raise ValueError("latent observation mask does not cover physical frames")
    return expanded[:, :frames]


def _traceable_zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def future_bond_distance_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    frame_mask: Tensor,
    observed_frames: Tensor,
    loss_mask: Tensor,
    bond_index: Tensor,
    abid: Tensor,
    eligible_samples: Tensor,
) -> FutureBondLoss:
    """Per-sample-equal future covalent-bond MSE in squared Angstroms.

    ``bond_index`` is fixed topology on the packed atom axis.  No radius graph
    or adjacency is inferred from predicted future coordinates.
    """

    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction must have shape [T,N,3]")
    if target.shape != prediction.shape:
        raise ValueError("target coordinates must match prediction")
    frames, atoms, _ = prediction.shape
    device = prediction.device
    frame_mask = torch.as_tensor(frame_mask, device=device, dtype=torch.bool)
    observed_frames = torch.as_tensor(observed_frames, device=device, dtype=torch.bool)
    loss_mask = torch.as_tensor(loss_mask, device=device, dtype=torch.bool).flatten()
    abid = torch.as_tensor(abid, device=device, dtype=torch.long).flatten()
    eligible_samples = torch.as_tensor(
        eligible_samples, device=device, dtype=torch.bool
    ).flatten()
    if frame_mask.ndim != 2 or frame_mask.shape[1] != frames:
        raise ValueError("frame_mask must have shape [B,T]")
    if observed_frames.shape != frame_mask.shape:
        raise ValueError("observed_frames must match frame_mask")
    if loss_mask.numel() != atoms or abid.numel() != atoms:
        raise ValueError("loss_mask and abid must align to coordinate atoms")
    batch_size = int(frame_mask.shape[0])
    if eligible_samples.numel() != batch_size:
        raise ValueError("eligible_samples must have one value per sample")
    if abid.numel() and (torch.any(abid < 0) or torch.any(abid >= batch_size)):
        raise ValueError("abid contains an invalid sample id")
    bonds = torch.as_tensor(bond_index, device=device, dtype=torch.long)
    if bonds.numel() == 0:
        return FutureBondLoss(
            loss=_traceable_zero(prediction),
            eligible_samples=int(eligible_samples.sum().item()),
            applicable_samples=0,
            valid_pair_frames=0,
            registered_bonds=0,
        )
    if bonds.ndim != 2 or bonds.shape[0] != 2:
        raise ValueError("bond_index must have shape [2,E]")
    if torch.any(bonds < 0) or torch.any(bonds >= atoms):
        raise ValueError("bond_index contains an out-of-range atom")
    source, destination = bonds
    bond_sample = abid.index_select(0, source)
    if not torch.equal(bond_sample, abid.index_select(0, destination)):
        raise ValueError("bond_index contains a cross-sample bond")
    valid_bond = loss_mask.index_select(0, source) & loss_mask.index_select(0, destination)
    future_frames = frame_mask & ~observed_frames
    pair_mask = future_frames.index_select(0, bond_sample).transpose(0, 1)
    pair_mask = pair_mask & eligible_samples.index_select(0, bond_sample).unsqueeze(0)
    pair_mask = pair_mask & valid_bond.unsqueeze(0)
    predicted_length = torch.linalg.vector_norm(
        prediction[:, source] - prediction[:, destination], dim=-1
    )
    target_length = torch.linalg.vector_norm(
        target[:, source] - target[:, destination], dim=-1
    )
    squared_error = ((predicted_length - target_length) / 1.0).square()
    sample_index = bond_sample.unsqueeze(0).expand(frames, -1)
    sums = prediction.new_zeros((batch_size,))
    counts = prediction.new_zeros((batch_size,))
    selected_index = sample_index[pair_mask]
    sums.scatter_add_(0, selected_index, squared_error[pair_mask])
    counts.scatter_add_(0, selected_index, torch.ones_like(squared_error[pair_mask]))
    applicable = counts > 0
    if not bool(torch.any(applicable)):
        loss = _traceable_zero(prediction)
    else:
        loss = (sums[applicable] / counts[applicable]).mean()
    return FutureBondLoss(
        loss=loss,
        eligible_samples=int(eligible_samples.sum().item()),
        applicable_samples=int(applicable.sum().item()),
        valid_pair_frames=int(pair_mask.sum().item()),
        registered_bonds=int(bonds.shape[1]),
    )


class FutureBondAuxiliary:
    """Build the round-2 endpoint decode and its weighted future-bond loss."""

    def __init__(
        self,
        *,
        adapter: StateDetailLatentAdapter,
        statistics: LatentStatistics,
        codec: nn.Module,
        coordinate_batch: Any,
        history_frames: int,
        lambda_bond: float,
        tau_threshold: float = 0.75,
    ) -> None:
        if not math.isfinite(float(lambda_bond)) or float(lambda_bond) < 0.0:
            raise ValueError("lambda_bond must be finite and non-negative")
        if not 0.0 <= float(tau_threshold) <= 1.0:
            raise ValueError("tau_threshold must lie in [0,1]")
        if int(history_frames) not in (4, 8):
            raise ValueError("future-bond supervision supports only H=4/H=8")
        for name in ("x", "frame_mask", "loss_mask", "bond_index", "abid"):
            if not hasattr(coordinate_batch, name):
                raise ValueError(f"coordinate batch is missing {name!r}")
        self.adapter = adapter
        self.statistics = statistics
        self.codec = codec
        self.coordinate_batch = coordinate_batch
        self.history_frames = int(history_frames)
        self.lambda_bond = float(lambda_bond)
        self.tau_threshold = float(tau_threshold)

    def raw_loss(
        self,
        prediction: LatentFields,
        flow_sample: FlowSample,
        normalized_batch: LatentBatch,
    ) -> FutureBondLoss:
        if normalized_batch.ratio != self.adapter.ratio or normalized_batch.mode != self.adapter.mode:
            raise ValueError("geometry auxiliary adapter and latent batch disagree")
        tau = torch.as_tensor(
            flow_sample.tau, device=normalized_batch.state_h.device, dtype=normalized_batch.state_h.dtype
        )
        if tau.shape != (normalized_batch.batch_size,):
            raise ValueError("flow tau must have one value per sample")
        eligible = tau >= self.tau_threshold
        if not bool(torch.any(eligible)):
            return FutureBondLoss(
                loss=_traceable_zero(prediction.state_h),
                eligible_samples=0,
                applicable_samples=0,
                valid_pair_frames=0,
                registered_bonds=int(torch.as_tensor(self.coordinate_batch.bond_index).shape[-1]),
            )
        endpoint = endpoint_from_velocity(
            flow_sample.interpolated,
            prediction,
            tau,
            sample_ids=normalized_batch.abid,
        )
        endpoint = apply_observation_clamp(
            endpoint, normalized_batch.fields, normalized_batch
        )
        physical_fields = self.statistics.inverse_fields(endpoint)
        physical_batch = normalized_batch.with_fields(physical_fields).zero_invalid()
        decoded_batch = physical_batch.with_fields(physical_batch.fields.to(dtype=torch.float32))
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if decoded_batch.state_h.device.type == "cuda"
            else nullcontext()
        )
        with context:
            latent = self.adapter.make_generated_latent(decoded_batch, decoded_batch.fields)
            output = self.codec.decode(latent)
            decoded = torch.as_tensor(output.x_hat, device=decoded_batch.state_h.device).float()
        target = torch.as_tensor(
            self.coordinate_batch.x, device=decoded.device, dtype=decoded.dtype
        )
        return future_bond_distance_loss(
            decoded,
            target,
            frame_mask=self.coordinate_batch.frame_mask,
            observed_frames=observed_frame_mask(normalized_batch),
            loss_mask=self.coordinate_batch.loss_mask,
            bond_index=self.coordinate_batch.bond_index,
            abid=self.coordinate_batch.abid,
            eligible_samples=eligible,
        )

    def __call__(
        self,
        prediction: LatentFields,
        flow_sample: FlowSample,
        normalized_batch: LatentBatch,
    ) -> Optional[tuple[Tensor, Mapping[str, Any]]]:
        # A literal no-op is required for the lambda=0 RF-compatibility check.
        if self.lambda_bond == 0.0:
            return None
        result = self.raw_loss(prediction, flow_sample, normalized_batch)
        weighted = result.loss * self.lambda_bond
        diagnostics: dict[str, Any] = {
            "geometry_bond_loss": result.loss.detach(),
            "geometry_weighted_loss": weighted.detach(),
            "geometry_lambda_bond": self.lambda_bond,
            "geometry_tau_threshold": self.tau_threshold,
            **result.diagnostics(),
        }
        return weighted, diagnostics
