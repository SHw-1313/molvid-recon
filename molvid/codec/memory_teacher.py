"""B split teacher boundary; never encode raw long history.

This private boundary is pending integration into prepare_frame_joint_batch
once A publishes its common commit. It is not yet a training backend switch.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import torch
from torch import Tensor
from ..data.batch import ClipBatch
from ..flow.source import repeat_last_frame_center
from ..geometry.coordinates import compute_masked_centroid_origin
from ..geometry.types import StaticTopologyMetadata
from ..latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from .frame import FrozenFrameTeacher, slice_clip_frames
from .learned_memory import RawHistoryObservation


@torch.no_grad()
def encode_teacher_at_observed_origin(teacher: FrozenFrameTeacher, frames: ClipBatch, observed_origin: Tensor) -> FrameLatentBatch:
    """Frozen target geometry in a caller-supplied observed coordinate frame.

    Exactly T supplied frames, h [T,N,128], v [T,N,3,128]. Origin [B,3]
    is Angstrom and MUST be derived from observed anchors, never targets.
    Topology registration and device transfer precede this function.
    """
    if observed_origin.shape != (frames.frame_mask.shape[0], 3):
        raise ValueError("observed origin must have shape [B,3]")
    if observed_origin.device != frames.x.device or not bool(torch.isfinite(observed_origin).all()):
        raise ValueError("observed origin must be finite and on the model device")
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError("teacher must remain frozen/eval")
    centered = frames.x - observed_origin[frames.abid][None]
    centered_batch = replace(frames, x=centered)
    scalar, vector = [], []
    # Do not let BF16 round physical clocks or the reference coordinate origin.
    with torch.autocast(device_type=frames.x.device.type, enabled=False):
        for start in range(0, frames.frames, teacher.frame_chunk_size):
            stop = min(start + teacher.frame_chunk_size, frames.frames)
            chunk = slice_clip_frames(centered_batch, start, stop)
            valid = chunk.frame_mask[chunk.abid].T
            if bool(chunk.frame_mask.any()):
                encoded = teacher.frame_encoder(chunk)
                h = encoded.h.float()
                v = encoded.v.float() + teacher.coordinate_stem(centered[start:stop].float()).float()
                scalar.append(torch.where(valid[..., None], h, 0))
                vector.append(torch.where(valid[..., None, None], v, 0))
            elif scalar:
                scalar.append(scalar[0].new_zeros(stop-start, frames.atom_count, scalar[0].shape[-1]))
                vector.append(vector[0].new_zeros(stop-start, frames.atom_count, 3, vector[0].shape[-1]))
            else:
                raise ValueError("first teacher chunk must have an observed/target valid frame")
    return FrameLatentBatch(torch.cat(scalar), torch.cat(vector), frames.time_ps.float(), frames.frame_mask, StaticTopologyMetadata.from_batch(frames))


@dataclass(frozen=True)
class PreparedMemoryObservation:
    raw_history: RawHistoryObservation
    geometry: ObservedContext  # exactly two complete teacher anchor frames
    teacher_observed_frames: int = 2


@torch.no_grad()
def prepare_memory_observation(teacher: FrozenFrameTeacher, raw_history: RawHistoryObservation, geometry_anchors: ClipBatch) -> PreparedMemoryObservation:
    """Inference boundary: observed coordinates only, no future argument."""
    expected = geometry_anchors.time_ps.new_tensor([-800, 0])[None].expand_as(geometry_anchors.time_ps)
    if geometry_anchors.frames != 2 or not bool(geometry_anchors.frame_mask.all()) or not torch.equal(geometry_anchors.time_ps, expected):
        raise ValueError("B geometry anchors must be exactly [-800,0] ps")
    topology = raw_history.topology
    for name, raw in (("abid", topology.abid), ("atype", topology.atom_type), ("block_id", topology.block_id), ("component_id", topology.component_id)):
        if not torch.equal(getattr(geometry_anchors, name), raw):
            raise ValueError(f"anchor/raw atom mapping differs: {name}")
    for anchor_slot, time in enumerate((-800, 0)):
        match = (raw_history.physical_time_ps == time) & raw_history.frame_mask
        if not bool((match.sum(1) == 1).all()):
            raise ValueError("raw observed history must contain each geometry anchor")
        frame = match.long().argmax(1)[topology.abid]
        atom = torch.arange(topology.num_atoms, device=frame.device)
        if not torch.equal(raw_history.coordinates[frame, atom], geometry_anchors.x[anchor_slot]):
            raise ValueError("geometry anchor coordinates must be the same raw observed frames")
    origin = compute_masked_centroid_origin(geometry_anchors.x, frame_mask=geometry_anchors.frame_mask,
                                            abid=geometry_anchors.abid, atom_mask=geometry_anchors.loss_mask)
    latent = encode_teacher_at_observed_origin(teacher, geometry_anchors, origin)
    geometry = ObservedContext(latent, geometry_anchors.x, origin, geometry_anchors.loss_mask.bool())
    return PreparedMemoryObservation(raw_history, geometry)


@dataclass(frozen=True)
class PreparedMemoryTargets:
    """Training-only target branch; never pass this object into generation."""
    target_future: FrameLatentBatch
    normalized_target: FrameLatentBatch
    source_center: FrameLatentBatch
    query: QuerySpec
    target_coordinates: Tensor
    teacher_target_frames: int


@torch.no_grad()
def prepare_memory_targets(teacher: FrozenFrameTeacher, observed: PreparedMemoryObservation, future: ClipBatch, normalizer) -> PreparedMemoryTargets:
    if bool((future.time_ps[future.frame_mask] <= 0).any()):
        raise ValueError("target frames must be strictly after the observed prefix")
    origin = observed.geometry.sample_origin
    topology = observed.geometry.topology
    for name, expected in (("abid", topology.abid), ("atype", topology.atom_type), ("block_id", topology.block_id), ("component_id", topology.component_id)):
        if not torch.equal(getattr(future, name), expected):
            raise ValueError(f"target/observed atom mapping differs: {name}")
    target = encode_teacher_at_observed_origin(teacher, future, origin)
    query = QuerySpec(target.time_ps, target.frame_mask)
    center = repeat_last_frame_center(normalizer.normalize(observed.geometry.latent), query)
    return PreparedMemoryTargets(target, normalizer.normalize(target), center, query, future.x, future.frames)
