"""B split teacher boundary; never encode raw long history.

The learned_memory branch in prepare_frame_joint_batch calls these boundaries.
Only frozen teacher features cross no-grad; raw features are learned later.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from ..data.batch import ClipBatch
from ..flow.source import repeat_last_frame_center
from ..geometry.coordinates import compute_masked_centroid_origin
from ..latent.types import FrameLatentBatch, ObservedContext, QuerySpec
from .frame import FrozenFrameTeacher
from .learned_memory import RawHistoryObservation


@torch.no_grad()
def encode_teacher_at_observed_origin(teacher: FrozenFrameTeacher, frames: ClipBatch, observed_origin: Tensor) -> FrameLatentBatch:
    """Frozen target geometry in a caller-supplied observed coordinate frame.

    Exactly T supplied frames, h [T,N,128], v [T,N,3,128]. Origin [B,3]
    is Angstrom and MUST be derived from observed anchors, never targets.
    Topology registration and device transfer precede this function.
    """
    with torch.autocast(device_type=frames.x.device.type, enabled=False):
        latent, _ = teacher(frames, reference_origin=observed_origin)
    return latent


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
    geometry = ObservedContext(latent, geometry_anchors.x, origin, geometry_anchors.loss_mask.bool(), raw_history)
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
