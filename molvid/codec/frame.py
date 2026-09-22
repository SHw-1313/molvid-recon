"""Frozen per-frame geometry target used by Frame Joint v1."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from ..data.batch import ClipBatch
from ..geometry.coordinates import center_coordinates
from ..geometry.types import StaticTopologyMetadata
from ..latent.types import FrameLatentBatch, ObservedContext


def slice_clip_frames(batch: ClipBatch, start: int, stop: int) -> ClipBatch:
    """Slice only the physical frame axis while retaining static topology."""

    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= batch.frames:
        raise ValueError("invalid ClipBatch frame slice")
    time = batch.time_ps[:, start:stop]
    return replace(
        batch,
        x=batch.x[start:stop],
        bpos=batch.bpos[start:stop],
        frame_mask=batch.frame_mask[:, start:stop],
        time_ps=time,
        delta_time_ps=time[:, 1:] - time[:, :-1],
    )


class FrozenFrameTeacher(nn.Module):
    """A strictly frozen TorchMD frame encoder plus coordinate vector stem."""

    frame_chunk_size = 4

    def __init__(self, frame_encoder: nn.Module, coordinate_stem: nn.Module) -> None:
        super().__init__()
        self.frame_encoder = frame_encoder
        self.coordinate_stem = coordinate_stem
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_codec(cls, codec: nn.Module) -> "FrozenFrameTeacher":
        if getattr(codec, "coordinate_stem", None) is None:
            raise ValueError("Frame Joint target codec must contain the centered coordinate stem")
        return cls(copy.deepcopy(codec.frame_encoder), copy.deepcopy(codec.coordinate_stem))

    def train(self, mode: bool = True):
        return super().train(False)

    def prepare_batch(self, batch: ClipBatch) -> None:
        self.frame_encoder.prepare_batch(batch)

    @torch.no_grad()
    def forward(self, batch: ClipBatch) -> tuple[FrameLatentBatch, Tensor]:
        """Encode exactly the supplied frames; no hidden future is accepted."""

        centered, origin = center_coordinates(
            batch.x,
            frame_mask=batch.frame_mask,
            abid=batch.abid,
            atom_mask=batch.loss_mask,
        )
        centered_batch = replace(batch, x=centered)
        scalar_chunks: list[Tensor] = []
        vector_chunks: list[Tensor] = []
        for start in range(0, batch.frames, self.frame_chunk_size):
            stop = min(start + self.frame_chunk_size, batch.frames)
            chunk = slice_clip_frames(centered_batch, start, stop)
            chunk_atom_mask = chunk.frame_mask.index_select(
                0, chunk.abid
            ).transpose(0, 1)
            if bool(torch.any(chunk.frame_mask)):
                encoded = self.frame_encoder(chunk)
                scalar = encoded.h
                vector = encoded.v + self.coordinate_stem(
                    centered[start:stop]
                ).to(dtype=encoded.v.dtype)
                scalar_chunks.append(scalar * chunk_atom_mask.unsqueeze(-1))
                vector_chunks.append(
                    vector * chunk_atom_mask.unsqueeze(-1).unsqueeze(-1)
                )
            else:
                if not scalar_chunks:
                    raise RuntimeError("the first frame-encoder chunk cannot be padding-only")
                scalar_chunks.append(
                    scalar_chunks[0].new_zeros(
                        stop - start, batch.atom_count, scalar_chunks[0].shape[-1]
                    )
                )
                vector_chunks.append(
                    vector_chunks[0].new_zeros(
                        stop - start,
                        batch.atom_count,
                        3,
                        vector_chunks[0].shape[-1],
                    )
                )
        h = torch.cat(scalar_chunks, dim=0)
        v = torch.cat(vector_chunks, dim=0)
        atom_frame_mask = batch.frame_mask.index_select(0, batch.abid).transpose(0, 1)
        latent = FrameLatentBatch(
            h=h * atom_frame_mask.unsqueeze(-1),
            v=v * atom_frame_mask.unsqueeze(-1).unsqueeze(-1),
            time_ps=batch.time_ps,
            frame_mask=batch.frame_mask,
            topology=StaticTopologyMetadata.from_batch(batch),
        )
        return latent, origin

    @torch.no_grad()
    def observed_context(self, batch: ClipBatch) -> ObservedContext:
        latent, origin = self(batch)
        return ObservedContext(
            latent=latent,
            coordinates=batch.x,
            sample_origin=origin,
            loss_mask=batch.loss_mask.to(dtype=torch.bool),
        )
