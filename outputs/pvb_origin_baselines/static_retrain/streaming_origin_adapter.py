"""Lazy clip-to-legacy-pair adapter for the unmodified original PVB entrypoint.

The adapter reads the current ``npz-v1`` clip mmap on demand and returns the
legacy pair dictionary expected by ``PVB_origin.data.collate_fn``.  It never
writes a converted dataset and never changes the reference repository.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_clip_dataset_class():
    module_path = Path("/workspace/PVB/data/clip_dataset.py")
    spec = importlib.util.spec_from_file_location("pvb_current_clip_dataset", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load current clip reader from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ClipMMapDataset


class StreamingClipPairDataset(torch.utils.data.Dataset):
    """Expose all clips or all adjacent pairs without materializing payloads.

    ``pairs_per_clip=1`` maps every clip to its native first transition
    (frame 0 -> frame 1), which matches the one-step PVB training contract and
    retains every current train/valid clip.  ``pairs_per_clip=15`` exposes all
    adjacent transitions from each T=16 clip for an explicitly larger run.
    """

    def __init__(self, clip_root: str | Path, *, pairs_per_clip: int = 1):
        super().__init__()
        if pairs_per_clip < 1:
            raise ValueError("pairs_per_clip must be positive")
        self.clip_root = Path(clip_root)
        ClipMMapDataset = _load_clip_dataset_class()
        self._clips = ClipMMapDataset(self.clip_root)
        self.pairs_per_clip = int(pairs_per_clip)
        self._specs: list[tuple[str, int, int, str]] = []
        for row in self._clips._clip_index_fields:
            if row is None:
                raise ValueError(f"clip index lacks atom/frame metadata: {self.clip_root}")
            sample_id, _start, _end, atoms, frames, bucket = row
            if frames < self.pairs_per_clip + 1:
                raise ValueError(
                    f"{self.clip_root}: requested {self.pairs_per_clip} pairs but clip has {frames} frames"
                )
            self._specs.append((str(sample_id), int(atoms), int(frames), str(bucket)))

    def __len__(self) -> int:
        return len(self._specs) * self.pairs_per_clip

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return divmod(index, self.pairs_per_clip)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_index, pair_index = self._locate(index)
        item = self._clips[clip_index]
        x = np.asarray(item["x"], dtype=np.float32)
        bpos = np.asarray(item["bpos"], dtype=np.float32)
        sample_id = str(item.get("sample_id", self._specs[clip_index][0]))
        record = {
            "atype": np.asarray(item["atype"], dtype=np.int64).tolist(),
            "btype": np.asarray(item["btype"], dtype=np.int64).tolist(),
            "x0": x[pair_index].tolist(),
            "b0": bpos[pair_index].tolist(),
            "x1": x[pair_index + 1].tolist(),
            "b1": bpos[pair_index + 1].tolist(),
            "edge_mask": np.asarray(item["edge_mask"], dtype=np.int64).tolist(),
            "mask": np.asarray(item["loss_mask"], dtype=np.bool_).tolist(),
            "bond_index": np.asarray(item["bond_index"], dtype=np.int64).tolist(),
            "clip_id": sample_id,
            "pair_index": pair_index,
            "source": str(item.get("source", "unknown")),
            "system_id": str(item.get("system_id", "unknown")),
            "replica": str(item.get("replica", "unknown")),
            "time_bucket_id": str(item.get("time_bucket_id", "unknown")),
            "delta_time_ps": float(np.asarray(item["delta_time_ps"])[pair_index]),
        }
        return record

    def get_len(self, index: int) -> int:
        clip_index, _pair_index = self._locate(index)
        return self._specs[clip_index][1]

    def get_origin(self, index: int) -> int:
        clip_index, _pair_index = self._locate(index)
        return clip_index

    def get_index_dict(self) -> dict[int, list[int]]:
        # ``same_origin`` is false for this run, but provide the original API.
        return {clip_index: list(range(clip_index * self.pairs_per_clip,
                                      (clip_index + 1) * self.pairs_per_clip))
                for clip_index in range(len(self._specs))}

    def close(self) -> None:
        close = getattr(self._clips, "close", None)
        if close is not None:
            close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_streaming_unidataset(path: str, *, pairs_per_clip: int = 1):
    """Factory used to monkey-patch only the original ``UniDataset`` symbol."""

    if not path.startswith("clip://"):
        raise ValueError(f"unexpected streaming dataset spec: {path}")
    return StreamingClipPairDataset(path[len("clip://"):], pairs_per_clip=pairs_per_clip)
