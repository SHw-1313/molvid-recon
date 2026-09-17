"""Output-local lazy adapter from current clip stores to original PVB pairs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _clip_dataset_class():
    name = "pvb_current_clip_dataset_for_origin_baseline"
    module_path = Path("/workspace/PVB/data/clip_dataset.py")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.ClipMMapDataset


class StreamingClipPairDataset(torch.utils.data.Dataset):
    """Lazy pairs; ``pairs_per_clip=1`` keeps every T=16 clip once."""

    def __init__(self, clip_root: str | Path, *, pairs_per_clip: int = 1):
        if pairs_per_clip < 1:
            raise ValueError("pairs_per_clip must be positive")
        self.clip_root = Path(clip_root)
        self._clips = _clip_dataset_class()(self.clip_root)
        self.pairs_per_clip = int(pairs_per_clip)
        self._specs = []
        for row in self._clips._clip_index_fields:
            if row is None:
                raise ValueError(f"missing clip metadata in {self.clip_root}")
            sample_id, _start, _end, atoms, frames, bucket = row
            if int(frames) < self.pairs_per_clip + 1:
                raise ValueError(f"not enough frames for {self.clip_root}: {frames}")
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
        return {
            "atype": np.asarray(item["atype"], dtype=np.int64).tolist(),
            "btype": np.asarray(item["btype"], dtype=np.int64).tolist(),
            "x0": x[pair_index].tolist(),
            "b0": bpos[pair_index].tolist(),
            "x1": x[pair_index + 1].tolist(),
            "b1": bpos[pair_index + 1].tolist(),
            "edge_mask": np.asarray(item["edge_mask"], dtype=np.int64).tolist(),
            "mask": np.asarray(item["loss_mask"], dtype=np.bool_).tolist(),
            "bond_index": np.asarray(item["bond_index"], dtype=np.int64).tolist(),
            "clip_id": str(item.get("sample_id", self._specs[clip_index][0])),
            "pair_index": pair_index,
            "source": str(item.get("source", "unknown")),
            "system_id": str(item.get("system_id", "unknown")),
            "replica": str(item.get("replica", "unknown")),
            "time_bucket_id": str(item.get("time_bucket_id", "unknown")),
            "delta_time_ps": float(np.asarray(item["delta_time_ps"])[pair_index]),
        }

    def get_len(self, index: int) -> int:
        clip_index, _ = self._locate(index)
        return self._specs[clip_index][1]

    def get_origin(self, index: int) -> int:
        clip_index, _ = self._locate(index)
        return clip_index

    def get_index_dict(self) -> dict[int, list[int]]:
        return {
            clip_index: list(range(clip_index * self.pairs_per_clip,
                                  (clip_index + 1) * self.pairs_per_clip))
            for clip_index in range(len(self._specs))
        }

    def close(self) -> None:
        self._clips.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_streaming_unidataset(path: str, *, pairs_per_clip: int = 1):
    if not path.startswith("clip://"):
        raise ValueError(f"expected clip:// dataset path, got {path}")
    return StreamingClipPairDataset(path[len("clip://"):], pairs_per_clip=pairs_per_clip)
