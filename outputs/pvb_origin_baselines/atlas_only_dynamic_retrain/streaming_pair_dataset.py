"""Read current npz-v1 trajectory clips as legacy adjacent frame pairs.

This is an output-scoped adapter for the unmodified PVB_origin trainer.  It
does not rewrite the clip stores: the index is read once, while each selected
clip payload is decoded only when ``__getitem__`` is called.  A T=16 clip
contributes its 15 adjacent pairs in deterministic clip/frame order.
"""

from __future__ import annotations

import bisect
import io
import json
import mmap
from pathlib import Path
from typing import Any

import numpy as np
import torch


_ARRAY_KEYS = (
    "x",
    "bpos",
    "atype",
    "btype",
    "edge_mask",
    "loss_mask",
    "bond_index",
)


class StreamingAdjacentPairDataset(torch.utils.data.Dataset):
    """Expose an npz-v1 clip root through the original PVB pair contract."""

    pair_policy = "all_adjacent_pairs"

    def __init__(self, root: str):
        super().__init__()
        self.root = Path(root)
        self.data_path = self.root / "data.bin"
        self.index_path = self.root / "index.txt"
        if not self.data_path.is_file() or not self.index_path.is_file():
            raise FileNotFoundError(
                f"expected npz-v1 clip store with data.bin/index.txt: {self.root}"
            )

        self._entries: list[tuple[str, int, int, int, int, str]] = []
        self._pair_ends: list[int] = []
        total = 0
        with self.index_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 6:
                    raise ValueError(
                        f"malformed clip index line {line_number} in {self.index_path}"
                    )
                sample_id, start, end, atoms, frames, bucket = fields[:6]
                start_i, end_i = int(start), int(end)
                atoms_i, frames_i = int(atoms), int(frames)
                if frames_i < 2 or atoms_i < 1 or end_i <= start_i:
                    raise ValueError(
                        f"invalid clip metadata at line {line_number}: {fields[:6]}"
                    )
                self._entries.append(
                    (sample_id, start_i, end_i, atoms_i, frames_i, bucket)
                )
                total += frames_i - 1
                self._pair_ends.append(total)

        self._data_file = self.data_path.open("rb")
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._cached_clip_index: int | None = None
        self._cached_record: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self._pair_ends[-1] if self._pair_ends else 0

    @property
    def clip_count(self) -> int:
        return len(self._entries)

    @property
    def pair_count(self) -> int:
        return len(self)

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        clip_index = bisect.bisect_right(self._pair_ends, index)
        previous_end = self._pair_ends[clip_index - 1] if clip_index else 0
        return clip_index, index - previous_end

    def get_len(self, index: int) -> int:
        clip_index, _ = self._locate(index)
        return self._entries[clip_index][3]

    def get_index_dict(self) -> dict[int, list[int]]:
        # The baseline uses same_origin=false.  Returning an empty mapping is
        # sufficient for the original wrapper and avoids materializing a
        # second copy of the large pair index.
        return {}

    def get_origin(self, index: int) -> int:
        clip_index, _ = self._locate(index)
        return clip_index

    def _decode_clip(self, clip_index: int) -> dict[str, Any]:
        if clip_index == self._cached_clip_index and self._cached_record is not None:
            return self._cached_record
        _, start, end, _, _, _ = self._entries[clip_index]
        payload = self._mmap[start:end]
        if payload[:2] != b"PK":
            raise ValueError(
                f"expected npz-v1 payload beginning with PK in {self.root}, "
                f"clip index {clip_index} starts with {payload[:2]!r}"
            )
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            record = json.loads(str(archive["metadata"].item()))
            record = dict(record)
            for key in _ARRAY_KEYS:
                record[key] = np.array(archive[key], copy=True)
        self._cached_clip_index = clip_index
        self._cached_record = record
        return record

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_index, frame = self._locate(index)
        record = self._decode_clip(clip_index)
        x = np.asarray(record["x"], dtype=np.float32)
        bpos = np.asarray(record["bpos"], dtype=np.float32)
        atype = np.asarray(record["atype"], dtype=np.int64)
        btype = np.asarray(record["btype"], dtype=np.int64)
        edge_mask = np.asarray(record["edge_mask"], dtype=np.int64)
        loss_mask = np.asarray(record["loss_mask"], dtype=np.bool_)
        bond_index = np.asarray(record["bond_index"], dtype=np.int64)
        if x.ndim != 3 or x.shape[0] < 2 or x.shape[2] != 3:
            raise ValueError(f"invalid x shape in {self.root}: {x.shape}")
        if bpos.shape != x.shape:
            raise ValueError(f"invalid bpos shape in {self.root}: {bpos.shape}")
        atom_count = x.shape[1]
        for name, value in (
            ("atype", atype),
            ("btype", btype),
            ("edge_mask", edge_mask),
            ("loss_mask", loss_mask),
        ):
            if value.ndim != 1 or value.shape[0] != atom_count:
                raise ValueError(f"invalid {name} shape in {self.root}: {value.shape}")
        if bond_index.ndim != 2 or bond_index.shape[0] != 2:
            raise ValueError(f"invalid bond_index shape in {self.root}: {bond_index.shape}")
        return {
            "atype": atype,
            "btype": btype,
            "edge_mask": edge_mask,
            "mask": loss_mask,
            "x0": x[frame],
            "x1": x[frame + 1],
            "b0": bpos[frame],
            "b1": bpos[frame + 1],
            "bond_index": bond_index,
        }

    def close(self) -> None:
        mmap_obj = getattr(self, "_mmap", None)
        if mmap_obj is not None:
            mmap_obj.close()
            self._mmap = None
        data_file = getattr(self, "_data_file", None)
        if data_file is not None and not data_file.closed:
            data_file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
