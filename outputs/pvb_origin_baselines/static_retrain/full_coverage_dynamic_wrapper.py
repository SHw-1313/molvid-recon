"""Output-local DynamicBatchWrapper that retains oversized records.

The reference implementation silently drops an item when its complexity is
above ``ubound_per_batch``.  This wrapper keeps that item as a singleton batch
so the clip-to-pair mapping remains one-to-one.  It does not change the
reference collator, model, trainer, or loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data.dataset_wrapper import DynamicBatchWrapper as _OriginDynamicBatchWrapper


OUTPUT_ROOT = Path("/workspace/PVB/outputs/pvb_origin_baselines/static_retrain")
AUDIT_PATH = OUTPUT_ROOT / "batch_coverage_audit.json"
_AUDITS: dict[str, dict] = {}


def _dataset_key(dataset) -> str:
    root = getattr(dataset, "clip_root", None)
    return str(root) if root is not None else repr(dataset)


def _write_audit() -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(_AUDITS, indent=2, sort_keys=True) + "\n")


class FullCoverageDynamicBatchWrapper(_OriginDynamicBatchWrapper):
    """Original dynamic batching with singleton retention for oversized items."""

    def _form_batch(self):
        last_batch_indexes = self.batch_indexes
        self.batch_indexes = []

        if not self.same_origin:
            np.random.shuffle(self.indexes)
            ordered_indexes = self.indexes
        else:
            for key in self.index_dict:
                np.random.shuffle(self.index_dict[key])
            grouped_index = list(self.index_dict.values())
            np.random.shuffle(grouped_index)
            self.indexes = [idx for group in grouped_index for idx in group]
            ordered_indexes = self.indexes

        cur_complexity = 0
        batch = []
        oversized = []
        for i in ordered_indexes:
            item_len = float(self.eval_func(self.dataset.get_len(i)))
            if item_len > self.ubound_per_batch:
                if batch:
                    self.batch_indexes.append(batch)
                    batch = []
                    cur_complexity = 0
                self.batch_indexes.append([i])
                oversized.append(i)
                continue

            origin_break = (
                self.same_origin
                and batch
                and self.dataset.get_origin(i) != self.dataset.get_origin(batch[-1])
            )
            if batch and (cur_complexity + item_len > self.ubound_per_batch or origin_break):
                self.batch_indexes.append(batch)
                batch = []
                cur_complexity = 0
            batch.append(i)
            cur_complexity += item_len
        if batch:
            self.batch_indexes.append(batch)

        if self.total_size is None:
            self.total_size = len(self.batch_indexes)
        elif len(self.batch_indexes) < self.total_size:
            self.batch_indexes += last_batch_indexes[: self.total_size - len(self.batch_indexes)]
        else:
            self.batch_indexes = self.batch_indexes[: self.total_size]

        key = _dataset_key(self.dataset)
        _AUDITS[key] = {
            "ubound_per_batch": self.ubound_per_batch,
            "items": len(self.dataset),
            "included_items_in_current_epoch": sum(len(x) for x in self.batch_indexes),
            "batches_in_current_epoch": len(self.batch_indexes),
            "oversized_items_in_untruncated_order": len(oversized),
            "max_item_complexity": max(
                (float(self.eval_func(self.dataset.get_len(i))) for i in ordered_indexes),
                default=0,
            ),
        }
        _write_audit()
