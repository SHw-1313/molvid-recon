"""Explicit single-frame records for decoder-only static auxiliary training."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .batch import (
    STATIC_TASK,
    STATIC_TIME_BUCKET_ID,
    TRAJECTORY_TASK,
    validate_clip_record,
)


STATIC_DECODER_TASK_TYPE = "static_decoder_aux"


def trajectory_record_to_static(
    record: Mapping[str, Any],
    *,
    frame_index: int,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Select one real trajectory frame without inventing a temporal transition.

    The returned record retains chemical/topology fields and carries explicit
    source provenance.  Its physical-time axis is the static origin ``[0]``;
    the original trajectory timestamp is metadata, not a zero-lag target.
    """

    source = validate_clip_record(record)
    if int(source["task"]) != TRAJECTORY_TASK:
        raise ValueError("trajectory_record_to_static requires a trajectory record")
    index = int(frame_index)
    if index < 0 or index >= int(source["x"].shape[0]):
        raise IndexError("static source frame index is out of range")
    if not bool(source["frame_mask"][index]):
        raise ValueError("static source frame must be valid")

    source_sample_id = str(source.get("sample_id", ""))
    result = dict(source)
    result.update(
        {
            "sample_id": (
                str(sample_id)
                if sample_id is not None
                else f"{source_sample_id}::static_frame_{index:06d}"
            ),
            "task": STATIC_TASK,
            "training_task_type": STATIC_DECODER_TASK_TYPE,
            "time_bucket_id": STATIC_TIME_BUCKET_ID,
            "time_ps": np.asarray([0.0], dtype=np.float32),
            "delta_time_ps": np.empty((0,), dtype=np.float32),
            "frame_mask": np.asarray([True], dtype=np.bool_),
            "x": np.asarray(source["x"][index : index + 1], dtype=np.float32),
            "bpos": np.asarray(source["bpos"][index : index + 1], dtype=np.float32),
            "static_source_sample_id": source_sample_id,
            "static_source_frame_index": index,
            "static_source_time_ps": float(source["time_ps"][index]),
            "physical_lag_applicable": False,
            "motion_available": False,
        }
    )
    return validate_clip_record(result)

