"""Explicit evaluation families; fixed-history never enters main RMSF."""

from typing import Any, Mapping


METRIC_FAMILY_SCHEMA = "molvid.metric_family.v1"


def metric_family(record: Mapping[str, Any], *, query_frames: int) -> dict[str, Any]:
    bucket = str(record.get("time_bucket_id", ""))
    explicit = record.get("view_kind", record.get("view_family"))
    if explicit == "fixed_history_h4_horizon_1200ps":
        explicit = "fixed_history"
    sample_id = str(record.get("sample_id", ""))
    fixed = bucket.startswith("fixed_history_") or "fixed_history" in sample_id
    if explicit is not None:
        if explicit not in {"ordinary", "legacy_uniform", "fixed_history"}:
            raise ValueError(f"unknown metric family: {explicit}")
        if fixed and explicit != "fixed_history":
            raise ValueError("metric family disagrees with fixed-history provenance")
        fixed = explicit == "fixed_history"
    if explicit is None and not fixed and not bucket.startswith("dt_"):
        raise ValueError("metric family needs explicit view or source time bucket")
    eligible = not fixed and query_frames >= 8
    return {
        "schema": METRIC_FAMILY_SCHEMA,
        "family": "fixed_history" if fixed else "ordinary",
        "query_frames": query_frames,
        "main_rmsf_available": eligible,
        "main_rmsf_reason": "ordinary_Q>=8" if eligible else (
            "fixed_history_excluded_from_main_RMSF" if fixed else "Q<8"
        ),
    }
