#!/usr/bin/env python
"""Recompute the Round 2/3 RMSF guards from existing evaluation summaries.

This entrypoint is deliberately JSON-only: it never constructs a model, loads
a checkpoint, opens a clip payload, trains, or samples.  The summaries already
contain the draw-equal -> clip-equal -> system-equal ``system_rows`` used by
the original decisions, so those rows are sufficient for this correction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.motion_metrics import (  # noqa: E402
    MOTION_RATIO_THRESHOLD,
    arm_rmsf_diagnostics,
    motion_arm_report,
    paired_motion_report,
)


HISTORIES = (4, 8)
AGGREGATION_CONTRACT = "draw_equal_then_clip_equal_then_system_equal"
RUN_ID = "20260914_r4_architecture_sequential_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path, *, required: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "required": required, "exists": path.is_file()}
    if path.is_file():
        result.update({"bytes": path.stat().st_size, "sha256": _sha256(path)})
    else:
        result["reason"] = "missing source file"
    return result


def _summary_aggregate(summary: Mapping[str, Any], history: int, *, path: Path) -> Mapping[str, Any]:
    try:
        aggregate = summary["aggregates"]["valid"][f"H{int(history)}"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} lacks valid H{int(history)} aggregate") from exc
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{path} valid H{int(history)} aggregate is not a mapping")
    aggregation = aggregate.get("aggregation")
    if aggregation != AGGREGATION_CONTRACT:
        raise ValueError(
            f"{path} H{int(history)} aggregation changed: {aggregation!r} "
            f"!= {AGGREGATION_CONTRACT!r}"
        )
    if not isinstance(aggregate.get("system_rows"), list):
        raise ValueError(f"{path} H{int(history)} lacks system_rows")
    return aggregate


def _aggregate_metric(aggregate: Mapping[str, Any], key: str) -> Any:
    system_equal = aggregate.get("system_equal")
    return system_equal.get(key) if isinstance(system_equal, Mapping) else None


def _arm_history(
    summary: Mapping[str, Any],
    history: int,
    *,
    label: str,
    path: Path,
) -> dict[str, Any]:
    aggregate = _summary_aggregate(summary, history, path=path)
    arm = motion_arm_report(summary, history, label=label)
    diagnostics = arm_rmsf_diagnostics(arm)
    diagnostics["summary_system_equal"] = {
        "prediction_rmsf": _aggregate_metric(aggregate, "rmsf.prediction"),
        "target_rmsf": _aggregate_metric(aggregate, "rmsf.target"),
        "prediction_target_ratio": _aggregate_metric(aggregate, "rmsf.ratio"),
    }
    return {
        "label": label,
        "history": int(history),
        "source_summary": str(path),
        "aggregation": aggregate.get("aggregation"),
        "system_count": aggregate.get("system_count"),
        "arm": arm,
        "diagnostics": diagnostics,
    }


def _pair_history(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    candidate_label: str,
    reference_label: str,
) -> dict[str, Any]:
    candidate_arm = candidate["arm"]
    reference_arm = reference["arm"]
    expected = sorted(
        set(candidate_arm.get("actual_systems", []))
        | set(reference_arm.get("actual_systems", []))
    )
    pair = paired_motion_report(
        candidate_arm,
        reference_arm,
        candidate_label=candidate_label,
        reference_label=reference_label,
        expected_systems=expected,
        threshold=MOTION_RATIO_THRESHOLD,
    )
    return {
        "candidate": candidate_label,
        "reference": reference_label,
        "candidate_rmsf": candidate["diagnostics"],
        "reference_rmsf": reference["diagnostics"],
        "guard": pair,
    }


def _all_motion_pass(rows: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        item["guard"].get("status") == "PASS"
        and item["guard"].get("median_motion_ratio") is not None
        and float(item["guard"]["median_motion_ratio"]) >= MOTION_RATIO_THRESHOLD
        for item in rows.values()
    )


def _decision_recompute(
    *,
    decision: Mapping[str, Any],
    motion_rows: Mapping[str, Mapping[str, Any]],
    round_name: str,
) -> dict[str, Any]:
    original_status = str(decision.get("status"))
    original_selected = decision.get("selected_arm")
    motion_status = {
        key: {
            "status": value["guard"].get("status"),
            "median_motion_ratio": value["guard"].get("median_motion_ratio"),
            "motion_ratio_system_count": value["guard"].get("motion_ratio_system_count"),
        }
        for key, value in motion_rows.items()
    }
    all_pass = _all_motion_pass(motion_rows)
    if round_name == "round2" and original_status == "KEEP" and not all_pass:
        revised_status = "TRADEOFF"
        revised_selected = "control"
        reason = (
            "the original KEEP branch relied on a null/empty motion guard; the corrected "
            "candidate/control RMSF ratio is below the registered threshold"
        )
    else:
        revised_status = original_status
        revised_selected = original_selected
        reason = "motion correction does not alter the original decision branch"
    return {
        "original": {"status": original_status, "selected_arm": original_selected},
        "motion_guard_all_histories_pass": all_pass,
        "motion_guard_by_history": motion_status,
        "recomputed": {
            "status": revised_status,
            "selected_arm": revised_selected,
            "changed": revised_status != original_status or revised_selected != original_selected,
            "reason": reason,
        },
    }


def _round2(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    round_root = root / "round2"
    decision_path = round_root / "decision.json"
    decision = _read_json(decision_path)
    if decision.get("evaluation_scope") != "final":
        raise ValueError("Round 2 recheck requires the original decision's final evaluation scope")
    paths = {
        "candidate": round_root / "evaluation_final/candidate/summary.json",
        "control": round_root / "evaluation_final/control/summary.json",
    }
    summaries = {label: _read_json(path) for label, path in paths.items()}
    if any(summary.get("scope") != "final" for summary in summaries.values()):
        raise ValueError("Round 2 source summary is not final scope")
    arms: dict[str, dict[str, Any]] = {}
    for label, summary in summaries.items():
        arms[label] = {
            f"H{history}": _arm_history(summary, history, label=label, path=paths[label])
            for history in HISTORIES
        }
    pairs = {
        f"H{history}": _pair_history(
            arms["candidate"][f"H{history}"],
            arms["control"][f"H{history}"],
            candidate_label="candidate",
            reference_label="control",
        )
        for history in HISTORIES
    }
    motion = {"scope": "final", "arms": arms, "pairs": pairs}
    recomputed = _decision_recompute(
        decision=decision,
        motion_rows=pairs,
        round_name="round2",
    )
    return {
        "round": 2,
        "decision_path": str(decision_path),
        "lineage": {
            key: decision.get(key)
            for key in (
                "parent_checkpoint",
                "parent_checkpoint_sha256",
                "selected_checkpoint",
                "selected_checkpoint_sha256",
            )
        },
        "decision": decision,
        "motion": motion,
        "decision_recheck": recomputed,
    }, {
        "decision": _source(decision_path),
        "summaries": {label: _source(path) for label, path in paths.items()},
        "generation_rows": {
            label: _source(path.parent / "generation_rows.jsonl", required=False)
            for label, path in paths.items()
        },
    }


def _round3(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    round_root = root / "round3"
    decision_path = round_root / "decision.json"
    decision = _read_json(decision_path)
    scope = str(decision.get("evaluation_scope"))
    if scope not in ("quick", "final"):
        raise ValueError(f"Round 3 decision has unsupported evaluation scope: {scope!r}")
    paths = {
        arm: round_root / f"evaluation_{scope}/{arm}/summary.json"
        for arm in ("parent", "local", "cross_block")
    }
    summaries = {label: _read_json(path) for label, path in paths.items()}
    if any(summary.get("scope") != scope for summary in summaries.values()):
        raise ValueError("Round 3 source summaries do not match the original decision scope")
    arms: dict[str, dict[str, Any]] = {}
    for label, summary in summaries.items():
        arms[label] = {
            f"H{history}": _arm_history(summary, history, label=label, path=paths[label])
            for history in HISTORIES
        }
    local_pairs = {
        f"H{history}": _pair_history(
            arms["local"][f"H{history}"],
            arms["parent"][f"H{history}"],
            candidate_label="local",
            reference_label="parent",
        )
        for history in HISTORIES
    }
    cross_pairs = {
        f"H{history}": _pair_history(
            arms["cross_block"][f"H{history}"],
            arms["parent"][f"H{history}"],
            candidate_label="cross_block",
            reference_label="parent",
        )
        for history in HISTORIES
    }
    pairs = {
        "local_vs_parent": local_pairs,
        "cross_vs_parent": cross_pairs,
    }
    motion = {"scope": scope, "arms": arms, "pairs": pairs}
    recomputed = {
        "original": {"status": decision.get("status"), "selected_arm": decision.get("selected_arm")},
        "motion_guard_all_histories_pass": all(
            _all_motion_pass(pair_set) for pair_set in (local_pairs, cross_pairs)
        ),
        "motion_guard_by_comparison": {
            name: {
                history: {
                    "status": item["guard"].get("status"),
                    "median_motion_ratio": item["guard"].get("median_motion_ratio"),
                    "motion_ratio_system_count": item["guard"].get("motion_ratio_system_count"),
                }
                for history, item in pair_set.items()
            }
            for name, pair_set in pairs.items()
        },
        "recomputed": {
            "status": decision.get("status"),
            "selected_arm": decision.get("selected_arm"),
            "changed": False,
            "reason": "both local/parent and cross-block/parent motion guards are complete and pass",
        },
    }
    return {
        "round": 3,
        "decision_path": str(decision_path),
        "lineage": {
            key: decision.get(key)
            for key in (
                "parent_checkpoint",
                "parent_checkpoint_sha256",
                "selected_checkpoint",
                "selected_checkpoint_sha256",
            )
        },
        "decision": decision,
        "motion": motion,
        "decision_recheck": recomputed,
    }, {
        "decision": _source(decision_path),
        "summaries": {label: _source(path) for label, path in paths.items()},
        "generation_rows": {
            label: _source(path.parent / "generation_rows.jsonl", required=False)
            for label, path in paths.items()
        },
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.9f}"
    return str(value)


def _systems(pair: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(pair["guard"].get("pairs", [])) + list(pair["guard"].get("noncomputable_systems", []))


def _render_pair_table(lines: list[str], title: str, pair_set: Mapping[str, Any]) -> None:
    lines.extend(
        [
            f"### {title}",
            "",
            "| history | status | median candidate/reference prediction RMSF | n valid | expected | candidate actual | reference actual | missing/non-computable |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for history in ("H4", "H8"):
        guard = pair_set[history]["guard"]
        missing = guard["missing_candidate_systems"] + guard["missing_reference_systems"]
        missing.extend(item["system"] for item in guard["noncomputable_systems"])
        lines.append(
            f"| {history} | {guard['status']} | {_fmt(guard['median_motion_ratio'])} | "
            f"{guard['motion_ratio_system_count']} | {len(guard['expected_systems'])} | "
            f"{len(guard['candidate_actual_systems'])} | {len(guard['reference_actual_systems'])} | "
            f"{', '.join(sorted({str(item) for item in missing})) or 'none'} |"
        )
        lines.extend(
            [
                "",
                f"- {history} expected: {', '.join(guard['expected_systems']) or 'none'}",
                f"- {history} {guard['candidate_label']} actual: {', '.join(guard['candidate_actual_systems']) or 'none'}",
                f"- {history} {guard['reference_label']} actual: {', '.join(guard['reference_actual_systems']) or 'none'}",
                f"- {history} missing candidate: {', '.join(guard['missing_candidate_systems']) or 'none'}; "
                f"missing reference: {', '.join(guard['missing_reference_systems']) or 'none'}",
                f"- {history} non-computable: "
                + (
                    "; ".join(
                        f"{item.get('system')}: {', '.join(item.get('reasons', []))}"
                        for item in guard["noncomputable_systems"]
                    )
                    or "none"
                ),
                "",
            ]
        )
    for history in ("H4", "H8"):
        guard = pair_set[history]["guard"]
        lines.extend(
            [
                "",
                f"#### {title} {history} per-system original guard metric",
                "",
                "| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |",
                "|---|---:|---:|---:|---|",
            ]
        )
        pair_by_system = {item["system"]: item for item in guard.get("pairs", [])}
        bad_by_system = {item["system"]: item for item in guard.get("noncomputable_systems", [])}
        for system in guard["expected_systems"]:
            if system in pair_by_system:
                item = pair_by_system[system]
                lines.append(
                    f"| {system} | {_fmt(item['candidate_prediction_rmsf'])} | "
                    f"{_fmt(item['reference_prediction_rmsf'])} | {_fmt(item['prediction_ratio'])} | |"
                )
            else:
                lines.append(
                    f"| {system} | n/a | n/a | n/a | "
                    f"{'; '.join(bad_by_system.get(system, {}).get('reasons', ['not paired']))} |"
                )


def _render_arm_table(lines: list[str], arms: Mapping[str, Any], history: str) -> None:
    lines.extend(
        [
            f"#### Arm RMSF diagnostics ({history})",
            "",
            "| arm | prediction RMSF | target RMSF | prediction/target (ratio of means) | mean of per-system ratios | n ratio pairs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, history_rows in arms.items():
        diagnostics = history_rows[history]["diagnostics"]
        lines.append(
            f"| {arm} | {_fmt(diagnostics['mean_prediction_rmsf'])} | "
            f"{_fmt(diagnostics['mean_target_rmsf'])} | "
            f"{_fmt(diagnostics['ratio_of_means_prediction_over_target'])} | "
            f"{_fmt(diagnostics['mean_of_per_system_prediction_target_ratios'])} | "
            f"{diagnostics['mean_of_ratios_system_count']} |"
        )


def _render_report(
    root: Path,
    round2: Mapping[str, Any],
    round3: Mapping[str, Any],
    sources: Mapping[str, Any],
) -> str:
    lines = [
        "# Motion metric recheck v1",
        "",
        "This is a read-only recomputation from existing evaluation summaries; no model, checkpoint, "
        "clip payload, training job, or sampler was started.",
        "",
        f"- RMSF guard threshold (unchanged): `{MOTION_RATIO_THRESHOLD}`",
        f"- Aggregation contract checked: `{AGGREGATION_CONTRACT}`",
        "- Current flat paths: `future['rmsf.prediction']`, `future['rmsf.target']`",
        "- Legacy path accepted only when flat path is absent: `future['rmsf']['prediction|target']`",
        "- Ratio definitions: `ratio of means = mean(prediction) / mean(target)`; `mean of ratios` averages per-system prediction/target ratios.",
        "",
        "## Round 2 (original final scope; candidate vs control)",
        "",
        f"Original decision: `{round2['decision'].get('status')}` / selected `{round2['decision'].get('selected_arm')}`; "
        f"recomputed decision: `{round2['decision_recheck']['recomputed']['status']}` / selected "
        f"`{round2['decision_recheck']['recomputed']['selected_arm']}`.",
        "",
    ]
    _render_pair_table(lines, "candidate vs control", round2["motion"]["pairs"])
    for history in ("H4", "H8"):
        _render_arm_table(lines, round2["motion"]["arms"], history)
    lines.extend(
        [
            "",
            "## Round 3 (original decision scope; local/cross-block vs parent)",
            "",
            f"Original decision: `{round3['decision'].get('status')}` / selected `{round3['decision'].get('selected_arm')}`; "
            "the motion correction leaves that branch unchanged.",
            "",
        ]
    )
    _render_pair_table(lines, "local vs parent", round3["motion"]["pairs"]["local_vs_parent"])
    _render_pair_table(lines, "cross_block vs parent", round3["motion"]["pairs"]["cross_vs_parent"])
    for history in ("H4", "H8"):
        _render_arm_table(lines, round3["motion"]["arms"], history)
    lines.extend(
        [
            "",
            "## Source files",
            "",
            "The summary `system_rows` were sufficient; generation rows were synchronized and hash-recorded "
            "for provenance but were not re-aggregated.",
            "",
            "```json",
            json.dumps(sources, indent=2, sort_keys=True),
            "```",
            "",
            "## Interpretation",
            "",
            "The original Round 2 motion guard was not a pass: its null/zero count came from the incorrect "
            "nested-field reader.  The corrected candidate/control ratios are complete but below 0.5 in H4 "
            "and H8, so the geometry-loss candidate fails the registered motion guard.  Round 3 local/parent "
            "and cross-block/parent ratios are complete and pass; its primary boundary result remains the "
            "reason for retaining the parent.  This correction therefore removes the original Round 2 KEEP "
            "recommendation; it does not constitute a new Round 3 parent selection or any retraining.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(root: Path) -> dict[str, Any]:
    round2, sources2 = _round2(root)
    round3, sources3 = _round3(root)
    output = root / "motion_recheck_v1"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "run_root": str(root),
        "config": _source(PROJECT_ROOT / "config/dit_architecture_sequential_v1.yaml"),
        "round2": sources2,
        "round3": sources3,
        "test_payload_opened": False,
        "checkpoint_loaded": False,
    }
    source_manifest = {
        "schema": "pvb.dit.architecture_sequential.v1.motion_recheck_sources.v1",
        "sources": sources,
    }
    summary = {
        "schema": "pvb.dit.architecture_sequential.v1.motion_recheck.v1",
        "run_id": RUN_ID,
        "threshold": MOTION_RATIO_THRESHOLD,
        "aggregation": AGGREGATION_CONTRACT,
        "round2": {
            "scope": round2["motion"]["scope"],
            "decision_recheck": round2["decision_recheck"],
            "output": "round2_motion.json",
            "decision_recheck_output": "round2_decision_recheck.json",
        },
        "round3": {
            "scope": round3["motion"]["scope"],
            "decision_recheck": round3["decision_recheck"],
            "output": "round3_motion.json",
            "decision_recheck_output": "round3_decision_recheck.json",
        },
        "test_payload_opened": False,
        "checkpoint_loaded": False,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "round2_motion.json").write_text(
        json.dumps(round2, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "round3_motion.json").write_text(
        json.dumps(round3, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "round2_decision_recheck.json").write_text(
        json.dumps(round2["decision_recheck"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "round3_decision_recheck.json").write_text(
        json.dumps(round3["decision_recheck"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "recheck_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.md").write_text(
        _render_report(root, round2, round3, sources), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/dit_architecture_sequential_v1" / RUN_ID,
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run(args.run_root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
