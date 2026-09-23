"""Select the P3 parent with the frozen P2 system-paired bootstrap rule."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from molvid.config import load_config


ARMS = ("B0", "G", "M", "GM")
MAIN_METRICS = ("bond_rmse_A", "residue_rmsf_mae_A", "msd_curve_mae_A2")
SAFETY_METRICS = ("clash_rate", "rmsf_pearson_error", "rmsf_spread_error")
RULE_VERSION = "frame_gm_p2_system_paired_bootstrap.v1"
FROZEN_RULE = {
    "version": RULE_VERSION,
    "main_errors": [
        "bond_rmse_A over all formal views",
        "residue_rmsf_mae_A over all formal views",
        "msd_curve_mae_A2 over fixed-history views",
    ],
    "bootstrap_unit": "system",
    "confidence_interval": "paired two-sided percentile 95% on candidate minus B0",
    "candidate_gate": "at least one main error upper CI < 0; no other main error lower CI > 0",
    "safety_gate": (
        "no paired clear worsening in clash rate, abs(1-system RMSF Pearson), "
        "or abs(log system RMSF spread ratio)"
    ),
    "tie_break": "lowest average rank across three main errors, then lower profiled GPU cost",
    "fallback": "B0 when no valid candidate passes",
    "posthoc_metric_changes_allowed": False,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--resolved-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=21208)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value: str, *, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {field} in {source}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} in {source}: {value!r}")
    return result


def _generated_rows(path: Path) -> list[dict[str, str]]:
    rows = [
        row for row in _read_csv(path)
        if row.get("path") == "generated" and row.get("clock") == "true"
    ]
    if not rows:
        raise ValueError(f"no generated/correct-clock rows in {path}")
    return rows


def _mean_by_system(
    rows: Sequence[Mapping[str, str]],
    metric: str,
    *,
    source: Path,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        system = str(row["system"])
        grouped.setdefault(system, []).append(
            _finite_float(str(row[metric]), field=metric, source=source)
        )
    return {system: float(np.mean(values)) for system, values in grouped.items()}


def _validate_metrics_run(
    metrics_dir: Path,
    expected_rows: int,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    metrics = _read_json(metrics_dir / "metrics.json")
    manifest = _read_json(metrics_dir / "run_manifest.json")
    if metrics.get("schema") != "molvid.frame_gm.metric_summary.v3":
        raise ValueError(f"unexpected metric schema in {metrics_dir}")
    if metrics.get("test_opened") is not False:
        raise ValueError(f"test-opened provenance is not false in {metrics_dir}")
    if metrics.get("historical_rows_mutated") is not False:
        raise ValueError(f"historical rows were mutated in {metrics_dir}")
    if metrics.get("aggregation") != "draw/seed mean -> anchor mean -> replica mean -> system equal":
        raise ValueError(f"unexpected aggregation in {metrics_dir}")
    if manifest.get("schema") != "molvid.frame_gm.metric_recompute_run.v1":
        raise ValueError(f"unexpected metric run-manifest schema in {metrics_dir}")
    if int(metrics.get("source_rows", -1)) != expected_rows:
        raise ValueError(f"expected {expected_rows} source rows in {metrics_dir}")
    if int(manifest.get("row_count", -1)) != expected_rows:
        raise ValueError(f"expected {expected_rows} output rows in {metrics_dir}")
    for name in ("rows.jsonl", "metrics.json", "per_system.csv", "run_manifest.json"):
        if not (metrics_dir / name).is_file():
            raise FileNotFoundError(metrics_dir / name)
    for name, field in (
        ("rows.jsonl", "output_rows_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("per_system.csv", "per_system_sha256"),
    ):
        if manifest.get(field) != _sha256(metrics_dir / name):
            raise ValueError(f"{name} SHA-256 differs from metric run manifest")
    source_run = Path(str(manifest.get("source_run", "")))
    protocol_path = source_run / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if metrics.get("source_protocol_sha256") != _sha256(protocol_path):
        raise ValueError(f"source evaluation protocol SHA-256 differs in {metrics_dir}")
    protocol = _read_json(protocol_path)
    if protocol.get("schema") != "molvid.frame_joint.multitime_eval.v1":
        raise ValueError(f"unexpected source evaluation protocol in {metrics_dir}")
    if protocol.get("checkpoint", {}).get("sha256") != expected_checkpoint_sha256:
        raise ValueError(f"source evaluation checkpoint differs in {metrics_dir}")
    if protocol.get("steps") != 16 or protocol.get("seeds") != [0, 1, 2]:
        raise ValueError(f"source evaluation sampler differs in {metrics_dir}")
    if protocol.get("paths") != ["persistence", "clean_latent_decoder_oracle", "generated"]:
        raise ValueError(f"source evaluation paths differ in {metrics_dir}")
    if protocol.get("expected_rows") != expected_rows or protocol.get("rows_written") != expected_rows:
        raise ValueError(f"source evaluation row contract differs in {metrics_dir}")
    paired_store = protocol.get("paired_store")
    paired_manifest = protocol.get("paired_manifest")
    if not isinstance(paired_store, Mapping) or not isinstance(paired_manifest, Mapping):
        raise ValueError(f"source paired-view provenance is missing in {metrics_dir}")
    if manifest.get("paired_store") != paired_store.get("path"):
        raise ValueError(f"metric/evaluation paired store differs in {metrics_dir}")
    store_path = Path(str(paired_store.get("path", "")))
    store_index_path = store_path / "index.txt"
    if not store_index_path.is_file():
        raise FileNotFoundError(store_index_path)
    actual_index_sha256 = _sha256(store_index_path)
    if (
        metrics.get("paired_store_index_sha256") != actual_index_sha256
        or paired_store.get("index_sha256") != actual_index_sha256
    ):
        raise ValueError(f"paired store index SHA-256 differs in {metrics_dir}")
    paired_manifest_path = Path(str(paired_manifest.get("path", "")))
    if not paired_manifest_path.is_file():
        raise FileNotFoundError(paired_manifest_path)
    actual_manifest_sha256 = _sha256(paired_manifest_path)
    if paired_manifest.get("sha256") != actual_manifest_sha256:
        raise ValueError(f"paired manifest SHA-256 differs in {metrics_dir}")
    paired_manifest_value = _read_json(paired_manifest_path)
    if paired_manifest_value.get("test_opened") is not False:
        raise ValueError(f"paired manifest opened sealed test data in {metrics_dir}")
    if expected_rows == 2352:
        if (
            paired_manifest_value.get("schema_version")
            != "molvid.frame_joint.multitime.eval_views.v1"
            or paired_manifest_value.get("history_frames") != [4, 8]
            or paired_manifest_value.get("views_per_trajectory") != 16
        ):
            raise ValueError(f"unexpected legacy paired-view family in {metrics_dir}")
        view_family = "legacy"
    elif expected_rows == 1392:
        if (
            paired_manifest_value.get("schema") != "molvid.frame_gm.fixed_history_views.v1"
            or paired_manifest_value.get("view_kind")
            != "fixed_history_h4_horizon_1200ps"
            or paired_manifest_value.get("observed_relative_time_ps") != [-300, -200, -100, 0]
        ):
            raise ValueError(f"unexpected fixed-history paired-view family in {metrics_dir}")
        view_family = "fixed"
    else:
        raise ValueError(f"unsupported formal row contract: {expected_rows}")
    return {
        "metric_manifest": str((metrics_dir / "run_manifest.json").resolve()),
        "metric_manifest_sha256": _sha256(metrics_dir / "run_manifest.json"),
        "evaluation_protocol": str(protocol_path.resolve()),
        "evaluation_protocol_sha256": _sha256(protocol_path),
        "paired_data": {
            "family": view_family,
            "store": str(store_path.resolve()),
            "store_index_sha256": actual_index_sha256,
            "manifest": str(paired_manifest_path.resolve()),
            "manifest_sha256": actual_manifest_sha256,
            "test_opened": False,
        },
        "protocol_identity": {
            key: protocol.get(key)
            for key in (
                "codec", "expected_rows", "generated_conditioning", "ids", "paired_manifest",
                "paired_store", "paths", "physical_time_axis", "rows_written", "seeds", "steps",
                "wrong_clock_dt", "wrong_clock_scoring",
            )
        },
    }


def _validate_view_grid(
    rows: Sequence[Mapping[str, str]],
    *,
    histories: Sequence[int],
    family: str,
    source: Path,
) -> tuple[list[str], dict[tuple[str, str, int, int], float]]:
    expected_views = {(lag, history) for lag in (100, 200, 300, 400) for history in histories}
    systems = sorted({str(row["system"]) for row in rows})
    keys: set[tuple[str, int, int]] = set()
    targets: dict[tuple[str, str, int, int], float] = {}
    for row in rows:
        system = str(row["system"])
        lag = int(row["lag_ps"])
        history = int(row["history_frames"])
        key = (system, lag, history)
        if key in keys:
            raise ValueError(f"duplicate generated formal view {key} in {source}")
        if (lag, history) not in expected_views:
            raise ValueError(f"unexpected generated formal view {key} in {source}")
        if int(row["replica_count"]) != 3:
            raise ValueError(f"formal view {key} does not contain three replicas")
        keys.add(key)
        targets[(family, *key)] = _finite_float(
            str(row["system_mean_rmsf_target_A"]),
            field="system_mean_rmsf_target_A",
            source=source,
        )
    expected_keys = {
        (system, lag, history)
        for system in systems
        for lag, history in expected_views
    }
    if keys != expected_keys:
        raise ValueError(f"generated formal view grid is incomplete in {source}")
    return systems, targets


def load_arm_values(
    legacy_dir: Path,
    fixed_dir: Path,
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Load one arm and reduce every formal view to one value per system."""
    legacy_provenance = _validate_metrics_run(
        legacy_dir, 2352, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    fixed_provenance = _validate_metrics_run(
        fixed_dir, 1392, expected_checkpoint_sha256=expected_checkpoint_sha256
    )
    legacy_path = legacy_dir / "per_system.csv"
    fixed_path = fixed_dir / "per_system.csv"
    legacy = _generated_rows(legacy_path)
    fixed = _generated_rows(fixed_path)
    if len(legacy) != 64 or len(fixed) != 32:
        raise ValueError(
            f"unexpected generated view counts: legacy={len(legacy)}, fixed={len(fixed)}"
        )
    legacy_systems, legacy_targets = _validate_view_grid(
        legacy, histories=(4, 8), family="legacy", source=legacy_path
    )
    fixed_systems, fixed_targets = _validate_view_grid(
        fixed, histories=(4,), family="fixed", source=fixed_path
    )
    if legacy_systems != fixed_systems:
        raise ValueError("legacy and fixed-history system sets differ")
    all_rows = [*legacy, *fixed]
    systems = legacy_systems
    if len(systems) != 8:
        raise ValueError(f"expected 8 systems, found {len(systems)}")
    expected_all = {system: 12 for system in systems}
    expected_fixed = {system: 4 for system in systems}
    counts_all = {system: 0 for system in systems}
    counts_fixed = {system: 0 for system in systems}
    for row in all_rows:
        counts_all[str(row["system"])] += 1
    for row in fixed:
        counts_fixed[str(row["system"])] += 1
    if counts_all != expected_all or counts_fixed != expected_fixed:
        raise ValueError("formal view coverage is not symmetric across systems")

    def mean(metric: str, rows: Sequence[Mapping[str, str]], source: Path) -> np.ndarray:
        values = _mean_by_system(rows, metric, source=source)
        if set(values) != set(systems):
            raise ValueError(f"missing systems for {metric}")
        return np.asarray([values[system] for system in systems], dtype=np.float64)

    # The two tables share the same columns. Source names in errors are only provenance.
    return {
        "systems": systems,
        "bond_rmse_A": mean("bond_rmse_A", all_rows, legacy_path),
        "residue_rmsf_mae_A": mean("residue_rmsf_mae_A", all_rows, legacy_path),
        "msd_curve_mae_A2": mean("msd_curve_mae_A2", fixed, fixed_path),
        "clash_rate": mean("clash_rate", all_rows, legacy_path),
        "rmsf_prediction_A": mean("system_mean_rmsf_prediction_A", all_rows, legacy_path),
        "rmsf_target_A": mean("system_mean_rmsf_target_A", all_rows, legacy_path),
        "rmsf_target_by_view": {**legacy_targets, **fixed_targets},
        "source": {
            "legacy_per_system": str(legacy_path.resolve()),
            "legacy_per_system_sha256": _sha256(legacy_path),
            "fixed_per_system": str(fixed_path.resolve()),
            "fixed_per_system_sha256": _sha256(fixed_path),
            "legacy_run": legacy_provenance,
            "fixed_run": fixed_provenance,
        },
    }


def _pearson_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = prediction - prediction.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(prediction * prediction, axis=1) * np.sum(target * target, axis=1)
    )
    result = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 0.0
    result[valid] = np.sum(prediction[valid] * target[valid], axis=1) / denominator[valid]
    return result


def _rmsf_safety_rows(
    prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pearson = _pearson_rows(prediction, target)
    prediction_std = np.std(prediction, axis=1)
    target_std = np.std(target, axis=1)
    spread = np.full(prediction_std.shape, np.nan, dtype=np.float64)
    valid = (prediction_std > 0.0) & (target_std > 0.0)
    spread[valid] = np.abs(np.log(prediction_std[valid] / target_std[valid]))
    return np.abs(1.0 - pearson), spread


def _ci(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if finite.size < max(100, int(0.95 * values.size)):
        raise ValueError("insufficient finite system-bootstrap replicates")
    return [float(value) for value in np.percentile(finite, [2.5, 97.5])]


def _rank(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda arm: (values[arm], arm))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        for arm in ordered[start:end]:
            ranks[arm] = average_rank
        start = end
    return ranks


def _validated_profile_costs(value: Mapping[str, Any]) -> dict[str, float]:
    if set(value) != set(ARMS):
        raise ValueError("profiled step costs must contain exactly the four frozen arms")
    result = {arm: float(value[arm]) for arm in ARMS}
    if any(not math.isfinite(cost) or cost <= 0.0 for cost in result.values()):
        raise ValueError("profiled step costs must be finite and positive")
    return result


def _validate_formal_provenance(
    formal_root: Path,
    resolved_path: Path,
    *,
    expected_step: int,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, float], dict[str, Any]]:
    resolved = _read_json(resolved_path)
    if resolved.get("selection_rule") != FROZEN_RULE:
        raise ValueError("resolved experiment selection rule differs from the frozen contract")
    formal_manifest_path = formal_root / "run_manifest.json"
    formal = _read_json(formal_manifest_path)
    if formal.get("schema") != "molvid.frame_gm.p2_formal_run.v1":
        raise ValueError("unexpected formal run-manifest schema")
    if formal.get("status") != "COMPLETE" or formal.get("sealed_test_opened") is not False:
        raise ValueError("formal run is incomplete or opened sealed test data")
    if formal.get("candidate_commit") != resolved.get("candidate_commit"):
        raise ValueError("formal candidate commit differs from resolved experiment")
    if formal.get("parent", {}).get("sha256") != resolved.get("parent", {}).get("sha256"):
        raise ValueError("formal parent differs from resolved experiment")
    if formal.get("successful_updates_per_arm") != expected_step:
        raise ValueError("formal run update count differs")
    formal_arms = formal.get("arms")
    resolved_configs = resolved.get("formal_training", {}).get("configs")
    if not isinstance(formal_arms, Mapping) or not isinstance(resolved_configs, Mapping):
        raise ValueError("formal arm/config provenance is missing")
    if set(formal_arms) != set(ARMS) or set(resolved_configs) != set(ARMS):
        raise ValueError("formal arm/config provenance differs")
    checkpoints: dict[str, dict[str, str]] = {}
    arm_provenance: dict[str, Any] = {}
    baseline_exposure: dict[str, Any] | None = None
    baseline_sampler: dict[str, Any] | None = None
    for arm in ARMS:
        arm_manifest = formal_arms[arm]
        resolved_config = resolved_configs[arm]
        if not isinstance(arm_manifest, Mapping) or not isinstance(resolved_config, Mapping):
            raise ValueError(f"formal provenance is malformed for {arm}")
        if arm_manifest.get("status") != "COMPLETE":
            raise ValueError(f"formal arm {arm} is incomplete")
        config_path = Path(str(resolved_config.get("path", "")))
        config_sha = _sha256(config_path)
        if (
            config_sha != resolved_config.get("sha256")
            or config_sha != arm_manifest.get("config_sha256")
            or str(config_path) != arm_manifest.get("config")
        ):
            raise ValueError(f"formal config provenance differs for {arm}")
        checkpoint = formal_root / arm / f"frame_joint_step_{expected_step:08d}.pt"
        checkpoint_sha = _sha256(checkpoint)
        if checkpoint_sha != arm_manifest.get("final_checkpoint_sha256"):
            raise ValueError(f"formal checkpoint provenance differs for {arm}")
        exposure = _read_json(formal_root / arm / "exposure.json")
        sampler = _read_json(formal_root / arm / "sampler_manifest.json")
        target = _read_json(formal_root / arm / "target_encoder_provenance.json")
        arm_resolved = _read_json(formal_root / arm / "resolved_config.json")
        reviewed_config = load_config(
            config_path, schema="molvid.frame_joint.train.v1"
        )
        actual_config = {
            key: value for key, value in arm_resolved.items() if key != "resolved"
        }
        if actual_config != reviewed_config:
            raise ValueError(
                f"formal runtime config differs from reviewed config for {arm}"
            )
        if exposure.get("successful_updates") != expected_step:
            raise ValueError(f"formal exposure is incomplete for {arm}")
        if sampler.get("test_opened") is not False:
            raise ValueError(f"formal sampler opened sealed test for {arm}")
        if target.get("code_commit") != resolved.get("candidate_commit"):
            raise ValueError(f"formal code provenance differs for {arm}")
        child = arm_resolved.get("resolved", {})
        if (
            arm_resolved.get("schema") != "molvid.frame_joint.train.v1"
            or child.get("total_updates") != expected_step
            or child.get("continuation_parent", {}).get("sha256")
            != resolved.get("parent", {}).get("sha256")
            or child.get("data_hash") != resolved.get("data", {}).get("derived_data_hash")
        ):
            raise ValueError(f"formal resolved config differs for {arm}")
        if baseline_exposure is None:
            baseline_exposure = exposure
            baseline_sampler = sampler
        elif exposure != baseline_exposure or sampler != baseline_sampler:
            raise ValueError(
                f"formal sampler schedule or exposure differs for {arm}"
            )
        checkpoints[arm] = {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha}
        arm_provenance[arm] = {
            "config": str(config_path),
            "config_sha256": config_sha,
            "exposure_sha256": _sha256(formal_root / arm / "exposure.json"),
            "sampler_manifest_sha256": _sha256(formal_root / arm / "sampler_manifest.json"),
            "target_encoder_provenance_sha256": _sha256(
                formal_root / arm / "target_encoder_provenance.json"
            ),
            "resolved_config_sha256": _sha256(formal_root / arm / "resolved_config.json"),
        }
    costs = _validated_profile_costs(resolved.get("profile_budget", {}).get("step_seconds", {}))
    profile_root = resolved_path.parent.parent / "profile"
    profile_provenance: dict[str, Any] = {}
    for arm in ARMS:
        profile_path = profile_root / f"{arm}.json"
        profile = _read_json(profile_path)
        if (
            profile.get("schema") != "molvid.frame_gm.p2_largest_sample_profile.v1"
            or profile.get("arm") != arm
            or profile.get("test_opened") is not False
            or profile.get("parent_sha256") != resolved.get("parent", {}).get("sha256")
            or profile.get("data_hash") != resolved.get("data", {}).get("derived_data_hash")
            or not math.isclose(float(profile.get("step_seconds", math.nan)), costs[arm], rel_tol=0, abs_tol=0)
        ):
            raise ValueError(f"profile artifact differs from resolved experiment for {arm}")
        profile_provenance[arm] = {
            "path": str(profile_path.resolve()),
            "sha256": _sha256(profile_path),
        }
    provenance = {
        "resolved_experiment": str(resolved_path.resolve()),
        "resolved_experiment_sha256": _sha256(resolved_path),
        "formal_run_manifest": str(formal_manifest_path.resolve()),
        "formal_run_manifest_sha256": _sha256(formal_manifest_path),
        "formal_arms": arm_provenance,
        "profile_artifacts": profile_provenance,
    }
    return resolved, checkpoints, costs, provenance


def select_from_system_values(
    arm_values: Mapping[str, Mapping[str, Any]],
    profile_cost_s: Mapping[str, float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Apply the frozen rule to already aggregated, paired system values."""
    if set(arm_values) != set(ARMS):
        raise ValueError(f"expected arms {ARMS}, found {sorted(arm_values)}")
    if bootstrap_samples < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    costs = _validated_profile_costs(profile_cost_s)
    systems = list(arm_values["B0"]["systems"])
    if len(systems) < 3:
        raise ValueError("at least three systems are required")
    required = (*MAIN_METRICS, "clash_rate", "rmsf_prediction_A", "rmsf_target_A")
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        if list(arm_values[arm]["systems"]) != systems:
            raise ValueError(f"system order differs for {arm}")
        arrays[arm] = {}
        for metric in required:
            value = np.asarray(arm_values[arm][metric], dtype=np.float64)
            if value.shape != (len(systems),) or not np.isfinite(value).all():
                raise ValueError(f"invalid {metric} values for {arm}")
            arrays[arm][metric] = value
        if not np.allclose(
            arrays[arm]["rmsf_target_A"], arrays["B0"]["rmsf_target_A"],
            rtol=1e-6, atol=1e-8,
        ):
            raise ValueError(f"RMSF targets are not paired for {arm}")

    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(systems), size=(bootstrap_samples, len(systems)))
    point: dict[str, dict[str, float]] = {}
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    for arm in ARMS:
        point[arm] = {metric: float(np.mean(arrays[arm][metric])) for metric in MAIN_METRICS}
        point[arm]["clash_rate"] = float(np.mean(arrays[arm]["clash_rate"]))
        pred = arrays[arm]["rmsf_prediction_A"]
        target = arrays[arm]["rmsf_target_A"]
        pearson_error, spread_error = _rmsf_safety_rows(pred[None, :], target[None, :])
        point[arm]["rmsf_pearson_error"] = float(pearson_error[0])
        point[arm]["rmsf_spread_error"] = float(spread_error[0])
        bootstrap[arm] = {
            metric: arrays[arm][metric][indices].mean(axis=1)
            for metric in (*MAIN_METRICS, "clash_rate")
        }
        pearson_rows, spread_rows = _rmsf_safety_rows(pred[indices], target[indices])
        bootstrap[arm]["rmsf_pearson_error"] = pearson_rows
        bootstrap[arm]["rmsf_spread_error"] = spread_rows

    ranks_by_metric = {
        metric: _rank({arm: point[arm][metric] for arm in ARMS})
        for metric in MAIN_METRICS
    }
    average_rank = {
        arm: float(np.mean([ranks_by_metric[metric][arm] for metric in MAIN_METRICS]))
        for arm in ARMS
    }
    arms: dict[str, Any] = {}
    eligible: list[str] = []
    for arm in ARMS:
        differences: dict[str, Any] = {}
        for metric in (*MAIN_METRICS, *SAFETY_METRICS):
            delta = bootstrap[arm][metric] - bootstrap["B0"][metric]
            differences[metric] = {
                "candidate_minus_B0": float(point[arm][metric] - point["B0"][metric]),
                "paired_bootstrap_95pct_ci": _ci(delta),
            }
        improved = [
            metric for metric in MAIN_METRICS
            if differences[metric]["paired_bootstrap_95pct_ci"][1] < 0.0
        ]
        main_worsened = [
            metric for metric in MAIN_METRICS
            if differences[metric]["paired_bootstrap_95pct_ci"][0] > 0.0
        ]
        safety_worsened = [
            metric for metric in SAFETY_METRICS
            if differences[metric]["paired_bootstrap_95pct_ci"][0] > 0.0
        ]
        passes = arm != "B0" and bool(improved) and not main_worsened and not safety_worsened
        if passes:
            eligible.append(arm)
        arms[arm] = {
            "point_errors": point[arm],
            "candidate_minus_B0": differences,
            "main_metric_ranks": {metric: ranks_by_metric[metric][arm] for metric in MAIN_METRICS},
            "average_main_error_rank": average_rank[arm],
            "profile_step_seconds": costs[arm],
            "clearly_improved_main_errors": improved,
            "clearly_worsened_main_errors": main_worsened,
            "clearly_worsened_safety_errors": safety_worsened,
            "passes_candidate_and_safety_gates": passes,
        }
    if eligible:
        ordered = sorted(eligible, key=lambda arm: (average_rank[arm], costs[arm]))
        selected = ordered[0]
        if len(ordered) > 1 and (
            average_rank[ordered[1]] == average_rank[selected]
            and costs[ordered[1]] == costs[selected]
        ):
            raise ValueError("eligible candidates tie on both frozen tie-break quantities")
        reason = (
            "selected among gate-passing candidates by lowest average rank across the "
            "three frozen main errors, then lower profiled step time"
        )
    else:
        selected = "B0"
        reason = "fallback to B0 because no candidate passed both frozen gates"
    return {
        "systems": systems,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "arms": arms,
        "eligible_candidates": eligible,
        "selected_arm": selected,
        "selection_reason": reason,
    }


def main() -> int:
    args = _args()
    resolved, checkpoints, costs, provenance = _validate_formal_provenance(
        args.formal_root,
        args.resolved_experiment,
        expected_step=args.expected_step,
    )
    rule = resolved["selection_rule"]
    arm_values: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_values[arm] = load_arm_values(
            args.legacy_root / arm,
            args.fixed_root / arm,
            expected_checkpoint_sha256=checkpoints[arm]["sha256"],
        )
    baseline_targets = arm_values["B0"]["rmsf_target_by_view"]
    for arm in ARMS[1:]:
        targets = arm_values[arm]["rmsf_target_by_view"]
        if set(targets) != set(baseline_targets) or any(
            not math.isclose(targets[key], baseline_targets[key], rel_tol=1e-6, abs_tol=1e-8)
            for key in baseline_targets
        ):
            raise ValueError(f"per-view RMSF targets are not paired for {arm}")
    for mode in ("legacy_run", "fixed_run"):
        identity = arm_values["B0"]["source"][mode]["protocol_identity"]
        if any(arm_values[arm]["source"][mode]["protocol_identity"] != identity for arm in ARMS[1:]):
            raise ValueError(f"formal evaluation protocol identity differs across arms for {mode}")
    selection = select_from_system_values(
        arm_values,
        costs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    for arm in ARMS:
        selection["arms"][arm]["checkpoint"] = checkpoints[arm]
        selection["arms"][arm]["metric_sources"] = arm_values[arm]["source"]
    selected = str(selection["selected_arm"])
    output = {
        "schema": "molvid.frame_gm.p2_selection.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rule_version": RULE_VERSION,
        "rule": rule,
        "provenance": provenance,
        "metric_direction": "all reported selection and safety quantities are errors; lower is better",
        "aggregation": (
            "saved draw/seed -> anchor -> replica -> system values; formal views mean within "
            "system; paired percentile bootstrap resamples systems"
        ),
        **selection,
        "parent": {
            "arm": selected,
            **checkpoints[selected],
            "mode": "changed-objective continuation with fresh optimizer, scheduler, cursor, and RNG",
        },
        "unresolved_issues": [
            "P2 selection is exploratory model selection, not confirmatory statistical evidence",
            "test split remains unopened",
        ],
        "test_opened": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "selected_arm": selected,
        "parent_sha256": checkpoints[selected]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
