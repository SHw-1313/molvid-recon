"""Validated RMSF readers and paired motion guard diagnostics.

The architecture runners aggregate evaluation rows into ``system_rows``.  The
current on-disk contract stores RMSF values in flattened keys such as
``future["rmsf.prediction"]``.  A small amount of historical output used the
older nested ``future["rmsf"]["prediction"]`` shape, so this module accepts
that shape only when the flattened key is absent.  If both representations are
present, they must agree exactly; otherwise the input is ambiguous and an
exception is raised instead of silently choosing one value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Mapping, Sequence


MOTION_RATIO_THRESHOLD = 0.5
DEFAULT_DENOMINATOR_EPSILON = 1.0e-8
RMSF_FIELDS = ("prediction", "target")


class MotionMetricError(ValueError):
    """Raised when one row contains contradictory RMSF representations."""


@dataclass(frozen=True)
class RMSFRead:
    """One validated scalar RMSF read, including a reason when unavailable."""

    field: str
    value: float | None
    source: str | None
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "source": self.source,
            "reason": self.reason,
        }


def _finite_scalar(value: Any) -> tuple[float | None, str | None]:
    """Return a finite JSON-number scalar without treating bool as numeric."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "non-numeric value"
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None, "non-finite numeric value"
    if not math.isfinite(number):
        return None, "NaN or infinite value"
    return number, None


def _format_path(context: str, path: str) -> str:
    return f"{context}: {path}" if context else path


def read_rmsf_value(
    row: Mapping[str, Any],
    field: str,
    *,
    context: str = "",
) -> RMSFRead:
    """Read one scalar RMSF field from a system row.

    The flattened path ``future["rmsf.<field>"]`` is the current contract.
    The nested ``future["rmsf"][field]`` path is accepted only as an explicit
    legacy form.  When both paths are present, finite values must be exactly
    equal; a finite/non-finite disagreement is also a conflict.
    """

    if field not in RMSF_FIELDS:
        raise ValueError(f"unsupported RMSF field: {field!r}")
    future = row.get("future")
    if not isinstance(future, Mapping):
        return RMSFRead(
            field=field,
            value=None,
            source=None,
            reason=_format_path(context, "missing future mapping"),
        )

    flat_key = f"rmsf.{field}"
    nested = future.get("rmsf")
    flat_present = flat_key in future
    nested_present = isinstance(nested, Mapping) and field in nested
    if not flat_present and not nested_present:
        if "rmsf" in future and not isinstance(nested, Mapping):
            reason = f"malformed legacy future['rmsf'] mapping for {field}"
        else:
            reason = f"missing future[{flat_key!r}] and legacy future['rmsf'][{field!r}]"
        return RMSFRead(
            field=field,
            value=None,
            source=None,
            reason=_format_path(context, reason),
        )

    flat_value, flat_reason = _finite_scalar(future.get(flat_key)) if flat_present else (None, None)
    nested_value, nested_reason = (
        _finite_scalar(nested.get(field)) if nested_present else (None, None)
    )

    if flat_present and nested_present:
        if flat_value is not None and nested_value is not None and flat_value == nested_value:
            return RMSFRead(field=field, value=flat_value, source="flat+legacy_equal", reason=None)
        if flat_value is None and nested_value is None:
            return RMSFRead(
                field=field,
                value=None,
                source="flat+legacy",
                reason=_format_path(
                    context,
                    f"invalid RMSF {field}: flat={flat_reason}, legacy={nested_reason}",
                ),
            )
        raise MotionMetricError(
            _format_path(
                context,
                f"conflicting RMSF {field}: flat={future.get(flat_key)!r}, "
                f"legacy={nested.get(field)!r}",
            )
        )

    if flat_present:
        return RMSFRead(
            field=field,
            value=flat_value,
            source="flat",
            reason=None
            if flat_value is not None
            else _format_path(context, f"invalid RMSF {field} at future[{flat_key!r}]: {flat_reason}"),
        )
    return RMSFRead(
        field=field,
        value=nested_value,
        source="legacy_nested",
        reason=None
        if nested_value is not None
        else _format_path(context, f"invalid RMSF {field} at future['rmsf'][{field!r}]: {nested_reason}"),
    )


def _summary_system_rows(summary: Mapping[str, Any], history: int) -> Sequence[Mapping[str, Any]] | None:
    aggregates = summary.get("aggregates")
    if not isinstance(aggregates, Mapping):
        return None
    valid = aggregates.get("valid")
    if not isinstance(valid, Mapping):
        return None
    aggregate = valid.get(f"H{int(history)}")
    if not isinstance(aggregate, Mapping):
        return None
    rows = aggregate.get("system_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return None
    return rows


def motion_arm_report(
    summary: Mapping[str, Any],
    history: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Validate the system rows for one arm and one history length."""

    rows = _summary_system_rows(summary, history)
    if rows is None:
        return {
            "label": label,
            "history": int(history),
            "row_count": 0,
            "actual_systems": [],
            "values": {},
            "invalid_systems": [
                {"system": None, "reasons": [f"missing valid H{int(history)} system_rows"]}
            ],
        }

    values: dict[str, dict[str, Any]] = {}
    actual_systems: list[str] = []
    invalid_systems: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            invalid_systems.append(
                {"system": None, "row_index": index, "reasons": ["system row is not a mapping"]}
            )
            continue
        raw_system = row.get("system")
        if raw_system is None or str(raw_system) == "":
            invalid_systems.append(
                {"system": None, "row_index": index, "reasons": ["missing system ID"]}
            )
            continue
        system = str(raw_system)
        context = f"{label} H{int(history)} system {system}"
        if system in values:
            invalid_systems.append(
                {"system": system, "row_index": index, "reasons": ["duplicate system ID"]}
            )
            continue
        prediction = read_rmsf_value(row, "prediction", context=context)
        target = read_rmsf_value(row, "target", context=context)
        reasons = [reason for reason in (prediction.reason, target.reason) if reason]
        actual_systems.append(system)
        values[system] = {
            "prediction": prediction.value,
            "target": target.value,
            "prediction_source": prediction.source,
            "target_source": target.source,
            "reasons": reasons,
        }
        if reasons:
            invalid_systems.append({"system": system, "row_index": index, "reasons": reasons})

    return {
        "label": label,
        "history": int(history),
        "row_count": len(rows),
        "actual_systems": sorted(actual_systems),
        "values": {system: values[system] for system in sorted(values)},
        "invalid_systems": invalid_systems,
    }


def _arm_value(
    arm: Mapping[str, Any],
    system: str,
    field: str,
) -> tuple[float | None, str | None]:
    values = arm.get("values")
    if not isinstance(values, Mapping) or system not in values:
        return None, f"missing {arm.get('label', 'arm')} system row"
    item = values[system]
    if not isinstance(item, Mapping):
        return None, f"malformed {arm.get('label', 'arm')} system row"
    value = item.get(field)
    if value is None:
        reasons = item.get("reasons")
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
            return None, "; ".join(str(reason) for reason in reasons)
        return None, f"missing/invalid {arm.get('label', 'arm')} RMSF {field}"
    number, reason = _finite_scalar(value)
    return (number, None) if number is not None else (None, reason)


def _arm_system_reasons(arm: Mapping[str, Any], system: str) -> list[str]:
    """Return row-validation reasons for a system, including missing targets."""

    reasons: list[str] = []
    invalid = arm.get("invalid_systems")
    if not isinstance(invalid, Sequence) or isinstance(invalid, (str, bytes)):
        return reasons
    for item in invalid:
        if not isinstance(item, Mapping) or str(item.get("system")) != system:
            continue
        item_reasons = item.get("reasons")
        if isinstance(item_reasons, Sequence) and not isinstance(item_reasons, (str, bytes)):
            reasons.extend(str(reason) for reason in item_reasons)
    return reasons


def paired_motion_report(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    candidate_label: str,
    reference_label: str,
    expected_systems: Sequence[str] | None = None,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
    threshold: float = MOTION_RATIO_THRESHOLD,
) -> dict[str, Any]:
    """Pair candidate/reference prediction RMSF and classify the guard.

    Missing or invalid systems make a non-empty comparison ``INCOMPLETE``;
    they never become a passing guard by omission.  If no pair has an usable
    denominator, the comparison is ``NOT_APPLICABLE`` and records each
    near-zero denominator.  A complete comparison is ``PASS`` or ``FAIL``
    according to the unchanged median-ratio threshold.
    """

    candidate_actual = {str(value) for value in candidate.get("actual_systems", [])}
    reference_actual = {str(value) for value in reference.get("actual_systems", [])}
    if expected_systems is None:
        expected = sorted(candidate_actual | reference_actual)
        expected_source = "union_of_candidate_and_reference_system_rows"
    else:
        expected = sorted({str(value) for value in expected_systems})
        expected_source = "caller_supplied"

    missing_candidate = sorted(set(expected) - candidate_actual)
    missing_reference = sorted(set(expected) - reference_actual)
    extra_candidate = sorted(candidate_actual - set(expected))
    extra_reference = sorted(reference_actual - set(expected))
    pairs: list[dict[str, Any]] = []
    noncomputable: list[dict[str, Any]] = []
    for system in expected:
        row_reasons = _arm_system_reasons(candidate, system) + _arm_system_reasons(reference, system)
        if row_reasons:
            noncomputable.append({"system": system, "reasons": row_reasons})
            continue
        candidate_prediction, candidate_reason = _arm_value(candidate, system, "prediction")
        reference_prediction, reference_reason = _arm_value(reference, system, "prediction")
        if candidate_reason or reference_reason:
            noncomputable.append(
                {
                    "system": system,
                    "reasons": [reason for reason in (candidate_reason, reference_reason) if reason],
                }
            )
            continue
        assert candidate_prediction is not None and reference_prediction is not None
        if abs(reference_prediction) <= float(denominator_epsilon):
            noncomputable.append(
                {
                    "system": system,
                    "reasons": [
                        f"{reference_label} prediction RMSF denominator {reference_prediction:.12g} "
                        f"is near zero (epsilon={float(denominator_epsilon):.12g})"
                    ],
                }
            )
            continue
        ratio = candidate_prediction / reference_prediction
        if not math.isfinite(ratio):
            noncomputable.append(
                {
                    "system": system,
                    "reasons": ["candidate/reference prediction RMSF ratio is non-finite"],
                }
            )
            continue
        pairs.append(
            {
                "system": system,
                "candidate_prediction_rmsf": candidate_prediction,
                "reference_prediction_rmsf": reference_prediction,
                "prediction_ratio": ratio,
            }
        )

    global_invalid = []
    for arm in (candidate, reference):
        invalid = arm.get("invalid_systems")
        if not isinstance(invalid, Sequence) or isinstance(invalid, (str, bytes)):
            continue
        for item in invalid:
            if not isinstance(item, Mapping):
                continue
            system = item.get("system")
            if system is not None and str(system) in expected:
                continue
            item_reasons = item.get("reasons")
            if isinstance(item_reasons, Sequence) and not isinstance(item_reasons, (str, bytes)):
                global_invalid.append(
                    {
                        "system": None if system is None else str(system),
                        "reasons": [str(reason) for reason in item_reasons],
                    }
                )
    if global_invalid:
        noncomputable.extend(global_invalid)
    if not expected:
        status = "NOT_APPLICABLE"
        status_reason = "no expected systems"
    elif noncomputable:
        near_zero_only = all(
            bool(item.get("reasons"))
            and all("near zero" in str(reason) for reason in item.get("reasons", []))
            for item in noncomputable
        )
        if not pairs and near_zero_only:
            status = "NOT_APPLICABLE"
            status_reason = "all expected reference prediction RMSF denominators are near zero"
        else:
            status = "INCOMPLETE"
            status_reason = "one or more expected systems are missing, invalid, or non-computable"
    else:
        value = float(median([item["prediction_ratio"] for item in pairs]))
        status = "PASS" if value >= float(threshold) else "FAIL"
        status_reason = f"median candidate/reference prediction RMSF ratio {value:.12g} {'>=' if status == 'PASS' else '<'} {float(threshold):.12g}"

    median_ratio = (
        float(median([item["prediction_ratio"] for item in pairs])) if pairs else None
    )
    return {
        "status": status,
        "status_reason": status_reason,
        "threshold": float(threshold),
        "denominator_epsilon": float(denominator_epsilon),
        "candidate_label": candidate_label,
        "reference_label": reference_label,
        "expected_source": expected_source,
        "expected_systems": expected,
        "candidate_actual_systems": sorted(candidate_actual),
        "reference_actual_systems": sorted(reference_actual),
        "missing_candidate_systems": missing_candidate,
        "missing_reference_systems": missing_reference,
        "extra_candidate_systems": extra_candidate,
        "extra_reference_systems": extra_reference,
        "noncomputable_systems": list(noncomputable),
        "pairs": pairs,
        "median_motion_ratio": median_ratio,
        "motion_ratio_system_count": len(pairs),
    }


def arm_rmsf_diagnostics(
    arm: Mapping[str, Any],
    *,
    denominator_epsilon: float = DEFAULT_DENOMINATOR_EPSILON,
) -> dict[str, Any]:
    """Summarize prediction/target RMSF, preserving ratio definitions."""

    values = arm.get("values")
    if not isinstance(values, Mapping):
        values = {}
    systems: list[dict[str, Any]] = []
    predictions: list[float] = []
    targets: list[float] = []
    ratios: list[float] = []
    ordered_values = sorted(
        ((str(system), item) for system, item in values.items()), key=lambda pair: pair[0]
    )
    for system, item in ordered_values:
        if not isinstance(item, Mapping):
            continue
        prediction = item.get("prediction")
        target = item.get("target")
        prediction_value, prediction_reason = _finite_scalar(prediction)
        target_value, target_reason = _finite_scalar(target)
        ratio = None
        ratio_reason = None
        if prediction_value is not None:
            predictions.append(prediction_value)
        if target_value is not None:
            targets.append(target_value)
        if prediction_reason or target_reason:
            ratio_reason = "; ".join(
                reason for reason in (prediction_reason, target_reason) if reason
            )
        elif target_value is not None and abs(target_value) <= float(denominator_epsilon):
            ratio_reason = (
                f"target RMSF denominator {target_value:.12g} is near zero "
                f"(epsilon={float(denominator_epsilon):.12g})"
            )
        elif prediction_value is not None and target_value is not None:
            ratio = prediction_value / target_value
            ratios.append(ratio)
        systems.append(
            {
                "system": system,
                "prediction_rmsf": prediction_value,
                "target_rmsf": target_value,
                "prediction_target_ratio": ratio,
                "prediction_target_ratio_reason": ratio_reason,
                "reasons": list(item.get("reasons", []))
                if isinstance(item.get("reasons"), Sequence)
                and not isinstance(item.get("reasons"), (str, bytes))
                else [],
            }
        )

    mean_prediction = sum(predictions) / len(predictions) if predictions else None
    mean_target = sum(targets) / len(targets) if targets else None
    ratio_of_means = None
    ratio_of_means_reason = None
    if mean_prediction is None or mean_target is None:
        ratio_of_means_reason = "missing finite prediction or target RMSF values"
    elif abs(mean_target) <= float(denominator_epsilon):
        ratio_of_means_reason = (
            f"mean target RMSF denominator {mean_target:.12g} is near zero "
            f"(epsilon={float(denominator_epsilon):.12g})"
        )
    else:
        ratio_of_means = mean_prediction / mean_target

    return {
        "label": arm.get("label"),
        "history": arm.get("history"),
        "system_count": len(values),
        "finite_prediction_system_count": len(predictions),
        "finite_target_system_count": len(targets),
        "mean_prediction_rmsf": mean_prediction,
        "mean_target_rmsf": mean_target,
        "ratio_of_means_prediction_over_target": ratio_of_means,
        "ratio_of_means_reason": ratio_of_means_reason,
        "mean_of_per_system_prediction_target_ratios": sum(ratios) / len(ratios)
        if ratios
        else None,
        "mean_of_ratios_system_count": len(ratios),
        "systems": systems,
    }


__all__ = [
    "DEFAULT_DENOMINATOR_EPSILON",
    "MOTION_RATIO_THRESHOLD",
    "MotionMetricError",
    "RMSFRead",
    "arm_rmsf_diagnostics",
    "motion_arm_report",
    "paired_motion_report",
    "read_rmsf_value",
]
