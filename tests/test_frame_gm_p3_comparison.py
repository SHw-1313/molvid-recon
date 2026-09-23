from __future__ import annotations

import numpy as np
import pytest

from tools.compare_frame_gm_p3 import compare_from_system_values


def _values() -> tuple[dict[str, object], dict[str, object]]:
    systems = [f"system_{index}" for index in range(8)]
    target = np.linspace(0.5, 1.5, 8)
    j0: dict[str, object] = {
        "systems": systems,
        "bond_rmse_A": np.full(8, 2.0),
        "residue_rmsf_mae_A": np.full(8, 1.0),
        "msd_curve_mae_A2": np.full(8, 3.0),
        "rmsf_target_A": target,
    }
    j1 = {
        key: value.copy() if isinstance(value, np.ndarray) else list(value)
        for key, value in j0.items()
    }
    return j0, j1


def test_bond_fork_requires_all_three_clear_directions() -> None:
    j0, j1 = _values()
    j1["bond_rmse_A"] = np.full(8, 1.5)
    j1["residue_rmsf_mae_A"] = np.full(8, 1.5)
    j1["msd_curve_mae_A2"] = np.full(8, 3.5)
    result = compare_from_system_values(
        j0, j1, bootstrap_samples=2000, bootstrap_seed=11
    )
    assert result["trigger_bond_keep_release"] is True
    assert all(result["trigger_components"].values())


@pytest.mark.parametrize(
    "metric,value",
    [
        ("bond_rmse_A", 2.0),
        ("residue_rmsf_mae_A", 1.0),
        ("msd_curve_mae_A2", 3.0),
    ],
)
def test_bond_fork_fails_closed_when_one_direction_is_not_clear(
    metric: str, value: float
) -> None:
    j0, j1 = _values()
    j1["bond_rmse_A"] = np.full(8, 1.5)
    j1["residue_rmsf_mae_A"] = np.full(8, 1.5)
    j1["msd_curve_mae_A2"] = np.full(8, 3.5)
    j1[metric] = np.full(8, value)
    result = compare_from_system_values(
        j0, j1, bootstrap_samples=2000, bootstrap_seed=13
    )
    assert result["trigger_bond_keep_release"] is False


def test_bond_fork_comparison_rejects_unpaired_or_nonfinite_inputs() -> None:
    j0, j1 = _values()
    j1["systems"] = ["different", *list(j1["systems"])[1:]]
    with pytest.raises(ValueError, match="same ordered set"):
        compare_from_system_values(j0, j1, bootstrap_samples=2000, bootstrap_seed=17)
    _, j1 = _values()
    j1["bond_rmse_A"] = np.asarray(j1["bond_rmse_A"])
    j1["bond_rmse_A"][0] = np.nan
    with pytest.raises(ValueError, match="invalid bond_rmse_A"):
        compare_from_system_values(j0, j1, bootstrap_samples=2000, bootstrap_seed=19)
