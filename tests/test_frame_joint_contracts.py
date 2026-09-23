from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from molvid.cli.train_frame_joint import (
    _exposure_state,
    _merge_exposure_states,
    _validate_exposure_state,
    _validate_frame_joint_config,
    _verified_derived_identity,
)
from molvid.config import load_config
from molvid.runtime import sha256_file
from molvid.training.joint import JointLossConfig, SampledAuxiliaryConfig
from tools.profile_frame_gm_p2 import _arm_from_config


def _valid_config() -> dict:
    return load_config(
        Path(__file__).parents[1] / "configs/frame_gm_p2_pilot_B0_260923.yaml",
        schema="molvid.frame_joint.train.v1",
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"typo": 1}), "unknown root keys"),
        (lambda value: value["model"].update({"geometry_enable": True}), "unknown model keys"),
        (lambda value: value["data"]["derived_train"][0].update({"records": 6912}), "unknown data.derived_train"),
        (lambda value: value["training"]["stages"][0].update({"update": 128}), "unknown training.stages"),
        (
            lambda value: value["training"]["stages"][0]["learning_rates"].update({"codec": 1e-4}),
            "unknown training.stages[0].learning_rates",
        ),
    ],
)
def test_frame_joint_config_rejects_unknown_keys(mutate, message: str) -> None:
    config = deepcopy(_valid_config())
    mutate(config)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _validate_frame_joint_config(config)


@pytest.mark.parametrize(
    ("section", "name"),
    [
        ("model", "geometry_enabled"),
        ("model", "motion_enabled"),
        ("training", "deterministic"),
        ("loss", "inherit_parent"),
    ],
)
def test_frame_joint_config_requires_real_booleans(section: str, name: str) -> None:
    config = deepcopy(_valid_config())
    config[section][name] = "false"
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_frame_joint_config(config)


def test_all_p2_configs_pass_strict_validation() -> None:
    root = Path(__file__).parents[1]
    for path in sorted((root / "configs").glob("frame_gm_p2_*_260923.yaml")):
        _validate_frame_joint_config(load_config(path, schema="molvid.frame_joint.train.v1"))


@pytest.mark.parametrize(
    ("name", "arm"),
    [
        ("frame_gm_p2_pilot_B0_260923.yaml", "B0"),
        ("frame_gm_p2_formal_G_260923.yaml", "G"),
        ("frame_gm_p2_formal_M_260923.yaml", "M"),
        ("frame_gm_p2_formal_GM_260923.yaml", "GM"),
    ],
)
def test_profile_config_name_identifies_arm(name: str, arm: str) -> None:
    assert _arm_from_config(Path(name)) == arm


def test_derived_identity_binds_index_and_record_count(tmp_path: Path) -> None:
    (tmp_path / "index.txt").write_text("one\ntwo\n", encoding="utf-8")
    index_hash = sha256_file(tmp_path / "index.txt")
    item = {"store_index_sha256": index_hash, "record_count": 2}
    manifest = {"store_index_sha256": index_hash, "record_count": 2}
    assert _verified_derived_identity(item, manifest, tmp_path, 2) == {
        "index_sha256": index_hash,
        "record_count": 2,
    }
    (tmp_path / "index.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="store index SHA-256 differs"):
        _verified_derived_identity(item, manifest, tmp_path, 2)
    with pytest.raises(ValueError, match="record count differs"):
        _verified_derived_identity(
            {**item, "store_index_sha256": sha256_file(tmp_path / "index.txt")},
            {**manifest, "store_index_sha256": sha256_file(tmp_path / "index.txt")},
            tmp_path,
            1,
        )


def _apply_exposure(
    trajectories: Counter[str], buckets: Counter[str], views: Counter[str], batch: tuple[str, str, str, int]
) -> int:
    trajectory, bucket, view, atom_frames = batch
    trajectories[trajectory] += 1
    buckets[bucket] += 1
    views[view] += 1
    return atom_frames


def test_continuous_and_strict_resume_exposure_are_exactly_equal() -> None:
    batches = (
        ("atlas_a_R1", "dt_100ps", "legacy_uniform|H4", 100),
        ("atlas_b_R2", "fixed_history_dt_200ps", "fixed_history|H4", 80),
        ("atlas_a_R1", "dt_400ps", "legacy_uniform|H8", 60),
    )
    continuous = (Counter(), Counter(), Counter())
    continuous_frames = sum(_apply_exposure(*continuous, batch) for batch in batches)
    continuous_state = _exposure_state(
        *continuous, successful_updates=3, valid_atom_frames=continuous_frames
    )

    before_resume = (Counter(), Counter(), Counter())
    before_frames = _apply_exposure(*before_resume, batches[0])
    checkpoint_state = _exposure_state(
        *before_resume, successful_updates=1, valid_atom_frames=before_frames
    )
    restored = _validate_exposure_state(checkpoint_state, successful_updates=1)
    resumed = (
        Counter(restored["trajectory_clip_exposure"]),
        Counter(restored["bucket_clip_exposure"]),
        Counter(restored["view_history_clip_exposure"]),
    )
    resumed_frames = int(restored["valid_atom_frames"])
    for batch in batches[1:]:
        resumed_frames += _apply_exposure(*resumed, batch)
    resumed_state = _exposure_state(
        *resumed, successful_updates=3, valid_atom_frames=resumed_frames
    )
    assert resumed_state == continuous_state
    assert _merge_exposure_states([resumed_state])["trajectory_clip_exposure"] == {
        "atlas_a_R1": 2,
        "atlas_b_R2": 1,
    }


def test_loss_contract_rejects_unknown_and_string_boolean() -> None:
    with pytest.raises(ValueError, match="unknown loss keys"):
        JointLossConfig.resolve({"generated_bond": 0.1, "near_bnd": 0.2})
    with pytest.raises(ValueError, match="must be a boolean"):
        JointLossConfig.resolve({"generated_bond": 0.1, "bond_enabled": "false"})


def test_sampled_auxiliary_config_is_explicit_and_frozen() -> None:
    config = deepcopy(_valid_config())
    config["sampled_auxiliary"] = {
        "enabled": True,
        "cadence": 8,
        "draws": 2,
        "euler_steps": 4,
        "activation_checkpoint": True,
        "feature_scales": "fit",
        "energy_weight": "calibrate",
        "observed_bond_weight": "calibrate",
        "calibration_batches": 8,
        "energy_gradient_target_ratio": 0.05,
        "observed_bond_gradient_target_ratio": 0.05,
    }
    _validate_frame_joint_config(config)
    for name, value in (
        ("cadence", 7),
        ("draws", 3),
        ("euler_steps", 8),
        ("calibration_batches", 7),
        ("energy_gradient_target_ratio", 0.1),
    ):
        invalid = deepcopy(config)
        invalid["sampled_auxiliary"][name] = value
        with pytest.raises(ValueError, match="frozen"):
            _validate_frame_joint_config(invalid)
    invalid = deepcopy(config)
    invalid["sampled_auxiliary"]["enabled"] = "true"
    with pytest.raises(ValueError, match="must be a boolean"):
        _validate_frame_joint_config(invalid)


def test_sampled_auxiliary_runtime_contract_and_cadence() -> None:
    scales = {
        "residue_rmsf_A": 1.0,
        "displacement_squared_A2": 2.0,
        "internal_distance_increment_A": 3.0,
        "internal_distance_increment_product_A2": 4.0,
    }
    enabled = SampledAuxiliaryConfig.resolve({
        "enabled": True,
        "cadence": 8,
        "draws": 2,
        "euler_steps": 4,
        "activation_checkpoint": True,
        "energy_weight": 0.25,
        "observed_bond_weight": 0.5,
        "feature_scales": scales,
    })
    assert [enabled.active_at(step) for step in range(16)] == [
        False, False, False, False, False, False, False, True,
        False, False, False, False, False, False, False, True,
    ]
    assert enabled.contract()["target_used_as_condition"] is False
    assert SampledAuxiliaryConfig.resolve(enabled.contract()) == enabled
    assert not SampledAuxiliaryConfig().active_at(7)
    invalid_source = enabled.contract()
    invalid_source["source"] = "future_target_plus_noise"
    with pytest.raises(ValueError, match="source contract differs"):
        SampledAuxiliaryConfig.resolve(invalid_source)
    invalid_target_boundary = enabled.contract()
    invalid_target_boundary["target_used_as_condition"] = True
    with pytest.raises(ValueError, match="must not use target"):
        SampledAuxiliaryConfig.resolve(invalid_target_boundary)
    invalid_integer = enabled.contract()
    invalid_integer["cadence"] = True
    with pytest.raises(ValueError, match="must be integers"):
        SampledAuxiliaryConfig.resolve(invalid_integer)
