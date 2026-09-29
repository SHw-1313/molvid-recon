"""Metadata-only tests; no CPU model execution or sealed data."""

from collections import Counter
from pathlib import Path

import pytest
import yaml

from molvid.cli.train_frame_joint import _validate_frame_joint_config
from molvid.data.sampling import ClipItemSpec, ClipSpecTable, TaskAwareClipBatchSampler
from molvid.evaluation.families import metric_family
from molvid.training.execution import execution_manifest, execution_schedule
from molvid.training.joint import JointLossConfig, SampledAuxiliaryConfig


def _specs():
    items = []
    for view in ("", "fixed_history_"):
        for lag in (100, 200, 400):
            for window in range(80):
                index = len(items)
                items.append(ClipItemSpec(index, 10, 16, "trajectory", f"{view}dt_{lag}ps",
                                          sample_id=f"atlas_system_R1_w{index:06d}"))
    return ClipSpecTable.from_specs(items)


def test_packed_shuffle_coverage_strata_and_resume():
    specs = _specs()
    sampler = TaskAwareClipBatchSampler(specs, max_tokens=160, seed=42, replacement=False)
    batches = sampler.global_batches
    assert sorted(i for b in batches for i in b) == list(range(len(specs)))
    transitions = sum(specs[a[0]].time_bucket_id != specs[b[0]].time_bucket_id
                      for a, b in zip(batches, batches[1:]))
    assert transitions > 100  # A group permutation has only five transitions.
    entries = execution_schedule(batches, specs, histories=(4, 8), seed=20260928, epoch=0)
    manifest = execution_manifest(entries)
    for key, counts in manifest["coverage"].items():
        assert counts["aux_updates"] == int(counts["updates"] / 8 + 0.5)
    histories = Counter((e.time_bucket_id, e.history_frames) for e in entries if e.view == "ordinary")
    for lag in (100, 200, 400):
        assert histories[(f"dt_{lag}ps", 4)] == histories[(f"dt_{lag}ps", 8)]
    restored = TaskAwareClipBatchSampler(specs, max_tokens=160, seed=42, replacement=False)
    restored.load_state_dict(sampler.state_dict())
    restored_entries = execution_schedule(restored.global_batches, specs, histories=(4, 8), seed=20260928, epoch=0)
    assert execution_manifest(restored_entries) == manifest
    # DDP entries are the same global list, then sharded by global index.
    ranks = [TaskAwareClipBatchSampler(specs, max_tokens=160, seed=42, replacement=False,
                                     num_replicas=2, rank=r, drop_last=True) for r in (0, 1)]
    assert tuple(map(tuple, ranks[0])) == batches[::2]
    assert tuple(map(tuple, ranks[1])) == batches[1::2]
    old = sampler.state_dict()
    old.pop("batch_order_version")
    with pytest.raises(ValueError, match="batch order"):
        restored.load_state_dict(old)


def test_family_and_endpoint_release_are_separate():
    fixed = metric_family({"time_bucket_id": "fixed_history_dt_100ps",
                           "view_kind": "fixed_history_h4_horizon_1200ps"}, query_frames=12)
    assert not fixed["main_rmsf_available"]
    assert metric_family({"time_bucket_id": "dt_400ps"}, query_frames=8)["main_rmsf_available"]
    assert not metric_family({"time_bucket_id": "dt_100ps"}, query_frames=3)["main_rmsf_available"]
    with pytest.raises(ValueError):
        metric_family({}, query_frames=12)
    loss = JointLossConfig.resolve({"generated_bond": 0.0, "bond_enabled": True})
    assert (loss.clean_bond, loss.near_bond, loss.generated_bond) == (0.1, 0.01, 0.0)
    assert SampledAuxiliaryConfig.resolve({"enabled": True, "euler_steps": 8,
                                          "observed_bond_weight": 1}).feature_scales is None


def test_public_cli_accepts_non_energy_branch_without_scales():
    path = Path(__file__).resolve().parents[1] / "configs/frame_gm_p3_formal_J0_260923.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["sampled_auxiliary"] = {"enabled": True, "observed_bond_weight": 1.0, "euler_steps": 8}
    _validate_frame_joint_config(raw)
    raw["sampled_auxiliary"]["energy_weight"] = 0.1
    with pytest.raises(ValueError, match="energy score requires"):
        _validate_frame_joint_config(raw)


def test_physical_configuration_does_not_invent_uncalibrated_tables():
    from molvid.losses.sampled_physical import PhysicalAuxiliaryConfig
    with pytest.raises(ValueError, match="explicit frozen"):
        PhysicalAuxiliaryConfig.resolve({"clash_weight": 1.0})
    with pytest.raises(ValueError, match="train-only scale"):
        PhysicalAuxiliaryConfig.resolve({"rmsf_weight": 1.0})
    with pytest.raises(ValueError, match="training-derived widths"):
        PhysicalAuxiliaryConfig.resolve({"bond_band_weight": 1.0})
    config = SampledAuxiliaryConfig.resolve({"enabled": True, "physical": {
        "displacement_weight": 0.1, "displacement_scale_A2": 1.0,
    }})
    assert config.feature_scales is None
    assert SampledAuxiliaryConfig.resolve(config.contract()) == config
    with pytest.raises(ValueError, match="historical J1"):
        SampledAuxiliaryConfig.resolve({"enabled": True, "observed_bond_weight": 0.1,
                                       "physical": {"displacement_weight": 0.1, "displacement_scale_A2": 1.0}})


def test_static_auxiliary_config_is_strict_and_internal_pool_is_frozen():
    path = Path(__file__).resolve().parents[1] / "configs/frame_gm_p3_formal_J0_260923.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["static_decoder_auxiliary"] = {
        "enabled": True,
        "source": "internal_md",
        "cadence": 4,
        "seed": 20260929,
        "max_atoms_per_update": 5000,
        "max_source_atoms": 1000,
        "expected_pool_size": 186,
        "exposure_target": {
            "static_updates": 250, "samples": 1000, "atoms": 800000,
            "supervised_atoms": 800000, "sample_tolerance": 20,
            "atom_tolerance": 20000, "supervised_atom_tolerance": 20000,
        },
        "perturbation": {
            "sigma_per_axis_angstrom": [0.05, 0.10, 0.20],
            "clean_probability": 0.20,
        },
        "loss": {"coordinate": 1.0, "bond": 0.1, "clash": 0.001, "gamma": 0.01},
        "gamma_calibration": {
            "path": "train-only-gamma.json",
            "sha256": "gamma-sha256",
            "parent_sha256": "released-parent-sha256",
            "code_commit": "code-commit",
            "reference_config_sha256": "reference-config-sha256",
            "recipe_release_sha256": "recipe-release-sha256",
            "auxiliary_recipe_sha256": "auxiliary-recipe-sha256",
        },
    }
    _validate_frame_joint_config(raw)
    raw["static_decoder_auxiliary"]["cadence"] = 5
    with pytest.raises(ValueError, match="cadence"):
        _validate_frame_joint_config(raw)
    raw["static_decoder_auxiliary"]["cadence"] = 4
    raw["static_decoder_auxiliary"]["expected_pool_size"] = 185
    with pytest.raises(ValueError, match="three-replica"):
        _validate_frame_joint_config(raw)


def test_external_static_config_requires_a_complete_eligibility_identity():
    path = Path(__file__).resolve().parents[1] / "configs/frame_gm_p3_formal_J0_260923.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["static_decoder_auxiliary"] = {
        "enabled": True,
        "source": "external_static",
        "cadence": 4,
        "seed": 20260929,
        "max_atoms_per_update": 5000,
        "max_source_atoms": 1000,
        "expected_pool_size": 2000,
        "exposure_target": {
            "static_updates": 250, "samples": 1000, "atoms": 800000,
            "supervised_atoms": 800000, "sample_tolerance": 20,
            "atom_tolerance": 20000, "supervised_atom_tolerance": 20000,
        },
        "perturbation": {},
        "loss": {"gamma": 0.01},
        "gamma_calibration": {
            "path": "train-only-gamma.json",
            "sha256": "gamma-sha256",
            "parent_sha256": "released-parent-sha256",
            "code_commit": "code-commit",
            "reference_config_sha256": "reference-config-sha256",
            "recipe_release_sha256": "recipe-release-sha256",
            "auxiliary_recipe_sha256": "auxiliary-recipe-sha256",
        },
        "external": {
            "store": "static-store",
            "store_index_sha256": "index-sha256",
            "source": "pdb",
            "split": "train",
            "eligibility_manifest": "eligibility.json",
            "near_homology_policy": "waived_by_user_20260929",
        },
    }
    with pytest.raises(ValueError, match="identity is incomplete"):
        _validate_frame_joint_config(raw)
