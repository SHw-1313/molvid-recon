"""Shared, auditable construction for the fixed-history C4/C8 capacity probe."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from molvid.checkpoints import load_frame_joint_inference
from molvid.config import load_config
from molvid.model import FrameJointModel
from molvid.runtime import sha256_file
from molvid.training.joint import (
    FrameJointTrainer,
    JointLossConfig,
    SampledAuxiliaryConfig,
    WarmStartReport,
)


CAPACITY_DEPTHS = (4, 8)
GEOMETRY_BLOCKS = {4: (2, 4), 8: (2, 4, 6, 8)}
PARENT_BLOCK_MAPPING = {4: (1, 2, 3, 4), 8: (2, 4, 6, 8)}
SOURCE_PATHS = ("molvid", "tools", "tests", "agent/gm_motion_memory_v3/runtime_paths.json")


def capacity_source_identity(*, require_clean: bool = True) -> dict[str, Any]:
    """Bind execution evidence to the exact committed capacity source tree."""

    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHS],
        cwd=root,
        text=True,
    ).splitlines()
    if require_clean and status:
        raise RuntimeError(
            "capacity acceptance evidence requires clean committed runtime source: "
            + "; ".join(status[:8])
        )
    return {
        "commit": commit,
        "tree": tree,
        "paths": list(SOURCE_PATHS),
        "clean": not status,
        "status": status,
    }


@dataclass(frozen=True)
class CapacityRelease:
    release_path: Path
    release: Mapping[str, Any]
    recipe_release_path: Path
    recipe_release: Mapping[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    configuration_path: Path
    configuration_sha256: str
    configuration: Mapping[str, Any]


def load_capacity_release(path: str | Path) -> CapacityRelease:
    """Validate the completed A handoff without opening sealed test data."""

    release_path = Path(path).resolve()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema") != "molvid.gm_memory.capacity_release.v1":
        raise ValueError("capacity release schema differs")
    if (
        release.get("status") not in {"released_interim_A1", "released_final_A1"}
        or release.get("selected_arm") != "A1"
    ):
        raise ValueError("capacity release is not the immutable evaluated A1 handoff")
    validation = release.get("validation")
    if (
        not isinstance(validation, Mapping)
        or int(validation.get("ordinary_records", 0)) < 1
        or int(validation.get("fixed_history_records", 0)) < 1
        or validation.get("guardrail_pass") is not True
    ):
        raise ValueError("capacity release lacks completed formal evaluation")
    if release.get("test_opened") is not False:
        raise ValueError("capacity release does not preserve the sealed-test boundary")

    recipe_release_path = release_path.with_name(str(release["source_release"]))
    recipe_release = json.loads(recipe_release_path.read_text(encoding="utf-8"))
    if recipe_release.get("schema") != "molvid.gm_memory.recipe_release.v1":
        raise ValueError("recipe release schema differs")
    if recipe_release.get("status") not in {
        "released_interim_A1",
        "final_A0_A3_formal_evaluated",
    }:
        raise ValueError("recipe release is not a completed evaluated handoff")
    if recipe_release.get("selected_arm") != "A1":
        raise ValueError("capacity and recipe releases select different arms")
    if recipe_release.get("review", {}).get("verdict") != "PASS":
        raise ValueError("released A recipe lacks independent PASS review")
    if recipe_release.get("test_opened") is not False:
        raise ValueError("recipe release does not preserve the sealed-test boundary")

    checkpoint_path = Path(str(release["checkpoint"]))
    checkpoint_sha256 = str(release["checkpoint_sha256"])
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError("released A1 checkpoint SHA-256 differs")
    configuration_path = Path(str(release["configuration"]))
    configuration_sha256 = str(release["configuration_sha256"])
    if sha256_file(configuration_path) != configuration_sha256:
        raise ValueError("released A1 configuration SHA-256 differs")
    configuration = load_config(
        configuration_path, schema="molvid.frame_joint.train.v1"
    )
    model = configuration["model"]
    if (
        int(model.get("depth", 0)) != 4
        or model.get("geometry_enabled") is not True
        or model.get("motion_enabled") is not True
    ):
        raise ValueError("released capacity parent is not the required four-block GM model")
    if sha256_file(configuration["codec"]["checkpoint"]) != str(release["teacher_sha256"]):
        raise ValueError("released target teacher SHA-256 differs")
    if sha256_file(configuration["statistics"]["path"]) != str(release["statistics_sha256"]):
        raise ValueError("released statistics SHA-256 differs")
    selected_checkpoint = recipe_release.get(
        "selected_checkpoint", recipe_release.get("A1_checkpoint", {})
    )
    if selected_checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("recipe and capacity checkpoint identities differ")
    if int(selected_checkpoint.get("updates", release["formal_updates"])) != int(
        release["formal_updates"]
    ):
        raise ValueError("recipe and capacity update counts differ")
    if int(selected_checkpoint.get("effective_epochs", release["effective_epochs"])) != int(
        release["effective_epochs"]
    ):
        raise ValueError("recipe and capacity effective epochs differ")
    if recipe_release.get("configuration", {}).get("sha256") != configuration_sha256:
        raise ValueError("recipe and capacity configuration identities differ")
    if recipe_release.get("teacher_sha256") != release.get("teacher_sha256"):
        raise ValueError("recipe and capacity teacher identities differ")
    if recipe_release.get("statistics_sha256") != release.get("statistics_sha256"):
        raise ValueError("recipe and capacity statistics identities differ")

    def require_equal(actual: Any, expected: Any, name: str) -> None:
        if float(actual) != float(expected):
            raise ValueError(f"released A1 numeric recipe differs for {name}")

    loss = configuration["loss"]
    ledger = recipe_release["loss_ledger"]
    for name in (
        "clean_coordinate",
        "clean_bond",
        "near_coordinate",
        "near_bond",
        "generated_bond",
    ):
        require_equal(ledger[name]["weight"], loss[name], name)
    require_equal(
        ledger["generated_bond"]["min_flow_time"],
        loss["generated_bond_min_flow_time"],
        "generated_bond_min_flow_time",
    )
    for name in ("near_coordinate", "near_bond"):
        require_equal(
            ledger[name]["min_flow_time"], loss["near_min_flow_time"], f"{name}_min_flow_time"
        )
    sampled = configuration["sampled_auxiliary"]
    auxiliary = recipe_release["auxiliary_recipe"]
    for name in ("draws", "euler_steps", "cadence"):
        require_equal(auxiliary[name], sampled[name], name)
    physical = sampled["physical"]
    for ledger_name, config_name in (
        ("sampled_clash", "clash_weight"),
        ("sampled_rmsf", "rmsf_weight"),
        ("sampled_displacement", "displacement_weight"),
        ("sampled_bond_band", "bond_band_weight"),
    ):
        require_equal(ledger[ledger_name]["weight"], physical[config_name], ledger_name)
    require_equal(ledger["sampled_rmsf"]["scale_A"], physical["rmsf_scale_A"], "rmsf_scale_A")
    require_equal(
        ledger["sampled_displacement"]["scale_A2"],
        physical["displacement_scale_A2"],
        "displacement_scale_A2",
    )
    return CapacityRelease(
        release_path=release_path,
        release=release,
        recipe_release_path=recipe_release_path,
        recipe_release=recipe_release,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        configuration_path=configuration_path,
        configuration_sha256=configuration_sha256,
        configuration=configuration,
    )


def load_parent_model(
    bundle: CapacityRelease, *, device: torch.device
) -> Mapping[str, Any]:
    loaded = load_frame_joint_inference(
        bundle.checkpoint_path,
        expected_sha256=bundle.checkpoint_sha256,
        codec_path=bundle.configuration["codec"]["checkpoint"],
        codec_sha256=bundle.configuration["codec"]["sha256"],
        device=device,
    )
    contracts = loaded["payload"]["contracts"]
    if contracts.get("loss") != bundle.recipe_release.get("loss_ledger", {}):
        # The release ledger is explanatory and has richer fields. Check the
        # exact numeric runtime contract separately below.
        released_numeric = {
            name: value["weight"]
            for name, value in bundle.recipe_release["loss_ledger"].items()
            if isinstance(value, Mapping) and "weight" in value
            and name in {"clean_coordinate", "clean_bond", "near_coordinate", "near_bond"}
        }
        for name, value in released_numeric.items():
            if float(contracts["loss"][name]) != float(value):
                raise ValueError(f"released loss coefficient differs for {name}")
    if contracts.get("data_hash") != bundle.recipe_release["data"]["derived_data_hash"]:
        raise ValueError("released A1 checkpoint data hash differs")
    return loaded


def build_capacity_trainer(
    bundle: CapacityRelease,
    parent: Mapping[str, Any],
    *,
    depth: int,
    device: torch.device,
    schedule_contract: Mapping[str, Any],
) -> tuple[FrameJointTrainer, WarmStartReport]:
    """Build one capacity arm and strictly map every parent block."""

    if depth not in CAPACITY_DEPTHS:
        raise ValueError(f"capacity depth must be one of {CAPACITY_DEPTHS}")
    model_config = bundle.configuration["model"]
    model = FrameJointModel.from_codec(
        parent["codec_artifact"].model,
        parent["statistics"].to(device=device),
        scalar_width=int(model_config["scalar_width"]),
        vector_width=int(model_config["vector_width"]),
        depth=depth,
        heads=int(model_config["heads"]),
        geometry_enabled=True,
        motion_enabled=True,
        geometry_block_indices=GEOMETRY_BLOCKS[depth],
    ).to(device)
    stages = bundle.configuration["training"]["stages"]
    learning_rates = {
        name: max(float(stage["learning_rates"][name]) for stage in stages)
        for name in ("history_encoder", "dit", "decoder")
    }
    trainer = FrameJointTrainer(
        model,
        loss_config=JointLossConfig.resolve(parent["payload"]["contracts"]["loss"]),
        learning_rates=learning_rates,
        weight_decay=float(bundle.configuration["training"].get("weight_decay", 0.01)),
        grad_clip=float(bundle.configuration["training"].get("grad_clip", 1.0)),
        stage="continuation",
        amp=bundle.configuration["training"].get("precision") == "bf16",
        data_hash=str(parent["payload"]["contracts"]["data_hash"]),
        teacher_artifact_sha256=str(bundle.configuration["codec"]["sha256"]),
        schedule_contract=dict(schedule_contract),
        sampled_config=SampledAuxiliaryConfig.resolve(
            bundle.configuration.get("sampled_auxiliary")
        ),
        checkpoint_flow_field=True,
    )
    report = trainer.load_warm_start(
        bundle.checkpoint_path,
        expected_sha256=bundle.checkpoint_sha256,
        dit_block_mapping=PARENT_BLOCK_MAPPING[depth],
    )
    if report.dit_block_mapping_one_based != PARENT_BLOCK_MAPPING[depth]:
        raise RuntimeError("capacity warm-start report lost its block mapping")
    expected_identity = () if depth == 4 else (1, 3, 5, 7)
    if report.identity_initialized_blocks_one_based != expected_identity:
        raise RuntimeError("capacity warm-start identity-block report differs")
    expected_new_geometry = () if depth == 4 else (2, 6)
    if report.zero_initialized_geometry_blocks_one_based != expected_new_geometry:
        raise RuntimeError("capacity warm-start new-geometry report differs")
    return trainer, report


def parameter_ledger(model: FrameJointModel) -> dict[str, Any]:
    def count(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    blocks = []
    for index, block in enumerate(model.dit.blocks):
        geometry = 0 if block.local_geometry is None else count(block.local_geometry)
        blocks.append({
            "block_one_based": index + 1,
            "parameters": count(block),
            "geometry_parameters": geometry,
            "geometry_enabled": block.local_geometry is not None,
        })
    return {
        "total_parameters": count(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "dit_parameters": count(model.dit),
        "geometry_parameters": sum(row["geometry_parameters"] for row in blocks),
        "blocks": blocks,
    }


__all__ = [
    "CAPACITY_DEPTHS",
    "GEOMETRY_BLOCKS",
    "PARENT_BLOCK_MAPPING",
    "CapacityRelease",
    "capacity_source_identity",
    "build_capacity_trainer",
    "load_capacity_release",
    "load_parent_model",
    "parameter_ledger",
]
