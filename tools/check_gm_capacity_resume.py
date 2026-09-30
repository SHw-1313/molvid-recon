"""Prove exact interrupted/resumed execution for both frozen capacity arms."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from molvid.cli.train_frame_joint import _validate_frame_joint_config
from molvid.runtime import atomic_write_json, sha256_file
from tools.gm_capacity_common import capacity_source_identity


def _assert_equal(first: Any, second: Any, path: str = "payload") -> int:
    """Require recursive value equality and return the number of tensors checked."""

    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        if not torch.equal(first, second):
            raise RuntimeError(f"resume differs at tensor {path}")
        return 1
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        if not np.array_equal(first, second):
            raise RuntimeError(f"resume differs at array {path}")
        return 0
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        if first.keys() != second.keys():
            raise RuntimeError(f"resume mapping keys differ at {path}")
        return sum(
            _assert_equal(first[key], second[key], f"{path}.{key}")
            for key in first
        )
    if (
        isinstance(first, Sequence)
        and isinstance(second, Sequence)
        and not isinstance(first, (str, bytes))
        and not isinstance(second, (str, bytes))
    ):
        if len(first) != len(second):
            raise RuntimeError(f"resume sequence length differs at {path}")
        return sum(
            _assert_equal(left, right, f"{path}[{index}]")
            for index, (left, right) in enumerate(zip(first, second))
        )
    if type(first) is not type(second) or first != second:
        raise RuntimeError(f"resume value differs at {path}: {first!r} != {second!r}")
    return 0


def _run(command: list[str], log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _train_command(
    config: Path,
    *,
    checkpoint_every: int,
    parent: Path | None = None,
    parent_sha256: str | None = None,
    resume: Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "molvid.cli.train_frame_joint",
        "--config",
        str(config),
        "--checkpoint-every",
        str(checkpoint_every),
    ]
    if resume is not None:
        command.extend(["--resume", str(resume), "--resume-sha256", sha256_file(resume)])
    elif parent is not None and parent_sha256 is not None:
        command.extend([
            "--warm-start",
            str(parent),
            "--warm-start-sha256",
            parent_sha256,
        ])
    else:
        raise ValueError("resume check requires a warm-start parent or resume checkpoint")
    if dry_run:
        command.append("--dry-run")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_identity = capacity_source_identity()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    if frozen.get("schema") != "molvid.capacity.frozen_configs.v1":
        raise ValueError("resume check requires frozen capacity configs")
    if frozen.get("source_identity") != source_identity:
        raise ValueError("resume check source differs from the frozen configs")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite resume evidence {args.output}")
    args.output.mkdir(parents=True)
    evidence: dict[str, Any] = {
        "schema": "molvid.capacity.resume_check.v1",
        "source_identity": source_identity,
        "frozen_configs": str(args.frozen.resolve()),
        "frozen_configs_sha256": sha256_file(args.frozen),
        "arms": {},
        "test_opened": False,
    }
    parent = Path(frozen["parent_path"])
    parent_sha256 = str(frozen["parent_sha256"])
    for arm in ("C4", "C8"):
        template_path = Path(frozen["arms"][arm]["pilot"]["config"])
        template = json.loads(template_path.read_text(encoding="utf-8"))
        arm_root = args.output / arm
        uninterrupted_output = arm_root / "uninterrupted"
        recovered_output = arm_root / "recovered"
        configs = {}
        for mode, output in (
            ("uninterrupted", uninterrupted_output),
            ("recovered", recovered_output),
        ):
            config = copy.deepcopy(template)
            config["training"]["stages"][0]["updates"] = 2
            config["training"]["gradient_diagnostics_every"] = 0
            config["training"]["output_root"] = str(output.resolve())
            _validate_frame_joint_config(config)
            path = arm_root / f"{mode}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, config)
            configs[mode] = path

        _run(
            _train_command(
                configs["uninterrupted"],
                checkpoint_every=1,
                parent=parent,
                parent_sha256=parent_sha256,
            ),
            arm_root / "uninterrupted.log",
        )
        _run(
            _train_command(
                configs["recovered"],
                checkpoint_every=1,
                parent=parent,
                parent_sha256=parent_sha256,
                dry_run=True,
            ),
            arm_root / "interrupted.log",
        )
        first_checkpoint = recovered_output / "frame_joint_step_00000001.pt"
        _run(
            _train_command(
                configs["recovered"],
                checkpoint_every=1,
                resume=first_checkpoint,
            ),
            arm_root / "resumed.log",
        )
        uninterrupted = uninterrupted_output / "frame_joint_step_00000002.pt"
        resumed = recovered_output / "frame_joint_step_00000002.pt"
        uninterrupted_payload = torch.load(uninterrupted, map_location="cpu", weights_only=False)
        resumed_payload = torch.load(resumed, map_location="cpu", weights_only=False)
        tensor_count = _assert_equal(uninterrupted_payload, resumed_payload)
        rank_state = resumed_payload["extra_state"]["rank_states"]["0"]
        evidence["arms"][arm] = {
            "uninterrupted_checkpoint": str(uninterrupted),
            "uninterrupted_checkpoint_sha256": sha256_file(uninterrupted),
            "resumed_checkpoint": str(resumed),
            "resumed_checkpoint_sha256": sha256_file(resumed),
            "payload_exact": True,
            "tensor_count": tensor_count,
            "model_state_exact": True,
            "optimizer_state_exact": True,
            "cursor_exact": True,
            "exposure_exact": True,
            "training_generator_exact": isinstance(
                rank_state.get("training_generator_state"), torch.Tensor
            ),
            "sampled_generator_exact": isinstance(
                rank_state.get("sampled_generator_state"), torch.Tensor
            ),
            "global_rng_exact": True,
            "checkpoint_flow_field": resumed_payload["contracts"]["optimization"][
                "checkpoint_flow_field"
            ],
        }
        if evidence["arms"][arm]["checkpoint_flow_field"] is not True:
            raise RuntimeError(f"{arm} resume check did not exercise flow checkpointing")
    evidence["passed"] = True
    atomic_write_json(args.output / "resume_check.json", evidence)
    print(json.dumps({
        "passed": True,
        "arms": {
            arm: {"tensor_count": row["tensor_count"], "payload_exact": True}
            for arm, row in evidence["arms"].items()
        },
        "output": str(args.output / "resume_check.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
