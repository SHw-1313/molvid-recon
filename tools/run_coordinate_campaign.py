"""Own-GPU queue: bounded real profile/resume gates, then fixed-final training."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from molvid.runtime import atomic_write_json, sha256_file


def assert_same(a, b, path="checkpoint"):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise ValueError(f"resume tensor differs: {path}")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError(f"resume array differs: {path}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f"resume keys differ: {path}")
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError(f"resume lengths differ: {path}")
        for index, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}.{index}")
    elif a != b:
        raise ValueError(f"resume value differs: {path}")


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    atomic_write_json(args.output / "status.json", dict(status="waiting_for_own_GPU_capacity_queue", prerequisite=str(args.wait_for)))
    try:
        while True:
            if args.wait_for.exists():
                status = json.loads(args.wait_for.read_text())
                if status.get("status") == "completed":
                    break
                if status.get("status") == "failed":
                    raise RuntimeError("upstream capacity queue failed")
            if time.monotonic() - started > 86400:
                raise TimeoutError("24 hour own-GPU wait budget")
            time.sleep(20)

        def call(name, command, timeout=22000):
            atomic_write_json(args.output / "status.json", dict(status="running", active=name, command=command))
            with (args.output / f"{name}.log").open("x") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)

        for seed in (71, 72):
            root = args.output / f"seed{seed}"
            config = args.parent_root / f"source_training/unit_{seed}.json"
            base = [sys.executable, "-u", "-m", "tools.train_coordinate_flow", "--config", str(config), "--seed", str(seed)]
            if seed == 71:
                call("largest_nominal_profile", base + ["--output", str(root / "profile"), "--max-steps", "1", "--profile-largest"])
                call("uninterrupted17", base + ["--output", str(root / "pilot17"), "--max-steps", "17"])
                checkpoint16 = root / "pilot17/coordinate_step_0016.pt"
                call("resume16to17", base + ["--output", str(root / "resume17"), "--max-steps", "17",
                     "--resume", str(checkpoint16), "--resume-sha256", sha256_file(checkpoint16)])
                left, right = root / "pilot17/coordinate_step_0017.pt", root / "resume17/coordinate_step_0017.pt"
                assert_same(torch.load(left, map_location="cpu", weights_only=False),
                            torch.load(right, map_location="cpu", weights_only=False))
                atomic_write_json(root / "real_resume_parity.json", dict(exact=True, checkpoint_sha256=[sha256_file(left), sha256_file(right)],
                    scope="whole model,optimizer,CPU/CUDA/flow/numpy/python RNG,cursor,contract", extra_optimizer_updates=1))
                extra = ["--resume", str(right), "--resume-sha256", sha256_file(right)]
            else:
                extra = []
            call(f"training_{seed}", base + ["--output", str(root / "formal")] + extra)
            checkpoint = root / "formal/coordinate_step_2048.pt"
            call(f"evaluation_{seed}", [sys.executable, "-m", "tools.evaluate_expanded_native",
                "--cache", str(args.parent_root / "development48_v2"), "--output", str(root / "expanded48"),
                "--arm", "coordinate", "--variant", f"coordinate_{seed}", "--checkpoint", str(checkpoint)])
            call(f"stereochemistry_{seed}", [sys.executable, "-m", "tools.audit_cached_stereochemistry",
                "--cache", str(args.parent_root / "development48_v2"), "--generated", str(root / "expanded48"),
                "--output", str(root / "stereochemistry")])
            call(f"paired_{seed}", [sys.executable, "-m", "tools.compare_expanded_native", "--generated",
                str(args.parent_root / "expanded_a1"), str(args.parent_root / "expanded_pvb"), str(root / "expanded48"),
                "--exposure", str(args.parent_root / "external_overlap_v3/summary.json"), "--output", str(root / "paired")])
        atomic_write_json(args.output / "status.json", dict(status="completed", default_promotion=False))
    except BaseException as error:
        atomic_write_json(args.output / "status.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wait-for", type=Path, required=True)
    run(p.parse_args())
