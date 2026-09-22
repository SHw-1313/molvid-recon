#!/usr/bin/env python3
"""Run a fresh read-only Codex review and preserve a commit-bound report.

This helper never trains, edits model code, upgrades Codex, or bypasses policy.
Exit codes: 0 PASS, 2 FIX_REQUIRED, 3 BLOCKED_ENV, 64 launcher/format failure.
Run under the repository's approved Python environment. Codex must be visible
there; otherwise use the equivalent host command in REVIEW_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def absolute_existing(value: str, *, directory: bool) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"expected absolute path: {value}")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"missing {'directory' if directory else 'file'}: {value}")
    return path.resolve()


def git(worktree: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(worktree), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def checked_request(path: Path) -> tuple[dict, Path, Path, str]:
    request_bytes = path.read_bytes()
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    request = json.loads(request_bytes)
    if request.get("schema") != "molvid.fresh_review.request.v1":
        raise ValueError("unsupported request schema")
    if request.get("stage") not in ("P2", "P3"):
        raise ValueError("only P2/P3 require this independent review")
    for key in ("base_commit", "candidate_commit"):
        if not SHA1.fullmatch(str(request.get(key, ""))):
            raise ValueError(f"{key} must be a full SHA; replace template placeholders")
    tree = absolute_existing(request["review_worktree"], directory=True)
    if git(tree, "rev-parse", "HEAD") != request["candidate_commit"]:
        raise ValueError("review worktree HEAD differs from candidate_commit")
    if git(tree, "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("review worktree must be clean")
    if git(tree, "rev-parse", "--verify", request["base_commit"] + "^{commit}") != request["base_commit"]:
        raise ValueError("base_commit not available")
    absolute_existing(request["evidence_root"], directory=True)
    for relative in request.get("spec_files", []):
        spec = (tree / relative).resolve()
        if not spec.is_relative_to(tree) or not spec.is_file():
            raise ValueError(f"missing or outside-worktree spec: {relative}")
    artifacts = request.get("artifacts", [])
    if not artifacts:
        raise ValueError("pilot/config evidence artifacts are required")
    for item in artifacts:
        artifact = absolute_existing(item["path"], directory=False)
        if not SHA256.fullmatch(str(item.get("sha256", ""))):
            raise ValueError(f"invalid artifact SHA256: {artifact}")
        if digest(artifact) != item["sha256"]:
            raise ValueError(f"artifact changed: {artifact}")
    output = Path(request["output_dir"])
    if not output.is_absolute():
        raise ValueError("output_dir must be absolute")
    output = output.resolve()
    if output.is_relative_to(tree):
        raise ValueError("review output must be outside the read-only code snapshot")
    return request, tree, output, request_sha256


def report_verdict(text: str, request: dict) -> str:
    def field(name: str) -> str:
        values = re.findall(rf"^{name}:\s*([^\n]+)$", text, flags=re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f"review must have exactly one {name} field")
        return values[0].strip()
    if field("REVIEW_STAGE") != request["stage"]:
        raise ValueError("report stage mismatch")
    if field("REVIEWED_COMMIT") != request["candidate_commit"]:
        raise ValueError("report candidate commit mismatch")
    verdict = field("VERDICT")
    if verdict not in ("PASS", "FIX_REQUIRED", "BLOCKED_ENV"):
        raise ValueError("invalid review verdict")
    if len(text.strip()) < 180:
        raise ValueError("review too short to contain usable evidence")
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model", help="Optional locally available model; default uses local config")
    parser.add_argument("--dry-run", action="store_true", help="Validate and render locally; do not invoke Codex")
    args = parser.parse_args()
    request_path = absolute_existing(str(args.request), directory=False)
    request, tree, output, request_sha256 = checked_request(request_path)
    prompt_path = tree / "agent/frame_gm_calibration_v2/REVIEW_PROMPT.md"
    prompt = prompt_path.read_text(encoding="utf-8")
    prompt += "\n\n## 本轮事实请求（不是通过结论）\n\n```json\n"
    prompt += json.dumps(request, ensure_ascii=False, indent=2) + "\n```\n"
    pending = output / "REVIEW.pending.md"
    report = output / "REVIEW.md"
    command = ["codex", "exec", "--sandbox", "read-only", "--ephemeral",
               "-C", str(tree), "--output-last-message", str(pending)]
    if args.model:
        command.extend(["--model", args.model])
    command.append("-")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_ONLY", "command": command,
                          "candidate_commit": request["candidate_commit"],
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "report_path": str(report)}, ensure_ascii=False, indent=2))
        return 0
    if not shutil.which("codex"):
        raise RuntimeError("Codex CLI unavailable in this environment; use documented host/fresh-agent route")
    output.mkdir(parents=True, exist_ok=True)
    names = [report, pending, output / "launcher.json", output / "rendered_prompt.md",
             output / "codex_stdout.log", output / "codex_stderr.log"]
    if any(path.exists() for path in names):
        raise ValueError("round output already exists; preserve it and choose a new round directory")
    (output / "rendered_prompt.md").write_text(prompt, encoding="utf-8")
    metadata = {"schema": "molvid.fresh_review.launch.v1", "status": "RUNNING",
                "stage": request["stage"], "candidate_commit": request["candidate_commit"],
                "request_sha256": request_sha256, "command": command,
                "conversation_inheritance": "none; new codex exec, no resume",
                "model_override": args.model}
    meta_path = output / "launcher.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""  # review consumes files, not training GPU slots
    with (output / "codex_stdout.log").open("w") as stdout, (output / "codex_stderr.log").open("w") as stderr:
        result = subprocess.run(command, input=prompt, text=True, cwd=tree,
                                env=environment, stdout=stdout, stderr=stderr, check=False)
    metadata["codex_exit_code"] = result.returncode
    metadata["status"] = "PROCESS_EXITED"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if result.returncode != 0 or not pending.is_file():
        raise RuntimeError("review process failed or report missing; inspect preserved logs, do not assume PASS")
    # Check again: results bind immutable candidate code and the same evidence.
    _, _, _, final_request_sha256 = checked_request(request_path)
    if final_request_sha256 != request_sha256:
        raise ValueError("review request changed during review; preserve output and rerun a new round")
    content = pending.read_text(encoding="utf-8")
    verdict = report_verdict(content, request)
    pending.replace(report)
    metadata.update({"status": "COMPLETE", "verdict": verdict,
                     "review_sha256": digest(report), "review_file": str(report)})
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "report": str(report),
                      "candidate_commit": request["candidate_commit"]}, ensure_ascii=False))
    return {"PASS": 0, "FIX_REQUIRED": 2, "BLOCKED_ENV": 3}[verdict]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Review launcher error: {error}", file=sys.stderr)
        raise SystemExit(64)
