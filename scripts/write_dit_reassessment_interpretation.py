#!/usr/bin/env python
"""Write compact required answers into an existing reassessment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dit_reassessment_interpretation import (
    derive_interpretation,
    render_interpretation_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-path", default="config/dit_source_reassessment_v2_neibu.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="neibu")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--gpu-pid", type=int, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    value = derive_interpretation(output_dir)
    interpretation_path = output_dir / "interpretation.json"
    temporary = interpretation_path.with_name(interpretation_path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(interpretation_path)

    report_path = output_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    report = report.replace("config/dit_source_reassessment_v2.yaml", args.config_path)
    report = report.replace("cuda:IDLE", args.device)
    report = report.replace(
        "enter-container  # then: conda activate torch-ito",
        f"ssh -tt {args.host} 'enter-container'\n"
        "conda activate torch-ito\n"
        "export CUDA_VISIBLE_DEVICES=0",
    )
    execution_line = (
        f"- Execution device: host `{args.host}`, physical GPU0 UUID `{args.gpu_uuid}`, "
        f"evaluation PID `{args.gpu_pid}`; container-visible device `{args.device}`."
    )
    if execution_line not in report:
        report = report.replace(
            "- Test payload opened: `false`.",
            execution_line + "\n- Test payload opened: `false`.",
            1,
        )
    section = "\n".join(render_interpretation_markdown(value))
    short_rollout_marker = "\n## Short rollout\n"
    interpretation_marker = "\n## Required scientific answers\n"
    if interpretation_marker in report:
        prefix, remainder = report.split(interpretation_marker, 1)
        _, suffix = remainder.split(short_rollout_marker, 1)
        report = prefix + "\n" + section + short_rollout_marker + suffix
    else:
        report = report.replace(
            short_rollout_marker,
            "\n" + section + short_rollout_marker,
            1,
        )
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "interpretation": str(interpretation_path),
        "report": str(report_path),
        "answers": value["answers"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
