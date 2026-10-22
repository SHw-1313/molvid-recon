#!/usr/bin/env bash
# Wait only for the supplied, already-running finite lanes. Never restart them.
set -euo pipefail
[[ $# -gt 0 ]] || { echo 'Usage: post_run.sh CONTAINER_LANE_PID [...]'; exit 2; }
task_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for lane in "$@"; do
 [[ $lane =~ ^[0-9]+$ ]] || { echo 'Numeric existing container PID required'; exit 2; }
 while kill -0 "$lane" 2>/dev/null; do sleep 30; done
done
bash "$task_dir/run_cpu.sh" --allow-partial
