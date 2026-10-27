#!/usr/bin/env bash
# Two finite independent continuation lists; no retry, resource expansion or old job changes.
set -u
cd /workspace/MOLVID-FLOW-KERNEL-20261027
export PYTHONPATH="$PWD"
if [ "${CONDA_DEFAULT_ENV:-}" != torch-ito ]; then exit 2; fi
lane="$1"
if [ "$lane" = 4 ]; then
  wait_arm=control; remaining=(h1 dense_history)
elif [ "$lane" = 2 ]; then
  wait_arm=history_mix; remaining=(lag1)
else exit 2; fi
receipt="runs/flow_kernel_20261027/${wait_arm}_execution.json"
# Receipt belongs only to this authorized finite study. Bound waiting to36h.
for ((i=0; i<4320; i++)); do
  state=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$receipt")
  if [ "$state" != running ]; then break; fi
  sleep 30
done
if [ "$state" = running ]; then exit 124; fi
for arm in "${remaining[@]}"; do
  CUDA_VISIBLE_DEVICES="$lane" python -u tasks/flow-kernel-20261027/run_train.py --arm "$arm" --allocator-GiB 32
  # A failed arm does not block another independent parent-initialized arm.
done
