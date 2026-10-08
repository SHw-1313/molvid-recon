#!/usr/bin/env bash
# Advance only the existing native80/native1000 panels on released goal GPUs.
# Identical CLI requests and the evaluator lock/reuse prevent duplicate draws.
set -euo pipefail
scheme=$1
gpu=$2
case "$scheme" in flow|edm) ;; *) exit 2;; esac
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
for seed in 71 72; do
 run="$root/seed$seed/$scheme"
 checkpoint="$run/train/coordinate_final.pt"
 while ! test -f "$checkpoint"; do sleep 30; done
 variant="coordinate_${scheme}_$seed"
 for lag in 80 1000; do
  cache=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
  if test "$lag" = 1000; then cache=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2; fi
  while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
  python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
   --data "$cache" --lag-ps "$lag" --variant "$variant" --output "$run/native$lag" > "$run/native${lag}_parallel.log" 2>&1
 done
done
