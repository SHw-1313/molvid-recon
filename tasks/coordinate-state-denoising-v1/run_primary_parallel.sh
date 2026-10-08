#!/usr/bin/env bash
# Advance the existing primary panels on released goal GPUs; no new samples.
# The evaluator's per-output lock/reuse prevents the original queue repeating
# these panels. Only complete fixed-final weights are eligible.
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
 while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
  --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
  --lag-ps 100 --variant "$variant" --output "$run/expanded48" > "$run/expanded48_parallel.log" 2>&1
 while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex \
  --complex-systems 64 --data /workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data \
  --variant "$variant" --output "$run/complex_validation64" > "$run/complex_validation64_parallel.log" 2>&1
done
