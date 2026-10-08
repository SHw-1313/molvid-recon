#!/usr/bin/env bash
# Invoke only after the original training lane completed BOTH seeds and its
# not-yet-running-model evaluation waiter was explicitly retired. Existing
# primary/time workers retain seed71 primary panels and both native time grids.
# Same forecasts; this lane advances seed72 protein, then seed71 short rollout.
set -euo pipefail
scheme=$1
gpu=$2
case "$scheme" in flow|edm) ;; *) exit 2;; esac
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
cd "$repo"
for seed in 71 72; do test -f "$root/seed$seed/$scheme/train/coordinate_final.pt"; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
run=$root/seed72/$scheme
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$run/train/coordinate_final.pt" --domain protein \
 --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
 --lag-ps 100 --variant "coordinate_${scheme}_72" --output "$run/expanded48" > "$run/expanded48_advanced.log" 2>&1
for seed in 71 72; do
 run=$root/seed$seed/$scheme
 if test "$seed" = 72; then
  python -u -m tools.evaluate_pretrained_bridge --checkpoint "$run/train/coordinate_final.pt" --domain complex \
   --complex-systems 64 --data /workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data \
   --variant "coordinate_${scheme}_$seed" --output "$run/complex_validation64" > "$run/complex_validation64_advanced.log" 2>&1
 fi
 # No second rollout owner: the original lane must be retired before launch.
 python -u -m tools.evaluate_complex_rollout --checkpoint "$run/train/coordinate_final.pt" \
  --data /workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/development16_rollout96 \
  --variant "coordinate_${scheme}_$seed" --output "$run/rollout" > "$run/rollout_advanced.log" 2>&1
done
