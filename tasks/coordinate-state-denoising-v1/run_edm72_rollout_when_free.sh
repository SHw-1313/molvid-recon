#!/usr/bin/env bash
# Sole owner after retiring advanced controller1144425 and its lock waiter.
# Existing seed72 complex evaluation on GPU5 must remain untouched.
set -euo pipefail
cd /workspace/molvid-coordinate-state-denoising-v1
run=runs/coordinate_state_denoising_v1/seed72/edm
test -f "$run/train/coordinate_final.pt"
test ! -e "$run/rollout"
while test -n "$(nvidia-smi -i 4 --query-compute-apps=pid --format=csv,noheader,nounits)"; do
 sleep 20
done
test ! -e "$run/rollout"
export CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
exec python -u -m tools.evaluate_complex_rollout \
 --checkpoint "$run/train/coordinate_final.pt" \
 --data /workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/development16_rollout96 \
 --variant coordinate_edm_72 --output "$run/rollout"
