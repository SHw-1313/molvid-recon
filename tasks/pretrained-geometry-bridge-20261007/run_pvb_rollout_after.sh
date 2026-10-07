#!/usr/bin/env bash
set -euo pipefail
predecessor=$1
gpu=$2
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
cd /workspace/molvid-pretrained-geometry-bridge-20261007
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=.:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126
base=/workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007
python -u -m tools.evaluate_complex_rollout --data "$base/development16_rollout96" \
 --pvb-checkpoint /data1/repo/PVB/model_ckpt/ckpt/pvb_misato.ckpt --variant released_pvb_rollout92 \
 --output "$base/released_pvb_rollout92" > "$base/released_pvb_rollout92.log" 2>&1
