#!/usr/bin/env bash
# Use the same authorized flow lane only after its paired work completes.
set -euo pipefail
predecessor=$1
gpu=$2
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
while ps -p "$predecessor" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do sleep 30; done
# Do not displace a pending repair/restart of the high-priority A/B lane.
while ! test -f "$root/seed72/flow/rollout/summary.json"; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
python -u -m tools.evaluate_pretrained_bridge --hv-baseline --domain complex \
 --complex-systems 64 --variant a1 \
 --data /workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data \
 --output "$root/hv_a1_complex64" > "$root/hv_a1_complex64.log" 2>&1
