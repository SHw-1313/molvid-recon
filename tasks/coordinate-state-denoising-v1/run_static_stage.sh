#!/usr/bin/env bash
# One explicitly paired followup, not continuation of MD-only checkpoints.
set -euo pipefail
scheme=$1
gpu=$2
predecessor=$3
case "$scheme" in flow|edm) ;; *) exit 2;; esac
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_static_pretrain_v1
run=$root/seed71/$scheme
if test "$predecessor" != none; then
 while ps -p "$predecessor" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do sleep 30; done
fi
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
cd "$repo"
mkdir -p "$run"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
python -u -m tools.train_coordinate_state --config "$root/recipes/${scheme}_static.json" \
 --seed 71 --output "$run/static" > "$run/static.log" 2>&1
python -u -m tools.train_coordinate_state --config "$root/recipes/${scheme}_md.json" \
 --seed 71 --warm-start "$run/static/coordinate_final.pt" --output "$run/md" > "$run/md.log" 2>&1
checkpoint=$run/md/coordinate_final.pt
variant=coordinate_static_${scheme}_71
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
 --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
 --lag-ps 100 --variant "$variant" --output "$run/expanded48" > "$run/expanded48.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
 --data /workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80 \
 --lag-ps 80 --variant "$variant" --output "$run/native80" > "$run/native80.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex --complex-systems 64 \
 --data /workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data \
 --variant "$variant" --output "$run/complex_validation64" > "$run/complex_validation64.log" 2>&1
