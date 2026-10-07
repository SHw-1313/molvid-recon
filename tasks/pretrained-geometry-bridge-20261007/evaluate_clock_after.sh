#!/usr/bin/env bash
set -euo pipefail
seed=$1
gpu=$2
predecessor=$3
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
cd /workspace/molvid-pretrained-geometry-bridge-20261007
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=.:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126
base=/workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007
root="$base/evaluation_seed$seed"
test -f "$root/expanded48/summary.json"
cache80=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
cache100=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$base/seed$seed/bridge_final.pt" --domain protein \
 --data "$cache80" --lag-ps 80 --variant "pretrained_bridge_$seed" --output "$root/native80" > "$root/native80.log" 2>&1
export CUDA_VISIBLE_DEVICES=''
script=/workspace/molvid-pretrained-geometry-bridge-20261007/tasks/pretrained-geometry-bridge-20261007/score_clock.py
cd /workspace/molvid-native-clock-training-20261005
PYTHONPATH=. python "$script" --root "$root" --cache80 "$cache80" --cache100 "$cache100" \
 --variant "pretrained_bridge_$seed" --output "$root/clock_response" > "$root/clock_response.log" 2>&1
cd /workspace/molvid-pretrained-geometry-bridge-20261007
initial="$base/initial_geometry"
while ! test -f "$initial/expanded48/summary.json" || ! test -f "$initial/complex_native80/summary.json"; do sleep 30; done
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference "$initial/complex_native80" --output "$root/complex_vs_initial_geometry.json" > "$root/compare_initial.log" 2>&1
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference "$initial/expanded48" --output "$root/protein_vs_initial_geometry.json" >> "$root/compare_initial.log" 2>&1
