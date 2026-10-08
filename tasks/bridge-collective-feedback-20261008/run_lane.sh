#!/usr/bin/env bash
# Two matched seeds across two lanes; never interrupt an existing GPU job.
set -euo pipefail
gpu=$1
predecessor=$2
shift 2
while ps -p "$predecessor" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
repo=/workspace/molvid-bridge-collective-feedback-20261008
study=/workspace/molvid-complex-validation64-20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$repo:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126"
root="$repo/runs/bridge_collective_feedback_20261008"
protein_data=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
cache80=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
complex_data="$study/runs/complex_validation64_20261007/data"
# Finish both assigned trainings before the longer trajectory-generation phase.
for assignment in "$@"; do
 seed=${assignment%%:*}; arm=${assignment#*:}
 run="$root/seed$seed/$arm"
 mkdir -p "$run"
 parent="/workspace/molvid-bridge-temporal-placement-20261007/runs/bridge_temporal_placement_20261007/seed$seed/train/bridge_final.pt"
 initialization=(--warmstart "$parent")
 if test -f "$run/train/checkpoint_latest.pt"; then initialization=(--resume "$run/train/checkpoint_latest.pt"); fi
 if ! test -f "$run/train/bridge_final.pt"; then
  python -u -m tools.train_pretrained_bridge --config "tasks/bridge-collective-feedback-20261008/$arm.json" \
   "${initialization[@]}" --seed "$seed" --output "$run/train" >> "$run/train.log" 2>&1
 fi
done
for assignment in "$@"; do
 seed=${assignment%%:*}; arm=${assignment#*:}
 run="$root/seed$seed/$arm"
 checkpoint="$run/train/bridge_final.pt"
 variant="bridge_cf1_${arm}_$seed"
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
  --data "$protein_data" --lag-ps 100 --variant "$variant" --output "$run/expanded48" > "$run/expanded48.log" 2>&1
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex \
  --complex-systems 64 --data "$complex_data" --variant "$variant" --output "$run/complex_validation64" \
  > "$run/complex_validation64.log" 2>&1
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
  --data "$cache80" --lag-ps 80 --variant "$variant" --output "$run/native80" > "$run/native80.log" 2>&1
done
