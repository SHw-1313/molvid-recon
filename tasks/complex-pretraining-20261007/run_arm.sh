#!/usr/bin/env bash
# Direct real training and generation; no discarded pilot or qualification run.
set -euo pipefail
arm=$1
seed=$2
gpu=$3
case "$arm" in atom|balanced);; *) exit 2;; esac
case "$seed" in 71|72);; *) exit 2;; esac
cd /workspace/molvid-complex-pretraining-20261007
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 15; done
root="/workspace/molvid-complex-pretraining-20261007/runs/complex_pretraining_20261007/${arm}_${seed}"
python -u -m tools.train_coordinate_flow --seed "$seed" --config "$root/recipe.json" --output "$root/train" > "$root/train.log" 2>&1
python -u -m tools.evaluate_complex_native \
 --data /workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2 \
 --checkpoint "$root/train/coordinate_step_2048.pt" --variant "complex_${arm}_${seed}" \
 --output "$root/complex_native80" > "$root/complex_evaluation.log" 2>&1
cache=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
python -u -m tools.evaluate_expanded_native --arm coordinate --variant "complex_${arm}_${seed}" \
 --checkpoint "$root/train/coordinate_step_2048.pt" --cache "$cache" --output "$root/expanded48" > "$root/protein_evaluation.log" 2>&1
CUDA_VISIBLE_DEVICES='' python -u -m tools.audit_cached_stereochemistry --cache "$cache" \
 --generated "$root/expanded48" --output "$root/stereochemistry" > "$root/chemistry.log" 2>&1
