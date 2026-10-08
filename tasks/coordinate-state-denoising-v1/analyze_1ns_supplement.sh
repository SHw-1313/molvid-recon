#!/usr/bin/env bash
# Complete cached1ns panels only; independent of slow complex generation.
# Main analysis retains full-path scores and external primary comparisons.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
baseline=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005
study=/workspace/molvid-complex-validation64-20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
for seed in 71 72; do
 flow=$root/seed$seed/flow
 edm=$root/seed$seed/edm
 for arm in "$flow" "$edm"; do
  while ! test -f "$arm/native1000/summary.json"; do sleep 30; done
 done
 python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py \
  --candidate "$edm/native1000" --reference "$flow/native1000" --output "$edm/protein1ns_vs_flow.json"
 python "$study/tasks/bridge-learning-curve-20261007/score_temporal_order.py" \
  --generated "$flow/native1000" "$edm/native1000" "$baseline/expanded_a1_1ns_v5" "$baseline/expanded_biokinema_1ns_v5" \
  --cache "$baseline/development48_v2" --output "$edm/temporal_order1ns"
 python tasks/coordinate-state-denoising-v1/score_motion_amplitude.py \
  --candidate "$edm/native1000" --reference "$flow/native1000" --domain protein \
  --data "$baseline/development48_v2" --train-scales "$root/train_scales_v2" --output "$edm/amplitude1ns_vs_flow.json"
done
