#!/usr/bin/env bash
# CPU-only analysis of complete cached forecasts, independent of GPU controllers.
set -euo pipefail
seed=$1
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1/seed$seed
clockrepo=/workspace/molvid-native-clock-training-20261005
motionrepo=/workspace/molvid-bridge-dynamic-prior-20261007
complex=/workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for scheme in flow edm; do
 for panel in expanded48 native80; do
  while ! test -f "$root/$scheme/$panel/summary.json"; do sleep 30; done
 done
 cd "$clockrepo"
 PYTHONPATH="$clockrepo" python "$repo/tasks/pretrained-geometry-bridge-20261007/score_clock.py" \
  --root "$root/$scheme" --variant "coordinate_${scheme}_$seed" \
  --cache80 /workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80 \
  --cache100 /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
  --output "$root/$scheme/clock80_100"
done
cd "$repo"
export PYTHONPATH="$repo"
python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
 --candidate "$root/edm/clock80_100/summary.json" --reference "$root/flow/clock80_100/summary.json" \
 --output "$root/edm/clock80_100_vs_flow.json"
for scheme in flow edm; do
 while ! test -f "$root/$scheme/complex_validation64/summary.json"; do sleep 30; done
done
python "$motionrepo/tasks/bridge-dynamic-prior-20261007/score_motion_distributions.py" \
 --generated "$root/flow/complex_validation64" "$root/edm/complex_validation64" "$complex/pvb" \
 --pair "coordinate_edm_$seed:coordinate_flow_$seed" \
 --pair "coordinate_flow_$seed:pvb_misato_released" --pair "coordinate_edm_$seed:pvb_misato_released" \
 --data "$complex/data" --output "$root/edm/motion_distributions64"
for scheme in flow edm; do
 while ! test -f "$root/$scheme/rollout/summary.json"; do sleep 30; done
 python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
  --candidate "$root/$scheme/rollout/summary.json" \
  --reference /workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/released_pvb_rollout92/summary.json \
  --output "$root/$scheme/rollout_vs_pvb.json"
done
python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
 --candidate "$root/edm/rollout/summary.json" --reference "$root/flow/rollout/summary.json" \
 --output "$root/edm/rollout_vs_flow.json"
