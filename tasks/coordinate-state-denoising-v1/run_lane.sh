#!/usr/bin/env bash
# Run only on an idle GPU, after the explicitly named existing controller exits.
set -euo pipefail
scheme=$1
gpu=$2
predecessor=$3
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
while ps -p "$predecessor" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
if test "$scheme" = edm; then
 python -u tasks/coordinate-state-denoising-v1/check_cuda_dataflow.py \
  --recipes "$root/train_scales_v2" --output "$root/cuda_dataflow.json" > "$root/cuda_dataflow.log" 2>&1
else
 while ! test -f "$root/cuda_dataflow.json"; do sleep 30; done
fi
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
# Both variants receive both seeds and identical fixed MD exposure. Complete
# training before expensive panels; never select favorable intermediate weights.
for seed in 71 72; do
 run="$root/seed$seed/$scheme"
 mkdir -p "$run"
 initialization=()
 if test -f "$run/train/checkpoint_latest.pt"; then initialization=(--resume "$run/train/checkpoint_latest.pt"); fi
 if ! test -f "$run/train/coordinate_final.pt"; then
  python -u -m tools.train_coordinate_state --config "$root/train_scales_v2/$scheme.json" \
   --seed "$seed" --output "$run/train" "${initialization[@]}" > "$run/train.log" 2>&1
 fi
done
for seed in 71 72; do
 run="$root/seed$seed/$scheme"
 checkpoint="$run/train/coordinate_final.pt"
 variant="coordinate_${scheme}_$seed"
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
  --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
  --lag-ps 100 --variant "$variant" --output "$run/expanded48" > "$run/expanded48.log" 2>&1
 python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex \
  --complex-systems 64 --data /workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007/data \
  --variant "$variant" --output "$run/complex_validation64" > "$run/complex_validation64.log" 2>&1
 for lag in 80 1000; do
  cache=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
  if test "$lag" = 1000; then cache=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2; fi
  python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
   --data "$cache" --lag-ps "$lag" --variant "$variant" --output "$run/native$lag" > "$run/native$lag.log" 2>&1
 done
 python -u -m tools.evaluate_complex_rollout --checkpoint "$checkpoint" \
  --data /workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/development16_rollout96 \
  --variant "$variant" --output "$run/rollout" > "$run/rollout.log" 2>&1
done
