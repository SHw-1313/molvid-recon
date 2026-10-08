#!/usr/bin/env bash
# Own one explicit arm; preserve every existing evaluation and external task.
set -euo pipefail
scheme=$1
gpu=$2
predecessor=$3
case "$scheme" in flow|edm) ;; *) exit 2 ;; esac
repo=/workspace/molvid-coordinate-temporal-order-v1
cd "$repo"
root=runs/coordinate_temporal_order_v1
run=$root/seed71/$scheme
test ! -e "$run"
mkdir -p "$run"
while ps -p "$predecessor" -o stat= | grep -q '^[[:space:]]*[^Z[:space:]]'; do sleep 20; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=.
python -u tasks/coordinate-temporal-order-v1/check_order_cuda.py --scheme "$scheme" \
 --protocol "$root/recipes/protocol.json" --output "$run/CUDA_order_result.json" > "$run/CUDA_order.log" 2>&1
python -u -m tools.train_coordinate_state --config "$root/recipes/${scheme}_md.json" \
 --warm-start "/workspace/molvid-coordinate-state-denoising-v1/runs/coordinate_static_pretrain_v1/seed71/$scheme/static/coordinate_final.pt" \
 --seed 71 --output "$run/md" > "$run/md.log" 2>&1
variant=coordinate_order_${scheme}_71
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$run/md/coordinate_final.pt" \
 --domain protein --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
 --lag-ps 100 --variant "$variant" --output "$run/expanded48" > "$run/expanded48.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$run/md/coordinate_final.pt" \
 --domain protein --data /workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80 \
 --lag-ps 80 --variant "$variant" --output "$run/native80" > "$run/native80.log" 2>&1
# Further complex/rollout evaluation is decided from complete paired results;
# do not silently create another full matrix or select partial checkpoints.
