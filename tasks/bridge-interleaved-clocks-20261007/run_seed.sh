#!/usr/bin/env bash
# Real equal-budget training; wait for the existing GPU owner, not a test gate.
set -euo pipefail
seed=$1
gpu=$2
predecessor=$3
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
repo=/workspace/molvid-bridge-interleaved-clocks-20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$repo:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126"
root="$repo/runs/bridge_interleaved_clocks_20261007/seed$seed"
mkdir -p "$root"
python -u -m tools.train_pretrained_bridge --config tasks/bridge-interleaved-clocks-20261007/train.json \
 --seed "$seed" --output "$root/train" > "$root/train.log" 2>&1
checkpoint="$root/train/bridge_final.pt"
variant="bridge_interleaved_$seed"
complex_data=/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2
protein_data=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
cache80=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex \
 --data "$complex_data" --variant "$variant" --output "$root/complex_native80" > "$root/complex.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
 --data "$protein_data" --variant "$variant" --output "$root/expanded48" > "$root/protein.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
 --data "$cache80" --lag-ps 80 --variant "$variant" --output "$root/native80" > "$root/native80.log" 2>&1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
reference=/workspace/molvid-complex-scoring-recovery-20261007/runs/complex_scoring_recovery_20261007/pvb
pvb_protein=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/expanded_pvb
baseline=/workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/evaluation_seed$seed
python tasks/complex-pretraining-20261007/score_ligand_geometry.py --generated "$root/complex_native80" \
 --data "$complex_data" --output "$root/ligand_geometry" > "$root/chemistry.log" 2>&1
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference "$reference" --output "$root/complex_vs_pvb.json"
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference "$pvb_protein" --output "$root/protein_vs_pvb.json"
while ! test -f "$baseline/expanded48/summary.json" || ! test -f "$baseline/complex_native80/summary.json"; do sleep 30; done
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference "$baseline/complex_native80" --output "$root/complex_vs_synchronized.json"
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference "$baseline/expanded48" --output "$root/protein_vs_synchronized.json"
python tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py \
 --generated "$root/complex_native80" "$baseline/complex_native80" "$reference" --data "$complex_data" \
 --pair "$variant:pretrained_bridge_$seed" --pair "$variant:pvb_misato_released" --output "$root/internal_paths"
python /workspace/molvid-complex-validation64-20261007/tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py \
 --generated "$root/expanded48" "$baseline/expanded48" "$pvb_protein" --cache "$protein_data" \
 --pair "$variant:pretrained_bridge_$seed" --pair "$variant:pvb" --output "$root/protein_paths"
cd /workspace/molvid-native-clock-training-20261005
PYTHONPATH=. python "$repo/tasks/pretrained-geometry-bridge-20261007/score_clock.py" \
 --root "$root" --cache80 "$cache80" --cache100 "$protein_data" --variant "$variant" \
 --output "$root/clock_response"
