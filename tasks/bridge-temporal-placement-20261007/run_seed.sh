#!/usr/bin/env bash
set -euo pipefail
seed=$1
gpu=$2
predecessor=$3
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
repo=/workspace/molvid-bridge-temporal-placement-20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$repo:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126"
root="$repo/runs/bridge_temporal_placement_20261007/seed$seed"
mkdir -p "$root"
python -u -m tools.train_pretrained_bridge --config tasks/bridge-temporal-placement-20261007/train.json \
 --seed "$seed" --output "$root/train" > "$root/train.log" 2>&1
checkpoint="$root/train/bridge_final.pt"
variant="bridge_temporal_placement_$seed"
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
control=/workspace/molvid-bridge-interleaved-clocks-20261007/runs/bridge_interleaved_clocks_20261007/seed$seed
python tasks/complex-pretraining-20261007/score_ligand_geometry.py --generated "$root/complex_native80" \
 --data "$complex_data" --output "$root/ligand_geometry" > "$root/chemistry.log" 2>&1
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference "$reference" --output "$root/complex_vs_pvb.json"
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference "$pvb_protein" --output "$root/protein_vs_pvb.json"
while ! test -f "$control/expanded48/summary.json" || ! test -f "$control/complex_native80/summary.json"; do sleep 30; done
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference "$control/complex_native80" --output "$root/complex_vs_matched_placement.json"
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference "$control/expanded48" --output "$root/protein_vs_matched_placement.json"
python tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py \
 --generated "$root/complex_native80" "$control/complex_native80" "$reference" --data "$complex_data" \
 --pair "$variant:bridge_interleaved_$seed" --pair "$variant:pvb_misato_released" --output "$root/internal_paths"
python /workspace/molvid-complex-validation64-20261007/tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py \
 --generated "$root/expanded48" "$control/expanded48" "$pvb_protein" --cache "$protein_data" \
 --pair "$variant:bridge_interleaved_$seed" --pair "$variant:pvb" --output "$root/protein_paths"
cd /workspace/molvid-native-clock-training-20261005
PYTHONPATH=. python "$repo/tasks/pretrained-geometry-bridge-20261007/score_clock.py" \
 --root "$root" --cache80 "$cache80" --cache100 "$protein_data" --variant "$variant" \
 --output "$root/clock_response"
