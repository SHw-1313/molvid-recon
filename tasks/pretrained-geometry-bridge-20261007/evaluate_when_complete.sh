#!/usr/bin/env bash
# Automatic real forecasts after the actual training owner releases its GPU.
set -euo pipefail
seed=$1
gpu=$2
cd /workspace/molvid-pretrained-geometry-bridge-20261007
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=.:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126
root=/workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007
checkpoint="$root/seed$seed/bridge_final.pt"
while ! test -f "$checkpoint"; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
result="$root/evaluation_seed$seed"
mkdir -p "$result"
complex_data=/workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2
protein_data=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
reference=/workspace/molvid-complex-scoring-recovery-20261007/runs/complex_scoring_recovery_20261007
variant="pretrained_bridge_$seed"
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain complex \
 --data "$complex_data" --variant "$variant" --output "$result/complex_native80" > "$result/complex.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --checkpoint "$checkpoint" --domain protein \
 --data "$protein_data" --variant "$variant" --output "$result/expanded48" > "$result/protein.log" 2>&1
export CUDA_VISIBLE_DEVICES=''
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$result/complex_native80" \
 --reference "$reference/pvb" --output "$result/complex_vs_released_pvb.json" > "$result/complex_compare.log" 2>&1
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$result/complex_native80" \
 --reference "$reference/complex_balanced_$seed/complex_native80" --output "$result/complex_vs_balanced.json" >> "$result/complex_compare.log" 2>&1
python tasks/complex-pretraining-20261007/score_ligand_geometry.py --generated "$result/complex_native80" \
 --data "$complex_data" --output "$result/ligand_geometry" > "$result/ligand_geometry.log" 2>&1
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$result/expanded48" \
 --reference /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/expanded_pvb \
 --output "$result/protein_vs_released_pvb.json" > "$result/protein_compare.log" 2>&1
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$result/expanded48" \
 --reference "$reference/complex_balanced_$seed/expanded48" \
 --output "$result/protein_vs_balanced.json" >> "$result/protein_compare.log" 2>&1
