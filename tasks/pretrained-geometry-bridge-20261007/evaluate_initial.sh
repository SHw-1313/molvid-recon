#!/usr/bin/env bash
# Learned bridge versus exactly its inherited, unadapted geometric prior.
set -euo pipefail
predecessor=${1:?existing container-side shared-observation owner PID}
gpu=${2:?GPU UUID}
while kill -0 "$predecessor" 2>/dev/null; do sleep 30; done
while test -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"; do sleep 20; done
cd /workspace/molvid-pretrained-geometry-bridge-20261007
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH=.:/workspace/molvid-geometry-scaling-20261005/.runtime/pvb_numpy126
root=/workspace/molvid-pretrained-geometry-bridge-20261007/runs/pretrained_geometry_bridge_20261007/initial_geometry
mkdir -p "$root"
config=tasks/pretrained-geometry-bridge-20261007/train.json
python -u -m tools.evaluate_pretrained_bridge --initial-config "$config" --domain complex \
 --data /workspace/molvid-complex-mixture-20261007/runs/complex_mixture_20261007/misato128_v2 \
 --variant inherited_geometry_joint --output "$root/complex_native80" > "$root/complex.log" 2>&1
python -u -m tools.evaluate_pretrained_bridge --initial-config "$config" --domain protein \
 --data /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2 \
 --variant inherited_geometry_joint --output "$root/expanded48" > "$root/protein.log" 2>&1
export CUDA_VISIBLE_DEVICES=''
python tasks/complex-pretraining-20261007/compare_complex.py --candidate "$root/complex_native80" \
 --reference /workspace/molvid-complex-scoring-recovery-20261007/runs/complex_scoring_recovery_20261007/pvb \
 --output "$root/complex_vs_released_pvb.json" > "$root/compare.log" 2>&1
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/expanded48" \
 --reference /workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/expanded_pvb \
 --output "$root/protein_vs_released_pvb.json" >> "$root/compare.log" 2>&1
python tasks/complex-pretraining-20261007/score_ligand_geometry.py --generated "$root/complex_native80" \
 --output "$root/ligand_geometry" > "$root/ligand_geometry.log" 2>&1
