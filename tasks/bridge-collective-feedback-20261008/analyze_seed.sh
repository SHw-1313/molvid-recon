#!/usr/bin/env bash
# Score completed full panels only. No additional model calls or selection.
set -euo pipefail
seed=$1
study=/workspace/molvid-complex-validation64-20261007
repo=/workspace/molvid-bridge-collective-feedback-20261008
root="$repo/runs/bridge_collective_feedback_20261008/seed$seed"
candidate="$root/collective"
control="$root/control"
parent="/workspace/molvid-bridge-temporal-placement-20261007/runs/bridge_temporal_placement_20261007/seed$seed"
cache=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
cache80=/workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80
complex_data="$study/runs/complex_validation64_20261007/data"
protein_pvb=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/expanded_pvb
complex_pvb="$study/runs/complex_validation64_20261007/pvb"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$study"
cd "$study"
while ! test -f "$candidate/expanded48/summary.json" || ! test -f "$control/expanded48/summary.json"; do sleep 30; done
for reference in control parent pvb; do
 case "$reference" in control) ref="$control/expanded48";; parent) ref="$parent/expanded48";; pvb) ref="$protein_pvb";; esac
 python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py \
  --candidate "$candidate/expanded48" --reference "$ref" --output "$candidate/protein_vs_$reference.json"
done
python tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py \
 --generated "$candidate/expanded48" "$control/expanded48" "$protein_pvb" --cache "$cache" \
 --pair "bridge_cf1_collective_$seed:bridge_cf1_control_$seed" --pair "bridge_cf1_collective_$seed:pvb" \
 --output "$candidate/protein_paths"
python tasks/bridge-learning-curve-20261007/score_temporal_order.py \
 --generated "$candidate/expanded48" "$control/expanded48" "$protein_pvb" --cache "$cache" \
 --output "$candidate/temporal_order100"
python "$repo/tasks/bridge-collective-feedback-20261008/score_spatial_cooperation.py" \
 --generated "$candidate/expanded48" "$control/expanded48" "$protein_pvb" --cache "$cache" \
 --pair "bridge_cf1_collective_$seed:bridge_cf1_control_$seed" --pair "bridge_cf1_collective_$seed:pvb" \
 --output "$candidate/spatial_cooperation100"
while ! test -f "$candidate/complex_validation64/summary.json" || ! test -f "$control/complex_validation64/summary.json"; do sleep 30; done
for reference in control parent pvb; do
 case "$reference" in control) ref="$control/complex_validation64";; parent) ref="$parent/complex_validation64";; pvb) ref="$complex_pvb";; esac
 for stratum in all old16 new48; do
  python tasks/complex-pretraining-20261007/compare_complex.py \
   --candidate "$candidate/complex_validation64" --reference "$ref" --systems 64 \
   --inventory "$complex_data/summary.json" --stratum "$stratum" --output "$candidate/complex_${stratum}_vs_$reference.json"
 done
done
python tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py \
 --generated "$candidate/complex_validation64" "$control/complex_validation64" "$complex_pvb" \
 --data "$complex_data" --systems 64 --pair "bridge_cf1_collective_$seed:bridge_cf1_control_$seed" \
 --pair "bridge_cf1_collective_$seed:pvb_misato_released" --output "$candidate/internal_paths64"
for arm in collective control; do
 python tasks/complex-pretraining-20261007/score_ligand_geometry.py \
  --generated "$root/$arm/complex_validation64" --data "$complex_data" --systems 64 --output "$root/$arm/ligand_geometry64"
 while ! test -f "$root/$arm/native80/summary.json"; do sleep 30; done
 (
  cd /workspace/molvid-native-clock-training-20261005
  PYTHONPATH=. python "$repo/tasks/pretrained-geometry-bridge-20261007/score_clock.py" \
   --root "$root/$arm" --cache80 "$cache80" --cache100 "$cache" \
   --variant "bridge_cf1_${arm}_$seed" --output "$root/$arm/clock_response"
 )
done
