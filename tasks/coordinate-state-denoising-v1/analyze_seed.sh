#!/usr/bin/env bash
set -euo pipefail
seed=$1
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
study=/workspace/molvid-complex-validation64-20261007
baseline=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005
flow=$root/seed$seed/flow
edm=$root/seed$seed/edm
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
cd "$repo"
for arm in "$flow" "$edm"; do
 while ! test -f "$arm/expanded48/summary.json"; do sleep 30; done
done
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$edm/expanded48" --reference "$flow/expanded48" --output "$edm/protein100_vs_flow.json"
for scheme in flow edm; do
 arm=$root/seed$seed/$scheme
 for reference in hv pvb; do
  ref=$root/hv_a1_raw100
  if test "$reference" = pvb; then ref=$baseline/expanded_pvb; fi
  python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$arm/expanded48" --reference "$ref" --output "$arm/protein100_vs_$reference.json"
 done
done
sources=("$flow/expanded48" "$edm/expanded48" "$root/hv_a1_raw100" "$baseline/expanded_pvb")
pairs=(--pair "coordinate_edm_$seed:coordinate_flow_$seed" --pair "coordinate_flow_$seed:a1" --pair "coordinate_edm_$seed:a1" --pair "coordinate_flow_$seed:pvb" --pair "coordinate_edm_$seed:pvb")
python "$study/tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py" --generated "${sources[@]}" --cache "$baseline/development48_v2" "${pairs[@]}" --output "$edm/protein_paths100"
python "$study/tasks/bridge-learning-curve-20261007/score_temporal_order.py" --generated "${sources[@]}" --cache "$baseline/development48_v2" --output "$edm/temporal_order100"
python tasks/coordinate-state-denoising-v1/score_motion_amplitude.py --candidate "$edm/expanded48" --reference "$flow/expanded48" --domain protein --data "$baseline/development48_v2" --train-scales "$root/train_scales_v2" --output "$edm/amplitude100_vs_flow.json"
for arm in "$flow" "$edm"; do
 while ! test -f "$arm/complex_validation64/summary.json"; do sleep 30; done
done
for scheme in flow edm; do
 arm=$root/seed$seed/$scheme
 for stratum in all old16 new48; do
  python "$study/tasks/complex-pretraining-20261007/compare_complex.py" --candidate "$arm/complex_validation64" --reference "$study/runs/complex_validation64_20261007/pvb" --systems 64 --inventory "$study/runs/complex_validation64_20261007/data/summary.json" --stratum "$stratum" --output "$arm/complex_${stratum}_vs_pvb.json"
 done
done
for stratum in all old16 new48; do
 python "$study/tasks/complex-pretraining-20261007/compare_complex.py" --candidate "$edm/complex_validation64" --reference "$flow/complex_validation64" --systems 64 --inventory "$study/runs/complex_validation64_20261007/data/summary.json" --stratum "$stratum" --output "$edm/complex_${stratum}_vs_flow.json"
done
python "$study/tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py" --generated "$edm/complex_validation64" "$flow/complex_validation64" "$study/runs/complex_validation64_20261007/pvb" --data "$study/runs/complex_validation64_20261007/data" --systems 64 --pair "coordinate_edm_$seed:coordinate_flow_$seed" --pair "coordinate_edm_$seed:pvb_misato_released" --pair "coordinate_flow_$seed:pvb_misato_released" --output "$edm/internal_paths64"
python tasks/coordinate-state-denoising-v1/score_motion_amplitude.py --candidate "$edm/complex_validation64" --reference "$flow/complex_validation64" --domain complex --data "$study/runs/complex_validation64_20261007/data" --train-scales "$root/train_scales_v2" --output "$edm/amplitude_complex_vs_flow.json"
for arm in "$flow" "$edm"; do
 while ! test -f "$arm/native1000/summary.json"; do sleep 30; done
done
for scheme in flow edm; do
 arm=$root/seed$seed/$scheme
 for ref in expanded_a1_1ns_v5 expanded_biokinema_1ns_v5; do
  python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$arm/native1000" --reference "$baseline/$ref" --output "$arm/protein1ns_vs_$ref.json"
 done
done
python "$study/tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py" --generated "$flow/native1000" "$edm/native1000" "$baseline/expanded_a1_1ns_v5" "$baseline/expanded_biokinema_1ns_v5" --cache "$baseline/development48_v2" --pair "coordinate_edm_$seed:coordinate_flow_$seed" --pair "coordinate_edm_$seed:a1" --pair "coordinate_flow_$seed:a1" --pair "coordinate_edm_$seed:biokinema" --pair "coordinate_flow_$seed:biokinema" --output "$edm/protein_paths1ns"
