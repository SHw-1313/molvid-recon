#!/usr/bin/env bash
# Full cached panels only: no extra forecasts and no coordinate modification.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_static_pretrain_v1/seed71
mdroot=$repo/runs/coordinate_state_denoising_v1
study=/workspace/molvid-complex-validation64-20261007
data=$study/runs/complex_validation64_20261007/data
pvb=$study/runs/complex_validation64_20261007/pvb
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
cd "$repo"
for scheme in flow edm; do
 for panel in expanded48 complex_validation64; do
  for base in "$root" "$mdroot/seed71"; do
   while ! test -f "$base/$scheme/$panel/summary.json"; do sleep 30; done
  done
 done
done
for stratum in all old16 new48; do
 python "$study/tasks/complex-pretraining-20261007/compare_complex.py" \
  --candidate "$root/edm/complex_validation64" --reference "$root/flow/complex_validation64" \
  --systems 64 --inventory "$data/summary.json" --stratum "$stratum" --output "$root/edm/complex_${stratum}_vs_flow.json"
done
sources=("$root/flow/complex_validation64" "$root/edm/complex_validation64" \
 "$mdroot/seed71/flow/complex_validation64" "$mdroot/seed71/edm/complex_validation64" "$pvb")
pairs=(--pair coordinate_static_edm_71:coordinate_static_flow_71 \
 --pair coordinate_static_flow_71:coordinate_flow_71 --pair coordinate_static_edm_71:coordinate_edm_71 \
 --pair coordinate_static_flow_71:pvb_misato_released --pair coordinate_static_edm_71:pvb_misato_released)
python "$study/tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py" --generated "${sources[@]}" \
 --data "$data" --systems 64 "${pairs[@]}" --output "$root/edm/internal_paths64"
python /workspace/molvid-bridge-dynamic-prior-20261007/tasks/bridge-dynamic-prior-20261007/score_motion_distributions.py \
 --generated "${sources[@]}" --data "$data" "${pairs[@]}" --output "$root/edm/motion_distributions64"
for domain in protein complex; do
 cache=$data
 panel=complex_validation64
 if test "$domain" = protein; then
  cache=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005/development48_v2
  panel=expanded48
 fi
 for comparison in edm_vs_flow flow_vs_md edm_vs_md; do
  scheme=${comparison%%_vs_*}
  reference=$mdroot/seed71/$scheme/$panel
  if test "$comparison" = edm_vs_flow; then reference=$root/flow/$panel; fi
  python tasks/coordinate-state-denoising-v1/score_motion_amplitude.py --candidate "$root/$scheme/$panel" \
   --reference "$reference" --domain "$domain" --data "$cache" --train-scales "$mdroot/train_scales_v2" \
   --output "$root/$scheme/amplitude_${domain}_${comparison}.json"
 done
done
for scheme in flow edm; do
 python -u tasks/coordinate-state-denoising-v1/score_complex_geometry.py \
  --generated "$root/$scheme/complex_validation64" --data "$data" --output "$root/$scheme/complex_geometry"
 for reference in "${scheme}71" pvb hv; do
  while ! test -f "$mdroot/complex_geometry/$reference/summary.json"; do sleep 30; done
  for stratum in all old16 new48; do
   python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py --candidate "$root/$scheme/complex_geometry/summary.json" \
    --reference "$mdroot/complex_geometry/$reference/summary.json" --stratum "$stratum" \
    --output "$root/$scheme/geometry_${stratum}_vs_$reference.json"
  done
 done
done
for stratum in all old16 new48; do
 python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py --candidate "$root/edm/complex_geometry/summary.json" \
  --reference "$root/flow/complex_geometry/summary.json" --stratum "$stratum" --output "$root/edm/geometry_${stratum}_vs_flow.json"
done
