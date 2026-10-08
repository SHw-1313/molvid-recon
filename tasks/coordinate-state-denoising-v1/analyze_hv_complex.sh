#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
study=/workspace/molvid-complex-validation64-20261007
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
cd "$repo"
while ! test -f "$root/hv_a1_complex64/summary.json"; do sleep 30; done
sources=("$root/hv_a1_complex64")
pairs=()
for seed in 71 72; do
 for scheme in flow edm; do
  arm=$root/seed$seed/$scheme
  while ! test -f "$arm/complex_validation64/summary.json"; do sleep 30; done
  for stratum in all old16 new48; do
   python "$study/tasks/complex-pretraining-20261007/compare_complex.py" \
    --candidate "$arm/complex_validation64" --reference "$root/hv_a1_complex64" \
    --systems 64 --inventory "$study/runs/complex_validation64_20261007/data/summary.json" \
    --stratum "$stratum" --output "$arm/complex_${stratum}_vs_hv.json"
  done
  sources+=("$arm/complex_validation64")
  pairs+=(--pair "coordinate_${scheme}_$seed:a1")
 done
done
python "$study/tasks/pretrained-geometry-bridge-20261007/score_internal_paths.py" \
 --generated "${sources[@]}" --data "$study/runs/complex_validation64_20261007/data" \
 --systems 64 "${pairs[@]}" --output "$root/hv_comparison_internal_paths64"
python /workspace/molvid-bridge-dynamic-prior-20261007/tasks/bridge-dynamic-prior-20261007/score_motion_distributions.py \
 --generated "${sources[@]}" --data "$study/runs/complex_validation64_20261007/data" \
 "${pairs[@]}" --output "$root/hv_comparison_motion_distributions64"
