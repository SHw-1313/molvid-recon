#!/usr/bin/env bash
# Cached forecast geometry only; no model execution or modification of outputs.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
data=/workspace/molvid-complex-validation64-20261007/runs/complex_validation64_20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
declare -A sources=(
 [pvb]="$data/pvb"
 [hv]="$root/hv_a1_complex64"
 [flow71]="$root/seed71/flow/complex_validation64"
 [flow72]="$root/seed72/flow/complex_validation64"
 [edm71]="$root/seed71/edm/complex_validation64"
 [edm72]="$root/seed72/edm/complex_validation64")
mkdir -p "$root/complex_geometry"
while true; do
 completed=0
 for name in pvb edm71 flow71 edm72 flow72 hv; do
  output=$root/complex_geometry/$name
  if test -f "$output/summary.json"; then completed=$((completed+1)); continue; fi
  if test -f "${sources[$name]}/summary.json"; then
   python -u tasks/coordinate-state-denoising-v1/score_complex_geometry.py \
    --generated "${sources[$name]}" --data "$data/data" --output "$output" \
    > "$root/complex_geometry/$name.log" 2>&1
   completed=$((completed+1))
  fi
 done
 if test "$completed" -eq 6; then break; fi
 sleep 30
done
for seed in 71 72; do
 for pair in "edm$seed:flow$seed" "edm$seed:pvb" "flow$seed:pvb" "edm$seed:hv" "flow$seed:hv"; do
  for stratum in all old16 new48; do
   python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
    --candidate "$root/complex_geometry/${pair%:*}/summary.json" \
    --reference "$root/complex_geometry/${pair#*:}/summary.json" --stratum "$stratum" \
    --output "$root/complex_geometry/${pair/:/_vs_}_$stratum.json"
  done
 done
done
