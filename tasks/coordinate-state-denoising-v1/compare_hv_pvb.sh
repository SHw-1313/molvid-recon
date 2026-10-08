#!/usr/bin/env bash
# Direct original-baseline comparison from existing full panels; no model calls.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
study=/workspace/molvid-complex-validation64-20261007
data=$study/runs/complex_validation64_20261007
cd "$repo"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
while ! test -f "$root/hv_a1_complex64/summary.json"; do sleep 30; done
for stratum in all old16 new48; do
 python "$study/tasks/complex-pretraining-20261007/compare_complex.py" \
  --candidate "$root/hv_a1_complex64" --reference "$data/pvb" \
  --systems 64 --inventory "$data/data/summary.json" --stratum "$stratum" \
  --output "$root/hv_a1_complex64/vs_pvb_$stratum.json"
done
while ! test -f "$root/complex_geometry/hv/summary.json"; do sleep 30; done
for stratum in all old16 new48; do
 python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
  --candidate "$root/complex_geometry/hv/summary.json" \
  --reference "$root/complex_geometry/pvb/summary.json" --stratum "$stratum" \
  --output "$root/complex_geometry/hv_vs_pvb_$stratum.json"
done
