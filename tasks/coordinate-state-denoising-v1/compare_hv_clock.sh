#!/usr/bin/env bash
# Reuse complete source-sampled caches; no additional neural calls.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_state_denoising_v1
cd "$repo"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
for seed in 71 72; do
 for scheme in flow edm; do
  source="$root/seed$seed/$scheme/clock80_100/summary.json"
  while ! test -f "$source"; do sleep 30; done
  python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py \
   --candidate "$source" --reference "$root/hv_clock80_100/summary.json" \
   --output "$root/seed$seed/$scheme/clock80_100_vs_hv.json"
 done
done
