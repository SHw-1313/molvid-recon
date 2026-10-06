#!/usr/bin/env bash
# CPU-only consumption of complete fixed-final predictions from live jobs.
set -euo pipefail
cd /workspace/molvid-complex-pretraining-20261007
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
root=/workspace/molvid-complex-scoring-recovery-20261007/runs/complex_scoring_recovery_20261007
result=/workspace/molvid-complex-pretraining-20261007/runs/complex_pretraining_20261007/full_comparisons_v1
mkdir -p "$result"
for arm in atom_71 balanced_71 balanced_72 atom_72; do
 seed=${arm##*_}
 candidate="$root/complex_$arm/complex_native80"
 control="$root/gq2048_$seed"
 while ! test -f "$candidate/summary.json" || ! test -f "$control/summary.json"; do sleep 30; done
 for reference in pvb gq persistence; do
  case "$reference" in
   pvb) reference_path="$root/pvb";;
   gq) reference_path="$control";;
   persistence) reference_path=persistence;;
  esac
  python tasks/complex-pretraining-20261007/compare_complex.py \
   --candidate "$candidate" --reference "$reference_path" --output "$result/${arm}_vs_${reference}.json"
 done
 python tasks/complex-pretraining-20261007/score_ligand_geometry.py \
  --generated "$candidate" --output "$result/${arm}_ligand_geometry"
done
for seed in 71 72; do
 python tasks/complex-pretraining-20261007/compare_complex.py \
  --candidate "$root/complex_balanced_$seed/complex_native80" \
  --reference "$root/complex_atom_$seed/complex_native80" --output "$result/balanced_vs_atom_${seed}.json"
done
