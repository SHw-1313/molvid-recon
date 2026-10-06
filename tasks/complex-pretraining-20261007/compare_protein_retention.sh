#!/usr/bin/env bash
# Consume the already-running full protein generations; no model calls.
set -euo pipefail
cd /workspace/molvid-complex-pretraining-20261007
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.
base=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005
mixed=/workspace/molvid-complex-scoring-recovery-20261007/runs/complex_scoring_recovery_20261007
shared=/workspace/molvid-shared-observation-geometry-20261007/runs/shared_observation_geometry_20261007
controls=/workspace/molvid-query-geometry-training-20261006/runs/query_geometry_training_20261006/formal_v2
result=/workspace/molvid-complex-pretraining-20261007/runs/complex_pretraining_20261007/protein_retention
compare_one() {
 local arm=$1 seed=$2 candidate control variant
 if test "$arm" = shared; then
  candidate="$shared/seed$seed/expanded48"
  control="$mixed/complex_balanced_$seed/expanded48"
  variant="shared_observation_$seed:complex_balanced_$seed"
 else
  candidate="$mixed/complex_${arm}_$seed/expanded48"
  control="$controls/gq_$seed/expanded48"
  variant="complex_${arm}_$seed:coordinate_gq_$seed"
 fi
 while ! test -f "$candidate/summary.json" || ! test -f "$control/summary.json"; do sleep 30; done
 python -u -m tools.compare_expanded_native \
  --generated "$candidate" "$control" "$base/expanded_a1" "$base/expanded_pvb" \
  --pair "$variant" --exposure "$base/external_overlap_v3/summary.json" \
  --output "$result/${arm}_$seed"
}
jobs=()
for arm in atom balanced shared; do
 for seed in 71 72; do
  compare_one "$arm" "$seed" &
  jobs+=("$!")
 done
done
failed=0
for job in "${jobs[@]}"; do wait "$job" || failed=1; done
exit "$failed"
