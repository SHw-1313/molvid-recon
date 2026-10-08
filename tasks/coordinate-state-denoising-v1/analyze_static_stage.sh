#!/usr/bin/env bash
# Reuse fixed full panels; report added static exposure, not a pure state ablation.
set -euo pipefail
repo=/workspace/molvid-coordinate-state-denoising-v1
root=$repo/runs/coordinate_static_pretrain_v1/seed71
md=$repo/runs/coordinate_state_denoising_v1/seed71
baseline=/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005
study=/workspace/molvid-complex-validation64-20261007
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$repo"
cd "$repo"
for scheme in flow edm; do
 while ! test -f "$root/$scheme/expanded48/summary.json"; do sleep 30; done
 for ref in md hv pvb; do
  case "$ref" in
   md) reference=$md/$scheme/expanded48;;
   hv) reference=$repo/runs/coordinate_state_denoising_v1/hv_a1_raw100;;
   pvb) reference=$baseline/expanded_pvb;;
  esac
  python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/$scheme/expanded48" \
   --reference "$reference" --output "$root/$scheme/protein100_vs_$ref.json"
 done
done
python tasks/pretrained-geometry-bridge-20261007/compare_proteins.py --candidate "$root/edm/expanded48" \
 --reference "$root/flow/expanded48" --output "$root/edm/protein100_vs_flow.json"
sources=("$root/flow/expanded48" "$root/edm/expanded48" "$md/flow/expanded48" "$md/edm/expanded48" "$baseline/expanded_pvb")
python "$study/tasks/pretrained-geometry-bridge-20261007/score_protein_paths.py" --generated "${sources[@]}" \
 --cache "$baseline/development48_v2" --pair coordinate_static_edm_71:coordinate_static_flow_71 \
 --pair coordinate_static_flow_71:coordinate_flow_71 --pair coordinate_static_edm_71:coordinate_edm_71 \
 --pair coordinate_static_flow_71:pvb --pair coordinate_static_edm_71:pvb --output "$root/edm/protein_paths100"
python "$study/tasks/bridge-learning-curve-20261007/score_temporal_order.py" --generated "${sources[@]}" \
 --cache "$baseline/development48_v2" --output "$root/edm/temporal_order100"
for scheme in flow edm; do
 while ! test -f "$root/$scheme/native80/summary.json"; do sleep 30; done
 PYTHONPATH=/workspace/molvid-native-clock-training-20261005 python tasks/pretrained-geometry-bridge-20261007/score_clock.py --root "$root/$scheme" \
  --cache80 /workspace/molvid-native-time-response-20261005/runs/native_time_response_20261005/development48_native80 \
  --cache100 "$baseline/development48_v2" --variant "coordinate_static_${scheme}_71" --output "$root/$scheme/clock80_100"
 python tasks/coordinate-state-denoising-v1/compare_summary_pairs.py --candidate "$root/$scheme/clock80_100/summary.json" \
  --reference "$md/$scheme/clock80_100/summary.json" --output "$root/$scheme/clock_vs_md.json"
done
for scheme in flow edm; do
 while ! test -f "$root/$scheme/complex_validation64/summary.json"; do sleep 30; done
 for ref in md pvb; do
  reference=$md/$scheme/complex_validation64
  if test "$ref" = pvb; then reference=$study/runs/complex_validation64_20261007/pvb; fi
  for stratum in all old16 new48; do
   python "$study/tasks/complex-pretraining-20261007/compare_complex.py" --candidate "$root/$scheme/complex_validation64" \
    --reference "$reference" --systems 64 --inventory "$study/runs/complex_validation64_20261007/data/summary.json" \
    --stratum "$stratum" --output "$root/$scheme/complex_${stratum}_vs_$ref.json"
  done
 done
done
