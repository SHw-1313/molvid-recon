#!/usr/bin/env bash
set -euo pipefail
# Run inside torch-ito, from this worktree. Cached-coordinate statistics only.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
task=tasks/coordinate-state-denoising-v1
md=runs/coordinate_state_denoising_v1
static=runs/coordinate_static_pretrain_v1/seed71
labels="$static/flow/amplitude100_vs_md_advanced.json"
protocol="$static/flow/expanded48/protocol.json"

for seed in 71 72; do
  python "$task/score_clock_amplitude.py" \
    --candidate "$md/seed$seed/edm/clock80_100" \
    --reference "$md/seed$seed/flow/clock80_100" \
    --amplitude "$labels" --protocol "$protocol" \
    --output "$md/seed$seed/edm/clock_amplitude_vs_flow.json"
done

# Existing primary analysis owns this output; do not duplicate its forecasts.
while [[ ! -s "$static/edm/clock80_100/summary.json" ]]; do sleep 30; done
python "$task/compare_summary_pairs.py" \
  --candidate "$static/edm/clock80_100/summary.json" \
  --reference "$static/flow/clock80_100/summary.json" \
  --output "$static/edm/clock_vs_flow.json"
for reference in md flow; do
  if [[ "$reference" == md ]]; then
    reference_root="$md/seed71/edm/clock80_100"
  else
    reference_root="$static/flow/clock80_100"
  fi
  python "$task/score_clock_amplitude.py" \
    --candidate "$static/edm/clock80_100" --reference "$reference_root" \
    --amplitude "$labels" --protocol "$protocol" \
    --output "$static/edm/clock_amplitude_vs_$reference.json"
done
