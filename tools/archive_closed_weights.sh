#!/usr/bin/env bash
# Host-only relocation of explicit, completed experiment files. Zero discarded bytes.
set -euo pipefail
workspace=/data4/users/sihao/workspace
archive=/data2/sihao/molvid-research-20261005/closed-weights-20261007
total=0
for relative in \
 molvid-data-scale-training-input-20261006/runs/data_scale_training_input_20261006/campaign_v1/seed71/pilot17/coordinate_step_0017.pt \
 molvid-data-scale-training-input-20261006/runs/data_scale_training_input_20261006/campaign_v1/seed71/resume17/coordinate_step_0017.pt \
 molvid-data-scale-training-input-20261006/runs/data_scale_training_input_20261006/campaign_v1/seed72/pilot17/coordinate_step_0017.pt \
 molvid-data-scale-training-input-20261006/runs/data_scale_training_input_20261006/campaign_v1/seed72/resume17/coordinate_step_0017.pt \
 molvid-static-update-budget-20261006/runs/static_update_budget_20261006/campaign_v1/budget_71/formal/coordinate_step_2048.pt \
 molvid-static-update-budget-20261006/runs/static_update_budget_20261006/campaign_v1/budget_72/formal/coordinate_step_2048.pt \
 molvid-coordinate-axial-readout-20261006/runs/coordinate_axial_readout_20261006/campaign_v1/axial_72/formal/coordinate_step_2048.pt \
 molvid-coordinate-axial-readout-20261006/runs/coordinate_axial_readout_20261006/campaign_v1/polar_72/formal/coordinate_step_2048.pt; do
 source_file="$workspace/$relative";target="$archive/$relative"
 test -f "$source_file";test ! -L "$source_file"
 test "$(stat -c %U "$source_file")" = sihao
 test "$(stat -c %h "$source_file")" = 1
 test ! -e "$target";test ! -e "$source_file.archive-link"
 if lsof -t -- "$source_file" 2>/dev/null;then exit 1;fi
 bytes=$(stat -c %s "$source_file")
 original_stat=$(stat -c '%s:%y:%i' "$source_file")
 mkdir -p "${target%/*}"
 cp --preserve=mode,timestamps --reflink=auto -- "$source_file" "$target"
 cmp -- "$source_file" "$target"
 test "$original_stat" = "$(stat -c '%s:%y:%i' "$source_file")"
 ln -s -- "$target" "$source_file.archive-link"
 mv -T -- "$source_file.archive-link" "$source_file"
 total=$((total+bytes))
done
printf 'Archived bytes=%s; discarded=0\n' "$total"
df -h /data4 /data2
