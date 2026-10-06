#!/usr/bin/env bash
# Host only. Do not archive live final/resume17 inputs or numerical source.
set -euo pipefail
run_root=/data4/users/sihao/workspace/molvid-data-scale-training-input-20261006/runs/data_scale_training_input_20261006/campaign_v1
archive_root=/data2/sihao/molvid-research-20261005/data-scale-prefix16-20261006
total=0
for seed in 71 72;do
  test "$(jq -r .status "$run_root/seed$seed/qualification.json")" = completed
  test "$(jq -r .actual_updates "$run_root/seed$seed/qualification.json")" = 35
  for arm in default16 pilot17;do
    source_file="$run_root/seed$seed/$arm/coordinate_step_0016.pt"
    target="$archive_root/seed$seed/$arm/coordinate_step_0016.pt"
    test -f "$source_file";test ! -L "$source_file"
    test "$(stat -c %U "$source_file")" = sihao
    test "$(stat -c %h "$source_file")" = 1
    test ! -e "$target";test ! -L "$target"
    test ! -e "$source_file.archive-link";test ! -L "$source_file.archive-link"
    if lsof -t -- "$source_file" 2>/dev/null;then exit 1;fi
    bytes=$(stat -c %s "$source_file")
    available=$(df -B1 --output=avail /data2 | tail -1 | tr -d ' ')
    test "$available" -gt "$((bytes+10*1024*1024*1024))"
    before=$(stat -c '%s:%y:%z:%i' "$source_file")
    mkdir -p "${target%/*}"
    cp --preserve=mode,timestamps --reflink=auto -- "$source_file" "$target"
    cmp -- "$source_file" "$target";sha256sum -- "$source_file" "$target"
    test "$before" = "$(stat -c '%s:%y:%z:%i' "$source_file")"
    if lsof -t -- "$source_file" 2>/dev/null;then exit 1;fi
    ln -s -- "$target" "$source_file.archive-link"
    mv -T -- "$source_file.archive-link" "$source_file"
    total=$((total+bytes))
  done
done
printf 'Archived bytes=%s; discarded bytes=0\n' "$total"
df -h /data4 /data2
