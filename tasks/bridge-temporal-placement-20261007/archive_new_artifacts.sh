#!/usr/bin/env bash
# Host only; these two new runs, closed atomic outputs only, no historical data.
set -euo pipefail
shopt -s nullglob
stage=/data4/users/sihao/workspace/molvid-bridge-temporal-placement-20261007/runs/bridge_temporal_placement_20261007
archive=/data1/sihao/molvid-derived/bridge-temporal-placement-20261007
archive_file() {
 local source=$1 destination=$2
 test -f "$source" && test ! -L "$source" || return 0
 test ! -e "$destination" || return 2
 cp --preserve=mode,timestamps --no-clobber -- "$source" "$destination"
 ln -s -- "$destination" "$source.archive-link"
 mv -Tf -- "$source.archive-link" "$source"
}
while true; do
 completed=0
 for seed in 71 72; do
  for domain in complex_native80 expanded48 native80; do
   destination="$archive/seed$seed/$domain"
   mkdir -p "$destination"
   for source in "$stage/seed$seed/$domain"/coordinates_[0-9][0-9][0-9][0-9].pt; do
    archive_file "$source" "$destination/${source##*/}"
   done
   if test -f "$stage/seed$seed/$domain/summary.json"; then completed=$((completed+1)); fi
  done
  if test -f "$stage/seed$seed/train/bridge_final.pt"; then
   mkdir -p "$archive/seed$seed/train"
   for name in bridge_final.pt checkpoint_latest.pt; do
    archive_file "$stage/seed$seed/train/$name" "$archive/seed$seed/train/$name"
   done
  fi
 done
 if (( completed == 6 )); then break; fi
 sleep 10
done
