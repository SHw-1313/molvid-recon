#!/usr/bin/env bash
# Host only: archive newly closed forecasts from this experiment, keep symlinks.
set -euo pipefail
shopt -s nullglob
stage=/data4/users/sihao/workspace/molvid-bridge-interleaved-clocks-20261007/runs/bridge_interleaved_clocks_20261007
archive=/data1/sihao/molvid-derived/bridge-interleaved-clocks-20261007
while true; do
 completed=0
 for seed in 71 72; do
  for domain in complex_native80 expanded48 native80; do
   destination="$archive/seed$seed/$domain"
   mkdir -p "$destination"
   for source in "$stage/seed$seed/$domain"/coordinates_[0-9][0-9][0-9][0-9].pt; do
    test ! -L "$source" || continue
    name="${source##*/}"
    test ! -e "$destination/$name" || exit 2
    mv --no-clobber -- "$source" "$destination/$name"
    ln -s -- "$destination/$name" "$source"
   done
   if test -f "$stage/seed$seed/$domain/summary.json"; then completed=$((completed+1)); fi
  done
 done
 if test "$completed" -eq 6; then break; fi
 sleep 10
done
