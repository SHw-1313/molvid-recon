#!/usr/bin/env bash
# Host-only storage migration of this experiment's closed artifacts.
set -euo pipefail
shopt -s nullglob
stage=/data4/users/sihao/workspace/molvid-bridge-collective-feedback-20261008/runs/bridge_collective_feedback_20261008
archive=/data1/sihao/molvid-derived/bridge-collective-feedback-20261008
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
  for arm in control collective; do
   root="$stage/seed$seed/$arm"
   for domain in expanded48 complex_validation64 native80; do
    destination="$archive/seed$seed/$arm/$domain"
    mkdir -p "$destination"
    for source in "$root/$domain"/coordinates_[0-9][0-9][0-9][0-9].pt; do
     archive_file "$source" "$destination/${source##*/}"
    done
    if test -f "$root/$domain/summary.json"; then completed=$((completed+1)); fi
   done
   if test -f "$root/train/bridge_final.pt"; then
    mkdir -p "$archive/seed$seed/$arm/train"
    for name in bridge_final.pt checkpoint_latest.pt; do
     archive_file "$root/train/$name" "$archive/seed$seed/$arm/train/$name"
    done
   fi
  done
 done
 if test "$completed" -eq 12; then break; fi
 sleep 10
done
