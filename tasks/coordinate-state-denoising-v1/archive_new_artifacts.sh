#!/usr/bin/env bash
# Host only; preserve every closed forecast and final checkpoint at its path.
set -euo pipefail
shopt -s nullglob
stage=/data4/users/sihao/workspace/molvid-coordinate-state-denoising-v1/runs/coordinate_state_denoising_v1
archive=/data1/sihao/molvid-derived/coordinate-state-denoising-v1
move_closed() {
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
  for scheme in flow edm; do
   root="$stage/seed$seed/$scheme"
   for domain in expanded48 complex_validation64 native80 native1000 rollout; do
    destination="$archive/seed$seed/$scheme/$domain"
    mkdir -p "$destination"
    for source in "$root/$domain"/coordinates_[0-9][0-9][0-9][0-9].pt; do
     move_closed "$source" "$destination/${source##*/}"
    done
    if test -f "$root/$domain/summary.json"; then completed=$((completed+1)); fi
   done
   if test -f "$root/train/coordinate_final.pt"; then
    mkdir -p "$archive/seed$seed/$scheme/train"
    for name in coordinate_final.pt checkpoint_latest.pt; do
     move_closed "$root/train/$name" "$archive/seed$seed/$scheme/train/$name"
    done
   fi
  done
 done
 if test "$completed" -eq 20; then break; fi
 sleep 20
done
