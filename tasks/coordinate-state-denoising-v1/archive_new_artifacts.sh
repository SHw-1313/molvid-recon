#!/usr/bin/env bash
# Host only; preserve every closed forecast and final checkpoint at its path.
set -euo pipefail
shopt -s nullglob
stage=/data4/users/sihao/workspace/molvid-coordinate-state-denoising-v1/runs/coordinate_state_denoising_v1
archive=/data1/sihao/molvid-derived/coordinate-state-denoising-v1
seeds=(71 72)
domains=(expanded48 complex_validation64 native80 native1000 rollout)
training_stages=(train)
expected_panels=20
case "${1:-md}" in
 md) ;;
 static)
  stage=/data4/users/sihao/workspace/molvid-coordinate-state-denoising-v1/runs/coordinate_static_pretrain_v1
  archive=/data1/sihao/molvid-derived/coordinate-static-pretrain-v1
  seeds=(71)
  domains=(expanded48 native80 complex_validation64)
  training_stages=(static md)
  expected_panels=6
  ;;
 *) exit 2;;
esac
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
 for seed in "${seeds[@]}"; do
  for scheme in flow edm; do
   root="$stage/seed$seed/$scheme"
   for domain in "${domains[@]}"; do
    destination="$archive/seed$seed/$scheme/$domain"
    mkdir -p "$destination"
    for source in "$root/$domain"/coordinates_[0-9][0-9][0-9][0-9].pt; do
     move_closed "$source" "$destination/${source##*/}"
    done
    if test -f "$root/$domain/summary.json"; then completed=$((completed+1)); fi
   done
   for training_stage in "${training_stages[@]}"; do
   if test -f "$root/$training_stage/coordinate_final.pt"; then
    mkdir -p "$archive/seed$seed/$scheme/$training_stage"
    for name in coordinate_final.pt checkpoint_latest.pt; do
     move_closed "$root/$training_stage/$name" "$archive/seed$seed/$scheme/$training_stage/$name"
    done
   fi
   done
  done
 done
 if test "$completed" -eq "$expected_panels"; then break; fi
 sleep 20
done
