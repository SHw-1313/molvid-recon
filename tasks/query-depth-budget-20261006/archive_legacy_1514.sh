#!/usr/bin/env bash
set -euo pipefail
workspace_root=/data4/users/sihao/workspace
archive_root=/data2/sihao/molvid-research-20261005/closed-legacy-1514-20261006
targets=(
"molvid_recon/runs/frame_joint_v1_multitime_260921/frame_joint_step_00022272.pt"
"molvid_recon/runs/frame_joint_v1_multitime_260921/frame_joint_step_00021272.pt"
"molvid_recon/runs/frame_joint_v1_192_newB_260921/frame_joint_step_00123199.pt"
"molvid_recon/runs/frame_joint_v1_192_newB_260921/frame_joint_step_00122199.pt"
"molvid_recon/runs/frame_joint_v1_192_newB_260921/frame_joint_step_00111573.pt"
"molvid_recon/runs/frame_joint_v1_192_newB_260921/frame_joint_step_00106260.pt"
"molvid_recon/runs/frame_joint_v1_48_frozen_control_260921/frame_joint_step_00002716.pt"
"molvid_recon/runs/frame_joint_v1_48_frozen_control_260921/frame_joint_step_00001358.pt"
"molvid_recon/runs/frame_joint_v1_48_bond_release_260921/frame_joint_step_00006790.pt"
"molvid_recon/runs/frame_joint_v1_48_bond_release_260921/frame_joint_step_00005432.pt"
"molvid_recon/runs/frame_joint_v1_48_bond_keep_260921/frame_joint_step_00006790.pt"
"molvid_recon/runs/frame_joint_v1_48_bond_keep_260921/frame_joint_step_00005432.pt"
"molvid_recon/runs/frame_joint_v1_48_joint_start_parent_260921/frame_joint_step_00005075.pt"
"molvid_recon/runs/frame_joint_v1_48_joint_start_parent_260921/frame_joint_step_00005000.pt"
"molvid_recon/runs/frame_joint_v1_48_pilot_260920/frame_joint_step_00013222.pt"
"molvid_recon/runs/frame_joint_v1_48_pilot_260920/frame_joint_step_00012222.pt"
"molvid_recon/runs/frame_joint_v1_ddp_stage_smoke_final_260920/frame_joint_step_00000004.pt"
"molvid_recon/runs/frame_joint_v1_ddp_stage_smoke_260920/frame_joint_step_00000004.pt"
"molvid_recon/runs/frame_joint_v1_cuda_260920/ddp_resume_step_00000001.pt"
"molvid_recon/runs/frame_joint_v1_review_fixes_260920/ddp_resume_step_00000001.pt"
"molvid_recon/runs/frame_joint_v1_review_child_cli_resume_260920_v2/frame_joint_step_00000002.pt"
"molvid_recon/runs/frame_joint_v1_tiny_corrected_resume_260920/frame_joint_step_00000012.pt"
"molvid_recon/runs/frame_joint_v1_tiny_corrected_260920/frame_joint_step_00000012.pt"
"molvid_recon/runs/frame_joint_v1_tiny_corrected_260920/frame_joint_step_00000009.pt"
)
total=0
for relative in "${targets[@]}"; do
    source_file="$workspace_root/$relative"; target="$archive_root/$relative"
    test -f "$source_file"; test ! -L "$source_file"
    test "$(stat -c %U "$source_file")" = sihao
    test "$(stat -c %h "$source_file")" = 1
    test "$(stat -c %Y "$source_file")" -lt "$(date -d '2026-10-05 00:00:00' +%s)"
    test ! -e "$target"; test ! -L "$target"
    test ! -e "$source_file.archive-link"; test ! -L "$source_file.archive-link"
    if lsof -t -- "$source_file" 2>/dev/null; then exit 1; fi
    total=$((total+$(stat -c %s "$source_file")))
done
available=$(df -B1 --output=avail /data2 | tail -1 | tr -d ' ')
test "$available" -gt "$((total+10*1024*1024*1024))"
for relative in "${targets[@]}"; do
    source_file="$workspace_root/$relative"; target="$archive_root/$relative"
    if lsof -t -- "$source_file" 2>/dev/null; then exit 1; fi
    before=$(stat -c '%s:%y:%z:%i' "$source_file")
    mkdir -p "${target%/*}"
    cp --preserve=mode,timestamps --reflink=auto -- "$source_file" "$target"
    cmp -- "$source_file" "$target"; sha256sum -- "$source_file" "$target"
    test "$before" = "$(stat -c '%s:%y:%z:%i' "$source_file")"
    if lsof -t -- "$source_file" 2>/dev/null; then exit 1; fi
    ln -s -- "$target" "$source_file.archive-link"
    mv -T -- "$source_file.archive-link" "$source_file"
    test "$(readlink "$source_file")" = "$target"
done
printf 'Archived bytes: %s; checkpoint bytes deleted: 0\n' "$total"
df -h /data4 /data2
