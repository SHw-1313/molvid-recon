#!/usr/bin/env bash
# Host-only recoverable relocation. No Python/model execution or broad deletion.
set -euo pipefail
archive_source_root=/data4/users/sihao/workspace/molvid-coordinate-chirality-training-20261005/runs/coordinate_chirality_training_20261005/formal_v1
archive_container_root=/workspace/molvid-coordinate-chirality-training-20261005/runs/coordinate_chirality_training_20261005/formal_v1
archive_destination_root=/data2/sihao/molvid-research-20261005/chirality-training-intermediates-v1
archive_queue=/data4/users/sihao/workspace/molvid-coordinate-chirality-training-20261005/runs/coordinate_chirality_training_20261005/formal_v1_training_queue_v1/status.json
archive_started=$(date +%s)
for archive_arm in control_71 chirality_71 chirality_72 control_72; do
    archive_run=$archive_source_root/$archive_arm
    while ! test -f "$archive_run/training_complete.json"; do
        if test -f "$archive_queue" && jq -e '.status == "failed"' "$archive_queue" >/dev/null; then
            printf 'Training queue failed before archiving %s\n' "$archive_arm"
            exit 1
        fi
        archive_now=$(date +%s)
        if (( archive_now - archive_started > 172800 )); then
            printf '48h archive wait budget expired\n'
            exit 1
        fi
        sleep 20
    done
    jq -e --arg expected "$archive_container_root/$archive_arm/formal/coordinate_step_2048.pt" \
        '.status == "completed" and .updates == 2048 and .checkpoint == $expected' \
        "$archive_run/training_complete.json" >/dev/null
    archive_final_sha=$(jq -r '.checkpoint_sha256' "$archive_run/training_complete.json")
    [[ "$archive_final_sha" =~ ^[0-9a-f]{64}$ ]]
    test "$(sha256sum "$archive_run/formal/coordinate_step_2048.pt" | cut -d ' ' -f1)" = "$archive_final_sha"
    jq -e --arg expected "$archive_final_sha" \
        '.status == "completed" and .cursor == 2048 and .checkpoint_sha256 == $expected' \
        "$archive_run/final_audit.json" >/dev/null
    archive_destination=$archive_destination_root/$archive_arm
    test ! -e "$archive_destination"
    mkdir -p "$archive_destination/pilot25" "$archive_destination/resume25"
    for archive_relative in pilot25/coordinate_step_0016.pt pilot25/coordinate_step_0025.pt resume25/coordinate_step_0025.pt; do
        archive_original=$archive_run/$archive_relative
        archive_copy=$archive_destination/$archive_relative
        test -f "$archive_original"
        test ! -L "$archive_original"
        test ! -e "$archive_original.archive-link"
        test ! -L "$archive_original.archive-link"
        test ! -e "$archive_copy"
        rsync -t --checksum "$archive_original" "$archive_copy"
        cmp "$archive_original" "$archive_copy"
        sha256sum "$archive_copy"
        stat -c '%s bytes' "$archive_copy"
        ln -s "$archive_copy" "$archive_original.archive-link"
        mv -T "$archive_original.archive-link" "$archive_original"
    done
    printf 'Recoverably archived completed %s intermediates; final/logs/predictions untouched\n' "$archive_arm"
done
