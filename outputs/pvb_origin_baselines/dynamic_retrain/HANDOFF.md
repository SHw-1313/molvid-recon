# Baseline D — handoff

This directory is the only destination for new baseline-D checkpoints, logs, metrics, wrappers, and run notes. `/data4/users/sihao/workspace/PVB_origin` is read-only for this task.

## Initial state

- Requested source checkpoint directory: `/data1/repo/PVB/ckpt/misato_from_author_pretrain`.
- Original PVB revision: `c08e5e3` (`submit version`).
- Pilot policy read from `agents/PILOT_COMMANDS.md`: seed `20260810`, 1000 optimizer steps, train on ATLAS `dt_100ps` plus MISATO `dt_80ps`, validate with at most 32 validation loader batches.
- Current clip stores are `npz-v1`, not original PVB gzip-JSON records; conversion is therefore a compatibility boundary that must be documented and checked.

## Evidence log

### 2026-08-20 — inspection started

- Status: original entrypoints and data formats identified; training not started.
- Files changed: this `TASKS.md`, this `HANDOFF.md`.
- Commands/evidence: see the assistant transcript and the eventual `commands.log`/`run_manifest.json` in this directory.
- Blockers: exact compatible pair-store availability and runtime compatibility are still being checked.
