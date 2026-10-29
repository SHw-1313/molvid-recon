# Task-route experiment, authorized 2026-10-29

Read /data4/users/sihao/workspace/molvid_recon/{AGENTS.md,HANDOFF.md,TASKS.md} after resume or updates (container /workspace/molvid_recon). These are the only PM authority; inherited local HANDOFF/TASKS are historical.

User authorizes two independent matched soft task-routing studies from L4096: shared versus task-dependent convex weights of the same two adapters. Base frozen, gradients through the downstream frozen layers retained. Same 512 Hmix updates, parent, exposure and saved random streams. Original source/ligand auxiliary unchanged. No memory/GAFF/causal/new loss in this task.

Independent end-to-end T cap57600 A100 GPU seconds (16h), including checks/failures/train/generation; initial profile allowance is included, not extra. Two 512 runs each initially capped10800s; remaining total reserved for finite existing-protocol evaluation. Six DEV/two seeds joint,Q12,Q4,original CF,H4Q1; endpoint three lags/four draws, limited compatibility only within remainder. Reuse existing workers, no new scheduler. Python only existing container torch-ito. Protect external tasks/old checkpoints/sealed TEST. One task README; distinguish dataset, physical dt, history, Q, sampler and reference window in scores.
