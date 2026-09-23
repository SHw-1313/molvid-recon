# Independent P3 r02 review

REVIEW_STAGE: P3  
REVIEWED_COMMIT: f6f33a95a08627e5c779ca099d075679e9bed5ac  
VERDICT: PASS

- Reviewer/session: fresh independent read-only reviewer, `/root/p3_independent_review_r02`.
- Base commit: `68abde6e1dc9196983cc5fd408e69118e60df41b`.
- Scope: targeted review of r01 issue R1, its complete fix response, corrected budget evidence, profiler semantics, and formal configuration symmetry.
- Code worktree: `/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2-review-p3-r02`.
- Environment: the initial sandbox command failed before reading files. Subsequent approved commands performed read-only inspection. No files were written, no training/tests/model execution were performed, and no dataset payloads or sealed test data were opened.

## Conclusion

R1 is closed. The corrected resolved experiment uses the measured ordinary-update time of `1.0679053366184235 s` and final sampled-update time of `6.8627921268343925 s`. Independent cadence and GPU-hour calculations reproduce every reported total.

The frozen two-arm formal run plus both conditional bond forks totals `19.487990774650953 GPU-hours` before the safety factor and `38.975981549301906 GPU-hours` after the frozen `2.0×` multiplier. This is below the `96 GPU-hour` P3 limit, with `57.024018450698094 GPU-hours` remaining against that limit.

No reproducible validity, budget, provenance, or runtime blocker was found in this targeted repair review. The exact J0/J1 configurations listed below may proceed from the reviewed candidate and recorded B0 parent.

## Required fixes

None.

### R1 — Closed

The stale `4.364293091930449 s` budget input has been removed. The replacement ordinary profile supplies the previously missing measurement; its measured value, script identity, final sampled profile, cadence counts, optional fork counts, GPU accounting, and conservative multiplier are explicitly bound in `resolved_experiment.json`.

The candidate commit and both formal configuration hashes are unchanged from r01. This repair changes budget evidence without changing the reviewed model implementation or formal exposure.

## Checks performed

### Identity and provenance

- Verified the request SHA-256 against the supplied digest.
- Verified every listed artifact SHA-256, including the complete prior review and fix response.
- Verified detached HEAD equals the requested candidate. The review worktree was clean initially and remained clean at the final check.
- Read the candidate repository instructions, plan, architecture, data/training/evaluation specification, review protocol, review prompt, and report template.
- Inspected the relevant candidate profiler, trainer, warm-start, training-loop, and data-loading implementations. This was a targeted R1 review, not a repeat of the entire r01 scientific-path review.

### Ordinary-update profiler

The hash-bound `profile_ordinary_largest.py`:

- Requires sampled auxiliary to be disabled and rejects a restricted `train_systems` configuration.
- Selects the maximum train-record `effective_tokens`, with deterministic sample-ID tie-breaking.
- Constructs the candidate `FrameJointModel`, resolves the parent loss contract, and loads the parent through the SHA-checked warm-start interface.
- Uses the pilot J0 configuration with `precision: bf16`; the candidate trainer implements CUDA BF16 autocast and rejects BF16 on a non-CUDA device.
- Synchronizes CUDA before timing and after `trainer.train_step`.
- Times an actual forward/loss computation, backward pass, gradient clipping, and optimizer update. In `molvid/training/joint.py`, successful-update counters advance after `optimizer.step()`.
- Checks that the sampled branch was inactive, all three trainable components had positive gradient norms, teacher gradients were zero, and the teacher state hash was unchanged.

The recorded ordinary profile satisfies those checks and reports step `1`, sampled branch inactive, and `386` loaded / `0` initialized warm-start tensors.

Both profiles identify the same train record:

- Dataset index: `67787`.
- Sample ID: `atlas_4yal_A_R3_dt_400ps_wf000061`.
- Size: `4,975 atoms × 16 frames = 79,600 atom-frames`.
- History: `4` frames.
- Device model: `NVIDIA A100-SXM4-80GB`.
- Identical parent SHA, derived-data hash, normalization-source hash, and initial teacher hash.

The candidate sampled profiler explicitly sets the successful-update counter to `7`, then measures update `8`; the recorded result confirms sampled branch active, K=2, four Euler steps, and activation checkpointing.

### Formal configuration symmetry

Direct comparison establishes that J0 and J1 share data configuration, codec/statistics identities, architecture, inherited base loss, seed, BF16 precision, deterministic setting, packing budget, trajectory cap, bucket weights, history order, optimizer settings, learning rates, stage, and `21,208` updates.

Their intended differences are the sampled-auxiliary block and output directory.

Each profiled pilot configuration differs from its corresponding formal configuration only in output directory and update count (`64` versus `21,208`). The profiler therefore uses the same relevant model, data, precision, and objective settings as the formal configuration.

### Cadence and GPU-hour arithmetic

`SampledAuxiliaryConfig.active_at` enables the branch when:

```text
enabled and (successful_updates + 1) % 8 == 0
```

Warm start resets both counters to zero. Consequently, formal J1 sampled updates are `8, 16, …, 21,208`.

Let:

```text
o = 1.0679053366184235 seconds
s = 6.8627921268343925 seconds
```

Independent recomputation gives:

| Run | Ordinary updates | Sampled updates | GPU-hours |
|---|---:|---:|---:|
| J0 formal | 21,208 | 0 | 6.291148994167646 |
| J1 formal | 18,557 | 2,651 | 10.558439238851683 |
| Each conditional fork | 2,320 | 331 | 1.319201270815813 |

Formulas:

```text
J0 = 21208 × o / 3600
J1 = (18557 × o + 2651 × s) / 3600
formal total = 16.84958823301933 GPU-hours

fork sampled count = floor(23859 / 8) − floor(21208 / 8) = 331
fork ordinary count = 2651 − 331 = 2320
each fork = (2320 × o + 331 × s) / 3600
two forks = 2.638402541631626 GPU-hours

profile total = 19.487990774650953 GPU-hours
2× conservative total = 38.975981549301906 GPU-hours
```

Each fork begins after successful update `21,208` and covers updates `21,209–23,859`, inclusive. Its first sampled update is `21,216`; its last is `23,856`. No off-by-one or cadence reset is introduced in the budget.

The budget correctly sums per-device execution time for two independent single-GPU arms. Two GPUs running in parallel reduce wall time; they do not divide the summed GPU-hours. The conditional fork remains trigger-only. Even its doubled allocation, `5.276805083263252 GPU-hours`, is below the specification’s maximum `19.2 GPU-hour` fork reserve.

### Formal-start and sealed-test status

The hash-bound request and corrected resolved experiment both record `formal_training_started: false` and `test_opened: false`. The command ledger contains pilot/profile/check commands and no formal launch command.

The inspected `_open_data` / `load_datasets` path opens train and validation payloads, requires the frozen manifest to declare test sealed, and requires derived training views to declare `split=train` and `test_opened=false`. The new profiler adds no test-data access.

These findings establish the status represented by the supplied evidence snapshot and the inspected execution path. This review did not perform a machine-wide audit of unlisted processes or filesystem artifacts.

## Non-blocking observations

The measured `step_seconds` covers the prepared-batch trainer update. Teacher/batch preparation, input loading, checkpoint writes, and process startup lie outside that timer. The `2×` budget is therefore a conservative planning allowance, not a measured end-to-end runtime guarantee. Continue the specification’s actual per-device elapsed-time accounting and checkpoint-boundary budget enforcement during formal execution.

The r01 calibration-provenance observation remains non-blocking and is unaffected by this evidence-only repair. Calibration artifacts outside this request were not reopened or independently revalidated.

## Exact approved identities

Paths below use:

```text
E = /data4/users/sihao/workspace/molvid-recon-gm-calibration-v2
C = /data4/users/sihao/workspace/molvid-recon-gm-calibration-v2-review-p3-r02
```

All listed digests were independently verified.

| File | SHA-256 |
|---|---|
| `E/agent/frame_gm_calibration_v2/reviews/P3/r02/review_request.json` | `080948ee5d384b1631f5956a2bdddeb59cae5bb3ea59e772c9b0be9f010b779b` |
| `E/agent/frame_gm_calibration_v2/reviews/P3/r01/REVIEW.md` | `6ec94450c85f94c40562606f05e705016be023f7016880edda157725fc26e1fa` |
| `E/agent/frame_gm_calibration_v2/reviews/P3/r01/FIX_RESPONSE.md` | `b3be32b070e174484d9508293e876d7593b1d98869abd1bcb097bdf4c558138f` |
| `E/runs/frame_gm_calibration_v2/P3/r01/evidence/resolved_experiment.json` | `d089b30b04c2dac1c2031b74d105d6239378e155d1b7ab31fdd3578fc7d0647a` |
| `E/runs/frame_gm_calibration_v2/P3/r01/profile/largest_ordinary.json` | `3e4e67af374804bd86cb66cc63c176fbd7dbbf57c8b8ce6dae20357fd66a4d71` |
| `E/runs/frame_gm_calibration_v2/P3/r01/evidence/profile_ordinary_largest.py` | `328d873c2e62602e1b7991e375d1db1137b1ff5b007060967cb4c6d7d26f1cf0` |
| `E/runs/frame_gm_calibration_v2/P3/r01/profile/largest_sample.json` | `c58448d6372d4fefe16e1795e467b5487ff378024dda203092fff987badb5056` |
| `E/runs/frame_gm_calibration_v2/P3/r01/evidence/commands.sh` | `ecd8b5de67232d3884a2ae5f91d824e9f5388b0572749c25d2362bcbe45c0f6e` |
| `C/configs/frame_gm_p3_formal_J0_260923.yaml` | `e9cb92ba0235ccb61455e58249e50c3db51f09452a2293909682aa909897a8dc` |
| `C/configs/frame_gm_p3_formal_J1_260923.yaml` | `b05ef564d9f0c778fc31cccd3a0cbec042fc675044ff1fe37426a1f1d17e6a3f` |

The recorded B0 parent is `frame_joint_step_00021208.pt`, SHA-256 `ff8121aa47a7c1885b4ad59ece586dcc90a3d8a12c67d71fb767f586c6b36258`. That identity agrees across the supplied profiles and resolved experiment; the checkpoint itself was outside this targeted request and was not reopened.

## Main-session next step

Record this review and its hash in the clearance manifest, close R1, and proceed with the exact approved J0/J1 configurations at candidate `f6f33a95a08627e5c779ca099d075679e9bed5ac`, using the recorded B0 warm-start parent and `21,208` updates per arm.

Preserve the reviewed code snapshot and configuration hashes during execution. Keep conditional bond forks contingent on the frozen comparison trigger; if triggered, bind their concrete configurations and hashes in the run manifest while preserving the reviewed `2,651` updates per arm and inherited successful-update cadence.
