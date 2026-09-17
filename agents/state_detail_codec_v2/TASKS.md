# TASKS — zero-preserving state/detail temporal codec v2

Status legend: `[ ]` pending, `[-]` active, `[x]` complete, `[!]` blocked.

## Preparation

- [x] S200 inspect repository, branch, HEAD, remote, worktree, environment, current codec code,
  and prior three-system artifacts without discarding user changes — source/status, torch-ito
  environment, legacy source contracts, and immutable v1 outputs were audited.
- [x] S201 create/enter `feat/state-detail-codec-v2` from approved commit and update root
  `AGENTS.md` for this phase — target branch created from `045dbb8809e9d7aee418eb355b6477e05088d4fd`;
  root rules now name this phase and the mandatory operator stop.
- [x] S202 record exact old temporal encoder, decoder, `x_anchor`, model/checkpoint contracts,
  tests, and stored R1/R4 behavior before editing — see `outputs/state_detail_codec_v2/preflight/legacy_audit.json`.
- [x] S203 verify the exact TorchMD frame-encoder checkpoint/hash to be shared and frozen across
  all T0 controls — extracted 42-key 128-channel state from immutable v1 TorchMD step 4979;
  hash is recorded in the preflight audit.
- [x] S204 verify the three systems, all nine trajectory replicas, 441/117 clip split, indexes,
  timestamps, and artifact hashes; stop on mismatch — independent torch-ito audit confirmed
  exact IDs/windows/T=16/dt_100ps and zero overlap.

## Algebra and latent contracts

- [x] S210 implement/test reusable orthonormal R1/R2/R4 lifting and exact inverse lifting for h/v
  — strict FP32 identity fixtures pass at rtol=1e-6, atol=1e-7.
- [x] S211 implement structured state/detail latent schema, masks, clocks, coefficient order,
  capacity accounting, and serialization — 18 codec tests cover shapes, masks, clocks, order,
  and contract fields.
- [x] S212 implement explicit per-sample centroid/origin handling and remove every per-atom x0
  bypass from new codec controls — new latents expose only [B,3] origin and no x_anchor.
- [x] S213 version model/checkpoint contracts while preserving explicit legacy loading — v3 new
  contract round-trip and legacy checkpoint/output regression pass.

## State/detail encoder and decoder

- [x] S220 implement R1 C-wide state path and absent/masked detail semantics — shape/capacity and
  static tests pass.
- [x] S221 implement R2 C-wide state plus C-wide zero-preserving detail path — zero and dynamic
  detail tests pass.
- [x] S222 implement R4 C-wide state and three-detail-to-C zero-preserving encoder — exact
  Dmid,D01,D23 ordering and 8C capacity are tested.
- [x] S223 implement R2/R4 zero-preserving detail decoders and inverse lifting — identity and
  masked inverse fixtures pass.
- [x] S224 implement shared framewise equivariant centered-coordinate decoder without additive
  time motion or spatial refiner — SE(3), clock-invariance, and no-anchor tests pass.
- [x] S225 implement static T=1 full-state path and distinguish it from repeated-static T=16
  — T=1 invalid-detail and repeated-static-zero tests pass.

## Matched control

- [x] S230 implement R4 two-head matched pooling with two unstructured C-wide banks — 8C
  capacity and independent-bank gradient tests pass.
- [x] S231 implement its no-anchor four-frame expansion decoder and report natural parameters
  — contract and integration tests pass.
- [x] S232 expose exactly four new control names/configurations without changing legacy names
  — explicit mode validation and legacy default regression pass.

## Correctness tests

- [x] S240 pass lifting roundtrip, ordering, mask, capacity, T=1/T=16, partial/padded block, and
  irregular-clock tests — covered by the 18-test phase suite.
- [x] S241 pass exact repeated-static zero coefficient, zero latent-detail, zero decoded-feature
  motion, and zero coordinate-motion tests — all three state/detail ratios pass.
- [x] S242 pass moving-input detail utilization and finite/nonzero detail/decoder/coordinate
  gradient tests — scalar/vector/detail/coordinate principal parameters are finite and nonzero.
- [x] S243 pass scalar invariance, vector/coordinate SE(3), translation-origin, frame isolation,
  and sample isolation tests — phase suite passes.
- [x] S244 prove decoder has no target-coordinate argument and no new per-atom x0/reference path
  — API, contract, latent, and post-encode target-mutation tests pass.
- [x] S245 pass new checkpoint save/resume/config roundtrip and legacy output/checkpoint regression
  — phase and legacy tests pass.
- [x] S246 pass the full relevant suite, `py_compile`, source audit, and `git diff --check`
  — 107 passed; py_compile and diff check clean.

## T0-A bounded training smoke

- [x] S250 freeze the four-control micro protocol before viewing outcomes — runner fixes seed,
  FP32, 500 steps, common frozen frame checkpoint, and immutable three-clip selection.
- [x] S251 run one-batch/single-clip overfit for R1-SD, R2-SD, R4-SD, and R4-Matched-Pooling
  — remote CUDA run completed for all four.
- [x] S252 verify loss decrease, finite/nonzero new-module gradients, no hidden spatial updates,
  checkpoint step/resume, runtime, memory, and detail utilization — all four remote micro result
  records report pass, >30% loss reduction, unchanged frozen encoder, and resume pass.
- [x] S253 run one explicit unfrozen-encoder forward/backward smoke without starting a fifth
  comparison experiment — `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/unfreeze_smoke.json`
  reports `status=passed`, 41 nonzero spatial gradients, and changed frame-encoder state.

## T0-B three-system/nine-trajectory comparison

- [x] S260 freeze manifest, common encoder hash, loader coverage, optimizer, schedule, evaluator,
  seed, output paths, and device assignment before launch — final protocol is recorded in
  `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/protocol.json`; manifest
  contract is `.../manifest_contract.json`, with common frame-encoder SHA256
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a` and CUDA A100 device data.
- [x] S261 train R1-SD for exactly 30 complete epochs — final result is in
  `.../ratio1_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S262 train R2-SD for exactly 30 complete epochs — final result is in
  `.../ratio2_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S263 train R4-SD for exactly 30 complete epochs — final result is in
  `.../ratio4_state_detail/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S264 train R4-Matched-Pooling for exactly 30 complete epochs — final result is in
  `.../ratio4_matched_pooling/result.json`; 4,976 scheduled steps and exact epoch coverage passed.
- [x] S265 verify every run's exact coverage, complete holdout evaluations, logs, best/final
  checkpoints, and checkpoint resume — all four have 30 epoch rows, exact 441/117 protocol
  coverage, complete holdout evaluation, final/best checkpoints, and resume `4976 -> 4977`.
- [x] S266 generate rate/capacity, reconstruction, dynamics, block-offset/boundary, latent-bank,
  counterfactual, system-level, runtime, and memory reports/plots — aggregate JSON/CSV/Markdown
  plus loss, evaluation, block-detail, and performance plots are in the final run directory.

## Operator packet and mandatory stop

- [x] S270 fill `OPERATOR_REVIEW.md` with exact evidence paths and observed results — packet is
  complete and leaves the operator decision pending.
- [x] S271 append complete changed files, commands, hashes, tests, metrics, deviations, warnings,
  and limitations to `HANDOFF.md` — final dated handoff appended.
- [x] S272 set phase status to `WAITING_FOR_OPERATOR_REVIEW` — recorded in the operator packet,
  aggregate report, and final handoff.
- [x] S273 stop without preparing or running T1 or any DiT/static/full-data work — no T1 manifest
  or later architecture task was started.

## T1 — explicitly authorized 2026-09-04

- [x] S300 materialize and independently verify the 64-system 48/8/8 system-level split and
  frozen manifest — corrected token-valid manifest and 11,904 selected clip records are
  materialized under `outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`; exact
  counts/hashes are in `manifest.json` and `materialization.json`
- [x] S301 run four parallel 200-step profiles; verify throughput, peak memory, checkpoint
  resume, and projected complete-run duration — all four passed; 45,844 projected steps and
  approximately 2.48–3.66 h/control are recorded under `outputs/state_detail_codec_v2/t1/profiles_20260904`
- [-] S302 after the profile gate, run the four complete one-seed T1 controls in parallel on
  available idle local/`neibu` GPUs
- [ ] S303 use validation only to freeze the ratio-selection rule, then open test evaluation
- [ ] S304 after T1, run additional seeds only for the two controls ranked first by the frozen
  validation rule
- [ ] S305 append the T1 review packet, exact artifacts/hashes, limitations, and final
  `WAITING_FOR_OPERATOR_REVIEW` stop record

The operator explicitly authorized S300–S305 on 2026-09-04. GPU use may be parallelized across
idle devices, but no other user's process may be stopped or preempted. No later architecture
phase or T1 manifest broadening is allowed.

## Repair phase — T1 gates (2026-09-04)

The operator requested repair and repeat T0. The accepted Haar/state-detail algebra and the four
control contracts remain binding. These tasks do not authorize T1.

- [x] R200 audit the actual checkout, source commit, remote, branch, worktree, current contracts,
  and prior T0 evidence — checkout is `/data4/users/sihao/workspace/PVB`, remote is
  `https://github.com/SHw-1313/molvid.git`, source HEAD is `48bbff992e66cc5f351911e23f31750325ef3726`,
  and the pre-repair worktree was clean.
- [x] R201 reproduce the four operator-review headline rows and record a clean pre-repair hash
  set — old final future RMSD/dRMSD/bond RMSE are R1 `11.424228/10.154870/3.083183`, R2
  `11.241862/9.900528/3.122913`, R4 `11.138619/9.767301/3.107002`, matched
  `10.544774/9.306050/3.344215`; hashes are appended to the repair handoff.
- [x] R202 create the recommended `fix/state-detail-codec-v2-t1-gates` branch and append the
  repair hypotheses, gates, and stop condition to the phase records — completed before source
  implementation.
- [x] R210 replace frame-expanded/coordinate-dependent latent topology with a validated static
  N-axis chemical schema.
- [x] R211 add topology metadata tests for N-axis bounds, T-invariance, coordinate/frame
  independence, and exclusion of radius/distance/target data — static-topology and codec tests
  pass with explicit longer-T shape invariance.
- [x] R220 implement aligned/raw RMSD, pair-aware contacts, dynamic ACF, and explicit RMSF/
  aggregation semantics while preserving clearly named raw legacy diagnostics.
- [x] R221 add evaluator fixtures for rigid-transform invariance, pair identity, and temporal
  ordering; `tests/test_codec_evaluation.py` and `tests/test_state_detail_codec_v2.py` pass
  together (28 passed before the shape-invariance assertion, then the same focused suite is rerun).
- [x] R230 run the required cached-feature single-clip diagnostic and freeze an operator-visible
  R1 reconstruction threshold before inspecting repaired training results — corrected stage2c
  cached h/v and centered target are in the diagnostic directory; threshold is recorded in D24.
- [x] R231 implement the smallest permitted no-anchor reconstruction repair based on the diagnostic;
  retain the shared architecture and accepted capacity/zero-motion contracts — centered-vector
  stem is versioned in model-contract v4 and applied before all four control codecs.
- [x] R240 run the genuine one-sample R1 overfit with the frozen 1,000-step/D24 gate, curves,
  raw/aligned RMSD, dRMSD, bond RMSE, finite gradients, frozen encoder, and centroid/origin
  baseline — stage2c passed all frozen thresholds; final aligned/raw/dRMSD/bond are
  `0.084951/0.085038/0.111515/0.024777` Å and checkpoint resume `1000 -> 1001` passed.
- [x] R241 run bounded single-clip R2, R4-SD, and matched-pooling smoke only after R1 passes —
  stage3 passed for all three controls with zero-preserving detail, finite/nonzero gradients,
  unchanged frozen encoder, shared decoder contract, and checkpoint resume.
- [x] R250 repeat the exact three-system/nine-trajectory T0 only after all earlier repair gates
  pass; the authoritative run is
  `outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210`, using the canonical
  441/117 store, FP32, seed `20260903`, frozen `torchmd_et`, and all four controls for 30
  complete epochs/4,976 steps. No data, split, precision, backbone, or control definition was
  changed; the physical GPU mapping was recorded as `CUDA_VISIBLE_DEVICES=1`.
- [x] R260 generate the repaired reports/checkpoints, fill the new operator packet, append all
  hashes/commands/limitations, set `WAITING_FOR_OPERATOR_REVIEW`, and stop. The packet and
  aggregate reports are in the same run directory; external binaries/plots are present and
  hash-checked on this machine, while the scoped output directory is Git-ignored. No T1 work
  was started.

R210–R260 must not create a T1 manifest, select T1 systems, benchmark T1, or begin any later
architecture/data/training phase. If the frozen-feature/no-anchor combination remains undecodable
after the bounded repair, mark the repair blocked with exact evidence and stop without T0 repeat.

The preceding repair stop was satisfied by the historical packet. Its no-T1 restriction was
superseded for S300–S305 only by the explicit operator authorization recorded in D26; all other
scope and safety restrictions remain active.
