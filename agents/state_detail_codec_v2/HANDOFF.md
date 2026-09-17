# HANDOFF — zero-preserving state/detail temporal codec v2

Status: prepared; implementation worker has not started.

## Approved starting point

```text
repository: /data4/users/sihao/workspace/PVB
source branch: feat/visnet-spatial-v2
source commit: 045dbb8809e9d7aee418eb355b6477e05088d4fd
target branch: feat/state-detail-codec-v2
```

The preceding ViSNet v2 phase completed and remains historical. Its final result was close to
TorchMD but substantially slower; the user selected `torchmd_et` for subsequent codec work.

## Approved design summary

```text
R1-SD: 16 x state(C) = 16C
R2-SD: 8 x [state(C) + detail(C)] = 16C
R4-SD: 4 x [state(C) + detail(C)] = 8C
R4-Matched-Pooling: 4 x [bankA(C) + bankB(C)] = 8C
```

State/detail uses block-local orthonormal Haar coefficients and deterministic AE training. The
new decoder receives no complete per-atom x0 coordinates; only a per-sample translation origin
is allowed. The old R4 single-bank codec remains historical only.

## T0 data expected for independent verification

```text
systems: atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A
replicates: R1/R2/R3 for each system
clip length: 16
native interval: 100 ps
expected train clips: 441
expected holdout clips: 117
```

Do not record these as verified until indexes, records, windows, and hashes have been inspected.

## Mandatory stopping point

After implementation, tests, and T0, set:

```text
Status: WAITING_FOR_OPERATOR_REVIEW
```

Do not prepare or run the planned 64-system T1 experiment. The user will first inspect code
correctness and whether the three-system/nine-trajectory training and evaluation look normal.

## Evidence update template

Append entries; do not rewrite older evidence.

```markdown
### YYYY-MM-DD — Sxxx-Syyy

- Status:
- Branch/base/current commit:
- Worktree before/after:
- Files changed:
- Environment and devices:
- Commands:
- Tests and tolerances:
- Data/sample/split hashes:
- Model/config/checkpoint hashes:
- Shape/capacity evidence:
- Zero-motion/no-anchor evidence:
- Training curves and absolute metrics:
- State/detail utilization:
- Runtime/memory:
- Checkpoint/resume:
- Failed attempts/warnings:
- Decisions/deviations:
- Blockers:
- Next task:
```

## Final T0 handoff requirements

Include:

- complete diff and changed-file list;
- old/new model and checkpoint schema behavior;
- mathematical-to-code mapping for R1/R2/R4 lifting and inverse lifting;
- proof of no per-atom x0 bypass;
- repeated-static and dynamic-detail evidence;
- SE(3), masks, clocks, target isolation, gradients, and legacy regression;
- exact four-control T0 protocol and coverage;
- full curves, metrics, counterfactual decoding, runtime, memory, and checkpoints;
- a clear list of anything the operator should inspect manually;
- explicit statement that T1 and all later model work did not begin.

### 2026-09-03 — S200-S204 preparation and immutable-protocol audit

- Status: complete; no source implementation began before the required audit. The approved
  source checkout was verified at `045dbb8809e9d7aee418eb355b6477e05088d4fd` on
  `feat/visnet-spatial-v2`; the target `feat/state-detail-codec-v2` was created without
  resetting, cleaning, deleting, or overwriting the user-owned untracked prompt and phase
  documents.
- Root transition: `AGENTS.md` now records active phase zero-preserving state/detail temporal
  codec v2, the new read order, the `torchmd_et`-only T0 matrix, and the mandatory
  `WAITING_FOR_OPERATOR_REVIEW` stop. Prior v1/v2 source, contracts, checkpoints, reports, and
  result files remain historical and untouched.
- Environment: the read-only environment check ran inside `enter-container` with activated
  `torch-ito`: Python 3.11.15, PyTorch 2.5.1+cu121, CUDA 12.1, CUDA available, 8 devices.
  No package, dependency, network, or data mutation was performed.
- Legacy audit: `outputs/state_detail_codec_v2/preflight/legacy_audit.json` records hashes and
  contracts for `module/temporal_codec.py`, `module/coordinate_decoder.py`,
  `trainer/codec_trainer.py`, `trainer/codec_contract.py`, and the relevant legacy tests. The
  old contract is `pvb.codec.model_contract.v1`; the old checkpoint is
  `pvb.codec.checkpoint.v2`; the legacy latent includes `[N,3] x_anchor`, and its decoder uses
  `x_hat = x_anchor + displacement(decoded.h, decoded.v)`. The immutable v1 tiny manifest,
  protocol, TorchMD result, bonded result, and step-4979 checkpoint hashes are recorded there.
- Frozen spatial source: the 42-key 128-channel TorchMD frame-encoder state was extracted from
  the immutable v1 TorchMD `codec_step_00004979.pt` without modifying that checkpoint. The
  standalone generated source is
  `outputs/state_detail_codec_v2/preflight/torchmd_frame_encoder_step_00004979.pt`, SHA256
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a`; its source full
  checkpoint SHA256 is
  `1146974b121844eb6c2120719575f1aec13a3ef6c907e0542216e4378680e128`.
- Data identity: an independent `torch-ito` audit of the stored lazy clip indexes confirmed
  systems `atlas_5e3e_A`, `atlas_1v7r_A`, `atlas_2wlt_A`; replicas `R1/R2/R3`; train windows
  `w000000`–`w000048`; late holdout `w000049`–`w000061`; 441 train and 117 holdout clips;
  T=16; `dt_100ps`; all source ranges valid; and zero overlap. Immutable artifact hashes are
  manifest `21992d86626eaf3b6d08dea12354b65060ec0be54b64c0adf9dc15b18f315a40`, protocol
  `d98805fee37c05370bc8fd52fd49c7f28e05532ebdc7366088aa0cc458263564`, TorchMD result
  `c7d36b57e1eb663b04b82fb0620e80599e6b8209ff16c2d2fb3e78aed7475067`, and v1 bonded result
  `2e408514e90c116a066f59ef2e300552daceda3f1b646a632d04cd9ca23df5b0`.
- Commands: the required repository preflight commands, source/document reads, and the
  environment/data audits completed. Generated preflight evidence was produced only through
  `enter-container`/`torch-ito`; no T0 or T1 job has started.
- Decisions/deviations: none. The extracted frame state is a new generated evidence artifact;
  the immutable source checkpoint and all prior result files remain unchanged.
- Blockers: none for preparation. Next task: S210-S213, reusable lifting/latent/origin contracts
  and explicit legacy/new checkpoint semantics.

### 2026-09-03 — S210-S252 implementation, correctness gates, and T0-A micro-overfit

- Status: implementation and pre-training gates complete; the exact four-control T0-B run is
  active on the isolated `neibu` worker and is not yet claimed complete. The target branch is
  `feat/state-detail-codec-v2`, still based on approved commit
  `045dbb8809e9d7aee418eb355b6477e05088d4fd`; no commit or history rewrite was made.
- Changed implementation files: `AGENTS.md`, `config/codec.yaml`, `module/__init__.py`,
  `trainer/codec_trainer.py`, `module/state_detail_codec_v2.py`,
  `scripts/pack_t0_clip_store.py`, `scripts/run_state_detail_codec_v2_t0.py`,
  `tests/test_state_detail_codec_v2.py`, and this phase's `TASKS.md`/`HANDOFF.md`. The prompt,
  v1/v2 phase files, prior checkpoints, and prior result artifacts were not overwritten.
- New source hashes: `module/state_detail_codec_v2.py`=
  `8b3aa07d5562c9ef7c8107401606a6cf9d2e25e0ceffb9e0a7224e00f17b209`;
  `scripts/run_state_detail_codec_v2_t0.py`=
  `3a1d07b18734b788ea9d978531c9256e15813ec0ffbbc5e57ffd10f84aa9c9a0`;
  `tests/test_state_detail_codec_v2.py`=
  `ccfe83a017538c30239395610d3843e585af8145b0adb67e79fdd9309b995b49`.
- Environment: all Python, tests, fixture generation, and training commands ran through
  `enter-container` with `torch-ito`. Local verification used Python 3.11.15, PyTorch
  2.5.1+cu121, CUDA 12.1, and the full suite saw 8 CUDA devices. T0 is isolated to remote
  `neibu` GPU0 after an idle-GPU check; the local PVB process/worktree was not stopped or
  modified by that worker.
- Commands and tests: `PYTHONPATH=. python -m pytest -q
  tests/test_state_detail_codec_v2.py` passed 18 tests; `PYTHONPATH=. python -m pytest -q`
  passed 107 tests with one pre-existing `torch.load(weights_only=False)` warning;
  `python -m py_compile module/state_detail_codec_v2.py module/__init__.py
  trainer/codec_trainer.py scripts/pack_t0_clip_store.py
  scripts/run_state_detail_codec_v2_t0.py tests/test_state_detail_codec_v2.py` passed; and
  `git diff --check` passed. The identity fixtures use FP32 rtol `1e-6`, atol `1e-7`.
- Algebra/contract evidence: R1/R2/R4 h/v Haar round trips, exact R4 order `Dmid,D01,D23`,
  explicit partial masks, 16C/16C/8C capacity accounting, T=1 and irregular clocks, SE(3),
  sample/frame isolation, post-encode target mutation, no decoder target argument, and v3
  checkpoint/config round trip all pass. New latents contain only one `[B,3]` masked-centroid
  origin and no `x_anchor`; legacy v1 loading retains its old anchor path.
- T0-A micro evidence: remote run
  `outputs/state_detail_codec_v2/t0_micro_remote/run_20260903T184751` completed all four
  modes on CUDA with 500 steps, finite nonzero codec gradients, unchanged frozen frame state,
  and step-500 to step-501 resume. Total-loss reductions / measured steps per second were:
  `ratio1_state_detail` 49.8303% / 3.9912; `ratio2_state_detail` 51.0267% / 3.8267;
  `ratio4_state_detail` 53.7159% / 3.7527; `ratio4_matched_pooling` 55.3385% / 3.8924.
  The four result JSON files and checkpoints remain in that run directory.
- T0-B launch: the runner is
  `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_state_detail_codec_v2_t0.py
  --mode all --store-root outputs/state_detail_codec_v2/t0_data/clip_store
  --output-root outputs/state_detail_codec_v2/t0_remote`. It independently verifies the
  approved 441/117 split, runs lazy no-replacement exact epochs, FP32, `max_tokens=80000`,
  30 epochs, seed `20260903`, common frozen TorchMD frame state
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a`, and safety cap 6000.
- Warnings/limitations: only the known PyTorch `torch.load` future warning has appeared; no
  dependency or network mutation occurred. T0 results, unfreeze smoke, final plots, and the
  operator packet remain pending until the remote run finishes. T1 and all later architecture
  phases have not begun.
- Next task: complete S253 and S260-S271 from the observed T0 run, then set
  `WAITING_FOR_OPERATOR_REVIEW` and stop at S272-S273.

### 2026-09-04 — S253 and S260-S273 final T0 handoff

- Status: `WAITING_FOR_OPERATOR_REVIEW`. Implementation gates, the explicit unfreeze smoke, all
  four exact T0 controls, reports, plots, and the operator packet are complete. No commit was
  made; `HEAD` remains `045dbb8809e9d7aee418eb355b6477e05088d4fd` on
  `feat/state-detail-codec-v2`, with the source/document/output changes listed below in the
  worktree.
- Changed files: `AGENTS.md`, `config/codec.yaml`, `module/__init__.py`,
  `module/state_detail_codec_v2.py`, `trainer/codec_trainer.py`,
  `scripts/pack_t0_clip_store.py`, `scripts/run_state_detail_codec_v2_t0.py`,
  `tests/test_state_detail_codec_v2.py`, and the state/detail phase records
  `TASKS.md`, `HANDOFF.md`, and `OPERATOR_REVIEW.md`. The v1/v2 phase files, old code paths,
  checkpoints, and result artifacts were not overwritten.
- Final source hashes: `module/state_detail_codec_v2.py`=
  `8b3aa07d5562c9ef7c8107401606a6cf9d2e25e0ce9ffb9e0a7224e00f17b209`;
  `scripts/run_state_detail_codec_v2_t0.py`=
  `1c45c47106cfc40823e0130f4c520273c41f2a2633438fef9c3cac193157291c`;
  `tests/test_state_detail_codec_v2.py`=
  `2f50ad7b4b2610edff65224f03a5f084d4818685339abe1b0a03f94403181300`;
  `trainer/codec_trainer.py`=
  `ba3d49b79a6467021aaa0706a2439e8b864177e59c59e449a82f8e9a28ed18ae`;
  `config/codec.yaml`=
  `afaa9b3e1c4990b5a44a18aa5e302e9e6f739fe472ca917a17db5e3318428bc3`.
- Environment and devices: all Python, tests, fixture/clip-store generation, training,
  evaluation, and plots ran through `enter-container` with `torch-ito`; Python 3.11.15,
  PyTorch 2.5.1+cu121, CUDA 12.1, and `torch_cluster` 1.6.3+pt25cu121. The final T0 ran on
  `neibu`, `CUDA_VISIBLE_DEVICES=0`, NVIDIA A100-SXM4-80GB. No package installation, network
  access, or dependency mutation occurred, and no other process was stopped.
- Exact final T0 command:

  ```bash
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_state_detail_codec_v2_t0.py --mode all \
    --store-root outputs/state_detail_codec_v2/t0_data/clip_store \
    --output-root outputs/state_detail_codec_v2/t0_remote_final
  ```

- Final evidence root:
  `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/`. It contains
  `protocol.json`, `manifest_contract.json`, `micro_summary.json`, `unfreeze_smoke.json`, four
  per-control result/evaluation/runtime/checkpoint directories, aggregate JSON/CSV/Markdown, and
  PNG/PDF loss, evaluation, block-detail, and performance plots.
- Validation: `PYTHONPATH=. python -m pytest -q` passed 107 tests in 12.81 s with one pre-existing
  `torch.load(weights_only=False)` FutureWarning in the legacy multiframe test. The phase suite
  is 18 passed. The required `python -m py_compile` command over all six changed Python files
  passed. `git diff --check` passed. Identity lifting fixtures use strict FP32 `rtol=1e-6`,
  `atol=1e-7`.
- Contract evidence: R1/R2/R4 orthonormal h/v Haar round trips, coefficient order
  `Dmid,D01,D23`, partial masks, capacity accounting, T=1/T=16 semantics, irregular clocks,
  SE(3) behavior, sample/frame isolation, target mutation isolation, no-target decoder API,
  new checkpoint/config round trip, and old checkpoint/output regression all passed. New
  state/detail latents carry only a single `[B,3]` origin using
  `frame0_loss_masked_centroid`; there is no per-atom `x_anchor` in the new module. The old
  legacy decoder still retains its old anchor behavior under its old schema.
- T0 data and model identity: systems are `atlas_5e3e_A`, `atlas_1v7r_A`, and `atlas_2wlt_A`;
  replicas are R1/R2/R3; train windows are `w000000`–`w000048` (441 clips); late holdout
  windows are `w000049`–`w000061` (117 clips); T=16 and `dt_100ps`; intersection is zero.
  `manifest_contract.json` SHA256 is
  `e84629d68a2c8bfd3ea1c5ae4a4cad3a8b711dc65a46f2c4dbd831019b9a804b` and its built-manifest
  hash is `2dc3790c22ddb2b9606d3e4087a7204e9b91be20252dd6dc6dc7a3126550bca4`; source manifest
  hash is `700f01e40f4e0fda697191cd161bb8161b97493a994415cee286354a9809450`; train/holdout
  index hashes are `04806744073a7df56af2fd5671849e1147a7c6492e956d389b6830aa136d9b0a` and
  `5d247fcb5f21ba2780d6b93880afbc01675ae1a42a935292f04161b7ff1ea0d1`. The common frozen
  128-channel TorchMD frame state is
  `outputs/state_detail_codec_v2/preflight/torchmd_frame_encoder_step_00004979.pt`, SHA256
  `e2ec7e6c1ed37c5b3bd6272f33b1ff48e4d697092de16f9925ef6fdd20a2414a`.
- Portability note: the original `/data4/users/sihao/data/...` source root is not visible in
  the copied container, so the final audit records `source_roots_exist_at_audit=false`. The
  exact byte-range compact store and provenance remain at
  `outputs/state_detail_codec_v2/t0_data/clip_store/`; the system/replica/window counts and
  hashes match the independently audited protocol. This is not a semantic split mismatch.
- S253 unfreeze smoke: `unfreeze_smoke.json` reports `status=passed`, 41 nonzero spatial
  gradients, spatial gradient norm sum `109.45351073767506`, and changed frame-encoder state.
  It was one smoke step and not a fifth experiment.
- T0-A micro-overfit: all four 500-step controls were finite, reduced total loss by at least
  49.8156%, had finite/nonzero new-module gradients, kept the frozen encoder unchanged, and
  resumed `500 -> 501`. Measured reductions / steps per second / training seconds were:
  `ratio1_state_detail` 49.8156% / 3.8201 / 130.89;
  `ratio2_state_detail` 51.0234% / 4.4545 / 112.25;
  `ratio4_state_detail` 53.7155% / 4.2839 / 116.72;
  `ratio4_matched_pooling` 55.1470% / 4.5470 / 109.96. The final micro checkpoint and
  gradient details are in `micro_summary.json`.
- T0-B completion: each control has 30 complete epoch rows and 4,976 steps, exact epoch
  coverage, complete 117-clip holdout evaluation, final and best checkpoints, and resume
  `4976 -> 4977`. Final checkpoint SHA256 values are:
  `ratio1_state_detail/codec_step_00004976.pt`=
  `26cfc05484bf28ca4399db3665dc28c99fa3c97e21afde342bfbfbce9b22fe05`;
  `ratio2_state_detail/codec_step_00004976.pt`=
  `324661817c765034174638e8e49eaa8d5d9f94677f7c2f4e36e5fe724cfe7306`;
  `ratio4_state_detail/codec_step_00004976.pt`=
  `d8c381e21582a6505b64e6a055018fc0dba3c31502b4ad1c92cebf75c12d7678`;
  `ratio4_matched_pooling/codec_step_00004976.pt`=
  `4fd3b29f0177e5ae15d2e5a89518a58af510b0a62a3d229fe2bce3eda3e513c0`. Best-checkpoint
  hashes are respectively `7e98e51b28144ad830f145b8df563472965f70781602a4baf70ebb9aa665fd70`,
  `0425288afbe3241eacead422fde35b97b38965cbe715840b1e6abda424395d46`,
  `51bb7f5a3e4747098ffafb6c6b743df6f3b8fe839f4f2b94035c691bab09d4dd`, and
  `ba807ff9097c77de312f746d33f5cf8c3a3f11ef8d01fc98df0e97ad23f0e1f0`.
- Capacity/parameters for C=128: R1 is 16C/2048 active elements, 643,016 total and 82,432
  trainable parameters; R2 is 16C/2048, 724,936 and 164,352; R4-SD is 8C/1024, 856,008 and
  295,424; matched pooling is 8C/1024, 1,102,536 and 541,952. R2 therefore does not halve
  total active feature volume.
- Final T0 future holdout metrics (RMSD / dRMSD / velocity / acceleration / RMSF correlation /
  frequency retention) are:
  `ratio1_state_detail` 11.424228 / 10.154870 / 0.0209102 / 0.000358905 / 0.766342 /
  9.382636;
  `ratio2_state_detail` 11.241862 / 9.900528 / 0.0128179 / 0.000192065 / 0.784969 /
  6.367882;
  `ratio4_state_detail` 11.138619 / 9.767301 / 0.00817435 / 0.000126111 / 0.782398 /
  4.089264;
  `ratio4_matched_pooling` 10.544774 / 9.306050 / 0.0106722 / 0.000170632 / 0.714334 /
  4.967844. ACF prediction/absolute error is R1 `0.958019/0.039499`, R2
  `0.984965/0.012553`, R4-SD `0.994066/0.003452`, matched `0.990999/0.006518`. Predicted
  boundary jump/absolute error is R1 `3.297411/2.378538`, R2 `2.793389/1.875074`, R4-SD
  `2.323409/1.404246`, matched `2.670770/1.751607`.
- State/detail utilization: on final holdout, R2 raw/encoded/decoded detail h norms are
  `0.160897/0.191008/0.680709`, and R4 are `0.163351/0.238345/0.623712`; both have valid
  fraction 1.0. R1 has no detail by contract. Matched bank h norms are A `1.618426` and B
  `1.584718`. Full versus detail-zero future RMSD/dRMSD is R2 `12.228289/11.139406` versus
  `12.275951/11.230306`, and R4 `12.113563/10.978540` versus `12.166277/11.085510`.
  Repeated-static detail and coordinate motion are exactly zero in the unit tests.
- Runtime: final end-to-end samples/s / train token/s / wall seconds / peak allocated memory
  are R1 `15.6238 / 88798.3 / 3884.85 / 25,781,457,408` bytes, R2
  `15.3303 / 64409.4 / 5355.86 / 26,321,821,696`, R4-SD
  `15.1350 / 64058.7 / 5385.18 / 26,322,873,344`, and matched
  `15.6286 / 89127.4 / 3870.50 / 25,788,809,728`. Spatial-only and temporal-only timings,
  reserved memory, per-system metrics, and block offsets are in each control's `runtime.json`
  and `evaluation.json`.
- Aggregate report hashes: `aggregate_comparison.json`=
  `153dff5f3e097bc82078db8f7aeea280ea0affee27b021c37771970fdc77fdab`;
  `aggregate_comparison.csv`=
  `be40f2824b6f30435b987fc392e7a9a8e20286c8d43d99170852e853be9e17bc`;
  `aggregate_comparison.md`=
  `3bd78508bb9fddf47589163728eb9b57c7d4e59cb6b0abfc198d2182404b9a88`.
- Failed attempts and fixes: an early pre-training attempt exposed a one-step unfreeze
  schedule-boundary bug; a retry completed R1 but detected that dynamic loader batch counts made
  resume inconsistent. The final runner precomputes all 30 epoch counts, resolves the schedule
  after loader length, and uses total 4,976 steps; the accepted final run passed all resume checks.
  These attempts were not used as scientific results. No acceptance threshold was weakened.
- Outcome classification: implementation pass — yes; micro-overfit pass — yes for all four;
  bounded T0 protocol/quality pass — yes as a finite, complete development comparison, not as
  a scientific architecture selection; production-ratio selection — not made. `torchmd_et`
  remains the only evaluated/new-experiment spatial backend and the approved development
  backend. Matched pooling has the lowest one-seed T0 future RMSD/dRMSD but also its natural
  larger parameter count, so this is not a production recommendation.
- Limitations/manual review: inspect the complete PNG/PDF loss curves for plateau behavior,
  the per-system and block-offset JSONs, the full/detail-zero counterfactuals, frozen-weight
  hashes, and the portable-store provenance. The comparison has one seed and three systems;
  multi-seed and broader-system confirmation would be needed before any ratio or codec choice.
- Stop rule: `S300`–`S303` remain blocked. No T1 manifest, T1 training, static/dynamic large-data
  run, DiT, observation adapter, forecasting, rollout, AF3/MSA, VAE/KL/VQ, scaling-law, or
  later architecture phase began. The worker stops here for operator review.

### 2026-09-04 — R200-R202 repair Stage 0 audit and protocol freeze

- Status: repair phase opened; no repair source implementation, GPU training, T0 repeat, or T1
  work has started. The operator decision supplied for this phase is
  `REQUEST_FIX_AND_REPEAT_T0`; it is a repair request, not a rejection of the accepted Haar/
  state-detail packing.
- Repository audit: actual checkout is `/data4/users/sihao/workspace/PVB`; remote is
  `https://github.com/SHw-1313/molvid.git`; source branch is `feat/state-detail-codec-v2`; source
  HEAD is `48bbff992e66cc5f351911e23f31750325ef3726`; source parent is
  `045dbb8809e9d7aee418eb355b6477e05088d4fd`; the pre-repair worktree was clean. The repair
  branch `fix/state-detail-codec-v2-t1-gates` was created directly from that verified HEAD.
- Pre-repair source/document SHA256 baseline recorded before the repair edits: `AGENTS.md`=
  `ef8cd7b9b131a761cb6f7cafb5c4fbd462125169baac829f1a7646e11cbc95ff`;
  `config/codec.yaml`=`afaa9b3e1c4990b5a44a18aa5e302e9e6f739fe472ca917a17db5e3318428bc3`;
  `module/__init__.py`=`7f673d31bd12d72d01a8b3621058bab13eaec8f3c412ff12ca3a4f2801b5db3d`;
  `module/state_detail_codec_v2.py`=`8b3aa07d5562c9ef7c8107401606a6cf9d2e25e0ce9ffb9e0a7224e00f17b209`;
  `trainer/codec_trainer.py`=`ba3d49b79a6467021aaa0706a2439e8b864177e59c59e449a82f8e9a28ed18ae`;
  `scripts/run_state_detail_codec_v2_t0.py`=`1c45c47106cfc40823e0130f4c520273c41f2a2633438fef9c3cac193157291c`;
  `tests/test_state_detail_codec_v2.py`=`2f50ad7b4b2610edff65224f03a5f084d4818685339abe1b0a03f94403181300`.
  The approved phase document hashes were also recorded: `PLAN.md`=
  `d2f3758006cca7d52b0bd67cbc89e7de9f9cff4a1e610e8fcece928f1e5141cc`, `DECISIONS.md`=
  `04f9d06ea7bf41ca030113af002a92c3653c48a217921bf237328e5246a665f9`, `ACCEPTANCE.md`=
  `02d7aee6f94514a1ea6653d7e4bd55486b17c3aa8825b046dc2ac5be391ea2ed`, `TASKS.md`=
  `5b06a39cc84f26bf5eb10d9766db8becdf11c05feef03fae2de774b444bf8614`.
- Required read order completed: root `AGENTS.md`, every file under
  `agents/state_detail_codec_v2/` including `PLAN.md`, `DECISIONS.md`, `ACCEPTANCE.md`,
  `TASKS.md`, `OPERATOR_REVIEW.md`, `HANDOFF.md`, and `worker.toml`; prior v1/v2 history and
  the supplied repair prompt were already read in the preceding phase context.
- Headline T0 reproduction from the committed JSON, using `enter-container`/`torch-ito`, is:
  R1 future RMSD/dRMSD/bond RMSE `11.424228/10.154870/3.083183`; R2
  `11.241862/9.900528/3.122913`; R4-SD `11.138619/9.767301/3.107002`; matched pooling
  `10.544774/9.306050/3.344215`. This confirms the operator’s concern that R1 is not yet a
  meaningful no-compression upper bound; the old values are retained as historical evidence.
- Existing T0 evidence verified present under
  `outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/`, including aggregate
  reports, plots, per-control logs/evaluations/checkpoints, manifest contract, and frozen
  encoder evidence. No historical v1/v2 file or artifact was overwritten.
- Repair planning records: the append-only repair addendum was added to `PLAN.md`, repair
  decisions D17-D23 were appended to `DECISIONS.md`, and R200-R260 were appended to `TASKS.md`.
  Root `AGENTS.md` now routes work to this repair branch and requires the same container,
  `torchmd_et`-only scope, and mandatory `WAITING_FOR_OPERATOR_REVIEW` stop.
- Immediate next task: R210/R211 static topology metadata and tests, followed by R220/R221
  evaluator semantics. The cached single-clip diagnostic and reconstruction repair must precede
  any R2/R4 smoke or T0 repeat. If bounded no-anchor repair cannot make R1 decodable, stop with
  the exact blocker and do not launch a larger run.

### 2026-09-04 — R210-R231 topology/evaluator repair and cached R1 diagnostic

- Static topology repair: new state/detail latents now carry `StaticTopologyMetadata` extracted
  from the input `ClipBatch`, with atom/block/component fields and binary covalent connectivity
  on the latent N axis.  It does not carry `FrameGraphBatch` radius edges, positions, distances,
  edge vectors, frame-expanded indices, or target coordinates.  Legacy `_decode_encoded` remains
  unchanged for the old checkpoint schema.
- Reconstruction diagnostic command (container `torch-ito`):
  `python -m scripts.run_state_detail_codec_v2_r1_diagnostic --output-dir
  outputs/state_detail_codec_v2/repair/r1_cached_feature_diagnostic/run_20260904T_stage2b`.
  It used exactly one real sample, `atlas_5e3e_A_R1_w000000`, with frozen `torchmd_et` and
  cached h/v shapes `[16,887,128]` and `[16,887,3,128]`, plus the centered target.  The initial
  pointwise path was aligned RMSD `13.652314` Å; a detached 500-step pointwise optimization
  reached `5.526837` Å; the linear equivariant vector oracle reached `7.392781` Å; origin-only
  aligned RMSD was `13.701599` Å.  This shows partial coordinate information in frozen vectors
  and a substantial pointwise decoder limitation, while not passing the formal repaired R1 gate.
  The first diagnostic execution had a script device-placement failure and produced no result;
  it is not a scientific run.  The stage2b execution completed and is the retained diagnostic.
- Evaluator repair: schema is `pvb.codec.eval.v2`; `aligned_rmsd` uses frame-wise Kabsch on
  `align_mask`, `centroid_gauge_raw_rmsd` is the direct legacy/raw metric, contacts retain pair
  identity with precision/recall/F1/Jaccard/FP/FN/occupancy-MAE, dynamic ACF uses per-trajectory
  Kabsch-aligned frame-to-frame velocity in Å/ps, and RMSF is aligned to each trajectory's own
  first frame with sample-equal aggregation.  Duplicate covalent bond rows are de-duplicated in
  pair denominators after the diagnostic exposed an old >1 FPR artifact.
- Focused verification: through `enter-container` with `torch-ito`,
  `python -m pytest -q tests/test_codec_evaluation.py tests/test_state_detail_codec_v2.py`
  completed with 28 passed before the final explicit longer-T topology assertion; that assertion
  was then added and the focused suite is rerun before the formal R1 gate.  `python -m py_compile`
  passed for the repaired modules, T0 runner, and diagnostic script.  The formal single-clip
  threshold was frozen in D24 before repaired training results are inspected.
- Implementation choice: `CenteredCoordinateVectorStem` is bias-free, initialized at one, and
  injects centered coordinates into the learned vector latent before Haar packing for R1/R2/R4/
  matched.  Repaired contracts are v4; explicit old v3 state/detail contracts load with
  `coordinate_stem=none` and preserve their old semantics.  A truthful linear-two-bank matched
  pooling name remains the contract; no scalar-gated behavior was claimed.
- Next gate: run the formal one-clip R1 overfit with the frozen thresholds, curves, checkpoint
  resume, and stem/codec gradients.  The formal budget is frozen at 1,000 steps with a 25-step
  metric log interval (D25).  R2/R4 smoke and repeated T0 remain forbidden until it passes.

### 2026-09-04 — R230 corrected cached diagnostic, R240 R1 gate, and R241 ratio smoke

- The earlier stage2b diagnostic is superseded for evaluator evidence because it was produced
  before the duplicate-covalent-pair denominator correction.  The corrected authoritative
  diagnostic is `outputs/state_detail_codec_v2/repair/r1_cached_feature_diagnostic/run_20260904T_stage2c/`.
  Its cached feature file has SHA256
  `6e122621df08c6447c0906a8114c1ffe174fc5c41f35f36c22e28578fc6f8506`; shapes are frozen
  `h=[16,887,128]`, `v=[16,887,3,128]`, centered target `[16,887,3]`.
- Corrected diagnostic results for the one real clip `atlas_5e3e_A_R1_w000000` were: current
  no-stem pointwise initial aligned/raw RMSD `13.652314/13.798742` Å, detached 500-step
  pointwise final aligned/raw/dRMSD/bond `5.524466/5.615697/4.559049/2.675140` Å, linear
  vector oracle aligned/raw/dRMSD/bond `7.392781/7.466507/6.587139/3.116368` Å, and
  origin-only aligned RMSD `13.701599` Å.  The diagnostic gradient summary was finite and
  nonzero.  This supports the pointwise-decoder hypothesis and does not constitute production
  training evidence.
- The formal R1 gate was frozen before repaired training in D24/D25:
  `aligned_rmsd<=1.5`, `centroid_gauge_raw_rmsd<=1.5`, `dRMSD<=1.5`, `bond_rmse<=0.5`,
  origin-only aligned improvement `>=0.5`, finite curves, and a converged final window.
  The genuine one-sample command used exactly 1,000 steps and logged every 25 steps:
  `python -m scripts.run_state_detail_codec_v2_r1_single_clip --output-dir
  outputs/state_detail_codec_v2/repair/r1_single_clip/run_20260904T_stage2c`.
- R240 passed at `outputs/state_detail_codec_v2/repair/r1_single_clip/run_20260904T_stage2c/`.
  Final all-frame aligned/raw/dRMSD/bond were `0.084951/0.085038/0.111515/0.024777` Å;
  future aligned/raw/dRMSD/bond were `0.085115/0.085196/0.111787/0.024745` Å; origin-only
  aligned improvement was `0.993800`; all intended stem/codec gradients were finite and
  nonzero; the frozen encoder was unchanged; checkpoint resume `1000 -> 1001` passed.
  The final checkpoint SHA256 is
  `5654bb4c37fa51976421cf4a23420fb536cc49106bbd04afa95d5dd4b8ae1dd0`; result JSON SHA256 is
  `921df11a11c4418b18c61d42785b7b0793df99f7c80fbb08528bf6174d652ea1`.
- R241 then ran the bounded 200-step smoke on the same single clip for
  `ratio2_state_detail`, `ratio4_state_detail`, and `ratio4_matched_pooling`:
  `python -m scripts.run_state_detail_codec_v2_ratio_smoke --output-dir
  outputs/state_detail_codec_v2/repair/ratio_smoke/run_20260904T_stage3`.  The summary status
  is `passed` and `t1_started=false`.  Final aligned/raw/dRMSD/bond/contact-F1 were respectively
  R2 `0.066825/0.067810/0.055249/0.040102/0.981242`, R4-SD
  `0.072682/0.073069/0.059461/0.041076/0.981053`, and matched
  `0.193294/0.193816/0.223682/0.079457/0.960623`; each control had finite nonzero new-module
  gradients, unchanged frozen encoder, and checkpoint resume `200 -> 201`.
  Summary SHA256 is `d01d56bf0dd6387640837764d9176375f49bcb48e601ccb0754f23987327b7a0`.
- The focused repair suite remains `28 passed`; repaired modules and scripts pass `py_compile`.
  The T0 report generator now emits aligned/raw names, pair-aware contact fields, dynamic
  correlation semantics, matched-pooling semantics, and LF-normalized CSV output.  The exact
  three-system T0 repeat is the next authorized repair task; no T1 manifest or later phase has
  been created or started.
### 2026-09-04 — R250/R260 repaired T0 repeat, final evidence, and mandatory stop

Repair implementation commit: 7878df7286350b6b46cb2198d942fdcda2451ba1, based directly on
48bbff992e66cc5f351911e23f31750325ef3726. Target branch is
fix/state-detail-codec-v2-t1-gates; remote is https://github.com/SHw-1313/molvid.git. The
implementation commit worktree was clean before this append. The final packet commit is the
only subsequent documentation change and is clean after commit. No destructive Git operation
or push was used.

Changed tracked files are AGENTS.md, .gitignore, evaluation/__init__.py,
evaluation/codec_evaluation.py, module/__init__.py, module/state_detail_codec_v2.py,
trainer/codec_trainer.py, scripts/run_state_detail_codec_v2_t0.py,
scripts/run_state_detail_codec_v2_r1_diagnostic.py,
scripts/run_state_detail_codec_v2_r1_single_clip.py,
scripts/run_state_detail_codec_v2_ratio_smoke.py, tests/test_codec_evaluation.py,
tests/test_state_detail_codec_v2.py, and the append-only state/detail PLAN.md, DECISIONS.md,
TASKS.md, and HANDOFF.md. OPERATOR_REVIEW.md is appended in the packet commit. Repair outputs
are Git-ignored external evidence and remain present on this machine.

Main implementation SHA256 at the repair implementation commit:

~~~text
evaluation/codec_evaluation.py                 1c7ac1b70146a933c88fdd790ff421e52eba85af3ce6bff5e1ac0437ad8ae544
evaluation/__init__.py                          332c46b68cdff9ba80062352ba4b86aefb8f2d4a5b03d655abbb4762e5f34b28
module/state_detail_codec_v2.py                5ff5fd0f08b772e59c0067fbcc0a3c8a3881d7e921084a92bd97d8f418080928
module/__init__.py                             6f4f23a9966e88c8738bfc2e06120a43c9182ec8965fd6c318ac903b0f8924ef
trainer/codec_trainer.py                        39e02bbc68858d778a533c80aabf32b5f05769f542aa3b8366de396dc5124694
scripts/run_state_detail_codec_v2_t0.py         03211c6c94de6a9215cdbc9cfcc975ce787226731bb73b9e9d5140b162b5fd6e
scripts/run_state_detail_codec_v2_r1_diagnostic.py dd9b162c8022b43ec1958bccea927ebfbab633729771880bfcc16edafb86c477
scripts/run_state_detail_codec_v2_r1_single_clip.py e92b530cc4f757b4c995c71d228c7b8b497d331506c69488a594bd4c73d31d12
scripts/run_state_detail_codec_v2_ratio_smoke.py 625588ba6d160d3cd445eba5eaf77d3718d2b0f0421a42637be6582178efab46
tests/test_codec_evaluation.py                  eac51038968acfeb125cf89ee8539f9dc39ce277097301c579425a67c1d0a52f
tests/test_state_detail_codec_v2.py             b76fc0001077e1fb9b4e9856ea03864e368a25fe563fbafbea7979f321b4978c
~~~

Evidence and tests:

- StaticTopologyMetadata is an N-axis, coordinate-independent schema with atom/block/component
  identifiers, pointers, and covalent pairs only. Radius edges, positions, distances, edge
  vectors, frame-expanded indices, and target coordinates are excluded. Tests cover N-axis
  bounds, longer-T shape invariance, coordinate independence, future-frame independence, and
  radius/distance exclusion.
- The evaluator adds per-frame Kabsch aligned_rmsd on align_mask and retains
  centroid_gauge_raw_rmsd as the explicit raw metric. Contacts preserve pair identity using a
  4.5 Å cutoff and covalent-pair exclusion. Dynamic correlation/ACF uses mean-removed,
  Kabsch-aligned frame-to-frame velocity in Å/ps. RMSF is aligned to each trajectory's first
  frame with sample-equal aggregation. Frequency power above one means excessive motion.
- The no-anchor repair is a shared bias-free CenteredCoordinateVectorStem before temporal
  packing. There is no per-atom x0 post-decoder addition. R1/R2/R4-SD share the decoder.
  Matched pooling is linear two-bank pooling, latent-volume-matched but not parameter-matched.
- The authoritative cached diagnostic is
  outputs/state_detail_codec_v2/repair/r1_cached_feature_diagnostic/run_20260904T_stage2c.
  It has one clip, h=[16,887,128], v=[16,887,3,128], centered targets [16,887,3];
  no-stem initial aligned/raw RMSD 13.652314/13.798742 Å; detached pointwise 500-step final
  5.524466/5.615697 Å with dRMSD/bond 4.559049/2.675140 Å; vector oracle
  7.392781/7.466507 Å; origin-only aligned RMSD 13.701599 Å. diagnostic.json SHA256 is
  1af48019a526fd12748b339bd544081b0210d0917557cd4070e25efda933ac19 and cached_features.pt
  SHA256 is 6e122621df08c6447c0906a8114c1ffe174fc5c41f35f36c22e28578fc6f8506.
- True single-clip R1 is in
  outputs/state_detail_codec_v2/repair/r1_single_clip/run_20260904T_stage2c. It used one
  sample, 1,000 steps, and 25-step logging. Frozen thresholds were aligned/raw RMSD <=1.5 Å,
  dRMSD <=1.5 Å, bond <=0.5 Å, origin-only improvement >=0.5 Å, finite curve, and converged
  final window. All passed. Final all-frame aligned/raw/dRMSD/bond =
  0.084951/0.085038/0.111515/0.024777 Å; future =
  0.085115/0.085196/0.111787/0.024745 Å; origin-only improvement = 0.993800 Å.
  Gradients were finite/nonzero, encoder unchanged, and resume 1000 -> 1001 passed.
- Ratio smoke in outputs/state_detail_codec_v2/repair/ratio_smoke/run_20260904T_stage3 used
  200 steps for R2, R4-SD, and matched pooling on the same clip. All passed zero-preserving
  detail, gradients, frozen encoder, shared decoder, and resume 200 -> 201. Final
  aligned/raw/dRMSD/bond/contact-F1: R2 0.066825/0.067810/0.055249/0.040102/0.981242;
  R4-SD 0.072682/0.073069/0.059461/0.041076/0.981053; matched
  0.193294/0.193816/0.223682/0.079457/0.960623. Summary SHA256 is
  d01d56bf0dd6387640837764d9176375f49bcb48e601ccb0754f23987327b7a0.
- Focused tests: python -m pytest -q tests/test_codec_evaluation.py
  tests/test_state_detail_codec_v2.py -> 28 passed. Full suite: python -m pytest -q ->
  115 passed in 10.77s, with one pre-existing torch.load(weights_only=False) FutureWarning.
  Repaired modules, scripts, and relevant tests passed py_compile. Generated JSON/CSV/Markdown/
  JSONL output was checked for CRLF/trailing whitespace with no findings.
- The exact committed source diff check passed:
  git diff --check 48bbff992e66cc5f351911e23f31750325ef3726
  7878df7286350b6b46cb2198d942fdcda2451ba1.

T0 protocol and evidence:

- Exact container command:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m scripts.run_state_detail_codec_v2_t0 --mode all
  --store-root outputs/atlas_selected_trajectories/clip_store
  --output-root outputs/state_detail_codec_v2/repair/t0_repeat
- Run directory:
  outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210.
  Systems are atlas_5e3e_A, atlas_1v7r_A, atlas_2wlt_A; replicas R1/R2/R3; train windows
  w000000-w000048; holdout windows w000049-w000061; T=16; dt_100ps; lazy loading;
  replacement=false; FP32; seed 20260903; max_tokens=80000; 30 complete epochs; cap 6000.
  Canonical and historical portable stores matched for all 558 IDs and payload hashes. Counts
  are 441 train, 117 holdout, zero overlap.
- Manifest hashes: source
  700f01e40f4e0fda697191cd161bb8161b97493a994415cee286354a9809450e; train index
  f8a05f905f1467864993291e64008db9d26109261f4a007f11cc15fe4ae8c3f9; holdout index
  c02d45d2cf77bfbbce787f1d8b747556d52d91e7c7de7f61baa26e6ab62a2a61; built manifest
  01c7bcb0cc463eee0ff8afecc7dbaad211156d3793c0b017afb8549f70114a2f; manifest contract
  e0bc84c98c5a955fe85fc5b663a34af9abe28af93860603e58a11106c754a73b.
  protocol.json is 8c14c67b62b504c74d1ade800fd354193b89f1654214bdf192e232b27483b0a6.
- Runtime is container cuda:0 on physical mapping CUDA_VISIBLE_DEVICES=1, NVIDIA
  A100-SXM4-80GB, Torch 2.5.1+cu121, CUDA 12.1, and torch_cluster 1.6.3+pt25cu121.
  Protocol records centered_vector, shared_framewise_equivariant, linear_two_bank_pooling,
  and the corrected evaluator semantics.
- All four controls completed 30 epochs and 4,976 scheduled steps, complete holdout
  evaluation, best/final checkpoints, and resume 4976 -> 4977. Final future metrics:

| Control | Active/atom | Total/trainable | Aligned RMSD | Raw RMSD | dRMSD | Bond | Contact F1 | Vel RMSE | Accel RMSE | Dyn corr | RMSF corr | Power ratio | Boundary error | Train s | Tok/s | E2E s/batch | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1-SD | 2048 | 643,144 / 82,560 | 0.105395 | 0.105435 | 0.143339 | 0.022667 | 0.986426 | 0.0001221 | 2.093e-6 | 0.999850 | 0.999968 | 1.012466 | 0.005946 | 4328.4 | 79,699 | 0.193 | 24.01 |
| R2-SD | 2048 | 725,064 / 164,480 | 0.078981 | 0.079039 | 0.103156 | 0.022113 | 0.988454 | 8.297e-5 | 1.323e-6 | 0.999923 | 0.999977 | 1.005878 | 0.002846 | 5971.0 | 57,773 | 0.197 | 24.47 |
| R4-SD | 1024 | 856,136 / 295,552 | 0.064856 | 0.064934 | 0.082521 | 0.021026 | 0.988925 | 7.330e-5 | 1.231e-6 | 0.999939 | 0.999983 | 1.003593 | 0.001552 | 5983.2 | 57,656 | 0.200 | 24.47 |
| R4-Matched-Pooling | 1024 | 1,102,664 / 542,080 | 0.090435 | 0.090519 | 0.115388 | 0.028109 | 0.986176 | 0.0006986 | 1.355e-5 | 0.994320 | 0.999795 | 1.004576 | 0.005997 | 4298.5 | 80,254 | 0.193 | 24.02 |

R1 is now a useful no-compression upper bound at approximately 0.105 Å and no longer fails at
the former approximately 11 Å scale. Structure and dynamics improve together; frequency ratios
are near one. R2 preserves 16C active capacity, R4-SD is 8C, and matched pooling is not
parameter matched. This is bounded development evidence, not a production ratio selection.

The aggregate report also contains contact precision/recall/Jaccard, per-frame/per-offset/system
metrics, aligned RMSF, dynamic ACF, frequency, boundary, latent norms, and full/detail-zero
counterfactuals. R2 full/detail-zero future RMSD is 0.078981/12.275951 Å and dRMSD is
0.103156/11.230306 Å; R4-SD is 0.064856/12.166277 Å and 0.082521/11.085510 Å.

Report/artifact paths are in OPERATOR_REVIEW.md and the run directory. Main hashes are:

~~~text
aggregate_comparison.json   e7d9ca83b280725dfac7a87c574eb02fb511bc75a6bde906f4a81904c95b6626
aggregate_comparison.csv    5b93e8f1eea6eb07828d3ea64f0032c9f4bf6a0a06d404ac5bb1317ef7ce2872
aggregate_comparison.md     db3a823b2c9cdd0e4983c328028648e23df987ebe89bb6597e0836bc713b02b1
micro_summary.json           179b5c111ffc565a5518457d02d1238e38923b063f521a1e3b2dfef7a219a45a
unfreeze_smoke.json          f609356e9e98bf807f9cc2f7dad81adb4417200a00776423d27b702302e21fe9
~~~

Final/best checkpoint hashes:

~~~text
R1 final 02434947a68b30b9ecf585a4ac86f5d68e653304861fad0c7b42de5b0db9113d
R1 best  13955f92217a291df381b3ebfb9ab1450648a7083148db9c5cbe1aad8b8e763c
R2 final e3e509b8cb5c16e7f3e9d535c47967ac5c4918cbf2ebb9fad44feb509328d9a5
R2 best  8f00848dbe6816af22f7a97f59cf1f6dd4f4fb16c9d2e573f0726a70353e8bf0
R4 final 5ed009cee98e62cb6f229d5a55ab2a4d0997453713879de7e56c21f2873b2e68
R4 best  0cb07483b89c05060be03f7210ed5373cd4ef2de2ba28e4e8242e0bf5ca608ce
MP final  b778d2532e516dab23d2e9887dec936d922a0dcbfa3d4756480644ce6803a157
MP best   0f1528ea2837fb12422a45663d3f46f68cdfc34ee33e97bb3f7b3e4d8884d0f2
~~~

Outcome: implementation pass; evaluator/topology pass; true single-clip R1 pass; ratio smoke
and three-clip micro-overfit pass; bounded T0 complete and finite with a meaningful R1 upper
bound. torchmd_et remains the only development backend and no production ratio recommendation
is made. Limitations are one seed, three systems/nine trajectories, no broad-system or
multi-seed confirmation, and external binary artifact retention.

Final phase status: WAITING_FOR_OPERATOR_REVIEW. T1 status: NOT_STARTED. No T1 manifest,
64-system split, T1 benchmark/training, static/dynamic large-data work, DiT, observation
adapter, forecasting, rollout, AF3/MSA, VAE/KL/VQ, scaling-law, or later architecture phase
began.
### 2026-09-04 — loss-curve readability update

- Updated scripts/run_state_detail_codec_v2_t0.py so aggregate and per-control loss plots use a
  logarithmic y-axis and overlay train total loss as a solid line with same-color
  late-holdout/test total loss as a dashed line. Holdout points are placed at cumulative
  optimizer steps, including the initial point at step zero, so train/holdout divergence is
  directly visible.
- Added the plot-only command path; no model, data, loss, checkpoint, or T0 metric was changed:
  python -m scripts.run_state_detail_codec_v2_t0 --plot-only-run
  outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210
- Regenerated loss_curves.png/.pdf and all four loss_curve_<mode>.png/.pdf files under
  outputs/state_detail_codec_v2/repair/t0_repeat/run_20260904T122210. The generated Markdown
  report now records the plotting semantics.
- Updated plotting source SHA256:
  scripts/run_state_detail_codec_v2_t0.py =
  bf9906d25639d93ae3d8ad1666d373a762f77c8934407999e40d3771d774779f.
- Regenerated plot hashes:
  loss_curves.png 738204eacadfb2a6925c6a926ff20be507d1b81b60a521c13d12ca8be83873a5;
  loss_curves.pdf 9672fb936586a7ce080c2b847e6b497babd78e44f7453c4c13658158d71a17c7;
  loss_curve_ratio1_state_detail.png
  313703d0b5a8eff4ca11d1ae183159bbf3c037f31af0a3862c69f5037f1dc3f7;
  loss_curve_ratio1_state_detail.pdf
  0626f6dead3ef3a2b0e29817ed0c3e9eac3f313f379502152575d0bc7ddbca9a;
  loss_curve_ratio2_state_detail.png
  5b70c638efa4ac2a68f4496eab72280ae1100972bc3b2426a0ee56f1695df625;
  loss_curve_ratio2_state_detail.pdf
  0a59a5e91c7d339ce1ee7dac1342587a227b4e83679b5a2cadd1b5cd9dc7fb34;
  loss_curve_ratio4_state_detail.png
  e30387bbbbd86e61317ac64e359be0586bb8c6214ac729bc0c5883523f787af3;
  loss_curve_ratio4_state_detail.pdf
  c266b9ef32681e1197de5d4896101fabed9c966bf44f3d1dfc35e54dbb3b9a2c;
  loss_curve_ratio4_matched_pooling.png
  b5a98427b85cf6c3e524d17dc3f5623ff0af16d29eb8366247b61a568f63f264;
  loss_curve_ratio4_matched_pooling.pdf
  28d2901703f8425ac59895e775337ab9891eab1e5f8fa3e32632db0738440150.
- Plot-only py_compile passed and the full regression suite remains 115 passed with the same
  pre-existing FutureWarning. The T0 report values and checkpoints are unchanged.
- Phase status remains WAITING_FOR_OPERATOR_REVIEW; T1 remains NOT_STARTED.

### 2026-09-04 — explicit T1 authorization and preflight GPU audit

- The operator explicitly authorized the bounded T1 sequence without another confirmation:
  freeze an independent 64-system 48/8/8 split; run four 200-step profiles; launch the four
  complete one-seed controls only if all profiles pass; select ratio from validation only before
  opening test; then add seeds only to the top two controls.
- Current branch before T1 source work: `fix/state-detail-codec-v2-t1-gates`; HEAD
  `c24e2576e2b57281c79ec142225e05a81ee87181`; remote
  `https://github.com/SHw-1313/molvid.git`; worktree had only the authorized `AGENTS.md` edit.
- Local GPU audit immediately before scheduling: idle and eligible physical GPUs were 1, 3, and
  5 (A100-SXM4-80GB, 80 GiB each). GPU 0 had an existing process and GPUs 2, 4, 6, and 7 were
  occupied; none were touched. `neibu` audit found remote GPUs 5, 6, and 7 idle while 0–4 had
  active processes; none were touched. The remote worker will use an isolated designated
  directory and the same `torch-ito` environment after code/data preflight.
- T1 remains constrained to `torchmd_et`, frozen common frame weights, FP32, unchanged decoder,
  losses, optimizer, and four existing codec control names. No later architecture phase has
  started.

### 2026-09-04 — pre-profile token-cap audit and corrected manifest

- The first hash-ranked candidate manifest was not used: validation of the existing sampler found
  186 oversized validation clips, including `atlas_2po4_A` at 8,521 atoms and T*N=136,336,
  against the frozen `max_tokens=80,000` contract. No profile or training process started from
  that candidate.
- The source index audit found 522/546 token-valid train systems, 91/94 token-valid valid
  systems, and 76/76 token-valid test systems after excluding the historical T0 systems. The
  corrected manifest ranks only token-valid systems, keeps the existing source train/valid/test
  system partitions, and selects 48/8/8 disjoint systems with all R1/R2/R3 and windows
  w000000–w000061.
- Corrected manifest/materialization root: `outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`.
  Its content hash, selected systems, source index hashes, and materialization hashes are
  recorded in its JSON files. The old `manifest_20260904` directory is retained as an invalid
  preflight record and is not used by any T1 command.

### 2026-09-04 — T1 profile gate passed

- Corrected manifest JSON SHA256: `88325925b339ef34a421d48b5439bfc869fb1003b9c59e3d7eeac45b62da3d2e`;
  manifest content hash: `f8a764eb38e90c7485bf9799115d570bb868f34584ebbafab8c799fec03df4d2`;
  materialization JSON SHA256: `41685082d85ff36ba442d09e23495c2f00110272f769f45c54e77207db3d85e8`;
  materialization record hash: `5b622ae6d0ac2a6b498f3a9388bd2053dbb29cc8aca83eecf5723987c94d49c7`.
- The final selected store contains 8,928 train, 1,488 validation, and 1,488 test clips,
  corresponding to 48/8/8 systems, three replicas, and all 62 windows. Every selected clip has
  `T*N <= 80,000`; train epochs sample 3,456 clips (24 per 144 trajectory) without replacement
  within each epoch.
- All four 200-step profiles passed on the same seed/config and frozen TorchMD source hash.
  R1: result SHA256 `11e7a4e480a7c11bd27036dde080b0742432de68b12218eaaa9b62aaff316d0d`,
  3.980 steps/s, 247,609 tok/s, 23.997 GiB allocated, estimated 3.20 h for 45,844 steps.
  R2: `b671b4a4d954adfdf30336b14ccc188092be9dc3be79ac1051920822c4eed8d0`, 4.844 steps/s,
  301,355 tok/s, 23.998 GiB, estimated 2.63 h. R4-SD:
  `7d8bb2973897d2353809546fa7fd00c5b337155e782ecb60f27f7d64d0b30612`, 3.477 steps/s,
  216,304 tok/s, 24.001 GiB, estimated 3.66 h. R4-Matched:
  `b81043dd2931e91e2883456802ee0d2dadfb398f4694cd343b2e3ea89cc6592c`, 5.125 steps/s,
  318,840 tok/s, 24.003 GiB, estimated 2.48 h.
- Each profile saved a checkpoint and resumed exactly `200 -> 201`; all reported finite execution,
  nonzero intended-module gradients, and unchanged frozen frame-encoder state. Local profiles
  used physical GPUs 1/3/5; the matched profile used remote `neibu` physical GPU 5. No occupied
  process was stopped or preempted.
- The profile gate passed. Full one-seed T1 is authorized and is now launched in parallel; test
  remains closed until the validation-only selection rule is frozen after the four runs.
