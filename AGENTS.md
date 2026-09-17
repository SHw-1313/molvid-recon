# molvid project instructions

## Active implementation phase

The active phase is **R2/R4 state-detail latent DiT probe v1**.

Before editing code, read the immutable historical context and then the current prompt and phase
documents in this exact order:

1. root `AGENTS.md`;
2. the v1, v2, repair-prompt, and state/detail codec files named by the preceding root phase
   instructions, as read-only historical context;
3. `LUNA_DIT_STATE_DETAIL_PROBE_V1_PROMPT_260907.md`;
4. `agents/dit_state_detail_probe_v1/PLAN.md`;
5. `agents/dit_state_detail_probe_v1/DECISIONS.md`;
6. `agents/dit_state_detail_probe_v1/ACCEPTANCE.md`;
7. `agents/dit_state_detail_probe_v1/TASKS.md`;
8. `agents/dit_state_detail_probe_v1/HANDOFF.md`;
9. `agents/dit_state_detail_probe_v1/OPERATOR_REVIEW.md`.

The DiT phase PLAN, prepared DECISIONS, and ACCEPTANCE are binding. Preceding v1/v2/state-detail
phase files, source implementations, active T1 processes, T1 branches, manifests, checkpoints,
reports, and result artifacts are read-only historical or active-experiment evidence and must
remain reproducible. The active codec T1 checkout is `/data4/users/sihao/workspace/PVB`; do not
edit or run DiT work there. The dedicated target is
`/data4/users/sihao/workspace/molvid-dit-state-detail-probe-v1` on
`feat/dit-state-detail-probe-v1`, based on audited source commit
`23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541`.

Older cumulative files under `agents/PLAN.md`, `agents/TASKS.md`, `agents/DECISIONS.md`, and
`agents/HANDOFF.md` are historical context and must not be overwritten or repurposed.

## Repository and environment rules

- Keep all DiT work in the dedicated target branch/worktree. Preserve unrelated user changes.
- Run every Python command, test, fixture generation, smoke, evaluation, and plotting command
  only through `enter-container` with `conda activate torch-ito`.
- Do not install packages, upgrade dependencies, use arbitrary network access, download data,
  perform destructive Git operations, or commit generated checkpoints, large datasets, or binary
  plots unless explicitly requested.
- Serialize GPU-heavy commands and use only an audited idle GPU. Never kill, suspend, migrate,
  or preempt another process.
- Do not silently rebase onto changed codec semantics. If the audited source branch advances,
  record and inspect the difference before proceeding.

## Scope discipline

Only `ratio2_state_detail` and `ratio4_state_detail` are DiT candidates. `ratio1_state_detail`
and `ratio4_matched_pooling` are historical codec controls and are not DiT candidates. The codec,
TorchMD frame encoder, centered-vector stem, coordinate decoder, codec trainer/evaluator semantics,
and all T1 scripts are frozen and must not be rewritten.

This phase is limited to the focused latent adapter, rectified-flow objective, shared molecular
DiT, observation-mask plumbing, trainer/checkpoint contracts, evaluation/report plumbing,
correctness tests, and one bounded smoke per candidate. Use a separate output root under
`outputs/dit_state_detail_probe_v1/`; never write under active T1 outputs.

Do not start the real R2/R4 scientific pilot, full-T1-data DiT training, T1 test access, scaling
experiments, static mixing, AF3/MSA conditioning, VAE/KL/VQ, long rollout, H=2, energy guidance,
ensemble control, or any later architecture work. Do not rank R2 versus R4 from smoke results.
The final status must be `WAITING_FOR_T1_AND_OPERATOR_REVIEW`, followed by a stop.

## Documentation ownership

- This root transition is authoritative only on the dedicated DiT branch; do not edit root
  `AGENTS.md` in the active T1 checkout.
- Preceding phase documents and artifacts remain immutable historical evidence.
- `agents/dit_state_detail_probe_v1/PLAN.md`, `DECISIONS.md`, and `ACCEPTANCE.md` are prepared
  authority; do not rewrite them. Append DECISIONS only for real implementation-forced facts.
- Update `agents/dit_state_detail_probe_v1/TASKS.md` only when evidence exists.
- Append dated commands, tests, hashes, metrics, blockers, and next tasks to `HANDOFF.md`.
- Fill `OPERATOR_REVIEW.md` with observed implementation/CPU/smoke evidence at the end.

Do not weaken acceptance criteria, add compatibility fallbacks, or continue after the required
operator-review stop.

## Active pilot transition — R2/R4 state-detail latent DiT T1 pilot v1

This separate branch/worktree is explicitly authorized for the matched 64-system pilot described
in `agents/dit_state_detail_pilot_v1/PILOT_PROTOCOL.md`. It is based on repair commit
`28a499f4c03deb4647a6468a5c477bd8eda8d62f` from `feat/dit-state-detail-probe-v1`; the repair
branch remains independently reviewable and must not be modified from this worktree.

- The frozen manifest is `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`.
- Only the manifest's train and validation materializations may be opened. Never construct,
  read, hash, or evaluate `clip_store/test`; no test split access is authorized tonight.
- Only the completed, SHA256-verified T1 `ratio2_state_detail` and `ratio4_state_detail`
  checkpoints named in the pilot protocol may be loaded. Random, T0, incomplete, or guessed
  codec checkpoints are prohibited.
- The frozen codec and frame encoder stay in eval mode and frozen. The shared DiT width/depth/
  heads, optimizer policy, ordered schedule, H=4/H=8 schedule, and validation protocol are
  identical for R2 and R4. No H=2, H=0 training, DDP, static mixing, AF3/MSA, architecture
  changes, or test evaluation.
- Real-data pilot outputs belong under `outputs/dit_state_detail_pilot_v1/`; do not write under
  active T1 output roots or commit checkpoints, datasets, caches, or binary plots.
- Every Python, test, profile, training, evaluation, and plotting command still runs through
  `enter-container` with `conda activate torch-ito`. Use only audited idle GPUs and never
  preempt another process.
- Freeze one common step budget in `pilot_budget.json` only after both 200-step profiles pass;
  require the common budget to be at least 1000 steps. Validation RF loss selects checkpoints;
  do not declare an R2/R4 winner.
- The pilot branch's final status is `WAITING_FOR_OPERATOR_REVIEW`; test remains sealed.

## Active transition — DiT pilot diagnostics / factorized backend v2, 2026-09-09

The operator has authorized two independent implementation lanes. This transition supersedes previous active-phase task order, target worktree, output-root and stop instructions only for these lanes. All historical artifacts, frozen codec semantics, data-access boundaries and environment safety remain in force.

- Session A stays on `exp/dit-state-detail-t1-pilot-v1` in the existing pilot worktree. It must not create or switch branches. Read `LUNA_SESSION_A_DIT_DIAGNOSTICS_PROMPT_260909.md`, `agents/dit_parallel_v2/CONTRACT.md`, and every instruction file in `agents/dit_pilot_diagnostics_v1/`. Implement and run bounded checkpoint diagnostics; no training or backend edits.
- Session B uses the separate `perf/dit-factorized-backend-v2` branch/worktree from audited commit `a22f60c4ffd1f502ab300352a00eafe841b1f8d2`. Read `LUNA_SESSION_B_DIT_BACKEND_PROMPT_260909.md`, the shared CONTRACT, and every instruction file in `agents/dit_factorized_backend_v2/`. Implement numerical-equivalent acceleration and bounded CUDA tests/profiles; no scientific retraining or evaluator changes.
- New lane-local AGENTS/PLAN/ACCEPTANCE/TASKS/HANDOFF govern current work. Old `agents/AGENTS.md` codec-v1 task descriptions, causal-tokenization rules, CPU-first prescriptions and legacy phase stops are historical, not instructions to restart those experiments. Keep 16-frame block-local state/detail tokenization, R2/R4, H4/H8 and current model contracts.
- All Python and scientific numeric work uses the configured `enter-container` / `torch-ito` environment. New numerical acceptance tests and real-data model paths run on CUDA. Pure metadata, I/O and reporting may use CPU. A skipped CUDA test is not evidence.
- A and B own disjoint source files as specified in CONTRACT. Add only your own local root transition; do not edit another worktree. Read-only artifacts may be referenced by absolute path. Never open test clip payloads, overwrite old outputs, terminate unrelated processes or bypass permission failures.
- Implement first, then targeted CUDA checks, real-clip smoke, relevant regression and bounded evaluation/profile. Do not run the full suite as a prerequisite to implementation. Do not silently change physics, latent statistics, optimization objectives or sampling priors during a performance refactor.
- Local commits of owned code and small evidence are authorized. No automatic push, merge, PR or long training. Stop at `A_DIAGNOSTICS_READY_FOR_REVIEW` / `B_BACKEND_READY_FOR_REVIEW`, or report incomplete evidence honestly.
- `agents/dit_parallel_v2/NEXT_STAGE.md` is a future experiment design, not permission to start it. Both Gaussian-source and conditional-source arms retain the same observation history; no H0 training is authorized in this task.

## Active transition — DiT source A/B v1, 2026-09-10

The operator has authorized the next stage on the existing `perf/dit-factorized-backend-v2` worktree. Read `LUNA_DIT_SOURCE_AB_V1_PROMPT_260910.md` and the instruction files under `agents/dit_source_ab_v1/` before implementation. This transition supersedes previous active-phase task order, lane ownership restrictions, no-training stops and old pilot step limits only for the work explicitly defined here.

- Stay on B; do not create a branch/worktree or merge A wholesale. Import only the specified diagnostics files from the fixed A commit. A and all historical experiment artifacts remain read-only.
- Integration fixes, targeted CUDA checks, source-center validation, bounded R4 Gaussian/conditional-source training, evaluation and local commits are authorized. Once prerequisite correctness checks pass, continue without stopping at the previous A/B review states.
- Preserve frozen codec/geometry/decoder/statistics, 16-frame tokenization, R4, H4/H8 and train/validation-only data access. The new source experiment is defined by `EXPERIMENT.md`; no additional architecture or loss changes are authorized.
- Use enter-container/torch-ito, actual CUDA for numerical work, and audited idle GPUs. Metadata and I/O may use CPU; numerical CPU fallback is prohibited.
- Implement before testing; targeted CUDA checks precede real smoke and directly affected regression. Do not start with full-repository pytest.
- New artifacts belong under `outputs/dit_source_ab_v1/`. Do not overwrite old outputs or open test payloads. Do not push, create PRs, merge/rebase or start extra coding agents.
- Follow the new budget and final-status rules. Finish at `SOURCE_AB_V1_COMPLETE_FOR_REVIEW`, `SOURCE_AB_V1_PARTIAL_BUDGET` or `SOURCE_AB_V1_BLOCKED` with evidence.

## Active transition — Session A source checkpoint reassessment v2, 2026-09-14

This transition is the current user-authorized A session and supersedes the preceding
source-A/B implementation and training order only for this reassessment. Stay on
`perf/dit-factorized-backend-v2` at baseline `5c2754fcce44ed77dad77db09db709408fee7634`;
do not switch branches, create a worktree, import active B work, or train any model.

- Evaluate the existing R4 Gaussian/conditional `checkpoint_step020000.pt` files as the
  shared-adapter legacy diagnostic. Keep the codec, RF semantics, trainer semantics, model
  implementation, manifests, and statistics frozen; do not interpret the two checkpoint
  difference as an independent source ablation.
- Read the old source-A/B JSONL and checkpoint outputs only. Never open test payloads. New
  A-specific entrypoints, helpers, tests, metadata, compact predictions, plots, and reports
  belong under `outputs/dit_source_reassessment_v2/<run_id>/` and must not overwrite old
  outputs.
- The A-owned implementation scope is limited to reassessment/evaluation/reporting code and
  targeted correctness tests. Preserve unrelated user changes and do not edit B's future
  worktree or source files outside that scope.
- Use the original eight validation systems, eight train-selected clips, R1/window30, H4/H8,
  8/16/32-step Euler diagnostics, fixed-epsilon draw pairing, and the bounded H8 short rollout
  defined by the current user prompt. Train/validation only; test remains sealed.
- All model, scientific, test, smoke, evaluation, and plotting commands run through
  `enter-container` with `conda activate torch-ito` on one confirmed idle CUDA GPU. Metadata,
  I/O, and aggregation may use CPU. Do not share, preempt, or terminate another process.
- Finish with a self-contained `report.md`, per-clip/per-frame results, exact commands, a brief
  HANDOFF update, and a local commit if the evidence is complete; report unavailable CUDA or
  rollout data explicitly rather than fabricating coverage.

## Active transition — R4 DiT architecture sequential v1, 2026-09-14

This independent worktree is authorized to execute the four architecture experiments defined by
`agents/dit_architecture_sequential_v1/PLAN.md` and `EVALUATION.md`, strictly in the order
amplitude information, geometry supervision, cross-block temporal refinement, then corrupted
history. This transition supersedes older phase task directories, task order, and stop states only
for that explicitly defined work.

- Sequential implementation, necessary isolation/integration repairs, targeted CUDA checks,
  budgeted train/validation experiments, generation evaluation, and local commits are authorized.
- A new decoder-postprocessing temporal refiner may be added as an independent module. Frozen codec
  weights, encoding semantics, coordinate-decoder semantics, and latent statistics must not change.
- Keep all new artifacts under `outputs/dit_architecture_sequential_v1/`; the source-A/B,
  capacity/data, and all historical worktrees and artifacts remain read-only.
- Never use sealed test payloads. Use `enter-container` with `conda activate torch-ito` for every
  Python, test, model, training, evaluation, and plotting command, and use only an audited idle CUDA
  GPU for numerical evidence.
- Finish at `SEQUENTIAL_V1_COMPLETE`, `SEQUENTIAL_V1_PARTIAL_BUDGET`, or
