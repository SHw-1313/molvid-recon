REVIEW_LANE: A
REVIEWED_COMMIT: eec76455a7d9a36723e7c1a2aad1043bba553198
VERDICT: FIX_REQUIRED

# A common repair independent review, round 01

Base commit: cc74a385577e57c64c9623111e0db86e54e48770.
Reviewer: independent sub-agent `/root/common_review`, started without implementation conversation context.
Review date: 2026-09-28. HEAD was verified equal to the reviewed commit.

This is a static common-code review, not approval for formal training. Model code was not modified; no training, CUDA operation, CPU model execution, sealed-test access, dependency change, or external-worktree write was performed. The only reviewer-authored repository file is this report.

## Reviewed evidence and scope

- Read root `AGENTS.md`, `SESSION_REVIEW_PROMPT_260928.md`, task `AGENTS.md`, `EXPERIMENTS.md`, `IMPLEMENTATION_A.md`, `DATA_TRAIN_EVAL.md`, and `WORKFLOW.md`.
- Inspected the complete fixed diff and surrounding sampler, execution, trainer loss aggregation, checkpoint/resume, CLI configuration/calibration, and metric aggregation paths.
- Scope: packed-batch mixing; shared H/aux execution schedule and resume/DDP semantics; inherited loss overrides; Euler4/8; metric-family separation. The new A clash/motion-only/bond-band losses are outside this candidate and were not treated as implemented.
- Supplied factual validation: `tests/test_gm_common_schedule.py`: **2 passed**, metadata/configuration only. This reviewer did not rerun it. The committed validation record reports the same result and a passed `git diff --check`; its approval conclusions are not used as review evidence.
- Supplied environment fact: all eight GPUs have other users' processes. CUDA smoke, CUDA resume, backward, pilot and profiling have **not run**. No GPU was claimed. No pilot/profile/resolved formal configuration was supplied for review. Parent checkpoint and actual dataset/statistics identities were not independently opened or verified.
- Writer separately flagged the CLI feature-scale gate during review; finding R1 below was independently confirmed against the fixed source.

## Blocking findings

### R1 — P1: CLI still rejects an enabled non-ES branch without unused feature scales

**Location:** `molvid/cli/train_frame_joint.py:283–285`; compare `molvid/training/joint.py:145–149`.

**Trigger:** An otherwise valid CLI configuration has `sampled_auxiliary.enabled=true`, `energy_weight=0`, a positive observed-bond weight, and omits `feature_scales`. The same restriction would block the intended no-ES A branch once its new loss fields are connected.

**Evidence:** `_validate_frame_joint_config` unconditionally raises whenever `enabled and feature_scales is None`. The new `SampledAuxiliaryConfig.resolve` only requires scales when `energy_weight > 0`; the new test exercises this resolver directly, bypassing the CLI validator.

**Impact:** The public training entry point cannot exercise the newly supported optional-scale behavior. Users must invent unused ES scales or perform unnecessary feature fitting to start a non-ES run.

**Minimal fix:** Apply the same effective-energy-branch predicate in CLI validation and runtime resolution. Calibration/`fit` sentinels must still require the appropriate calibration inputs. Do not require ES scales for disabled ES.

**Minimal verification:** A metadata-only test through `_validate_frame_joint_config` with a complete valid config: enabled non-ES branch without scales succeeds; positive ES weight without scales fails; ES calibration/fit remains valid. On available CUDA, run one real enabled non-ES auxiliary backward without scales.

### R2 — P1: Independent rank aux masks change the effective loss strength with DDP world size

**Location:** `molvid/training/execution.py:67–71`, `molvid/cli/train_frame_joint.py:1141` and `:1199`, interacting with `molvid/training/joint.py:729–746`.

**Trigger:** `world > 1`, enabled sampled auxiliary, and one optimizer update has an active auxiliary entry on only some ranks. The new schedule chooses activation per packed global batch, then assigns adjacent entries to ranks, so this is an ordinary outcome.

**Evidence:** Inactive ranks have zero auxiliary applicable count. `_distributed_total` normalizes each loss by the count of active samples only (`local * world / global_counts`). For two equal-size rank batches with auxiliary gradients `g` and `0`, the active rank is scaled by two and DDP averaging yields `g`, rather than the `g/2` contribution of the explicitly masked global batch. Previously the cadence was shared by successful-update index, so all ranks activated together.

**Impact:** Approximately one eighth of rank-local entries are auxiliary-active, but the auxiliary has full normalized strength whenever *any* rank is active. With approximately independent masks the fraction of such optimizer updates approaches `1-(7/8)^world`, not `1/8`. Exact finite strata introduce correlations, but do not remove the demonstrated partial-rank amplification. Thus the cadence-amortized auxiliary/flow ratio changes with GPU count, contrary to shared exposure and weight semantics. The current metadata sharding test does not check this.

**Minimal fix:** Define and implement a single DDP objective for scheduled masks: either coordinate auxiliary activation at optimizer-update level with an appropriate stratified global-update schedule, or normalize masked rank contributions against the intended full eligible global population, preserving zeros for schedule-inactive samples. Keep structural ineligibility distinct from cadence inactivity. If DDP is not needed for this release, explicitly reject enabled-aux multi-rank execution until this is implemented, and document the limitation.

**Minimal verification:** First add a small algebra/metadata check for mixed active/inactive ranks and uneven batch counts. If enabling DDP, when two assigned GPUs become available compare a two-rank masked update with the explicitly defined reference objective and verify its accumulated auxiliary coefficient across a schedule. No additional DDP CUDA gate is needed for an explicitly single-rank-only release.

### R3 — P2: Inherited loss overrides are discarded when generated-bond calibration resolves

**Location:** `molvid/cli/train_frame_joint.py:832–842` and `:1178–1181`.

**Trigger:** Model warm start with `loss.inherit_parent=true` and explicit `generated_bond="calibrate"` (or null), reaching a joint/continuation stage with bond loss enabled.

**Evidence:** Initial parsing removes `inherit_parent` and correctly merges the parent's terms with explicit overrides. After calibration, however, the code rebuilds `resolved_loss` from `raw["loss"]`, removes only `calibration_target_ratio`, and passes it to `JointLossConfig.resolve`. That resolver rejects the retained `inherit_parent` key. Merely removing that key would still discard inherited nondefault clean/near coefficients and thresholds.

**Impact:** This supported inheritance/override combination fails after spending calibration work; a superficial unknown-key fix would silently change the remaining base objective. Ordinary endpoint Release with numeric zero does not enter this path and is not affected.

**Minimal fix:** Update only `generated_bond` on the already merged, resolved loss contract, preserving all inherited/explicit terms, and use that same contract for trainer state and resolved output.

**Minimal verification:** Extract or directly test the post-calibration mapping using nondefault parent clean/near weights and thresholds, explicit overrides, and a supplied calibrated scalar. Assert all unrelated resolved terms survive and no metadata key reaches `JointLossConfig.resolve`. A metadata/config test is sufficient for mapping behavior; a later CUDA calibration smoke verifies integration.

## Nonblocking notes and limits

1. Packed-batch shuffling is performed after homogeneous packing, preserving the batch shape constraint. Deterministic schedule RNG is separate from torch generators; H assignment is balanced within ordinary shape/bucket groups; aux counts explicitly round per `(view, bucket, H)`. The inspected A ordinary groups all use sixteen frames. H balance here is balance in batch counts, not a proof of exact 50/50 clip counts for variable packing; report actual clip exposure as planned.
2. The CLI records execution hashes at checkpoints and rejects mismatches. Sampler version changes reject old order contracts. Source noise and main flow draws use separate explicit generators; Frame Joint fixes dropout at zero. Nonfinite loss/gradient raises before optimizer/cursor advance, so recovery is rollback to a saved checkpoint, not an in-place overflow retry. Actual CUDA replay remains unverified.
3. Euler4/8 is accepted by the model, config resolver, CLI validator and calibration call, and appears in the sampled contract. The CLI's existing resume policy reconstructs sampled settings from the checkpoint rather than validating conflicting requested values. Thus a raw resume config requesting Euler8 may still execute the checkpoint's Euler4. This is not a new regression in the diff, but callers must inspect the resolved contract; explicitly rejecting conflicting overrides would make exact-resume intent clearer.
4. Fixed-history Q12 is now excluded from `main_rmsf`, and family aggregation is separated; ordinary lag/H buckets remain distinct. `all_diagnostic_metrics` can still contain fixed-history RMSF, which is acceptable only as diagnostics. This change does not itself complete the required cross-grid fixed-time summaries, prespecified weighted main aggregate, bootstrap comparisons, or all new geometry metrics; those remain downstream evaluation work and were not certified here.
5. The existing tests prove neither actual CUDA integration nor DDP gradient semantics. They also do not test CLI scale validation or the inherited-calibration path identified above. Their passing status should not be described as complete common-path validation.

## Minimum follow-up and permitted continuation

Fix R1–R3 (or explicitly disable the unsupported DDP mode for R2), commit a new candidate, and request focused independent review against this report. Run the small configuration/algebra tests described above first. When an authorized GPU is available, run the scoped CUDA auxiliary backward and exact cursor/RNG resume smoke, then the planned pilot/profile; do not substitute CPU model execution or seize occupied GPUs.

Read-only inspection, implementation of these fixes, metadata checks, and downstream branch development with the limitations recorded can continue now. This verdict does not authorize or certify formal A training, production/scientific parity, complete A loss behavior, or capacity/recipe release.
