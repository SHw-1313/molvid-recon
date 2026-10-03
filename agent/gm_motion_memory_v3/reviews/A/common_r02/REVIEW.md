REVIEW_LANE: A
REVIEWED_COMMIT: 5a745960f673d5397b883c2b129f39cf57593856
BASE_COMMIT: eec76455a7d9a36723e7c1a2aad1043bba553198
VERDICT: PASS_STATIC_COMMON_FIXES_ONLY
FORMAL_TRAINING_VERDICT: NOT_REVIEWED_NOT_APPROVED

# Common repair focused independent review, round 02

Reviewer session: `/root/common_review_r02`, fresh sub-agent without implementation conversation context. Date: 2026-09-28.

Scope is limited to R1–R3 from common_r01 and regressions directly caused by their fixes. All candidate source, previous review, fix response, tests and committed validation evidence were read with `git show` at the exact candidate above. The working tree contains ongoing A development and was not treated as reviewed code. Root AGENTS was read at the candidate. Task AGENTS.md, WORKFLOW.md, IMPLEMENTATION_A.md and DATA_TRAIN_EVAL.md are absent from the candidate tree; their present task-directory files were read only as external task specifications, not as candidate implementation evidence.

No blocking source defect remains in these three fixes for the explicitly single-process CLI protocol. This is static acceptance of the common fixes only, not a formal-training PASS, A-loss review, numerical validation or scientific parity claim.

## Findings against prior issues

### R1 — closed by source inspection

`molvid/cli/train_frame_joint.py:283–286` now requires scales only for enabled positive or calibrated energy weight. The earlier numeric validation rejects invalid weights before this predicate; the calibration sentinel short-circuits the float conversion. Enabled observed-bond-only configurations with absent scales now pass this gate and reach the compatible `SampledAuxiliaryConfig.resolve` predicate (`molvid/training/joint.py:146–149`). Positive energy without scales still fails. The existing `fit` placeholder and paired weight-calibration paths remain intact (`train_frame_joint.py:854–897`, followed by scale fitting/calibration before training).

This does not certify the forthcoming clash-only branch: its new configuration/loss implementation is outside the fixed candidate and this review.

### R2 — closed by explicitly disabling the unsupported training protocol

`molvid/cli/train_frame_joint.py:723–727` rejects all `world != 1` execution before dataset loading, model construction or optimizer updates. Thus mixed rank-local auxiliary masks cannot reach this CLI's DDP loss aggregation. The limitation is explicit in both the exception and common_r01/FIX_RESPONSE.md and matches the task's one-process-per-arm option. Generic sampler sharding support is not evidence of numerical DDP support; DDP training remains unsupported.

Nonblocking improvement: move the WORLD_SIZE guard before `_distributed_device`. Currently `_distributed_device` (`:341` onward) can set the CUDA device and initialize NCCL before the guard executes, and the guard lies outside the cleanup `try/finally`. A malformed multi-rank launch can therefore fail or wait in initialization before producing the intended unsupported-protocol error. A properly launched multi-rank run still cannot train, so this does not reopen the original auxiliary-weighting defect.

### R3 — closed by source inspection

The initial loss mapping removes `inherit_parent`, merges the parent's complete contract with explicit overrides, and replaces a requested calibration sentinel with a provisional numeric zero (`train_frame_joint.py:834–851`). Post-calibration now copies this merged `loss_mapping`, removes control keys, and changes only `generated_bond` (`:1184–1188`). Inherited nondefault clean/near coefficients and thresholds, and explicit per-term overrides, therefore survive. The trainer receives the resulting resolved contract, and `resolved_config.json` is updated from that same trainer contract (`:1192–1193`).

The prior report mentioned explicit null alongside `"calibrate"`; the CLI already rejects explicit null during numeric validation. This patch does not add null support. The supported `"calibrate"` path is sufficient to reproduce and close R3; use that spelling for CLI configuration.

## Evidence limits and remaining checks

- Candidate diff changes only the three CLI locations plus the preserved r01 review and fix response; no new regression tests were added in this fix commit.
- The supplied committed result is two passing metadata/configuration tests. This reviewer inspected their source but did not rerun them. They exercise schedule metadata and the runtime optional-scale resolver; they do not exercise the CLI scale gate, the multi-process rejection, or inherited post-calibration mapping. Targeted metadata regression cases for those three paths are still desirable and must not be represented as already passed.
- No Python, model, CUDA, training, profiling, calibration or GPU command was run by this reviewer. No sealed data or checkpoints were opened. The reported environment has all eight GPUs occupied by other users; this reviewer did not independently remeasure availability.
- Actual no-ES auxiliary backward, calibration integration and exact CUDA cursor/RNG replay remain unverified. Pilot/profile, A-specific numerical behavior, formal configuration identity and full evaluation contracts are outside this review.

Common integration and further development may proceed with the single-process restriction recorded. Formal A training still requires the scoped CUDA evidence, pilot/profile and independent review of the actual A candidate/configuration. This report must not be reused as approval of a different code commit or of uncommitted A changes.
