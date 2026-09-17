# TASKS — R2/R4 state-detail latent DiT probe v1

Status legend: `[ ]` pending, `[-]` active, `[x]` complete, `[!]` blocked.

## Preparation and isolation

- [x] D000 audit repository path, remote, branch, HEAD, worktree status, active processes, GPU
  occupancy, environment, and current T1 status without modifying anything
- [x] D001 create or enter the isolated `feat/dit-state-detail-probe-v1` worktree from the audited
  source commit; stop if existing contents cannot be reconciled safely
- [x] D002 update root `AGENTS.md` on the new branch to name this phase, its read order, scope, and
  mandatory stop; do not modify root instructions in the active T1 worktree
- [x] D003 record exact R2/R4 codec schemas, checkpoint candidates, public encode/decode APIs,
  topology fields, masks, clocks, and forbidden raw-detail/target fields
- [x] D004 run the existing full test suite before implementation and record pre-existing failures

## Contracts and latent adapter

- [x] D010 add a versioned DiT batch/latent contract for R2 and R4 with explicit `[K,N,C]` and
  `[K,N,3,C]` shapes
- [x] D011 implement state/detail channel packing and scalar/vector input/output projections without
  xyz mixing or state/detail token duplication
- [x] D012 reconstruct generated `StateDetailLatent` objects with `raw_detail_h/v=None` and decode
  through the frozen public codec API
- [x] D013 implement ratio-specific, mask-aware train-only latent statistics and deterministic
  normalization/inverse normalization
- [x] D014 serialize and hash codec, stats, data, ratio, adapter, and model contracts

## Flow and observation masks

- [x] D020 implement arbitrary-rank rectified-flow interpolation and velocity targets for the four
  normalized scalar/vector fields
- [x] D021 implement equal four-field mask-normalized loss and per-field diagnostics
- [x] D022 implement H=0/4/8 frame-to-latent observation-mask conversion for R2/R4
- [x] D023 reject partial observed blocks and prove no coefficient depending on an unobserved frame
  is exposed as clean
- [x] D024 implement observed-token clamping and invalid-token zeroing during every Euler step
- [x] D025 implement deterministic 8-step and 16-step seeded sampling

## Molecular DiT

- [x] D030 implement flow-time, physical-block-time/span, ratio, observation, and static chemistry
  conditioning
- [x] D031 implement ragged atom-to-block pooling, per-sample block spatial attention, and block-to-
  atom broadcast without atom-level dense attention
- [x] D032 implement bidirectional per-atom temporal scalar/vector attention over K
- [x] D033 implement scalar/vector FFN and AdaLN-Zero residual modulation
- [x] D034 implement four-field output heads with scalar invariance and vector equivariance
- [x] D035 expose one configurable R2/R4 model constructor with identical size policy

## Trainer, evaluation, and configuration

- [x] D040 implement trainer support for frozen codecs, field-normalized RF loss, AMP policy,
  gradient clipping, logging, and checkpoint resume
- [x] D041 implement evaluation separating codec-oracle error, generated-latent error, and the DiT
  generation gap
- [x] D042 report future RMSD/dRMSD, bonds, contacts, clashes, velocity/acceleration, dynamic
  correlation, RMSF, frequency, observation boundary, diversity diagnostics, and runtime
- [x] D043 add the frozen v1 configuration and fail-fast validation for unsupported options
- [x] D044 add a bounded smoke runner that cannot open T1 test or exceed 100 steps/15 minutes

## Tests

- [x] D050 pass R2/R4 shape, packing, no-raw-detail, normalization, contract, and decode tests
- [x] D051 pass rectified-flow equations, mask normalization, deterministic seed, and Euler clamp
  tests
- [x] D052 pass SE(3), isotropic-noise rotation, no-xyz-mixing, padding, ragged sample, and topology
  isolation tests
- [x] D053 pass physical-time and H=0/4/8 observation/future-isolation tests
- [x] D054 pass checkpoint/config/stats/codec-hash round-trip tests
- [x] D055 pass existing codec/T1 regressions, full suite, `py_compile`, source audit, and
  `git diff --check`

## Bounded execution and stop

- [x] D060 audit idle GPUs without killing or preempting existing work; skip CUDA smoke if no device
  is safely available
- [x] D061 run at most one 100-step/15-minute R2 smoke and record gradients, hashes, resume,
  throughput, memory, and decode evidence
- [x] D062 run at most one 100-step/15-minute R4 smoke under the identical policy
- [x] D063 generate the operator review packet and classify code/CPU/CUDA gates separately
- [x] D064 set status to `WAITING_FOR_T1_AND_OPERATOR_REVIEW` and stop

## Observed evidence

- D000-D004: audited target branch is feat/dit-state-detail-probe-v1 at
  23c6dbdd89a7b92c58b33edc9cf76aa82d0c5541; baseline full suite was 115 passed with one
  pre-existing torch.load warning. Source T1 worktree remained read-only.
- D010-D055: focused DiT tests passed 25; legacy/state-detail/evaluator regression selection passed
  45; final full-suite result is recorded in HANDOFF.md. py_compile and git diff --check passed.
- D060-D062: GPU 7 was idle at each launch. R2 and R4 smoke reports are under
  outputs/dit_state_detail_probe_v1/{ratio2_state_detail,ratio4_state_detail}/ and use the
  immutable T0-valid payload from the audited source worktree. Both are execution evidence only.
- D063-D064: HANDOFF.md and OPERATOR_REVIEW.md contain the final gate classification and stop
  status. No T1 test split was opened.

## Planned later pilot — not authorized

- [!] D100 resolve operator-approved T1 R2/R4 checkpoints and train-only latent-stat inputs
- [!] D101 build/freeze the real pilot manifest without opening T1 test
- [!] D102 train matched R2 and R4 small DiTs
- [!] D103 evaluate H=4/H=8 generation and the predeclared quality/compute rule
- [!] D104 select the default codec for scaling and plan the next model-size/data-size stage

D100–D104 require a later explicit operator authorization even if D000–D064 pass.

## Review-fix repair record — 2026-09-07

- [x] D070 repair vector normalization, vector FFN nonlinearities, scalar/vector interaction, and
  nonzero-gate SO(3) tests without changing the factorized DiT architecture
- [x] D071 restore the codec-compatible loss-masked origin for H>0 and fixed zero origin for H=0;
  add future-mutation, masked-atom, and clamped-gauge tests
- [x] D072 make conditional evaluation history-aware for observed, future, boundary, and full
  diagnostic intervals, including sliced temporal metrics and ragged/loss masks
- [x] D073 add actual CUDA BF16 autocast policy and activation-dtype coverage; reject CPU AMP
- [x] D074 persist and reconstruct all six latent-statistics tensors with provenance and hash;
  validate all latent fields and topology/sample contracts
- [x] D075 add explicit CUDA correctness coverage for forward/backward, rotation, checkpoint
  recovery, sampling, history-aware evaluation, and gradient groups
- [x] D076 rerun the repaired focused and complete CPU suites, compile checks, diff check, and
  forbidden-input/source audit
- [x] D077 run the audited-GPU correctness test and verify the repaired branches under real CUDA
  BF16 autocast
- [x] D078 run exactly one final two-step H=4 smoke for each candidate with 24 combined sampling
  evaluations, fresh statistics recovery, exact clamping, and corrected throughput accounting
- [x] D079 append the repair evidence and operator review, stage only source/tests/docs, and
  create the focused repair commit; no scientific pilot is authorized in this phase
