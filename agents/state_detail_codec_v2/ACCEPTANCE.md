# ACCEPTANCE — zero-preserving state/detail temporal codec v2

## A. Repository and scope gate

- work starts from `feat/visnet-spatial-v2` commit
  `045dbb8809e9d7aee418eb355b6477e05088d4fd` on a new phase branch;
- pre-existing user changes and all v1/v2 artifacts are preserved;
- all Python/training/evaluation commands use `enter-container` and `torch-ito`;
- no package/dependency/network/data-download mutation occurs;
- every new run uses `torchmd_et` and no ViSNet comparison is added;
- no T1, DiT, observation-mask, VAE, AF3, rollout, scaling, or full-data work begins.

## B. Lifting and capacity gate

- R1, R2, and R4 fixed lifting and inverse lifting recover deterministic FP32 h/v inputs within
  `rtol <= 1e-6`, `atol <= 1e-7` when learned bottlenecks are replaced by identity fixtures;
- R4 coefficient order is exactly `Dmid,D01,D23` and round-trips without permutation;
- vector operations never mix Cartesian axes;
- R1 produces 16 state-only tokens and 16C active feature volume for T=16;
- R2 produces 8 state/detail tokens and 16C active feature volume;
- R4-SD and R4-Matched-Pooling each produce 4 two-bank tokens and 8C active feature volume;
- reports distinguish token reduction from active latent elements.

## C. Mask, clock, and task-semantics gate

- invalid frames never contribute to valid state/detail coefficients;
- token, block-frame, detail, and component masks have documented shapes and semantics;
- static T=1 uses a full C-wide state and `detail_valid=false`;
- repeated-static T=16 uses valid detail equal to zero;
- T=16 works for all ratios, and partial/padded synthetic cases either behave according to the
  explicit mask contract or fail before training with an actionable error;
- actual `time_ps` is preserved and irregular clocks are not replaced by frame indices;
- static data are never repeated into training trajectories.

## D. Zero-preserving gate

For state/detail controls in FP32 evaluation mode:

- exact repeated frame features yield exactly zero analytical detail coefficients;
- encoded detail norm is `<= 1e-7` absolute;
- decoded detail coefficients are `<= 1e-7` absolute;
- decoded within-block h/v differences are `<= 1e-6` absolute;
- decoded coordinate motion is `<= 1e-6` angstrom absolute after accounting for masks;
- changing timestamps alone cannot make a zero-detail input move;
- zero preservation is obtained without a consistency loss;
- dynamic synthetic input produces nonzero detail and all principal detail modules receive
  finite, nonzero gradients.

If normal floating-point behavior demonstrates that a listed absolute tolerance is impossible,
record raw evidence and stop for operator review; do not silently relax the gate.

## E. Coordinate and target-isolation gate

- no new latent contains `[N,3]` first-frame/reference coordinates;
- only a documented `[B,3]` translation origin may bypass the latent;
- the origin is computed by a masked centroid rule and shifts equivariantly under translation;
- decoded centered coordinates come from latent scalar/vector features;
- decoder APIs accept no target coordinates;
- mutating target tensors after encoding cannot change decoding;
- scalar outputs are rotation/translation invariant and vector/coordinate outputs satisfy the
  declared SE(3) behavior;
- no frame or sample contaminates another;
- old checkpoints retain their explicit old per-atom-anchor semantics.

## F. Configuration and compatibility gate

- all four new controls are constructible by explicit names;
- ratio, mode, widths, coefficient order, masks, no-anchor policy, and origin rule round-trip
  through model/checkpoint contracts;
- accepted configuration fields change real computation and unsupported combinations fail before
  optimizer construction;
- old temporal/model/checkpoint paths pass regression tests;
- ViSNet v1/v2 backends and result files are not modified semantically;
- no spatial refiner is active in a locked control.

## G. T0-A training-smoke gate

For each of the four controls:

- real CUDA forward/backward/optimizer execution is finite;
- the selected TorchMD encoder hash is identical and its parameters stay unchanged while frozen;
- new packer/decoder parameters receive finite, nonzero gradients;
- a bounded overfit loss decreases materially from initialization;
- state/detail controls show nonzero dynamic detail utilization;
- checkpoint saves and resumes for at least one additional optimizer step;
- runtime, peak allocated/reserved memory, and parameter count are recorded;
- one separate smoke confirms gradients can reach TorchMD when deliberately unfrozen.

This gate asserts execution and trainability, not scientific quality.

## H. T0-B data and training gate

- exactly three approved ATLAS systems and all three replicas per system are present;
- the stored split is independently verified as 441 train and 117 fixed holdout clips, or the
  phase stops on mismatch;
- T=16 and native `dt_100ps` are verified from data rather than assumed;
- all four runs use the same manifest, common frozen encoder, seed, loader coverage, optimizer,
  staged losses, and evaluator;
- each run completes exactly 30 nonreplacement train epochs and complete holdout evaluation;
- logs cover the entire run and include best/final checkpoint hashes;
- best/final checkpoints resume successfully;
- NaN, Inf, hidden target access, silent sample replacement, and undocumented fallback are absent.

## I. T0 evaluation gate

The review packet contains per-control and per-system absolute values/curves for:

- RMSD/dRMSD overall, per frame, and by block offset;
- bond/contact/clash;
- velocity/acceleration;
- RMSF and one lagged/ACF statistic;
- frequency retention and block-boundary jump;
- state/detail or bank utilization;
- full versus detail-zero state/detail decoding;
- token count, active elements, parameters, throughput, wall time, and memory;
- separated spatial/packer/decoder performance;
- repeated-static zero-motion evidence;
- all warnings, failed attempts, and limitations.

Three-system results are labeled implementation/training sanity and are not used to select the
production ratio.

## J. Operator-stop gate

- `OPERATOR_REVIEW.md` points to the exact diff, tests, reports, plots, logs, and checkpoints;
- `HANDOFF.md` contains reproducible commands and hashes;
- `TASKS.md` marks S300–S303 blocked;
- final phase status is `WAITING_FOR_OPERATOR_REVIEW`;
- no 64-system manifest is created and no T1 job is launched.

Passing A–I does not waive J. The operator pause is mandatory.
