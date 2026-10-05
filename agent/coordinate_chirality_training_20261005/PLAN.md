# Matched coordinate handedness continuation — independent candidate

Parent6b0b754, not a modification of ongoing coordinate/native-clock jobs.
Reuse the existing CoordinateFlowModel and coordinate training CLI. No new
architecture, source distribution, decoder, time module, static mixing or
motion-amplitude penalty. Add only optional observed-reference Cα signed-volume
margin on differentiable actual-source trajectories, no posthoc repair.

Proposed formal comparison:each original coordinate seed71/72 fixed2048 parent
initializes both control(weight0) and chirality arms, fresh identical optimizers.
Same next2048 ordered training batches (global schedule indices2048:4096), all192
training proteins, same H4/H8/query clocks, FP32,LR schedule and clipping. Both
main flow generators seed=training_seed+18000. Auxiliary reuses the observed-only
main source, rebuilds history condition after RF backward and preserves global
RNG. One combined optimizer step; gradients are never detached between Euler
steps. Same sample/update exposure does NOT imply same FLOPs (auxiliary costs
additional16-step forward/backward calls). Frozen teacher weights stay frozen.

Planned auxiliary cadence every8 updates, exact16 Euler steps to match inference,
margin.1,reference min.2Å³. Loss is dimensionless; its coefficient carries the
RF loss's effective Å² scale. **Weight is not selected yet**:Q12 train-only
gradient/memory gate is still queued in the diagnostic branch. Do not infer
Q12 safety from Q2. Require train-only calibration, largest-query profile,
actual16-update/resume checks and fixed-final full48 chemistry/motion assessment.

No formal configurations frozen or training submitted at initial implementation.
Do not call fixture tests or zero-update probes an efficacy result. Original
development48 is not independent baseline validation. Sealed8/reserve96 remain
closed; competitive generalization/complex/thermodynamic claims remain unmet.

## Preregistered train-only coefficient rule,19:03

Before Q12 results exist, set coefficient `min(1, .10 / max_ratio)`, where
max_ratio is the maximum measured raw auxiliary/RF gradient-norm ratio across
DiT/readout and all16 paired rows of the completed seed71 Q12/Euler16 probe.
Require finite ratios, nonzero signal, no updates and matching auxiliary/
inference metrics. This caps the largest measured contribution at10% (or less
under coefficient cap1), not a guarantee for all192 proteins or the other seed.
Do not tune against development scores or assume the small probe determines
the optimal weight. Keep this coefficient fixed across both training seeds.

`tools.freeze_coordinate_chirality_training` will verify probe inputs, hash
both parent weights and qualify actual next2048 exposure/all192 coverage, then
freeze four configurations through the existing CLI. It does not submit model
training. Actual largest-query profiling and resume gates remain mandatory.

Freezer additionally emits separately tagged disposable profile configs, each
same-seed chirality recipe but explicit cadence1. Profile tool executes one real
largest-query update for each seed's selected2048 schedule; no saved profile
weights or promotion. Both gates must show actual auxiliary execution and finite
gradient/loss. These2 discarded engineering updates are not formal training.
