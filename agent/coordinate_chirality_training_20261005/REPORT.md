# Coordinate handedness continuation — implementation, not efficacy

Independent branch `exp/coordinate-chirality-training-20261005`, parent6b0b754.
Only this branch's existing training step/CLI changed; active coordinate
evaluation, queued Q12 probe and native-clock jobs remain in their original
worktrees. No original baseline source/weights or sealed data modified.

Implementation:optional loss after RF backward, observed-only 16-step source
rollout, one combined gradient clip/AdamW update, no integration-step detach.
Main source reused without consuming its generator; global RNG restored after
auxiliary computation. Original teacher stays frozen. Targets detached and
restricted to RF interpolation/supervision; auxiliary sees no future geometry.
No neural inference architecture changed. The existing CLI now accepts explicit
coordinate-parent model-only warm start, SHA/source/model/seed qualification,
next2048 schedule slice and checkpoint16/final-only policy for the new phase.
All actual settings appear in the resolved checkpoint contract.

10 targeted tests passed in6.44s via torch-ito, including4 real CUDA training-
fixture tests:aux changes weights but not main RF loss/RNG; future mutation does
not affect the auxiliary; inactive cadence equals zero-weight model/optimizer
exactly; active auxiliary exact model/optimizer/all-RNG resume. Fixtures use
2 Euler steps for bounded numerical tests; **they are not formal16-step protein
training or evidence that geometry/motion improved**.

Full Q12 calibration still waiting onGPU5 evaluation. Formal coefficient,
configs and experiment training are not yet frozen/launched. Added freezer
uses the preregistered maximum-gradient-budget rule and actual ordered train
metadata qualification, without opening development or sealed targets.

Expanded targeted suite:20 passed in6.61s, including calibration rejection
(Q2,wrong solver,updates,missing/duplicate pairs,NaN/no gradients,solver mismatch)
and inherited zero-auxiliary exact CUDA optimizer/RNG resume. All three CLI
entrypoints imported and displayed help successfully without CUDA allocation.

19:07 submission, source321f67c, session3478 in torch-ito:
wait up to4h for diagnostic `probe_q12_v1/summary.json`, stop on recorded probe
failure; then CPU freezer creates `runs/coordinate_chirality_training_20261005/formal_v1`.
Only after qualified freezing, GPU5 executes both disposable largest-query
profiles. Logs `/tmp/molvid-coordinate-chirality-{freeze,profile}-v1.log`.
Current state is **waiting**; no frozen weight/configs or real protein optimizer
updates from this branch yet. The formal model-training/resume/evaluation lane
is not submitted by this command and remains the next implementation step.

## Formal controller implemented,19:15

`tools.run_coordinate_chirality_training` now joins profile completion to all
four formal arms, active-auxiliary exact resume verification, final CPU checkpoint
audit and real full48 CUDA forecasts. It checks that each paired recipe differs
only in the registered weight, not data/parent/LR/time/solver/budget. Native100ps
geometry comparisons include A1/originalPVB/corresponding coordinate parent and
explicit chirality-minus-control. Chemistry and the existing native-time
worktree's ordered-motion scorer are reused, no baseline retraining.

Resume gate upgraded to25 versus16→25 so step24 actually exercises the auxiliary.
Successful formal trajectory retains2048 updates;9 replay verification updates
per arm are separately accounted. Queue completion is execution completion, not
automatic scientific promotion or independent/complex/thermodynamic success.

18 CPU queue/calibration tests passed2.41s, including paired-recipe confound
rejection and both-profile coverage. Host archive script passed `bash -n` syntax
check; it has not relocated any of this campaign's not-yet-created weights.
Its exact targets and recoverability rules are registered in PLAN.md.

19:15 both automatic services submitted, source8c7fff3:

- Container session89404 / PID2778420:full training/evaluation controller,
  GPU5,OMP/MKL4,log `/tmp/molvid-coordinate-chirality-training-v1.log`.
  Sibling status `runs/coordinate_chirality_training_20261005/formal_v1_training_queue_v1/status.json`
  confirms `waiting_for_own_GPU_profiles`. Future formal_v1 still nonexistent,
  so the waiter does not invalidate the freezer's fresh-output gate.
- Host session58826 / PID2778693:recoverable completed-intermediate archive
  service,log `/tmp/molvid-chirality-archive-v1.log`,waiting for successful final
  training/audit markers. No files relocated by this service yet.

This submits the complete conditional workflow, not a completed training run.
At submission original coordinate72 evaluation remained live; native data
materializer passed187/192 and its GPU6/7 pipelines remain separate.
