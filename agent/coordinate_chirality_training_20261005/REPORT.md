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
