# Coordinate stereochemistry intervention — registered18:42

Independent branch from static-gradient diagnostic dfa04f2; original coordinate
seed72 training runtime untouched. Motivation:full48 coordinate71 improves
bond/clashes but increases Cα handedness flips, so no promotion is justified.

Implement an observed-reference relative signed-volume hinge, margin.1,
reference minimum absolute volume.2Å³. Both are fixed before this experiment;
not selected to improve development predictions. Require supplied Cα-N/C/CB
bonds and complete unmasked atom identities. No penalty for correctly oriented
volume above margin; no displacement/RMSF suppression or target-coordinate
restoration. It is a differentiable soft constraint, not guaranteed chemistry.
Automatic centers cover protein Cα, **not** ligand or all side-chain R/S.

First real diagnostic is train-only, zero optimizer updates:8 hash-selected
training proteins from the existing train cache, H4/Q2, fixed trained coordinate71
2048 checkpoint, FP32. Two CFM interpolation times.25/.75 use only training
targets for RF supervision. Separately generate from observed-only source via
differentiable Euler4, compare auxiliary gradients with the RF gradient; retain
Euler16 no-grad generation as a numerical/solver diagnostic, not validation.
Seed20261005 selection and per-system source seeds4000+i are registered.
Measure raw source/interpolation/one-step endpoint/generated handedness,
auxiliary magnitude, per-module gradient norms/cosines and frozen teacher
boundaries. Require model state bitwise unchanged and no validation/test input.
Short Q2 is only a gradient/memory gate, not Q12 effectiveness or kinetics.

No formal auxiliary weight selected yet. Do not label this a training success.
If usable gradients exist, follow with a bounded train-only scale/profile and
matched continuation control/aux arms with independent seeds and Q12 fixed-final
evaluation. Keep native-clock changes and static mixing out of this comparison.

## Solver-matched follow-up registration,18:49

Completed Euler4 gradient gate showed fewer flips than its Euler16 inference
reference, so an Euler4-only training auxiliary cannot be assumed to target
the actual sampling error. Repeat the **same8 train proteins/H4/Q2/noise/weights**
with auxiliary gradients through all16 Euler steps (`probe_v3 --aux-steps16`).
Same15-minute budget and20GiB process cap. No new data or weight update, and no
development selection. Keep v2 intact; v3 rows use `generated_auxiliary` with
the actual step count in protocol, rather than the older `generated4` key.
