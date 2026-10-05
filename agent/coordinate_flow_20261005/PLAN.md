# Coordinate-state flow prototype — 2026-10-05

Parent `6441ce3`, independent `exp/coordinate-flow-20261005`; ongoing capacity
and source-scale training stays in `molvid-geometry-scaling-20261005` unchanged.
No competitive result or default promotion is claimed.

## Question and limits

A1 predicts 128 scalar + 128 vector channels per atom/frame (512 real values)
from independent feature noise, although its deterministic target encoder
lifts 3 coordinate values per atom into those channels. This is not a spatial
compression bottleneck. Train-only measurements found highly correlated
vector channels. Off-manifold generation is a plausible failure mechanism,
not an established unique cause; limited data/exposure and learning objectives
remain competing explanations. A1 clean reconstruction is not evidence for
robust decoding of its generated feature distribution.

Replace the ODE state with XYZ, retaining compatible observed-history,
physical-time, DiT and frozen spatial-encoder weights. Current XYZ is encoded
at every flow evaluation; an equivariant readout predicts XYZ velocity.
Retire the old latent decoder, explicitly initialize new velocity to zero.
This is a changed-architecture warm start with a fresh optimizer, not an
exact continuation or a single-factor source-scale ablation.

## Boundaries and gates

- Coordinates: Angstrom; query clock: physical ps; RF progress: dimensionless.
- Source: last observation + iid Cartesian Gaussian, initial sigma 1 Angstrom
  per axis. No hidden future, independent feature noise or oracle correction.
- Dynamic input: observed-only ClipBatch, arbitrary valid future QuerySpec.
- Frozen encoder weights, but differentiable current-query coordinates and
  multi-step actual-source rollout. Observed encoding remains no-grad.
- Static input: independently corrupted single structure; clean structure
  is a loss target only. Same spatial DiT/readout, no history/physical-time
  modules or invented zero-lag trajectory. Static mixing is NOT yet trained.
- CUDA gates: frozen-forward parity, input/trunk/history gradients, dense
  SO(3)+translation, packed-system isolation, padding, static time bypass,
  target-loss detachment and initial identity-source behavior.
- Real-data gate after fixture: train-only small system, real teacher graph,
  full model forward/backward and fresh checkpoint reload; report exact
  updates/peak VRAM/runtime. No claim based solely on tiny fixture tests.
- Then a bounded train-only overfit feasibility experiment before a matched
  multi-system campaign; record source corruption difficulty and motion
  amplitudes. A very short run cannot reject the architectural hypothesis.
- Formal evaluation must use the existing system-equal paired48 development
  protocol, chemistry/chirality and native multi-lag motion, plus independent
  confirmation later. Posthoc loss/seed/checkpoint selection is not permitted.

## Known limitations

Frozen spatial features may be poorly calibrated for noisy coordinates and
their statistics were fitted to clean frames. Re-encoding every step costs
more than latent-only flow. Existing DiT local geometry uses observed
reference context, while the query encoder uses the current state. Legacy
history only in this first prototype; no claim of learned-memory/Mamba
compatibility. Static/dynamic balancing, complexes, equilibrium distribution
and kinetics are open, not solved by a change of ODE coordinates.

## Registered first full-data campaign

Two independent fresh A1 warm starts, seeds71/72,2048 updates each, all192
training systems. Reuse the exact source-training unit71/72 ordered clip/H
schedule and data identity; no hidden future input. Objective is coordinate
CFM only, not a geometry-repair penalty; static auxiliary is deliberately off
to establish the new generation path first. FP32, source sigma1 Angstrom,
DiT5e-5/history2.5e-5/new readout5e-4,5% warmup/cosine to10%.
Architecture, objective, new readout and precision differ from latent A1;
this candidate comparison does not isolate a single mechanistic cause.

Own GPU5 queue waits for all C4/C8 capacity evaluations to finish. Before
formal seed71:one disposable update on the largest nominal atom-frame batch;
then uninterrupted17 versus16→17 exact full-checkpoint resume gate. Failure
aborts the queue with a persistent error, never silently reduces batch/data.
The accepted step17 continues to2048; seed72 starts fresh. Save16/1024/final
plus resume evidence, no validation-selected best checkpoint. Each run has
a6h budget and8GiB disk reserve; wait budget24h. Do not count the duplicate
resume update or disposable profile as distinct training exposure.

After each final:48 systems×2 anchors×2 seeds native100ps, H4/Q12, Euler16,
same raw-coordinate metrics as A1/PVB, followed by CA stereochemistry and
system-paired comparison. Missing/OOM cases make evaluation incomplete;
no CPU fallback, no partial-case ranking. Internal ordered-motion rescoring
uses the parent campaign's evaluator after these artifacts complete.
