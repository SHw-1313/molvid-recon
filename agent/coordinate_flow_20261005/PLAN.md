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
