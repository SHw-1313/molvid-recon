# Coordinate-flow evidence — 2026-10-05

Independent candidate; NOT a promoted model or demonstrated quality gain.
See [registered scope](PLAN.md).

Implemented `molvid/flow/coordinate.py`: XYZ-only ODE, current-coordinate
frozen encoder with active input gradients, observed history and physical
time, shared static spatial trunk, explicit velocity readout. The previous
512-value-per-atom feature state and latent decoder are not sampled/used here.

## CUDA fixture checks

Five targeted tests passed in 8.07 s via `enter-container` / `torch-ito`, on
our GPU5, co-resident with our capacity evaluation, process memory cap2.5%.
These are numerical fixture checks, NOT protein training or a speed benchmark.
Initial attempt:3 passed/2 failed because the synthetic parent was completely
zero-output/zero-gated and therefore did not expose deep gradients. The fixture
now explicitly activates those test-only weights (the actual A1 parent already
has trained weights). The same assertions then passed; no tolerance relaxed.

Coverage: teacher old-forward exact parity with input autograd alternative;
dense SO(3)/translation; system/padding isolation; four-step actual-source
backprop; target-loss detachment; static history/time bypass; initial source
identity. Real teacher/graph and checkpoint gate subsequently passed below.

`tools/pilot_coordinate_flow.py` is a bounded real-data gate:4 whole-model
updates on one smallest train-cache system, H4/Q2; actual-source two-step and
static backward checks without extra updates; fresh exact checkpoint forecast
reload. It is not a geometry/motion experiment and cannot establish efficacy.

## Real-data gate completed

`runs/coordinate_flow_20261005/real_gate_v1`:4 real full-model updates,
18.756 s including two-step rollout backward, static backward and exact fresh
checkpoint forecast reload. Peak allocated4,039,456,256 bytes. Checkpoint SHA
`e602c0e73bfcb3a614c2435468e4be4c43efc2e37f3f25cab9bad07d2dc5c98d`.
First zero-readout update correctly reaches only readout; subsequent updates
reach DiT/history. Frozen teacher never receives parameter gradients.
Static backward reaches DiT (norm .01927), not history; no static optimizer
update yet. Actual-source input gradient norm .24097. Fresh forecast exact.
Optimizer resume was not tested and this tiny checkpoint is not a formal parent.

Next fixed learnability gate: fresh A1 warm start, one smallest train-cache
clip, H4/Q2,512 updates, Cartesian RF only. LR DiT5e-5/history2.5e-5/readout5e-4;
three fixed held-out noise seeds, Euler16, initial versus fixed final only.
Unadapted A1 and persistence are descriptive references, not matched-training
controls. A memorized single clip cannot validate kinetics/generalization.
