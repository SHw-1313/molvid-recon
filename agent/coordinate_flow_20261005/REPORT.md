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
identity. Real teacher/graph and checkpoint gates remain to be run.

`tools/pilot_coordinate_flow.py` is a bounded real-data gate:4 whole-model
updates on one smallest train-cache system, H4/Q2; actual-source two-step and
static backward checks without extra updates; fresh exact checkpoint forecast
reload. It is not a geometry/motion experiment and cannot establish efficacy.
