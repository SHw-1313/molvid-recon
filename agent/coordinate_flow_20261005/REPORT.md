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

Completed `overfit512_v1`:512 real updates,524.39 s, peak1,386,487,808 bytes;
checkpoint SHA `ce7aa5ec5290d004c47ab4580959934f9a271361d03d81f89ec70ec89951d5df`.
Final three noise draws:bond .13033/.13832/.13607 Å; coordinate RMSE
.52662/.56507/.57956 Å. Initial source bond1.60/1.63/1.59 Å; unadapted A1
single draw bond .39748 Å. Persistence bond .04179 Å but coordinate RMSE1.39771
Å and zero motion. Thus geometry alone would misleadingly prefer persistence.
This remains **one memorized train clip, not improved generalization**.

## Full-data candidate preparation

Added coordinate RF trainer using the exact source71/72 data/H execution
schedule, including packed systems, padding and physical clocks; target
coordinates enter only interpolation/loss. Frozen teacher, FP32, RF-only;
fresh A1 warm start, readout5e-4/DiT5e-5/history2.5e-5,5% warmup/cosine over2048.
This changes architecture, objective and precision versus source-unit training;
it is an architecture candidate, not a one-factor causal attribution.
No static or sampled-physical losses in this first full-data candidate.
One-clip overfit checkpoint is NOT reused. Formal runs are not launched yet.

Six CUDA tests now pass in13.73 s, including exact optimizer/RNG next-update
resume on a packed two-system fixture. Saved logs, including initial failed
fixture attempt, are under `runs/coordinate_flow_20261005/fixture_evidence/`.
Real full-dataset resume and largest-batch feasibility remain to be checked.

Pre-launch audit found generic `history_encoder.requires_grad_(True)` had
re-enabled unused inverse-Haar decoder parameter flags. Replaced it with the
existing `set_trainable(True)` contract, preserving their frozen non-role.
Those weights had no forward path/gradient/optimizer updates in the earlier
gate/overfit; their numerical results remain valid, but the earlier gate's
nominal trainable-parameter count includes unused flags. Added an explicit
regression assertion. Do not present unused weights as trained capacity.

`tools/run_coordinate_campaign.py` now implements the registered own-GPU
wait/profile/real-resume/final-training/evaluation sequence. CLI import/help
checks passed in torch-ito. Queue `full192_v1` is launched and waiting for the
parent's GPU5 capacity queue to complete; no formal optimizer update yet.
Final pre-launch CUDA6 regression passed in9.46s after the trainability fix.
See PLAN for changed factors and fixed budgets. Formal quality unavailable.
