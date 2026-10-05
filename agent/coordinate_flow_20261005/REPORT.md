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
## Storage gate failure and corrected restart

`full192_v1` failed before any optimizer update: data2 is readable but mounted
read-only INSIDE the container. Merely resolving a path was not a sufficient
writeability test. Preserve the controller traceback and unused links as the
failed attempt; no active checkpoint was moved or deleted, no mount changed.
After a real temporary-directory write check in the container and checking
15.49GiB free, restart identical code/data/config as `full192_v2` on writable
data4. The BioKinema unused read-only links were separately unlinked; their
target directories and all result files remain intact.

v2 largest nominal79,600atom-frame/H8 profile completed a real CUDA update:
sample atlas_4yal_A_R2_dt_400ps_w000001,39,800 valid query-atoms,
4.68249s, peak29,753,665,024bytes. This disposable profile update is not counted
as formal training or evidence of quality. Real17/resume16→17 gates follow.

v2 real17/resume16→17 **passed bitwise** over model, optimizer, all RNG,
cursor and contract. Formal continuation logged updates18–48 (31 successful
updates), then OOM on the next forward. The original capacity lane's
checkpoint6000 evaluation was concurrently using47.52GiB on GPU5; it is not
this campaign's completed final-capacity queue and was not stopped. A profile
maximizing total16-frame atom cost selectedH8 and missed the larger H4 query
activation cost. v2 has no completed formal2048 checkpoint; no quality claim.

Optional DiT-block non-reentrant activation recomputation implemented, default
off; static/inference semantics and model state keys unchanged. Three distinct
test blocks cover forward/RNG identity, gradient and update numerical parity;
all7 targeted CUDA tests passed in9.12s at2.5% process memory cap. The corrected
profile selects largest query-atom-frame cost. Register fresh-source-contract
`full192_v3` with recomputation, same192 data/H schedule, FP32 objective, two
seeds and fixed2048 budget. v2 discarded updates are not counted in that budget.

v3 is now in real formal training (observed through step60, past v2's failed
step49). The maximum-query profile uses H4/59,700 query-atoms, sample
atlas_4yal_A_R1_dt_100ps_w000007:6.53485s, peak16,305,031,168bytes. Its real
17/resume16→17 whole-checkpoint comparison is again bitwise exact. Step17 SHAs:
45a75480…52b1 uninterrupted,685d48a4…e7621 resumed (serialization bytes differ;
all compared tensor/state/RNG values agree). A CPU comparison of the saved v2
and v3 step17 checkpoints finds ZERO differing floating model-state tensors
and identical flow RNG despite the execution optimization; no fresh model call
was used for that audit. CUDA test log is retained under
`runs/coordinate_flow_20261005/fixture_evidence/checkpoint_blocks_cuda7.log`.
Formal peak through step60 is15,341,581,312bytes. No generated-validation result
exists yet; the single-clip overfit numbers are not substituted for one.
