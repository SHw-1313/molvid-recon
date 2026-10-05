# Coordinate handedness continuation — implementation, not efficacy

## Frozen recipes, completed real profiles and formal start,19:39

Q12 prerequisite completed with0 updates/unchanged model. `formal_v1` now
exists and its four paired recipes are frozen. Registered rule selected
handedness_weight=.2637673163334546, every8 updates, actual-source Euler16,
margin.1,minimum observed-reference volume.2Å³. Maximum Q12 probe gradient
ratio=.3791220284228847; coefficient×maximum=.1 on those train8 measurements,
not a bound throughout training or for another seed. No development tuning.

Largest-query profiles both completed real GPU auxiliary+RF updates:
seed71 59,700 atom-queries,72.470s,peak57,792,257,536B;
seed72 29,850 atom-queries,49.291s,peak38,447,456,256B.
Both profile updates were discarded, no profile checkpoint used for training.
`profile_queue/status.json` is completed. This proves execution feasibility,
not geometry/motion improvement.

Automatic formal controller has passed control71 prefix25 versus exact
resume16to25 and entered `control_71_formal` onGPU5. Its final budget is2048
new updates from that seed's coordinate2048 parent. Remaining control/chirality
arms, full evaluations and paired audits remain queued. The manifest's
`formal_training_submitted:false` is the immutable freezer-time snapshot;
live state is sibling `formal_v1_training_queue_v1/status.json`. Its historical
17-step minimum was strengthened by the controller to25 steps, which exercises
active auxiliary step24 in the chirality arms. No recipe was changed.

Older paragraphs below are historical snapshots; no efficacy claim yet.

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

## Control71 training completed; full48 evaluation active,20:26

The weight-zero control has reached2048 real optimizer updates. Its final
checkpoint SHA256 is
`4346e024289719cc7ad6f7de3acbafae547eae7b213000df8b1fc5cff13028bd`.
`control_71/final_audit.json` independently confirms cursor2048,348 optimizer
states,102 runtime hashes and all floating model values finite; this audit
made0 model calls. `formal/summary.json` records2023 continuation updates in
2243.691252832301 seconds, in addition to the completed25-update prefix.
That time is not the wall time of the entire2048-update experiment.

The live controller is now `control_71_evaluate`, running actual CUDA full48
forecasts from that exact final checkpoint. No chirality-minus-control effect
can yet be claimed; the auxiliary arm and remaining seed are pending.

The completed-intermediate archive service moved only this control's
`pilot25/coordinate_step_0016.pt`, `pilot25/coordinate_step_0025.pt` and
`resume25/coordinate_step_0025.pt` to
`/data2/sihao/molvid-research-20261005/chirality-training-intermediates-v1/control_71`.
Original paths are symlinks; independent in-container byte-hash reads passed.
Total655,024,602 bytes relocated recoverably, not deleted. Respective SHA256:

- `5c52e04d0189b114bc6080197fe2fd9bd7f6c2be660175ca6075efc7a965de59`
- `321d94438ece8ef3a84c0fb609785f1f7f4d1de32c5df13d6543ae682067a09b`
- `a0e88659eb606ce5d2a9a20d27d461f2f55cef5c04911b9b34d653e65bc5a37f`

Final weights, logs, predictions and raw trajectories were left in place.

## Full control71 result: more training improves geometry but damps motion,20:57

Full48/192 CUDA forecasts completed in2357.486894702539s,0 failures. CPU
`control71_training_effect_v1` compares all matched draws against coordinate71,
A1 and original PVB;0 new neural calls. Existing chemistry audit completed;
the unchanged native-time worktree scorer produced
`control71_ordered_training_effect_v1` (48 systems,480 rows including MD,
persistence and shuffled-MD controls;0 neural calls). No partial-case ranking.

|Metric|coordinate71 parent2048|control71 +2048|original PVB|
|---|---:|---:|---:|
|bond RMSE Å|.163022|.115100|.040154|
|clashes/1000 atoms|50.912283|29.469197|1.371463|
|MSD curve MAE Å²|2.048901|2.329854|2.092712|
|residue RMSF MAE Å|.278153|.335054|.323675|
|RMSF amplitude ratio|.841226|.736375|.887260|
|CA handedness flip fraction|.006707|.003876|.020883|
|internal variogram MAE Å²|.208504|.106078|.093062|
|internal fluctuation RMS error Å|.097257|.058810|.075604|
|increment correlation MAE|.052467|.051934|.113386|

Control-minus-parent paired system95%CIs:bond−.047922[−.049879,−.046077],
clash−21.443086[−26.287642,−17.111388],MSD+.280953[+.228892,+.338566],
residueRMSF+.056901[+.046808,+.066520],amplitude−.104852[−.110520,−.098909].
Control-minus-PVB MSD+.237142[+.071936,+.427043]; residueRMSF+.011379
[−.005259,+.028400] is inconclusive, whereas bond/clash gaps remain clear.

This is NOT uniformly worse motion: internal-distance statistics improve while
MSD/RMSF and amplitude deteriorate. Geometry/internal fluctuation improvements
cannot establish correct global/concerted motion or thermodynamics. The
additional-training arm uses a fresh optimizer and next ordered2048 exposure;
it tests that continuation package, not a causal effect of parameter count.
Only training seed71 is available; system CIs do not establish training-seed
reproducibility. No joint-success or competitive-model promotion.

The original automatic queue has moved through chirality71's25 versus16→25
exact-resume gate and is now running its2048 formal continuation. Existing
candidate/control recipes remain frozen despite this observed development
tradeoff. Next independent diagnostic should test16/32/64 Euler convergence
on fixed training-only systems before adding another distributional objective.
