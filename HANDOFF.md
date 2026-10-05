# Refactor handoff

## Active independent coordinate-state prototype

Worktree `molvid-coordinate-flow-20261005`, parent `6441ce3`.
Read `agent/coordinate_flow_20261005/{PLAN,REPORT}.md`.
Core implemented; six CUDA gates including exact optimizer/RNG resume passed.
Real4-update gate and one-train-clip512 overfit completed; neither establishes
generalization. `full192_v1` queue is launched, waiting for parent's GPU5
capacity evaluations to complete. Profile/real-resume gates precede training.
Do not modify running parent experiments. Overfit checkpoint is not a parent.

## Active continuous campaign — 2026-10-05

Branch `exp/geometry-scaling-20261005`, worktree `molvid-geometry-scaling-20261005`.
Authoritative current status: [geometry/scaling report](agent/geometry_scaling_20261005/REPORT.md).
User authorizes continued scientific experiments without stage confirmation.
Goal is NOT achieved; no default promotion. C8 exact continuation is running on
GPU6; expanded capacity queue on GPU5 (C4@11000 complete, C8@11000 running).
A1/PVB48 and Gcoord48 completed: reject projection/all Gcoord due motion and
chemistry regressions. C4 extra training improves geometry/motion but still
far from PVB clash quality. BioKinema native1ns pilot and I/O audit completed.
Train-only source-scale inference interventions FAILED; no sampler promotion.
Optional source contract implemented, CPU30/CUDA4 gates passed, old real A1 draw
bitwise identical. Bounded consistent-source full-model retraining is running:
unit/v02 × seeds71/72 ×2048 updates (8192 PLANNED, not yet completed).
GPU7 seed71 v02 then unit; GPU6 seed72 unit then v02 waits C8 completion.
Each lane has final-only48 evaluation/chemistry/paired comparison queued.
v02_71 real16-step checkpoint CPU reload/optimizer audit passed. Details and
exact queue paths are in the report. Do not confuse these with completed
171744-parameter Gcoord head updates. Source framework defaults remain unit.
Baseline train overlap invalidates an independent-test claim for dev48/reserve96.
All experiments use `enter-container` / `torch-ito`; GPU0–4 belong to other work.
Parent Cα handoff below is historical, not the active worktree.

## Current authoritative handoff — 2026-10-05

Branch `exp/ca-detail-20261005`, worktree `molvid-ca-detail-20261005`.
Read [Cα detail report](agent/ca_detail_20261005/REPORT.md) and
[execution audit](agent/ca_detail_20261005/EXECUTION_AUDIT.md) first.
96 frozen-A1 draws; clean/mixed × two training seeds × 512 updates; 288 paired
scores and 48 parameter-free orientation-control scores completed on CUDA.
Three targeted CUDA tests passed. All four initial/final checkpoints freshly
reloaded and scored; historical U/K evidence also checked. No joint win:
learned recovery improves total geometry but worsens inter-residue bonds and
motion amplitude; generated-CA frame transport increases clashes. Oracle
frame transport diagnoses missing orientation, not deployable model success.
Artifacts: `runs/ca_detail_20261005/`; sealed test closed, no default promotion.
No experiment from this task remains running. Later scientific proposals are
not implemented. All sections below are inherited historical handoffs.

## Parent review follow-up — 2026-10-05

Branch `fix/review-followup-20261005`, worktree `molvid-review-followup-20261005`.
Read [completed review follow-up](agent/review_followup_20261005/REPORT.md) and
[fix response](agent/review_followup_20261005/FIX_RESPONSE.md) first. N0 repairs,
72 frozen-A1 draws, stable three-arm S aggregation, 27 U/K head fits, three
ridge fits and 12-reference audit are complete. Scientific stop conditions,
not missing user confirmation, prevent static scale-up/default Mamba/U-K loss.
Artifacts: `runs/review_20261005/aggregate-v2.json` plus each stage directory.
No sealed data or original worktree modified; no checkpoint promoted.
All older sections below are historical handoffs at their stated dates.

## Independent static-response branch (2026-10-04)

This worktree is `exp/static-response-abcd-20261004`, based on `946917b`.
It contains only the shared-static-path transplant plus a matched decoder-only
gradient control, train-only module-budget calibration, and paired generated
validation. No Mamba changes were imported; do not confuse this branch with
`feat/d2-mamba-20261004`. See [current report](agent/static_response_20261004/REPORT.md).
At that historical cutoff only S was selected; U/K were subsequently completed.
## U probe (2026-10-04)

Branch `exp/u-future-probe-20261004`, base `946917b`. Shared A1 feature-cache
producer is active; U trains last-only / history-mean / history-mean-variance
heads, not the generator. See [protocol and evidence](agent/u_probe_20261004/REPORT.md).
Do not interpret these probes as new memory modules or dynamic-generation gains.
## K probe (2026-10-04)

Branch `exp/k-finite-time-20261004`, base `946917b`. K consumes only the
read-only shared A1 feature cache, not U model predictions. K0 fits finite-time
response kernels; K1 compares decoded A1 samples with real MD and predictors.
K2 is disabled. See [protocol and evidence](agent/k_probe_20261004/REPORT.md).

## ABCD integration (2026-10-04, current worktree)

This isolated worktree is `integration/abcd-20261004`, merged in A → D → C → B
order; original source worktrees and live B/C8 jobs were not modified. The
selected scientific baseline remains A1/4-block/legacy history+G+M. B learned
memory, C capacity support and D decoder-only static training are optional;
their union is not a scientifically validated combined model. B M-long and
C8 are still external running experiments, not blockers for this code merge.

Final numerical checks: 75 targeted tests passed without skips; one real A1
inference sample matches the original A tree byte-for-byte; real BF16 K2
streamed backward passes existing tolerances; B M-short/C4/three D artifacts
strict-load; C4→C8 initialization and flow-checkpoint gradients pass. This does
not close the older full production-migration gates below. See the complete
[audit, source identities and evidence](agent/abcd_integration_20261004/REPORT.md).

The bounded generation diagnostic completed 16 validation clips / 192 rows:
Euler16→32 increases clash by 13.90% while MSD MAE improves 3.94%; zero-history
and zero-M damage geometry. Do not treat more solver steps as a geometry fix
or retire memory as unused. These are short-clip/OOD sensitivity results,
not trained ablations or full baseline comparisons.

User clarified that D must pretrain the shared generative trunk, not just the
decoder. [D2 design](docs/static_backbone_pretraining_v2.md) and
[physical-time Mamba design](docs/mamba_temporal_v1.md) are ready for staged
implementation, but not implemented/trained in this merge. Use only
`/data4/users/sihao/workspace/PVB_origin` (`c08e5e3`) as original PVB reference;
the separate `workspace/PVB` is user-modified. `/data4` has about 7.7 GB free:
reserve a safe checkpoint destination before new checkpoint-heavy training.

## Earlier refactor/calibration handoff (historical context)

Current stage: Frame GM Calibration v2 P1--P3 is complete on the dedicated
`feat/frame-gm-calibration-v2` worktree. The earlier operator-directed history
cleanup remains intact: the active surface is `molvid/`, `configs/`, `tools/`,
`benchmarks/`, eight flat-only tests and `results_archive/`. On 2026-09-20
all archive-directory paths were purged from local Git history; that cleanup
did not rerun historical experiments.

Environment: all Python, plotting and scientific checks use `enter-container` / `torch-ito`. The 116-test CUDA pass predates test archival and history cleanup; per user direction, no pytest was rerun now. The retained result bundle contains 388 hash-verified structured files/images, 43 original figures and 55 generated loss curves. It excludes checkpoints, clip stores, logs and sealed test data. `evaluation_metrics.csv` distinguishes final, quick and unspecified scopes. Historical capacity-parent sampler parity is exact, but later-manifest optimizer/data continuation, legacy-runner metrics, artifact export and complete P5 production cutover remain unverified.

## Frame Joint v1 (2026-09-20)

The shallow `molvid/` implementation, targeted CUDA checks, strict resume and
continuation checks, true largest-48-system memory profile, and the original
3-system/9-trajectory tiny run are complete. The historical tiny checkpoint is
`runs/frame_joint_v1_tiny_corrected_260920/frame_joint_step_00000012.pt`
(SHA-256 `c7e18d55a7d5e8bebf29c2dbee31470bcd9dfe5ec73003bb03919ef522622c7e`);
the final-code H4/H8 evaluation is under its `evaluation_current/` directory. A two-GPU
four-step smoke also crossed decoder warmup into flow-start without hanging.

The independent review R1--R6 was then verified and fixed. Current evidence is
under `runs/frame_joint_v1_review_fixes_260920/`: BF16 joint steps and decoder
local-branch startup pass, asymmetric-threshold DDP differs from the single-GPU
update by `2.59e-8`, continuation child ordinary resume and schedule rejection
pass, and motion metric availability is explicit. The retained targeted suite
is now 20 tests. See the appended response in `agent/frame_joint_v1/REVIEW.md`.

The step-12 tiny checkpoint predates the local-message initialization fix. It
remains a frozen pipeline/evaluation artifact, but it is not a valid parent for
new scientific training and does not validate the repaired local topology
branch. At the close of the Frame Joint v1 task, no post-review tiny training
and no Frame Joint v1 48/192 long training had been started; the later Frame GM
Calibration v2 experiments below are separate warm-start studies.

Do not use any existing tiny directory as a new training parent. Full
evidence, commands and limitations are in `agent/frame_joint_v1/HANDOFF.md`;
the deferred training instructions are in `agent/frame_joint_v1/TRAIN_PROMPT.md`.

## Frame GM Calibration v2 (2026-09-24)

P1 repaired temporal metric semantics, rebuilt all dependent schema-v3 fields,
created native-time fixed-history views, and diagnosed a large source-noise /
train-motion scale mismatch without changing the frozen source distribution.
P2 then ran symmetric B0/G/M/GM training for 21,208 successful updates per arm
on 192 systems. No non-baseline arm passed the predeclared joint gate, so B0
was selected (checkpoint SHA-256
`ff8121aa47a7c1885b4ad59ece586dcc90a3d8a12c67d71fb767f586c6b36258`).

P3 ran J0/J1 from that same B0 parent with exact paired main exposure. J1 used
two differentiable four-step source samples every eight successful updates.
Across eight valid systems, J1 minus J0 changed bond RMSE by `-0.08184 Å`
(95% paired bootstrap CI `[-0.08707, -0.07705]`) and fixed-history MSD curve
MAE by `+0.49850 Å²` (`[0.42882, 0.57977]`). Residue RMSF MAE changed by
`+0.01967 Å` (`[-0.02448, 0.07637]`). Because the frozen optional-fork rule
required all three directions to be clear, no bond keep/release fork was run.

P2 and P3 independent reviews are closed with PASS. All formal evaluations were
observed-only Euler16 generation with seeds 0/1/2; sealed test data was not
opened. This is exploratory validation evidence, not production parity. Full
architecture, metrics, costs, hashes and limitations are in
`docs/frame_gm_calibration_v2.md`; operational details are in
`agent/frame_gm_calibration_v2/HANDOFF.md`.
# Active continuous geometry/scaling work — 2026-10-05

Branch `exp/geometry-scaling-20261005`, parent `a8da6de`.
Read [report](agent/geometry_scaling_20261005/REPORT.md) and
[registered plan](agent/geometry_scaling_20261005/PLAN.md).
User authorized continued scientific implementation and real experiments,
no stage-confirmation pauses. Goal not achieved; no new default model.

Live outputs under `runs/geometry_scaling_20261005`:
GPU5 expanded A1/raw+projection; GPU6 real C8 continuation13575;
GPU7 expanded original PVB, followed automatically by Gcoord16-step pilot
and static/mixed71/72 formal2048-step jobs. Gcoord queue checks PVB execution
completion before launching. Keep other GPUs/processes untouched.
Preserve every failure artifact; do not count CPU audits or coordinate
optimization as neural training. Final reports must separate updated module,
updates, systems, compute and completed versus queued work.
# Active execution correction — 2026-10-05

Current source0782453 adds optional block recomputation;7 CUDA gates passed.
`runs/coordinate_flow_20261005/full192_v3` is the CURRENT GPU5 campaign, using
`--checkpoint-blocks`, fresh A1, same192 data/H exposure and two2048 budgets.
Largest-query H4 profile16.3GB and real17 exact-resume passed; formal seed71
observed through60. v1 failed on container-readonlydata2 (0 updates); v2 passed
real-resume but OOM after18–48 because original C8@6000 evaluation sharesGPU5.
Do not interrupt that original lane. Keep both failures; never aggregate their
discarded updates into v3. All outputs now use writabledata4, not data2 links.
Runtime model files must stay frozen while the strict-contract training runs.
Final-only48 evaluation and seed72 follow automatically. No quality result yet.
