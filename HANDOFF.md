# Refactor handoff

## Independent static-response branch (2026-10-04)

This worktree is `exp/static-response-abcd-20261004`, based on `946917b`.
It contains only the shared-static-path transplant plus a matched decoder-only
gradient control, train-only module-budget calibration, and paired generated
validation. No Mamba changes were imported; do not confuse this branch with
`feat/d2-mamba-20261004`. See [current report](agent/static_response_20261004/REPORT.md).
S/U/K are not all implemented: only S is selected for this first experiment.

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
