# Refactor handoff

Current stage: operator-directed history cleanup. The active surface is
`molvid/`, `configs/`, `tools/`, `benchmarks/`, eight flat-only tests and
`results_archive/`. On 2026-09-20 all archive-directory paths were purged from
all local Git history; no historical experiment was rerun.

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
branch. No post-review tiny training and no 48/192 long training was started.

Do not use any existing tiny directory as a new training parent. Full
evidence, commands and limitations are in `agent/frame_joint_v1/HANDOFF.md`;
the deferred training instructions are in `agent/frame_joint_v1/TRAIN_PROMPT.md`.
