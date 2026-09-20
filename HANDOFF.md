# Refactor handoff

Current stage: operator-directed history cleanup. The active surface is
`molvid/`, `configs/`, `tools/`, `benchmarks/`, eight flat-only tests and
`results_archive/`. On 2026-09-20 all archive-directory paths were purged from
all local Git history; no historical experiment was rerun.

Environment: all Python, plotting and scientific checks use `enter-container` / `torch-ito`. The 116-test CUDA pass predates test archival and history cleanup; per user direction, no pytest was rerun now. The retained result bundle contains 388 hash-verified structured files/images, 43 original figures and 55 generated loss curves. It excludes checkpoints, clip stores, logs and sealed test data. `evaluation_metrics.csv` distinguishes final, quick and unspecified scopes. Historical capacity-parent sampler parity is exact, but later-manifest optimizer/data continuation, legacy-runner metrics, artifact export and complete P5 production cutover remain unverified.
