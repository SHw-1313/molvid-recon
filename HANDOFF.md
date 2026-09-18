# Refactor handoff

Current stage: operator-directed root cleanup. The active surface is `molvid/`,
`configs/`, `tools/`, `benchmarks/`, eight flat-only tests and `results_archive/`.
Legacy source, runners, phase notes, config, old/new parity tests and 490 tracked
output files were moved intact into `old/`; no historical experiment was rerun.

Environment: all Python, plotting and scientific checks use `enter-container` / `torch-ito`. The 116-test CUDA pass was before this batch's legacy fixture relocation; per user direction, no pytest was rerun now. The archive script packaged 388 structured source files/images, copied 43 original figures, drew 55 loss curves and verified every archive member SHA-256. It excludes checkpoints, clip stores, logs and sealed test data. `evaluation_metrics.csv` distinguishes final, quick and unspecified scopes. Historical capacity-parent sampler parity is exact, but later-manifest optimizer/data continuation, old-runner metrics, artifact export and complete P5 production cutover remain unverified. Historical runners that name moved test paths should be reproduced at their original Git commit, not called from this checkout.
