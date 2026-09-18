# Refactor handoff

Current stage: P4d5 sample/evaluation YAML complete; P5 audit remains. Isolated local branch `refactor/flat-layout` starts at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`. All P1–P4 runtime files are in flat `molvid/`; no legacy runner or file has been retired. The old source worktree remains read-only. See `docs/migration.md` for exact parity evidence.

Environment: interactive `enter-container` works; activate `torch-ito` there. Old-path P0 CUDA suite: 60 passed; combined P1–P4d CUDA suite: 106 passed. Four real codec controls and the historical 166-moment DiT baseline passed strict state migration gates; short observed-only CLI sample/rollout/evaluation and old/new metrics passed on tiny CUDA clips. Full historical DiT generation, real-system rollout, raw preprocessing and historical data/sampler continuation remain `not_run`. Original artifacts remain untouched; old runners stay live.
