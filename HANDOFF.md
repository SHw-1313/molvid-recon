# Refactor handoff

Current stage: P0 complete; P1 is next. Isolated local branch `refactor/flat-layout` starts at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`; no production edits or legacy deletion yet. The old source worktree is read-only. See `docs/migration.md` for symbol ownership, active entrypoints, verified artifact hashes and fresh CUDA reference numbers.

Environment: interactive `enter-container` works; activate `torch-ito` there. A TTY-wrapped invocation ran 60 targeted old-path tests on CUDA GPU 5 (all passed) and recorded fixed tiny codec and DiT references. Original artifacts remain untouched. Next: migrate P1 data/geometry and run its targeted checks before P2. The codec best checkpoint contains optimizer/sampler state but no explicit RNG state; account for this when specifying exact resume parity.
