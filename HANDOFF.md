# Refactor handoff

Current stage: P1a data sub-checkpoint complete; P1 itself is incomplete. Isolated local branch `refactor/flat-layout` starts at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`. Only new `molvid/data` modules and tests have been added; no legacy code has been retired. The old source worktree is read-only. See `docs/migration.md` for precise evidence.

Environment: interactive `enter-container` works; activate `torch-ito` there. Old-path P0 CUDA suite: 60 passed. New-path P1a data suite: 18 passed. Original artifacts remain untouched. Next: migrate remaining P1 files, compare old/new fixed data/geometry behavior and complete P1 gate before P2. The codec best checkpoint contains optimizer/sampler state but no explicit RNG state; account for this when specifying exact resume parity.
