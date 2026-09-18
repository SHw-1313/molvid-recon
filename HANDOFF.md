# Refactor handoff

Current stage: P1b geometry/chemistry sub-checkpoint complete; P1 is still incomplete. Isolated local branch `refactor/flat-layout` starts at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`. New `molvid/data`, `molvid/geometry` and `molvid/equivariant.py` modules have been added; no legacy code has been retired. The old source worktree is read-only. See `docs/migration.md` for evidence.

Environment: interactive `enter-container` works; activate `torch-ito` there. Old-path P0 CUDA suite: 60 passed. New-path P1 data/geometry suite: 24 passed, including CUDA graph and distance-bond tests. Original artifacts remain untouched. Next: trajectory I/O/preprocess, manifest, runtime/config and real-store parity before P1 gate. The codec best checkpoint contains optimizer/sampler state but no explicit RNG state; account for this when specifying exact resume parity.
