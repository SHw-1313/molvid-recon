# Refactor handoff

Current stage: P2a spatial extraction complete; P2 codec/checkpoint migration is next. Isolated local branch `refactor/flat-layout` starts at source HEAD `d8f674aad692c8426cf9240d07b3384f7a043888`. All P1 runtime files and new spatial files are in flat `molvid/`; no legacy runner or file has been retired. The old source worktree remains read-only. See `docs/migration.md` for exact parity evidence.

Environment: interactive `enter-container` works; activate `torch-ito` there. Old-path P0 CUDA suite: 60 passed. New-path full P1 suite: 31 passed. Real frozen manifest/validation hashes, first train/valid decoded records, one raw ATLAS sample and six CUDA graph tensors matched old behavior; test payload stayed sealed. Original artifacts remain untouched. Next: P2 codec weight/optimizer migration, same-input tensor/gradient comparison and checkpoint gate. Codec best checkpoint has optimizer/sampler state but no explicit RNG state.
