# Refactor tasks

- [x] P0: identify the selected HEAD, source symbols, active paths, artifact identities and historical numerical context.
- [x] P0: run fixed tiny old-path CUDA forward/loss/gradient/update/resume reference in `enter-container` / `torch-ito`.
- [x] P1: flat package, data and geometry; 31 targeted tests plus real artifact/graph parity; local checkpoint.
- [x] P1a: data contract, storage and sampling sub-checkpoint (18 new-path tests); full P1 remains open.
- [x] P1b: geometry, chemistry and equivariant operators sub-checkpoint (24 combined tests); full P1 remains open.
- [x] P2: spatial encoder, codec and weight/optimizer migration; 15 P2 tests, 46 combined tests; local checkpoint.
- [x] P2a: TorchMD/frame encoder extraction with exact deterministic CUDA forward and gradient parity; local checkpoint.
- [ ] P3: latent, condition, DiT and flow; run parity and future-isolation checks and checkpoint.
- [x] P3a: latent/conditioning extraction, 10 old/new and future-mutation tests; local checkpoint.
- [x] P3b: DiT reference/factorized extraction, 6 CUDA parity tests and 62-test combined regression; local checkpoint.
- [ ] P4a–P4d: losses/training, generation/evaluation, CLI/tools; check one step, resume and frozen gradients after each substep.
- [ ] P5: whole-chain checks, import isolation and verified retirement only.

Detailed evidence, unresolved items and the next gate are kept in `docs/migration.md`.
