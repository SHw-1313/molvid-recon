# Refactor tasks

- [x] P0: identify the selected HEAD, source symbols, active paths, artifact identities and historical numerical context.
- [x] P0: run fixed tiny old-path CUDA forward/loss/gradient/update/resume reference in `enter-container` / `torch-ito`.
- [x] P1: flat package, data and geometry; 31 targeted tests plus real artifact/graph parity; local checkpoint.
- [x] P1a: data contract, storage and sampling sub-checkpoint (18 new-path tests); full P1 remains open.
- [x] P1b: geometry, chemistry and equivariant operators sub-checkpoint (24 combined tests); full P1 remains open.
- [ ] P2: spatial encoder, codec and weight/optimizer migration; run parity checks and checkpoint.
- [ ] P3: latent, condition, DiT and flow; run parity and future-isolation checks and checkpoint.
- [ ] P4a–P4d: losses/training, generation/evaluation, CLI/tools; check one step, resume and frozen gradients after each substep.
- [ ] P5: whole-chain checks, import isolation and verified retirement only.

Detailed evidence, unresolved items and the next gate are kept in `docs/migration.md`.
