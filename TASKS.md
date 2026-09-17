# Refactor tasks

- [x] P0: identify the selected HEAD, source symbols, active paths, artifact identities and historical numerical context.
- [ ] P0: run fixed tiny old-path CUDA forward/loss/gradient/update/resume reference in `enter-container` / `torch-ito` (environment access blocked).
- [ ] P1: flat package, data and geometry; run targeted checks and checkpoint.
- [ ] P2: spatial encoder, codec and weight/optimizer migration; run parity checks and checkpoint.
- [ ] P3: latent, condition, DiT and flow; run parity and future-isolation checks and checkpoint.
- [ ] P4a–P4d: losses/training, generation/evaluation, CLI/tools; check one step, resume and frozen gradients after each substep.
- [ ] P5: whole-chain checks, import isolation and verified retirement only.

Detailed evidence, unresolved items and the next gate are kept in `docs/migration.md`.
