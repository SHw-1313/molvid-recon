# Refactor tasks

- [x] P0: identify the selected HEAD, source symbols, active paths, artifact identities and historical numerical context.
- [x] P0: run fixed tiny old-path CUDA forward/loss/gradient/update/resume reference in `enter-container` / `torch-ito`.
- [x] P1: flat package, data and geometry; 31 targeted tests plus real artifact/graph parity; local checkpoint.
- [x] P1a: data contract, storage and sampling sub-checkpoint (18 new-path tests); full P1 remains open.
- [x] P1b: geometry, chemistry and equivariant operators sub-checkpoint (24 combined tests); full P1 remains open.
- [x] P2: spatial encoder, codec and weight/optimizer migration; 15 P2 tests, 46 combined tests; local checkpoint.
- [x] P2a: TorchMD/frame encoder extraction with exact deterministic CUDA forward and gradient parity; local checkpoint.
- [x] P3: latent, condition, DiT and flow; 68-test combined parity and future-isolation gate; local checkpoint.
- [x] P3a: latent/conditioning extraction, 10 old/new and future-mutation tests; local checkpoint.
- [x] P3b: DiT reference/factorized extraction, 6 CUDA parity tests and 62-test combined regression; local checkpoint.
- [x] P3c: flow/source/sampling extraction, 6 old/new and inference-isolation tests; local checkpoint.
- [x] P4a1: reconstruction losses and time-bucket normalization; 4 CUDA parity tests; local checkpoint.
- [x] P4a: codec one-step/frozen behavior, strict new-format RNG/optimizer/cursor resume and rejection rollback; 76 combined tests; local checkpoint.
- [x] P4b1: clean/noisy DiT batch separation and frozen future-bond decode gradients; local checkpoint.
- [x] P4b2: DiT trainer, strict historical 166-moment conversion and new-format resume; 82 combined tests; local checkpoint.
- [ ] P4a–P4d: losses/training, generation/evaluation, CLI/tools; check one step, resume and frozen gradients after each substep.
- [x] P4c1: observed-prefix sample and short autoregressive rollout; 3 deterministic CUDA tests; local checkpoint.
- [ ] P5: whole-chain checks, import isolation and verified retirement only.

Detailed evidence, unresolved items and the next gate are kept in `docs/migration.md`.
