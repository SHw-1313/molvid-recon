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
- [x] P4a–P4d: losses/training, generation/evaluation, CLI/tools; one-step, resume, frozen-gradient and short-rollout gates passed.
- [x] P4c1: observed-prefix sample and short autoregressive rollout; 3 deterministic CUDA tests; local checkpoint.
- [x] P4c2: geometry/motion metric parity, corrected RMSF reader; 15 targeted generation/metric tests; local checkpoint.
- [x] P4c3: latent/evaluation runner/report, observed-only scoring isolation; 5 tests; P4c local checkpoint.
- [x] P4d1: preprocess CLI and source-manifest helper; exact selection/manifest parity; local checkpoint.
- [x] P4d2: current codec train CLI, dry run and SHA-checked resumed second CUDA step; local checkpoint.
- [x] P4d3: DiT train CLI with real codec and synthetic frozen manifest; exact resumed/uninterrupted two-step CUDA gate; local checkpoint.
- [x] P4d4: SHA-checked DiT inference loader, observed-only sample/evaluate CLI, 106-test combined CUDA gate; local checkpoint.
- [x] P4d5: versioned sample/evaluate YAML, config path relocation and targeted 10-test gate; local checkpoint.
- [ ] P5: whole-chain checks, import isolation and verified retirement only.

Detailed evidence, unresolved items and the next gate are kept in `docs/migration.md`.
