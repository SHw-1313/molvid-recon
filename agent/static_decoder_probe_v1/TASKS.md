# Static decoder probe v1 status

- [x] Read task prompt, repository handoff/migration documents, A/B/C task
  specifications, and coordination release/resource files.
- [x] Create isolated `feat/static-decoder-probe-v1` worktree from
  `cc74a385577e57c64c9623111e0db86e54e48770`.
- [x] Replay A's published common commits, then the independently reviewed A
  stable source through local `cc1809d`; never edit A's worktree.
- [x] Implement static record selection, perturbation-consistent input batch,
  decoder-only loss/update, gradient calibration, and explicit contracts.
- [x] Run targeted CUDA FP32/BF16/largest-sample smoke and the 128-update P2
  engineering pilot.
- [x] Preserve a fresh-context independent review of fixed commit `73ba705`.
- [x] Implement responses to its orchestration, gamma, optimizer/RNG, external
  eligibility, and traceability findings; fixed-candidate review is pending.
- [x] Lock evaluated A1 recipe release and audit external-static availability.
- [ ] Run A1 gamma/smoke evidence and obtain independent PASS on the fixed
  candidate.
- [ ] Run matched three-arm short screen within eight GPU-hours.
- [ ] Evaluate and deliver manifests, figures, costs, and `report.md`.

Current gate: A1 is released and evaluated.  The existing PDB-derived static
store lacks allowed near-homology exclusion metadata against A validation and
sealed-test identities, so the external arm is fail-closed.  Under the task
specification the baseline/internal paired screen may proceed after A1 CUDA
evidence and independent re-review; no external result will be fabricated.
