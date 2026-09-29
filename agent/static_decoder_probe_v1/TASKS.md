# Static decoder probe v1 status

- [x] Read task prompt, repository handoff/migration documents, A/B/C task
  specifications, and coordination release/resource files.
- [x] Create isolated `feat/static-decoder-probe-v1` worktree from
  `cc74a385577e57c64c9623111e0db86e54e48770`.
- [x] Replay only A's published common commits (`a681da6`, `228592c` in this
  branch); do not import the active A loss candidate.
- [ ] Implement static record selection, perturbation-consistent input batch,
  decoder-only loss/update, gradient calibration, and explicit contracts.
- [ ] Run targeted CUDA FP32/BF16/largest-sample smoke and 100--200-update pilot.
- [ ] Commit fixed candidate and preserve independent review request/report.
- [ ] Resolve review findings and rerun affected checks.
- [ ] Lock A recipe release and external-static data availability.
- [ ] Run matched three-arm short screen within eight GPU-hours.
- [ ] Evaluate and deliver manifests, figures, costs, and `report.md`.

Current dependency: A reports `formal_A1_training_running` and
`recipe_release_ready=false`.  This blocks only the formal matched screen, not
implementation, internal-data pilot, profiling, or review preparation.

