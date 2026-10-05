# Molvid

Current diagnostic branch: `exp/coordinate-static-balance-20261005`.
[Train-only static/dynamic gradient plan](agent/coordinate_static_balance_20261005/PLAN.md).
No training updates or default promotion in this probe; inherited campaigns
below run in their original worktrees.

Current independent worktree: `exp/coordinate-flow-20261005`.
[Coordinate-state prototype and evidence](agent/coordinate_flow_20261005/REPORT.md).
No default promotion; parent campaigns below continue in their own worktree.

Current worktree (2026-10-05): `exp/geometry-scaling-20261005`.
[Continuous geometry/scaling campaign](agent/geometry_scaling_20261005/REPORT.md)
and [registered plan](agent/geometry_scaling_20261005/PLAN.md).
C8 exact next-update parity passed; real continuation and expanded48 native
forecast evaluations are running. Current-coordinate geometry recovery is an
independent matched static/mixed candidate, not a promoted default.

Parent worktree: `exp/ca-detail-20261005`.
[Generated Cα / oracle detail-recovery experiment](agent/ca_detail_20261005/REPORT.md)
and [GPU execution audit](agent/ca_detail_20261005/EXECUTION_AUDIT.md).
Completed 96 frozen-A1 draws, four 512-update detail heads and paired validation.
No joint geometry/motion win; no default promotion. Local-frame orientation is
a completed post-pilot diagnostic, not a trained replacement or independent test.

Parent worktree: `fix/review-followup-20261005`.
[Independent-review fixes and completed N0–N4 diagnostics](agent/review_followup_20261005/REPORT.md),
[engineering fix response](agent/review_followup_20261005/FIX_RESPONSE.md).
E1–E5 repaired; S/U/K tools consolidated, A1 remains the scientific reference.
No Mamba/default checkpoint promotion or U/K generator loss. Sections below
describe the inherited branch history, not unfinished current work.

Independent static-response branch (2026-10-04):
[design, controls and evidence](agent/static_response_20261004/REPORT.md).
Starts directly from the ABCD integration; S0/Sdec/Strunk isolate static
gradient routing without Mamba, capacity or external-data changes.
Independent U branch: [frozen-history future-moment probe](agent/u_probe_20261004/REPORT.md).
No generator updates, S/D2 or Mamba changes; fixed ABCD/A1 parent.
Independent K branch: [finite-time response-kernel probe](agent/k_probe_20261004/REPORT.md).
K0 predictors plus bounded K1 diagnosis; no K2 or generator updates.

Current code lives in the shallow root package [`molvid/`](molvid/). Use the
versioned templates in [`configs/`](configs/) and the current inspection tools
in [`tools/`](tools/). New run artifacts go in `runs/` (ignored by Git).

2026-10-04 integration: [ABCD audit and merge handoff](agent/abcd_integration_20261004/REPORT.md).
The selected scientific baseline remains A1/4-block; B/C/D are retained optional
capabilities, not jointly promoted experiments. Next-stage designs:
[shared-trunk static pretraining](docs/static_backbone_pretraining_v2.md) and
[physical-time Mamba](docs/mamba_temporal_v1.md). Shared-static support is now
implemented here; Mamba remains an isolated candidate outside this tree.

```bash
enter-container
conda activate torch-ito
python -m molvid.cli.preprocess --help
python -m molvid.cli.train_codec --config configs/codec_train.yaml --dry-run
python -m molvid.cli.train_dit --config configs/dit_train.yaml --dry-run
python -m molvid.cli.sample --config configs/sample.yaml
python -m molvid.cli.evaluate --config configs/evaluate.yaml
```

Fill in checkpoint hashes, approved codec, frozen manifest and observed-prefix
paths before sampling. Only observed coordinates reach the generator; held-out
future coordinates belong to evaluation. Full production cutover still depends
on the missing real-artifact checks in [`docs/migration.md`](docs/migration.md).

- [`results_archive/`](results_archive/) — curated historical tables, figures
  and a hash-indexed raw-result bundle. Superseded source and full experiment
  outputs were purged from Git history after this bundle was verified.
- [`TASKS.md`](TASKS.md) and [`HANDOFF.md`](HANDOFF.md) — outstanding gates and
  handoff status.
- [`docs/frame_gm_calibration_v2.md`](docs/frame_gm_calibration_v2.md) —
  completed P1--P3 geometry, motion and sampled-distribution calibration study.
