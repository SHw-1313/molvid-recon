# Molvid

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
[physical-time Mamba](docs/mamba_temporal_v1.md); neither is implemented yet.

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
