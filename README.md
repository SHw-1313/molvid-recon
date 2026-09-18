# Molvid

Current code lives in the shallow root package [`molvid/`](molvid/). Use the
versioned templates in [`configs/`](configs/) and the current inspection tools
in [`tools/`](tools/). New run artifacts go in `runs/` (ignored by Git).

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
  and hash-indexed raw results.
- [`old/`](old/) — historical source, runners, tests, phase notes and untouched
  experiment outputs. These are archived evidence, not active entrypoints.
- [`TASKS.md`](TASKS.md) and [`HANDOFF.md`](HANDOFF.md) — outstanding gates and
  handoff status.
