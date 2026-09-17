# Baseline C pilot-policy restart

This restart supersedes the unlaunched `origin_train_config_retry.yaml` plan. It uses `origin_train_config_pilot500.yaml`:

- `ubound_per_batch=5000`
- `max_batches=500` in each source wrapper (up to about 1,000 train optimizer batches across ATLAS and MISATO)
- `max_epoch=1`
- seed `20260810`, `pairs_per_clip=1`
- host GPU 7 exposed as process-local GPU 0

The command uses `run_origin_train.py` with the output-local lazy clip adapter and the untouched reference `DynamicBatchWrapper`, `collate_fn`, `DynamicTrainer`, and `dyVAE._train`. The reference wrapper's documented behavior retains only records whose complexity is at most 5,000 for a batch; records above that bound are not put into pilot batches, preventing a 20k-atom singleton OOM. The resulting selected/skipped counts must be reported from the output audit before interpreting validation metrics.
