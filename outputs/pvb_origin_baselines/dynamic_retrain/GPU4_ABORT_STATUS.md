# Baseline D GPU4 run — aborted

Status: **aborted; do not use as a result**.

- Launch: 2026-08-20 15:58:08 Asia/Shanghai.
- Reference: `/workspace/PVB_origin` (clean checkout; no edits).
- Command used `--gpus 4` with the container-path config
  `baseline_d_train_container.yaml`.
- Policy was `ubound_per_batch=5000`, `max_batches=500` per source,
  `max_epoch=1`, giving 1,000 planned optimizer steps.
- The run shared host GPU4 with the user's main-thread A workload. The user
  reported sustained OOM/full utilization; D was stopped by Ctrl-C before
  validation. The log reached about step 867/1000 and ended in
  `loss.backward()` with `KeyboardInterrupt`.
- No D validation metric or completed D checkpoint exists from this run.
- The earlier same-command path attempt failed before training because
  `/data4/...` was read-only inside the container; the container-path config
  fixed only that path issue and was not an algorithmic change.

Required restart condition: wait for A to finish, confirm host GPU7 is idle,
then launch with `CUDA_VISIBLE_DEVICES=7` and original-process `--gpus 0`.
