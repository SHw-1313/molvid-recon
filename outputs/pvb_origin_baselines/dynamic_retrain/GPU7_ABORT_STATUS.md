# Baseline D GPU7 run — aborted

Status: **aborted; do not use as a result**.

- Launch attempt: 2026-08-20 16:14:37 Asia/Shanghai.
- `CUDA_VISIBLE_DEVICES=7` mapped host GPU7 to process cuda:0 as requested at
  that time, but host GPU7 was concurrently occupied by C. The GPU reached
  about 58.5 GiB and the original trainer emitted repeated
  `CUDA out of memory, skip batch` warnings.
- D was stopped by Ctrl-C at about DataLoader batch 162/1000. It did not reach
  validation and its partial output is not a baseline result.
- C was not stopped or modified.

The next run uses host GPU4, confirmed free, with the same pair bound and
training policy.
