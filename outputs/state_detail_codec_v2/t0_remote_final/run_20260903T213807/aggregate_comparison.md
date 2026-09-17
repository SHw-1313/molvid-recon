# State/detail codec v2 T0 comparison

Status: **WAITING_FOR_OPERATOR_REVIEW**.

The four controls use the same frozen TorchMD frame encoder, exact 441/117 lazy split, FP32 losses, and one seed.

| Mode | Ratio | Active elements/atom | Params | Trainable | Future RMSD | Future dRMSD | Bond RMSE | Velocity RMSE | Accel RMSE | RMSF corr | Frequency | Detail norm | Train min | Tok/s | E2E s/batch | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ratio1_state_detail | 1 | 2048 | 643,016 | 82,432 | 11.4242 | 10.1549 | 3.08318 | 0.0209102 | 0.000358905 | 0.766342 | 9.38264 | n/a | 64.75 | 88798.3 | 0.192 | 24.01 |
| ratio2_state_detail | 2 | 2048 | 724,936 | 164,352 | 11.2419 | 9.90053 | 3.12291 | 0.0128179 | 0.000192065 | 0.784969 | 6.36788 | 0.19100824652648554 | 89.26 | 64409.4 | 0.196 | 24.51 |
| ratio4_state_detail | 4 | 1024 | 856,008 | 295,424 | 11.1386 | 9.7673 | 3.107 | 0.00817435 | 0.000126111 | 0.782398 | 4.08926 | 0.2383449328381841 | 89.75 | 64058.7 | 0.198 | 24.52 |
| ratio4_matched_pooling | 4 | 1024 | 1,102,536 | 541,952 | 10.5448 | 9.30605 | 3.34422 | 0.0106722 | 0.000170632 | 0.714334 | 4.96784 | n/a | 64.51 | 89127.4 | 0.192 | 24.02 |

## Fixed protocol

- Manifest: 441 train / 117 late holdout, intersection=0, T=16, dt_100ps.
- Loader: lazy mmap, replacement=false, exact epoch coverage, FP32, max_tokens=80000, 30 complete epochs, safety cap=6000 steps.
- Seed=20260903; spatial_backbone=torchmd_et; common frozen frame state=/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/preflight/torchmd_frame_encoder_step_00004979.pt.
- Loss schedule is resolved after the loader length is frozen: 0%-10% coordinate/local/bond; 10%-30% adds velocity; 30%-100% adds acceleration.

## Evidence paths

- Aggregate JSON: `/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.json`
- Aggregate CSV: `/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.csv`
- Plots: `/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curves.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curves.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/evaluation_curves.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/evaluation_curves.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/block_detail_metrics.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/block_detail_metrics.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/performance_summary.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/performance_summary.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio1_state_detail.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio1_state_detail.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio2_state_detail.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio2_state_detail.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio4_state_detail.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio4_state_detail.pdf, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio4_matched_pooling.png, /workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/loss_curve_ratio4_matched_pooling.pdf`
- Micro: `/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/micro`
- Unfreeze smoke: `/workspace/PVB_state_detail_codec_v2/outputs/state_detail_codec_v2/t0_remote_final/run_20260903T213807/unfreeze_smoke.json`

## Stop condition

No T1 manifest, T1 training, static/dynamic large-data run, DiT, observation adapter, forecasting, rollout, or later architecture phase was started.
