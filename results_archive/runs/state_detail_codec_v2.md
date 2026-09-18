# state_detail_codec_v2

生成时间由运行路径标识推断；下表按运行而非按原始文件逐行展开。

| 运行 | 时间 | 数据设置 | 模型设置 | 训练设置 | 结果 | 主要原始记录 |
| --- | --- | --- | --- | --- | --- | --- |
| preflight | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |  |
| t0_remote_final/run_20260903T213807 | 2026-09-03 21:38:07 | frames_per_clip=16; time_bucket_id=dt_100ps | model_type=trainer.codec_trainer.PVBCodecModel; mode=ratio1_state_detail; ratio=1; backbone=torchmd_et; spatial_backbone=torchmd_et | epochs=30; steps=4976; seed=20260903; precision=fp32; max_tokens=80000 | status=passed; bond_rmse=3.0498554574400196; total=49.45879677014473 | state_detail_codec_v2/t0_remote_final/run_20260903T213807/aggregate_comparison.json, state_detail_codec_v2/t0_remote_final/run_20260903T213807/micro/ratio1_sta… |

## 图

- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__block_detail_metrics.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__block_detail_metrics.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__evaluation_curves.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__evaluation_curves.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio1_state_detail.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio1_state_detail.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio2_state_detail.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio2_state_detail.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio4_matched_pooling.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio4_matched_pooling.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio4_state_detail.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curve_ratio4_state_detail.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curves.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curves.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio1_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio1_state_detail__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio2_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio2_state_detail__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio4_matched_pooling__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio4_matched_pooling__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio4_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__micro__ratio4_state_detail__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__performance_summary.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__performance_summary.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio1_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio1_state_detail__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio2_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio2_state_detail__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio4_matched_pooling__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio4_matched_pooling__train_metrics.loss.png)
- [state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio4_state_detail__train_metrics.loss.png](../figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__ratio4_state_detail__train_metrics.loss.png)

[原始文件清单](../raw_manifest.csv) · [评估指标数据](../evaluation_metrics.csv)
