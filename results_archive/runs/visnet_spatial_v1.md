# visnet_spatial_v1

生成时间由运行路径标识推断；下表按运行而非按原始文件逐行展开。

| 运行 | 时间 | 数据设置 | 模型设置 | 训练设置 | 结果 | 主要原始记录 |
| --- | --- | --- | --- | --- | --- | --- |
| micro_overfit/run_20260901T181240 | 2026-09-01 18:12:40 | data_root=/workspace/PVB/outputs/atlas_selected_trajectories/clip_sto… | 未记录 | steps=500; seed=20260901; precision=fp32; max_tokens=80000 | status=running | visnet_spatial_v1/micro_overfit/run_20260901T181240/protocol.json, visnet_spatial_v1/micro_overfit/run_20260901T181240/summary.json |
| micro_overfit/run_20260901T190657 | 2026-09-01 19:06:57 | data_root=/workspace/PVB/outputs/atlas_selected_trajectories/clip_sto… | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=torchmd_et; spatial_backbone=torchmd_et | steps=500; seed=20260901; precision=fp32; max_tokens=80000 | status=passed; total=0.3131781816482544 | visnet_spatial_v1/micro_overfit/run_20260901T190657/protocol.json, visnet_spatial_v1/micro_overfit/run_20260901T190657/summary.json, visnet_spatial_v1/micro_ov… |
| overview | 未记录 | 未记录 | 未记录 | 未记录 | 未记录 |  |
| tiny_overfit/run_20260901T192557 | 2026-09-01 19:25:57 | data_root=/workspace/PVB/outputs/atlas_selected_trajectories/clip_sto…; frames_per_clip=16; time_bucket_id=dt_100ps | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=torchmd_et; spatial_backbone=torchmd_et | epochs=30; steps=4979; lr=0.0001; seed=20260901; precision=fp32; max_tokens=80000 | status=passed; bond_rmse=0.1763439779714257; total=0.30615650728727 | visnet_spatial_v1/tiny_overfit/run_20260901T192557/aggregate_comparison.json, visnet_spatial_v1/tiny_overfit/run_20260901T192557/protocol.json, visnet_spatial_… |

## 图

- [visnet_spatial_v1__micro_overfit__run_20260901T181240__torchmd_et__train_metrics.loss.png](../figures/visnet_spatial_v1__micro_overfit__run_20260901T181240__torchmd_et__train_metrics.loss.png)
- [visnet_spatial_v1__micro_overfit__run_20260901T190657__torchmd_et__train_metrics.loss.png](../figures/visnet_spatial_v1__micro_overfit__run_20260901T190657__torchmd_et__train_metrics.loss.png)
- [visnet_spatial_v1__micro_overfit__run_20260901T190657__visnet_bonded__train_metrics.loss.png](../figures/visnet_spatial_v1__micro_overfit__run_20260901T190657__visnet_bonded__train_metrics.loss.png)
- [visnet_spatial_v1__micro_overfit__run_20260901T190657__visnet_radius__train_metrics.loss.png](../figures/visnet_spatial_v1__micro_overfit__run_20260901T190657__visnet_radius__train_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_torchmd_et.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_torchmd_et.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_visnet_bonded.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_visnet_bonded.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_visnet_radius.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__eval_metrics_visnet_radius.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__loss_curves.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__loss_curves.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__performance_summary.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__performance_summary.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__torchmd_et__epoch_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__torchmd_et__epoch_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__torchmd_et__train_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__torchmd_et__train_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_bonded__epoch_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_bonded__epoch_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_bonded__train_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_bonded__train_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_radius__epoch_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_radius__epoch_metrics.loss.png)
- [visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_radius__train_metrics.loss.png](../figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__visnet_radius__train_metrics.loss.png)

[原始文件清单](../raw_manifest.csv) · [评估指标数据](../evaluation_metrics.csv)
