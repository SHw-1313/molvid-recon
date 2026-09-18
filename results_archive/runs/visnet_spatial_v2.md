# visnet_spatial_v2

生成时间由运行路径标识推断；下表按运行而非按原始文件逐行展开。

| 运行 | 时间 | 数据设置 | 模型设置 | 训练设置 | 结果 | 主要原始记录 |
| --- | --- | --- | --- | --- | --- | --- |
| micro_overfit | 2026-09-02 17:46:49 | data_root=/workspace/PVB_visnet_v2_worker/outputs/atlas_selected_traj… | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=visnet_v2_bonded; spatial_backbone=visnet_v2_bonded | steps=500; seed=20260902; precision=fp32; max_tokens=80000 | status=passed; total=0.33591245611508685 | visnet_spatial_v2/micro_overfit/neibu_run_20260902T174649/protocol.json, visnet_spatial_v2/micro_overfit/neibu_run_20260902T174649/summary.json, visnet_spatial… |
| micro_overfit/run_20260902T170456 | 2026-09-02 17:04:56 | data_root=/workspace/PVB/outputs/atlas_selected_trajectories/clip_sto… | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=visnet_v2_bonded; spatial_backbone=torchmd_et | steps=500; seed=20260902; precision=fp32; max_tokens=80000 | status=passed; total=0.32297465205192566 | visnet_spatial_v2/micro_overfit/run_20260902T170456/protocol.json, visnet_spatial_v2/micro_overfit/run_20260902T170456/summary.json, visnet_spatial_v2/micro_ov… |
| micro_overfit_radius | 2026-09-02 22:02:53 | data_root=/workspace/PVB_visnet_v2_worker/outputs/atlas_selected_traj… | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=visnet_v2_radius; spatial_backbone=visnet_v2_radius | steps=500; seed=20260902; precision=fp32; max_tokens=80000 | status=passed; total=0.4000779092311859 | visnet_spatial_v2/micro_overfit_radius/neibu_run_20260902T220253/protocol.json, visnet_spatial_v2/micro_overfit_radius/neibu_run_20260902T220253/summary.json, … |
| tiny_overfit | 2026-09-02 18:59:30 | frames_per_clip=16; time_bucket_id=dt_100ps | model_type=trainer.codec_trainer.PVBCodecModel; mode=topology; ratio=1; backbone=torchmd_et; spatial_backbone=visnet_v2_bonded | epochs=30; steps=4979; lr=0.0001; seed=20260901; precision=fp32; max_tokens=80000 | status=passed; bond_rmse=0.1763439779714257; total=0.30615650728727 | visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/aggregate_comparison.json, visnet_spatial_v2/tiny_overfit/neibu_run_20260902T185930/protocol.json, vis… |

## 图

- [visnet_spatial_v2__micro_overfit__neibu_run_20260902T174649__visnet_v2_bonded_lmax2__train_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit__neibu_run_20260902T174649__visnet_v2_bonded_lmax2__train_metrics.loss.png)
- [visnet_spatial_v2__micro_overfit__run_20260902T170456__torchmd_et__train_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit__run_20260902T170456__torchmd_et__train_metrics.loss.png)
- [visnet_spatial_v2__micro_overfit__run_20260902T170456__visnet_bonded__train_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit__run_20260902T170456__visnet_bonded__train_metrics.loss.png)
- [visnet_spatial_v2__micro_overfit__run_20260902T170456__visnet_v2_bonded_lmax1__train_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit__run_20260902T170456__visnet_v2_bonded_lmax1__train_metrics.loss.png)
- [visnet_spatial_v2__micro_overfit_radius__neibu_run_20260902T220253__visnet_v2_radius_lmax1__resume_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit_radius__neibu_run_20260902T220253__visnet_v2_radius_lmax1__resume_metrics.loss.png)
- [visnet_spatial_v2__micro_overfit_radius__neibu_run_20260902T220253__visnet_v2_radius_lmax1__train_metrics.loss.png](../figures/visnet_spatial_v2__micro_overfit_radius__neibu_run_20260902T220253__visnet_v2_radius_lmax1__train_metrics.loss.png)
- [visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__eval_metrics_comparison.png](../figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__eval_metrics_comparison.png)
- [visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__loss_curves_comparison.png](../figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__loss_curves_comparison.png)
- [visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__performance_summary_comparison.png](../figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__performance_summary_comparison.png)
- [visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__visnet_v2_bonded__epoch_metrics.loss.png](../figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__visnet_v2_bonded__epoch_metrics.loss.png)
- [visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__visnet_v2_bonded__train_metrics.loss.png](../figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__visnet_v2_bonded__train_metrics.loss.png)

[原始文件清单](../raw_manifest.csv) · [评估指标数据](../evaluation_metrics.csv)
