# dit_architecture_sequential_v1

生成时间由运行路径标识推断；下表按运行而非按原始文件逐行展开。

| 运行 | 时间 | 数据设置 | 模型设置 | 训练设置 | 结果 | 主要原始记录 |
| --- | --- | --- | --- | --- | --- | --- |
| 20260914_r4_architecture_sequential_v1/baseline | 2026-09-14 | data_hash=9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304eb…; system_count=8; clip_count=8 | mode=disabled; ffn_norm_source=post_adaln; depth=4; source_mode=conditional | successful_updates=4500 | status=PASS; aligned_rmsd=3.6652429941267757; bond_rmse=3.5263224069196806; contact_f1=0.3560098294831647 | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/baseline/evaluation_quick/baseline/summary.json, dit_architecture_sequential_v1/20260914_… |
| 20260914_r4_architecture_sequential_v1/overview | 2026-09-14 | 未记录 | 未记录 | 未记录 | status=FAIL; selected_arm=candidate; total=4 | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/motion_recheck_v1/recheck_summary.json, dit_architecture_sequential_v1/20260914_r4_archit… |
| 20260914_r4_architecture_sequential_v1/round1 | 2026-09-14 | data_hash=9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304eb… | 未记录 | successful_updates_per_arm=5250 | status=REJECT; selected_arm=control; primary_metric=system-equal per-atom future RMSF absolute error | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round1/decision.json |
| 20260914_r4_architecture_sequential_v1/round2 | 2026-09-14 | data_hash=9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304eb…; system_count=8; clip_count=72 | 未记录 | successful_updates_per_arm=5000 | status=PASS; selected_arm=candidate; primary_metric=system-equal future free-generation bond RMSE; aligned_rmsd=2.0304858799464403; bond_rmse=0.5433211738490225; contact_f1=0.7773102586303483 | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/decision.json, dit_architecture_sequential_v1/20260914_r4_architecture_sequential_… |
| 20260914_r4_architecture_sequential_v1/round3 | 2026-09-14 | data_hash=9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304eb…; system_count=8; clip_count=8 | 未记录 | successful_updates_per_arm=5000 | status=KEEP_PARENT; selected_arm=parent; primary_metric=system-equal abs(log((boundary/within)_prediction / (bounda…; aligned_rmsd=1.6873282488260513; bond_rmse=0.542794066160297; contact_f1=0.7798494346276674 | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/decision.json, dit_architecture_sequential_v1/20260914_r4_architecture_sequential_… |
| 20260914_r4_architecture_sequential_v1/round4 | 2026-09-14 | data_hash=9daaf83fe5ee862634f7d1d3530e730adb4a8529ed37a2bc2fd330304eb…; system_count=8; clip_count=8 | 未记录 | successful_updates_per_arm=5000 | status=REJECT; selected_arm=control; primary_metric=E_roll_bond: mean max(0, bond_generated_prefix - bond_true_…; aligned_rmsd=1.6336344652407875; bond_rmse=0.4703942833239208; contact_f1=0.792656208627464 | dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round4/decision.json, dit_architecture_sequential_v1/20260914_r4_architecture_sequential_… |

## 图

此组未发现可画的训练 loss 序列或原有图；单次评估保留数据。

[原始文件清单](../raw_manifest.csv) · [评估指标数据](../evaluation_metrics.csv)
