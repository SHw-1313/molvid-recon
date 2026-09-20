# 关键实验结果

按“生成时间 → 数据设置 → 模型设置 → 训练设置 → 结果”阅读。时间仅由运行目录名推断；没有可靠时间的记录标为“未记录”。这里不重算指标、不把 quick 和 final 评估合并，也不把不同实验的 loss 直接排名。长训练序列只给曲线，单次/所选 checkpoint 的评估给数值表。

| 生成时间 | 数据设置 | 模型设置 | 训练设置 | 结果与来源 |
| --- | --- | --- | --- | --- |
| 2026-08-26 09:44:44 | Atlas 选定轨迹的 topology-bond overfit | ratio1 无 temporal / ratio1 temporal / ratio4 temporal | FP32、500 steps、最多 80k tokens | 三组最终 total loss 分别 **0.6897 / 0.3210 / 0.4037**，均较初值下降；只说明此 overfit。 [原始总结][atlas-top] · [loss 图][atlas-top-plot] |
| 2026-08-26 10:56:36 | 同一选定轨迹，distance-only bond | 同上三组 | FP32、500 steps、最多 80k tokens | 最终 total loss **0.7260 / 0.3049 / 0.3963**，均较初值下降。 [原始总结][atlas-distance] · [loss 图][atlas-distance-plot] |
| 2026-09-01 19:25:57 | 16-frame Atlas tiny overfit/holdout | TorchMD ET、ViSNet radius/bonded | 30 epochs、LR 1e-4、FP32 | 开发后端建议 **TorchMD ET**；其 holdout future dRMSD **0.6448 Å**，ViSNet radius/bonded 为 **0.6822/0.6798 Å**。 [原始比较][visnet-v1] · [epoch 图][visnet-v1-plot] |
| 2026-09-02 18:59:30 | 同类 16-frame tiny overfit/holdout | TorchMD ET 与 ViSNet v2 bonded 等 | 30 epochs、LR 1e-4、FP32 | v2 bonded future dRMSD **0.6463 Å**、future bond RMSE **0.1608 Å**；其 `tiny_quality_pass=false`，不能据单个指标宣称胜出。 [原始比较][visnet-v2] · [epoch 图][visnet-v2-plot] |
| 2026-09-03 21:38:07 | T0 16-frame、100 ps clip | state/detail R1、R2、R4 与 matched-pooling 对照 | 原协议 30 epochs、FP32 | R4 state/detail 最终 future RMSD **11.1386 Å**、bond RMSE **3.1070 Å**。这是 `final_*` 评估，原记录另列 best-checkpoint SHA，不能误作 best-checkpoint 复评。 [原始比较][codec-t0] · [loss 图][codec-t0-plot] |
| 2026-09-08 | 冻结的 train/valid manifest；test 未打开 | state/detail DiT R2 与 R4 | target 4500 steps、seed 20260907、H4/H8 | 两份 pilot summary 均为 `PASS`；末条 validation RF total 为 **0.7757 / 0.8261**。但两份 summary 的 `completed_steps=4100` 与末条 validation `step=4500` 不一致，不能据此认定完整 4500-step resume。 [R2][pilot-r2] · [R4][pilot-r4] · [验证曲线][pilot-plot] |
| 2026-09-10 | source A/B 记录的主聚合为 1 system、52 samples | R4 DiT；Gaussian 与 conditional source | 共同固定 20k steps；不跨 source 按 RF loss 排名 | conditional/Gaussian future aligned RMSD **2.0777/2.1646 Å**、bond RMSE **0.7049/0.8945 Å**、contact F1 **0.7379/0.7124**；原协议仍标记为待复核，不作因果胜出结论。 [原始总结][source-ab] · [验证曲线][source-ab-plot] |
| 2026-09-14 | 8-system quick valid；train/valid 隔离 | C48 来源的 R4 factorized DiT baseline | 4500 successful updates | quick valid 的 H4/H8 aligned RMSD **3.7211/3.7501 Å**、bond RMSE **3.5417/3.5308 Å**；见下表，不与 round2 final scope 混比。 [checkpoint 总结][dit-baseline] · [quick 数据][dit-baseline-eval] |
| 2026-09-14 | 相同顺序实验中的分轮 valid/rollout | R1 amplitude、R2 geometry、R3 refiner、R4 corrupted history | R1 每臂 5250 updates；R2–R4 每臂 5000 | 决策依次 **REJECT / KEEP candidate / KEEP_PARENT / REJECT**；每轮以自己的主指标和 guardrail 决策。下表列所选 checkpoint 的评估数据。 [R1][r1] · [R2][r2] · [R3][r3] · [R4][r4] |
| 2026-09-14 | source checkpoint reassessment | 已有 R4 checkpoint | 只读复评；未训练 | 记录状态为 `A_SOURCE_REASSESSMENT_V2_PARTIAL_NO_CUDA_EVALUATION`，不提取虚构的 CUDA 指标。 [原始总结][reassessment] |
| 未记录 | Atlas + MISATO pair baseline | 原静态/动态条件 A–D | 原始配置见归档 | 记录的 valid overall loss：C **1.8152**、D **1.3128**；这是不同条件的汇总，不当成当前 flat-package 对照。 [原始总结][origin] · [训练曲线][origin-plot] |

## 所选 checkpoint 的评估数据（不另画图）

下列 H4/H8 数值是各原始评估文件中 `valid.*.system_equal` 的 aligned RMSD / bond RMSE，单位 Å。`quick` 与 `final` 是原协议的不同评估范围；这里只存数据，不把它们拼成一条曲线。

| 阶段 / 所选臂 | checkpoint SHA-256 前缀 | 范围 | H4 aligned / bond | H8 aligned / bond | 原始评估 |
| --- | --- | --- | ---: | ---: | --- |
| baseline | `808eb51e5c59` | quick | 3.7211 / 3.5417 | 3.7501 / 3.5308 | [summary][dit-baseline-eval] |
| R2 candidate | `7fec9b339201` | final | 2.0305 / 0.5433 | 1.8976 / 0.5297 | [summary][r2-eval] |
| R3 parent | `7fec9b339201` | quick | 1.6982 / 0.5349 | 1.6683 / 0.5229 | [summary][r3-eval] |
| R4 control | `994e69e65315` | quick | 1.6735 / 0.4772 | 1.6467 / 0.4656 | [summary][r4-eval] |

R4 的选择依据是后续生成前缀的 `E_roll_bond`，不是上表单段 bond RMSE：control **0.14554**、candidate **0.14970**，所以原决策为 REJECT candidate。[原始决策][r4]

其余 `g5`、`atlas_only_modified` 等只有图或逐步日志、没有足以复原完整设置/最终评估的元数据；已保留在[实验组索引](README.md)及[原始包清单](raw_manifest.csv)，没有补造结论。更细的单次评估标量见 [evaluation_metrics.csv](evaluation_metrics.csv)，`scope=unspecified` 不等于 final。所有训练曲线均来自原始 history/metrics，原有图一并保留在 `figures/`。

[atlas-top]: runs/atlas_selected_trajectories.md
[atlas-top-plot]: figures/atlas_selected_trajectories__overfit_three_systems__run_20260826T094444__plots__loss_curves.png
[atlas-distance]: runs/atlas_selected_trajectories.md
[atlas-distance-plot]: figures/atlas_selected_trajectories__overfit_three_systems_distance_only__run_20260826T105636__plots__loss_curves.png
[visnet-v1]: runs/visnet_spatial_v1.md
[visnet-v1-plot]: figures/visnet_spatial_v1__tiny_overfit__run_20260901T192557__torchmd_et__epoch_metrics.loss.png
[visnet-v2]: runs/visnet_spatial_v2.md
[visnet-v2-plot]: figures/visnet_spatial_v2__tiny_overfit__neibu_run_20260902T185930__visnet_v2_bonded__epoch_metrics.loss.png
[codec-t0]: runs/state_detail_codec_v2.md
[codec-t0-plot]: figures/state_detail_codec_v2__t0_remote_final__run_20260903T213807__loss_curves.png
[pilot-r2]: runs/dit_state_detail_pilot_v1.md
[pilot-r4]: runs/dit_state_detail_pilot_v1.md
[pilot-plot]: figures/dit_state_detail_pilot_v1__full_20260908_S4500__ratio4_state_detail__seed20260907_S4500__pilot_summary.validation.loss.png
[source-ab]: runs/dit_source_ab_v1.md
[source-ab-plot]: figures/dit_source_ab_v1__20260910_r4_source_ab_v1__conditional__validation_history.loss.png
[dit-baseline]: runs/dit_architecture_sequential_v1.md
[dit-baseline-eval]: runs/dit_architecture_sequential_v1.md
[r1]: runs/dit_architecture_sequential_v1.md
[r2]: runs/dit_architecture_sequential_v1.md
[r3]: runs/dit_architecture_sequential_v1.md
[r4]: runs/dit_architecture_sequential_v1.md
[r2-eval]: runs/dit_architecture_sequential_v1.md
[r3-eval]: runs/dit_architecture_sequential_v1.md
[r4-eval]: runs/dit_architecture_sequential_v1.md
[reassessment]: runs/dit_source_reassessment_v2.md
[origin]: runs/origin_baselines.md
[origin-plot]: figures/origin_baselines__summary__loss_curves.png
