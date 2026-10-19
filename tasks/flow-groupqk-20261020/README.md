# Flow A：化学单元共享未来 Q/K 的匹配小试

2026-10-20 已完成实现、针对检查、两臂各512更新、24条原始Q50、ComplexMD0.3.1共用评分及播放器导出。**保留为小幅正向研究候选，不替换主线。** 共享Q/K改善了六例蛋白总项、主链及含侧链的键/角异常、碰撞、局部平面与CA时间相关，幅度大致保持；但CA RMSF空间Pearson六例下降，MISATO手性和部分极端异常退化，严重几何及欠幅仍未解决。单seed、六个开发体系的结果不证明最初的时间路由假设就是根因。

[原始50帧对照播放器](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/comparison_viewer/index.html) · [完整比较摘要](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/comparison/comparison_summary.json) · [逐draw指标](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/comparison/draw_metrics.csv) · [运行/成本回执](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/execution_summary.json)

## 实际改动与匹配范围

独立分支 `research/flow-groupqk-20261020`，基点 `2f50820`，科学实现提交 `1f37a05`，未push。主要修改 [bridge_history.py](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/molvid/codec/bridge_history.py) 与 [coordinate_state.py](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/molvid/flow/coordinate_state.py)：空间编码和历史attention后，对标量h按sample/component/block做masked mean，再经原norm、物理时间及Q/K计算组级未来时间权重；每个未来时点保留自己的权重行。权重广播给所属原子，h/v Value、输出投影和残差均逐原子。向量只做原有通道映射。历史attention、CollectiveFeedback、后两层空间、输出头、source、目标、loss、35NFE采样器不变；`query_qk_mode=atom`为兼容默认。

蛋白使用现有残基分组。真实TRAIN 3M35有1629蛋白重原子/223残基block和37配体原子/单block；DEV4J8T有1046蛋白原子/128残基block和28配体原子/单block。配体block覆盖整个柔性配体，故本轮按edge_mask逐原子fallback，没有强制整配体共享路由。

两臂同静态RF父模型 `/data1/sihao/molvid-derived/coordinate-static-pretrain-v1/seed71/flow/static/coordinate_final.pt`（2048静态更新、dynamic_examples=0），seed71，各512动态更新、lr1e-4/warmup128、原128 ATLAS+128 MISATO和H4/Q12或H8/Q8。配置仅query_qk_mode不同；没有额外几何loss。trainer只复用已有stop-after32保存/恢复机制，最终updates始终512，scheduler不变。前32包含在512中。

两臂各6例×2draw，生成seed17/29，H4/Q50、18区间/35NFE；ATLAS100ps与MISATO80ps原生网格。六例参考MD/拓扑/阈值沿用主线。**历史2048步static→A及其续训不是本轮匹配对照，不能拿本次512结果直接声称超过完整主线。**

## 几何：绝对值与未解决部分

下表均为atom→group，先均值同体系两draw，再在数据集内体系等权。异常比例单位为%；碰撞分母为合格非键原子对，不能与键/角分母混看。主链重原子N/CA/C/O/OXT；“含侧链项”包含任一其他蛋白重原子，因而包含CA–CB与主侧链邻角。

| 指标 | ATLAS | MISATO |
|---|---:|---:|
| 蛋白键异常% | 11.5657→10.9946 | 8.6047→8.1417 |
| 蛋白角异常% | 10.8441→10.4647 | 8.5926→8.3677 |
| 主链键异常% | 9.7390→9.3839 | 7.8902→7.5528 |
| 主链角异常% | 5.9328→5.7620 | 4.4487→4.3583 |
| 含侧链键异常% | 13.3301→12.5513 | 9.2867→8.7056 |
| 含侧链角异常% | 14.9382→14.3868 | 11.9464→11.6131 |
| 蛋白碰撞比例% | 0.04920→0.04585 | 0.02254→0.02131 |
| 蛋白局部平面异常% | 4.1460→4.0208 | 3.0189→2.9025 |
| 蛋白手性异常% | 0.55809→0.55591 | **0.31886→0.32897** |

蛋白总/主链/含侧链键角、碰撞、平面均六例改善；这是小幅收益，并非化学有效轨迹已解决。两臂全部12draw的蛋白键和角首坏均为未来frame0。极端值不全部改善：MISATO配体键max excess为2.983474→3.390580Å，蛋白手性max excess为1.153578→1.674150相对有向体积；这里是数据集所有draw的最大值，不是平均最大值。完整mean/max excess和逐例方向在比较摘要。

现有原始定位（原子索引从0开始）：group/5TV1 seed17的蛋白键首坏frame0=80ps，关联884个原子，例含5/6=ARG1 NE/CZ、16/17=VAL2 C/O、18/19/20=PHE3 N/CA/CB；该draw键异常9.06419%、max excess2.303868Å，最大值不声称发生在首坏帧。group/2XSB seed29配体手性首坏frame38=3120ps，索引3331/3332/3333/3340为MOL425 C4/C5/C6/O4。播放器可定位这些原子；本轮没有追加WebGL人工播放验证。

## 运动与时间保护

RMSF幅度比以MD=1为参照；更大不自动更好。柔区由参考MD残基质心RMSF上四分位固定，未按模型结果重新选区。CA向量ACF比较lag1/2/3，分别对应ATLAS100/200/300ps、MISATO80/160/240ps；PCA和contact ACF沿用共用引擎。未新增长轨迹或把旧clock/order协议结果混入本轮。

| 指标 | ATLAS atom→group | MISATO atom→group |
|---|---:|---:|
| CA RMSF幅度比 | 0.61176→0.60970 | 0.67650→0.67520 |
| 固定柔区质心RMSF幅度比 | 0.42678→0.42602 | 0.46370→0.46297 |
| CA RMSF空间Pearson | **0.63493→0.62754** | **0.44799→0.44061** |
| CA向量ACF RMSE | 0.47651→0.46319 | 0.42185→0.41163 |
| PCA1 ACF RMSE | 0.61803→0.60914 | 0.70075→0.69998 |
| 接触数ACF RMSE | N/A，无配体 | 0.206214→0.202683 |

CA ACF六例改善，PCA ACF五例改善（2XSB从0.573064升至0.575376）。Contact ACF三例4J8T/2XSB/5TV1分别0.146484→0.144977、0.180751→0.173583、0.291406→0.289490。CA RMSF Pearson六例均下降，不能只报时间相关改善。

| 体系 | CA幅度比 atom→group | 固定柔区质心幅度比 atom→group | CA ACF RMSE atom→group | Backbone位移均值Å atom→group（MD） |
|---|---:|---:|---:|---:|
| 1whz | .96536→.96196 | .71278→.71127 | .32305→.31109 | .93740→.93683 (.97607) |
| 3vnb | .55508→.55214 | .36816→.36646 | .49470→.47992 | 1.16688→1.16968 (2.10483) |
| 6e5y | .31482→.31502 | .19940→.20034 | .61179→.59858 | 1.37579→1.39021 (2.72609) |
| 4J8T | .83206→.82960 | .55313→.55143 | .35797→.34746 | .87438→.87599 (1.04954) |
| 2XSB | .67659→.67443 | .47375→.47256 | .34670→.33643 | 1.01099→1.01474 (1.18682) |
| 5TV1 | .52084→.52158 | .36421→.36492 | .56087→.55100 | 1.14423→1.16045 (1.56214) |

这里的backbone位移是现有全局对齐后相对首个MD未来帧的RMSD时间均值，不是sliding lag-MSD；与唯一未来的配对RMSD也不独立裁定动力学好坏。总体幅度宏平均改变不足0.4%，没有明显靠压幅换收益的证据；但6e5y仍严重欠幅（CA约0.315、固定柔区约0.200），本次并未恢复真实大运动。

## 未来attention残差

正式24draw中被动收集detach统计，没有额外生成或修改模型算式。下表六例等权、先均值两seed；相对RMS为query-attention输出残差/对应输入，潜特征单位，不能解读为Å。完整72行及sum-square/count见[残差CSV](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/comparison/query_residual_summary.csv)。

| flow阶段 | 相对h RMS atom→group | 相对v RMS atom→group | 绝对h残差RMS | 绝对v残差RMS |
|---|---:|---:|---:|---:|
| early [0,1/3) | 4.306%→3.919% | 5.239%→5.195% | .77569→.70350 | .16945→.16850 |
| middle [1/3,2/3) | 4.402%→4.013% | 6.602%→6.869% | .93002→.87746 | .20339→.21253 |
| late [2/3,1] | 3.809%→3.670% | 10.190%→10.245% | .80464→.84148 | .32120→.33136 |

六例early相对h/v均降、middle相对h降/v升；late相对h均降，v除6e5y微降外五例微升。不是所有时间更新都缩小，也不能据此证明共享时间来源是几何改善的唯一机制；输入特征自身也变了，且RMSF空间Pearson退化需要保留。

## 验证、成本与复现

4项必要CUDA检查通过，覆盖组内共享权重而Value/输出不同、sample/component隔离、mask/空组有限行为、逐帧物理时间、梯度、旋转等变、默认atom输出兼容、静态parent严格加载与真实TRAIN3M35原source35NFE。有效蛋白原子1629，路由组260=223残基+37配体单原子。没有为本次结果追加测试矩阵。

两臂最终逐步data/noise/progress相同，generator/torch/CUDA/python/numpy RNG及mixture/exposure七项最终状态一致。静态parent的temporal输出为零，保留正式初始化；最终group Q/K更新范数1.06071/1.18720、Value h/v .86441/.78589、output h/v .28877/.60480均有限且超过名义weight-decay-only。atom Q/K .96244/.98646。参数更新只证明训练路径打开，不证明科学收益。详见[最终检查](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/checks/group_512.json)。

原RF明确重分配余额14325秒；实际GPU进程合计7263.558秒=**2.017655 A100 GPUh**：CUDA26.513秒、训练4929.603秒、24draw生成2307.442秒，零失败/重试。GPU6已释放；未用7061.442秒不自动用于新实验。CPU prepare/score/compare均exit0，总墙时422.975秒、CPU428.099秒，新增GPU0；[CPU回执](/data4/users/sihao/workspace/MOLVID-FLOW-GROUPQK-20261020/runs/flow_groupqk_20261020/process_logs/cpu_execution.json)。EDM低噪声的独立预算和运行未受影响。

配置、命令分别保留在本任务configs和run/process_logs；原始checkpoint为run/{atom,group}/coordinate_final.pt，原始Q50为run/short50_generated/{atom,group}/{case}/coordinates_seed{17,29}.pt，协议记录parent/实际模型源/物理时间/35NFE/残差。初始atom32开始时HEAD仍为基点但使用随后提交1f37a05的同一冻结实现；恢复及group记录1f37a05。最终只补任务说明与CPU检查器本地提交，不push、不修改科学文件。管理状态仍只由PM维护molvid_recon三文件。
