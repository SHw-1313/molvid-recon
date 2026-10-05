# Molvid 几何与规模连续实验（2026-10-05，进行中）

目标尚未达到：没有证据足以宣布几何/运动同时显著改善或能与 PVB、
BioKinema 竞争。用户已授权阶段间自动推进；失败诊断保留，不等待确认。
默认科学参考仍为 A1，不把新模块自动合入默认模型。

## 已完成的可靠证据

- 实际 checkpoint SHA 核查确认已有 C4/C8 比较都是11000更新，训练曝光、
  主 RNG、辅助 RNG 完全匹配。`retained_identity.json` 的 C4=13000 是旧元数据。
  我曾据该旧文件错误判断不等步数，随后已向用户更正；最终采用
  `capacity_audit_v3/summary.json`，没有重跑已完成的全部评估。
- C4→C8：键长 RMSE .26477→.25907 Å；MSD MAE3.26567→3.15696 Å²；
  残基 RMSF MAE .29836→.28102 Å；但 clash/1000atoms119.23→126.79，
  severity .03579→.04021。增大规模有部分收益，不是几何/运动共同胜利。
- A1 的8个训练体系潜变量诊断是真实 CUDA 推理，38.48秒，8个新生成 draw，
  **0训练更新**。128维向量通道有效秩：clean1.13、motion2.27；
  这不是整分子的内在维数。源噪声总范数约19.59/atom-frame，而真实运动
  1.25–4.08，支持继续研究表示和源分布匹配，不足以证明单一因果。
- 无未来目标的坐标投影控制：8训练体系，35.91秒，32坐标优化步，
  **0神经训练更新**。键长 .26161→.05432 Å、clash96.22→59.79；
  MSD3.29543→3.33304略坏、RMSF幅度比 .85171→.83428下降。
  因而不能把几何投影直接宣传为运动提升。
- 冻结48新开发体系和96预留确认候选：MMseqs30%identity/50%双向coverage，
  每个干净序列簇一个，排除202个已知训练/开发曝光体系。已生成开发集
  288片段：2anchors × 100/400/1000ps × 48体系；原生10ps精确取帧。
  确认集及旧 sealed test 轨迹未读取；完整预训练/结构家族排重仍待完成。
- C8 原源码和本分支从11000到11001下一步：775模型张量、2136优化器张量、
  RNG/训练状态精确一致。随后已自动开始真实全模型续训至原计划13575。
- PVB 原始仓库与发布版 ATLAS checkpoint 已实际 CUDA 推理；模型约997万参数。
  原库未改动；进程隔离 NumPy1.26.4 解决旧 `np.compat.long` 兼容问题。
  不是全环境降级。原子顺序、类型、键图核对通过。

## 正在运行 / 排队

|实验|实际工作|状态文件（以下相对 runs/geometry_scaling_20261005）|
|---|---|---|
|C8 continuation|全模型真实训练11000→13575，原目标/数据/优化器/RNG|`c8_continuation_v2/formal/train_metrics.jsonl`|
|expanded A1|48体系、2anchors、2seeds；原始与固定坐标投影分列|`expanded_a1/rows.json`|
|expanded PVB|同一观测末帧，原生 H1、100ps、SDE10、12步，无 refiner|`expanded_pvb/rows.json`|
|Gcoord static/mixed|各2seeds×2048真实几何模块更新，192静态训练体系；mixed加32体系真实生成样本|`gcoord_campaign/queue.json`|

Gcoord 等待本轮 PVB 释放 GPU7，再自动16步执行检查及4个正式训练。
它用当前预测坐标逐层重建半径图，和此前 S 静态 latent 梯度路由不同。
只训练新几何模块，**不声称在训练 DiT 或新的动力学生成器**。
通过几何/运动共同检验后才考虑整合到主干。CUDA检查已覆盖稠密旋转、
平移、跨体系隔离、padding、identity初始化与反向传播。
首次梯度检查发现最后一层存在不会影响坐标输出的隐层更新参数，已删除。

## 外部对照边界

新48开发体系不能直接称为对所有基线独立的测试集。PVB只有 H1，Molvid
使用 H4，应同时报告条件和计算量不同。短1.2ns片段不是热力学收敛证明；
单条随机轨迹对单条 MD 的路径 RMSE 也不是生成分布正确性的充分指标。
正式竞争结论需要补充独立体系、原生物理时间、长程/多副本、分布与时间
相关量，以及 MD 对 MD 的抽样误差参考。

官方来源核实：

- [PVB 官方代码与静态/动态统一目标说明](https://github.com/yaledeus/PVB)。
- [BioKinema 官方复现实验](https://github.com/IDEA-XL/BioKinema/blob/main/experiments/README.md)：
  短程/复合物 sqrt 与长程 beta=.25 不应混用。
- [BioKinema ATLAS 原生复现协议](https://github.com/IDEA-XL/BioKinema/blob/main/experiments/atlas_benchmark/README.md)：
  发布 sqrt 权重，MSA，1ns步长100ns rollout；本轮100ps短片段不可冒称此协议。

发现之前已下载并验证正式 sqrt 权重（SHA c96ad408…dece39），位于
`/workspace/molvid-integration-abcd-20261004/inputs/baselines/`，并有原生
2ns执行 smoke。`/data1/repo/BioKinema/checkpoints/biokinema.pt` 的 SHA 是
78434720…99c1b，是不同文件，不能用文件名相近代替身份验证。

## 失败及未完成项

保留：第一次 latent deterministic reporting失败、旧 retained metadata
容量审计失败、因已有配对结果而中止的冗余容量评估、续训 parity 比较
checkpoint键名错误、开发材料首版 atype键名错误、PVB首版 NumPy错误。
这些都不是已完成实验，不计入有效样本或训练结果。

仍需：完整扩展评估、Gcoord正式训练及同样本验证、C8最终checkpoint比较、
原始基线预训练曝光审计、BioKinema匹配物理时间的公平比较、长程动力学
和热力学验证。没有自动把局部改善升级为最终成功。
