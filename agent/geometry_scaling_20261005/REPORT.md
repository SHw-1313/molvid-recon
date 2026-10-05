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
- PVB 扩展48体系已完成192次真实生成，1371.6秒，0训练更新。
  体系等权：bond RMSE .04015 Å、clash/1000atoms1.3715、MSD MAE2.0927 Å²、
  RMSF幅度比 .88726。不能与旧8体系均值直接比较，等待同48体系 A1 配对统计。
- CA/N/C/CB 有向体积审计发现 PVB 约2.09% Cα手性翻转（MD为0）；
  因此漂亮的键长/clash 不等于全部化学正确。该指标不是全分子 R/S 指派。
  仅第一个开发片段中，A1→固定投影的翻转率 .254%→3.150%，
  这是单片段预警，不能与 PVB48均值作优劣比较。
- Gcoord四个正式模型已完成，每个171,744参数、2048次更新、192个静态训练体系，
  合计8192更新；static/mixed各两种子。mixed每组512个生成辅助更新。
  全部优化器状态step2048、日志1..2048、权重有限性通过CPU审计；配对组静态
  曝光完全一致，首次干预第4步之前三步精确一致。各运行135–214秒是小模块
  训练，不是全DiT训练。静态输入来自MD快照，不冒称外部PDB预训练。
- BioKinema发布sqrt权重原生1ns、12未来帧pilot已完成，约47.3秒，0训练。
  MSA实际深度4738；698原子逐帧对应。只用观测首帧求一个SO(3)+平移，
  撤销原生整体增强后首帧最大误差3.24e-6 Å。CPU输出审计通过，可在本片段
  计分，但没有跨体系性能结论。原库/模型权重没有改动。
- A1真实权重CUDA检查：隐藏未来坐标改写不改变输出；显式源尺度(1,1)与旧
  默认采样逐位一致。源尺度改动只作为训练集推理分布诊断，尚无重训练结果。

## 正在运行 / 排队

|实验|实际工作|状态文件（以下相对 runs/geometry_scaling_20261005）|
|---|---|---|
|C8 continuation|全模型真实训练11000→13575，原目标/数据/优化器/RNG|`c8_continuation_v2/formal/train_metrics.jsonl`|
|expanded A1|48体系、2anchors、2seeds；原始与固定坐标投影分列|`expanded_a1/rows.json`|
|expanded PVB|已完成48体系、192次生成|`expanded_pvb/summary.json`|
|Gcoord static/mixed|已完成4×2048真实更新|`gcoord_campaign_v2/queue.json`、`gcoord_training_audit/summary.json`|
|Gcoord expanded|GPU7等A1完成；用相同缓存draw评估4个最终模型|`gcoord_evaluation_queue_v2/status.json`|
|expanded capacity|GPU5等A1；依次C4/C8@11000、C4/C8@13575|`expanded_followup_queue_v3/status.json`|
|source-bank diagnostic|GPU7等Gcoord验证；8训练体系×2seeds×4源尺度，0更新|`source_bank_diagnostic/`（尚未启动模型）|
|BioKinema native1ns pilot|已完成并通过I/O审计；单体系，不是benchmark|`biokinema_matched1ns_audit/summary.json`|

Gcoord 用当前预测坐标逐层重建半径图，和此前 S 静态 latent 梯度路由不同。
只训练新几何模块，**不声称在训练 DiT 或新的动力学生成器**。
通过几何/运动共同检验后才考虑整合到主干。CUDA检查已覆盖稠密旋转、
平移、跨体系隔离、padding、identity初始化与反向传播。
首次梯度检查发现最后一层存在不会影响坐标输出的隐层更新参数，已删除。
首个真实pilot因旧128类型表不覆盖蛋白残基118–137失败，0更新；已使用
共享138类型词表，边界检查和CUDA测试通过后重启v2，失败产物保留。
当前距离/中心力结构为O(3)等变，不能凭架构独自选择正确手性；后续验证
把手性/近平面退化作为否决条件，不以几何两项单独晋级。

## 外部对照边界

新48开发体系不能直接称为对所有基线独立的测试集。PVB只有 H1，Molvid
使用 H4，应同时报告条件和计算量不同。短1.2ns片段不是热力学收敛证明；
单条随机轨迹对单条 MD 的路径 RMSE 也不是生成分布正确性的充分指标。
正式竞争结论需要补充独立体系、原生物理时间、长程/多副本、分布与时间
相关量，以及 MD 对 MD 的抽样误差参考。

官方动态split元数据审计（`external_overlap_v3`）：开发48中BioKinema train46、
val1、test1；PVB train26。预留96中Bio train88、val3、test5，PVB train56。
因而这96也不能直接充当公平基线独立确认集。已识别16个已知动态训练
序列簇干净候选，但尚不能证明外部静态预训练独立；未打开其轨迹目标。
本模型192训练体系相对PVB790/Bio1266存在明显动态数据覆盖差距，不能把
差距完全归结于模型参数。已仅按元数据冻结512训练体系候选（192+320），
保护开发/确认/外部测试簇；**尚未材料化、未开始512体系训练**。

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

仍需：完整扩展评估、Gcoord同样本验证、C8最终checkpoint比较、
原始基线完整静态预训练曝光审计、BioKinema多体系匹配物理时间比较、长程动力学
和热力学验证。没有自动把局部改善升级为最终成功。
