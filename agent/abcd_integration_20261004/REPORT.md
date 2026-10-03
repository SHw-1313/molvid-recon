# ABCD 合并审计与下一代模型设计

日期：2026-10-04。工作树：`/workspace/molvid-integration-abcd-20261004`，分支 `integration/abcd-20261004`。这是代码集成和研究路线审计，不是四条实验都已完成、也不是超过 PVB/BioKinema 的声明。sealed test 未打开。

## 结论先行

默认科学基线保留 **A1、4 个 DiT blocks、legacy history+G+M**。B 的完整胜负未定；C8 仍由原工作树运行，不阻塞合并；旧 D 只是 decoder-only 对照，**没有完成用户要求的静态结构主干预训练**。保留这些可复现实验入口，不把它们未经验证地一起开启。

最严重的问题不是少一层 attention 或少一个 loss，而是生成路径与容易的重建路径不一致、几何与运动目标互相牵制，以及缺少同协议 baseline 证据。Mamba 有明确的时间表示改进空间，但不单独解决生成几何问题。新 D 必须让静态梯度穿过共享 DiT，不能继续只优化 decoder。

## 1. 最终代码与实验状态分开记录

| Lane | 本次纳入的源码 HEAD | 已知实验结果 | 合并后定位 |
|---|---|---|---|
| A motion/geometry | `e5e5fe0f559cb50d1d2ad9ab42f07692277db0bc`；科学实现 `af60c05` | A0–A3 完成，各 10,606 updates；A1 被选中 | 默认 recipe 与实际 source 辅助损失 |
| B learned memory | `5092dcbe0e70f8864d72e0d13d3301db7d46929c` | M-short 9,741 完成；M-long 仍在续跑，尚无成对最终选择 | 可选 history backend，不替换默认 |
| C capacity | `e4beede3bbcbed68cfb059bd5686e6613147b340` | C4 13,575 完成；C8 运行中 | 合并深度扩展、warm start、checkpointing 支持；不认定 8 层胜出 |
| D static decoder | `29e9645`；实现 `07b6ea76`，最终评估 `abd8f7b` | 三臂 500 dynamic updates，静态臂另有 125 static steps | 已完成的有限 pilot；仅作 D2 的 decoder-only 对照 |

A1 checkpoint：`/workspace/molvid-recon-gm-motion-v3/runs/gm_motion_memory_v3/A/r16/configs/formal/A1/frame_joint_step_00010606.pt`，SHA-256 `52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`。

B 的 artifact-only 分支 `artifacts/learned-memory-v1-step5394` (`172362b`) 是较早的部分状态，不覆盖正式代码或冒充最新终局。原 B 工作树的 tracked runtime metadata 与 untracked 文档均保留。共享 coordination JSON 可能滞后，源码、正式 checkpoint、实际进程/日志分别核对，不能只看旧 HANDOFF。

## 2. 最小风险合并顺序与冲突处理

从共同点 `cc74a385577e57c64c9623111e0db86e54e48770` 新建隔离 worktree，没有切换或编辑任何 live training 工作树，也没有移动原 `molvid_recon` 主分支。

| 顺序 | Merge commit | 风险控制 |
|---|---|---|
| A | `7da861d` | 先建立完成评估的 A1；保留最终 artifact/report，不让其它分支携带的旧 A 结果覆盖 |
| D | `3b88cb6` | 静态 decoder 路径是显式 optional，默认不启用；保留其控制组语义 |
| C | `f70e388` | 同时保留 D 的静态构造和 C 的 flow-field checkpointing；扩层映射/geometry block indices 显式入 contract |
| B | `f85896b` | 最后处理重叠最大的 model/history/trainer；strict loader、warm-start retirement、raw-history 路径均保留 |

随后 `3aad9be` 保留不同后端已有的反向传播数值顺序，并加入设计/诊断工具。共享 `_train_step_streamed` 中，legacy 使用 A 原有的逐分支 condition 重算，B 使用一次 observed graph 加梯度 proxy。为了代码看起来统一而强行让 legacy 也走 B 的 proxy，会改变 BF16 梯度累加；本次没有将这种额外数值改动混入合并。

另一个实际冲突：D 允许有限 generation seeds，C 的 evaluation reuse/coverage 校验曾假定固定行数。合并后从 protocol 的 seeds 推导所需 draw 数，增加“D seed scope 可复用、缺一 draw 必须拒绝”的回归检查；没有降低 protocol/hash/budget 校验。

合并接受的是各能力的兼容实现，不是 A+B+C+D 的全组合训练验证。尤其 B 的 history schedule 与 A 不同，不能仅设置几个开关就称为公平 A1+B 比较。

## 3. 为什么现在的进展仍然很有限

### 3.1 A 的改善不足以证明核心问题已解决

以下是 A 的同协议 ordinary validation；不是与 PVB/BioKinema 的横向排名。

| 指标 | A0 | A1 selected | A2 motion | A3 motion+band |
|---|---:|---:|---:|---:|
| bond RMSE / Å | 0.293627 | 0.291442 | 0.314318 | 0.292752 |
| clashes / 1,000 atoms | 152.359 | 146.159 | 187.297 | 164.642 |
| residue RMSF MAE / Å | 0.344480 | 0.348561 | 0.309478 | 0.311098 |
| 跨体系 RMSF spread ratio | 0.291876 | 0.285463 | 0.373607 | 0.370106 |

A1 clash 仅改善 4.07%，severity 改善 5.11%，RMSF MAE 反而增加 1.18%；没有达到原先约 30% 的理想几何改善。A2 提升运动，但 clash/severity 恶化 22.93%/29.10%。A3 部分挽回几何，仍未解除 trade-off。高 system Pearson（约 0.90）与很低的 spread 可以并存：体系排序对，并不代表运动幅度和分布对。

这不是再选一个最好看的指标就能掩盖的结果。A1 是当前较安全的折中，不是强模型。只有一个 training seed、八个 validation systems，bootstrap 不能代替独立训练重复。来源：[A 最终报告](../gm_motion_memory_v3/report.md)。

### 3.2 生成 latent → 几何流形，是首要待定位瓶颈

旧 D 的 dynamic clean-latent oracle bond RMSE 约 0.0205 Å，而生成路径约 0.30 Å，差一个数量级。说明“给 decoder 正确 latent，它能解码”不等于“flow 能生成 decoder 熟悉的 latent”。这是明确的路径差距，但还不能单独区分 RF 误差、solver 误差、latent manifold 和 decoder 鲁棒性各占多少。

teacher 每 atom/frame 的 h128 + v(3×128) 有 512 个数，却监督三维坐标；坐标 vector stem 初始重复投影到多通道。这里存在冗余、相关性和 off-manifold source 的风险，**尚未测 rank/covariance，不能直接断言维度过大或 frozen teacher 是性能上限**。PVB 自身也有坐标均值旁路，不能仅凭存在 shortcut 就判 molvid 更差。

当前 RF 主目标在 latent 空间，clean/near 部分几何训练又有 decoder-only/detach 边界。A 的 actual-source clash 才真正经过采样路径，但 cadence 约 1/8、K2、Euler8。active weighted loss 小不自动等于梯度弱；应看 shared-trunk 梯度范数、夹角、clip 后更新量，不能直接因 loss 数字小把权重放大。

本次补充 solver/memory/oracle 诊断用于缩小上述解释空间，结果见第 5 节。后续应把结构预训练和几何约束放在**实际生成路径**，而不是继续只强化 easy reconstruction。

### 3.3 时间表征有重叠，也有过早压缩

- legacy history 的 temporal difference、Haar/detail gates 与独立 M 都编码运动变化；H4 被压成 1 个 memory token，H8 为 2 个，再在各 DiT 层反复读取。
- 13 维 physical-time features 被多处各自投影。100 ps 网格上部分 Fourier 项重复，但线性/log 项仍有信息；不能把整个时钟误称为完全 alias。
- B 在 raw history 上做轻量空间编码，再 residue pooling、25 个手工物理分析 tokens、8 个 learned slots。它增加覆盖，却仍可能在建模前丢掉组内相反方向运动；八个 slots 没有时间顺序，不能直接交给 Mamba 当八帧。
- DiT temporal 和 decoder temporal 是两套时间混合器。它们不是必然重复：一个生成 latent、一个恢复坐标；但必须消融，确认 decoder 是否只是用平滑补偿错误 latent。

因此 Mamba 的重点是统一真实时钟、保留有序历史和多尺度状态、消除多套运动摘要，不是简单把 attention 名称替换掉。Q8/Q12 很短，不保证 Mamba 更快；收益需要 matched-capacity、matched-exposure 检验。

### 3.4 几何也不是只有 bond/clash

G 与 decoder 的局部消息主要使用 observed 几何和共价 1-hop/1–3 拓扑，尚缺显式随生成坐标更新的非键相互作用归纳偏置。模型仍有空间 attention，不能说没有全局相互作用；但通用 attention 不自动等价于排斥体积、手性、键角和扭转势。提高 RMSF 时靠错误键角/局部扩张获得幅度，是必须排除的失败模式。

这条问题线与时间表示独立。先诊断 generated 几何、再按一次一个因素比较非键消息/结构约束；不要把更多 loss、Mamba、8 层和外部 PDB 同时改变，最终无法归因。

### 3.5 D 当前结论与用户真正目标错位

旧 D 用 500 dynamic + 125 static optimizer steps，仅训练 decoder；internal MD 静态臂使 ordinary clash/severity 降低 8.38%/10.11%，但 RMSF MAE/MSD MAE 增加 0.85%/1.77%。external PDB 对 generated ordinary geometry 没有确定收益。

在 static sigma=0.1 Å/axis 时，raw noisy coordinate MSE 约 0.009983 Å²，而 internal/external 输出为 0.059454/0.053098 Å²。虽然相对旧 baseline 大幅变好，**仍不是净去噪器**；sigma=0.2 时也差于 raw input。只与旧 decoder 比较会产生过强的进展印象。来源：[D 最终报告](../static_decoder_probe_v1/report.md)。

旧 D 还多做了 optimizer steps，其 near-homology 控制曾被豁免。它回答的是“额外 decoder 辅助训练有没有用”，不是“共享主干静态预训练是否能迁移动力学”。不能据此判静态主干预训练无效。

## 4. D2 与 Mamba 的具体改进方案

完整设计分别在 [静态共享主干 D2](../../docs/static_backbone_pretraining_v2.md) 和 [物理时间 Mamba](../../docs/mamba_temporal_v1.md)。两者目前都是**设计完成，尚未实现/训练**；不会悄悄改变本次兼容合并的科学基线。

D2 的关键取舍：

1. static Q1 和 dynamic history+Q8/Q12 使用**同一套** DiT spatial/G/FFN、输入输出投影与 decoder；static 不复制成 H4，也不赋假物理时间。
2. static 经过 corrupted condition → RF/shared trunk → generated endpoint → decoder。clean teacher 只供 target/loss，不泄露 clean geometry 给条件。
3. static 显式跳过 history/M/physical-time/future-temporal/decoder-temporal，避免大量单结构训练把时间模块推向零运动。flow progress `s` 仍然共享，它不是物理时间。
4. 每个 global update 累积 dynamic + 小 static batch，**一次 AdamW step**。dynamic 的采样曝光、成功步数、LR 不变，static RNG/cursor 独立。不要延续旧 D 的额外 optimizer-step 混杂。
5. 先按体系/结构平均各自 loss，再用至少 16 个 train batch pairs 标定共享主干梯度。初始 static/dynamic 梯度范数比 `ρ=0.25`，两侧试 `0.10/0.50`；不是 25% 样本或 loss。同时记录逐层 cosine、decoder/DiT 梯度与 clip 系数。
6. 当前 A1 先做 joint continued-pretraining 对照；真正 static-pretrain→dynamic 的样本效率问题另从共同初始化比较。static-only 阶段不更新 temporal 参数；dynamic 阶段保留 static replay。
7. 首轮 D0 dynamic-only、D1 decoder-only、D2-I internal shared、D2-E external shared，保持 4 blocks；然后才做 Mamba×D2 的 2×2 交互。预算优先独立结构与训练 seeds，不先扩大 loss 数量。

Mamba 最终替换 history、独立 M、history-query 读出、future temporal、decoder temporal 与分散 clock 建模；保留 spatial/G、teacher/stats 和 flow-s 调制。实际 ps 间隔乘内容选择性速率，标量门控对 vector xyz 共享，state FP32；padding 不推进状态；query 从相同 observed state 独立读出；noisy-future state 不跨 Euler 步缓存。先 history 再 future/readout 再 decoder/clock，避免一口气替换后无从定位退步。

## 5. 合并验证与补充实验

验证证据正在收尾；这里只列出已完成与待完成边界，不把初版检查冒充最终检查。

- 九个与 A/B/C/D 接口直接相关的文件共 75 个针对性 tests 已在中间合并版通过；最后保留 legacy 反向数值顺序后，将重跑这些相关测试。
- 原始 A 工作树与合并树，同一 A1 checkpoint、真实 validation clip `atlas_2ejn_B_R1_a000000_dt_100ps_H4`、Euler16、seed0，generated coordinates SHA-256 均为 `5cfabe39aaecd52716102d852c3b94c208d5efeafd7727386ab222835c87f32b`，逐字节一致。只是一个真实样本，不宣称全生产 parity。
- 最终代码的真实 train-clip BF16/K2/Euler8 full-vs-streamed 梯度检查、B/C4/D strict artifact loads 待收尾；不加载 C8 live checkpoint。
- 额外生成诊断冻结八个 valid systems，每体系 R1/anchor0/H4、lags 100/400 ps，seeds 0/1、Euler 8/16/32，另有 oracle、persistence、zero-history、zero-M，共 16 records/192 rows。zero-condition 是 OOD 敏感性干预，**不是 trained ablation**；这也不是完整 A validation。

所有 Python/model 检查都在 `enter-container` + `torch-ito`，CUDA GPU4。只在新 `runs/abcd_integration_20261004/` 写小型诊断结果；没有新长期训练、没有复制数据、没有修改正在使用的依赖。当前 `/data4` 仅约 7.7 GB 空闲，后续训练须先指定充足的 checkpoint 位置或明确保留策略，不能冒险耗尽 B/C 所用文件系统。

## 6. PVB / BioKinema 对照边界

**PVB 仅以用户指定的 `/data4/users/sihao/workspace/PVB_origin` 为原始源码基准**，commit `c08e5e3cd49d45c6d748387e78224843bd356f50`，读取时 worktree clean。`workspace/PVB` 已深度修改，不能充当原始 baseline。本次没有编辑二者。[PVB 官方仓库](https://github.com/yaledeus/PVB)。

原始 PVB 的静态/动态数据参与同一生成桥接路线；不能把旧 D decoder-only 当成等价实现。对原模型的优劣需要真实 checkpoint 和匹配协议，不能把本地 epoch0/step1 smoke 权重当发布模型。

BioKinema 源码参照 commit `62109dd49b325a7d9a22357303220c5f2b302b48`。本机 `/data1/repo/BioKinema/checkpoints/` 确有 `biokinema.pt` 和 kinase/misato fine-tuned 权重；没有下载或加载这些多 GB artifact。其 Pairformer 输入/预计算表征、coarse/fine 生成与训练数据需一并匹配。base config 的 `alpha_acf=0`，不能仅因代码含 ACF loss 就说发布模型都用了它。[官方数据和训练说明](https://github.com/IDEA-XL/BioKinema/blob/main/docs/data_and_training.md)。

下一步 head-to-head 必须冻结共同原子集合、Å/ps、split/pretraining overlap、observed prefix、forecast horizons、alignment、generation seeds 和 sampling cost。单帧 baseline 分别报告共同 H1 任务及各自合法最佳条件，不让它默默少看历史。短 clip 上不能声称 slow kinetics 已恢复；需足够长度的 ACF/VACF、接触/二面角分布和长 rollout 稳定性。

目前能够严格说的是 molvid 的内部瓶颈很明显；**尚没有本次同协议实测证明落后多少，更没有超过这两个 baseline 的证据**。应将“超过”变为冻结评测协议下的结果，而不是对模块名称的期待。
