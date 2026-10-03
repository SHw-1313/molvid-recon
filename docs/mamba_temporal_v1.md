# Molvid：统一物理时间 Mamba 方案

基线：`integration/abcd-20261004`，ABCD 合并点 `f85896b`。这是可以在合并树上逐阶段实现的设计，**尚未实现或训练 Mamba，不能把预期收益当成结果**。默认科学基线仍为 A1/4-block/原 history+M；B 与 C 的实验结论不作为实施依赖。

静态单结构进入共享主干的接口与梯度平衡见 [D2 联合预训练方案](static_backbone_pretraining_v2.md)。static 显式跳过物理时间模块；不把单帧当零运动轨迹训练 Mamba。

## 目标与不应混淆的对象

统一所有**物理时间序列**的可学习建模：observed history、history 到 query 的读出、future 内部相互作用、decoder 时间混合。`physical_time_ps` 与 flow progress `s∈[0,1]` 保持两个接口。`s` 是每次向量场求值的噪声条件，不是 MD 时钟，不对 Euler 的 8/16/32 个步骤做 Mamba 扫描。保留其小型 MLP/AdaLN 调制。

“全部换 Mamba”应指替换可学习时间混合器，而不是删去真实时间坐标、空间消息或所有 MLP。纯粹将 Fourier/MLP 换为标准 Mamba 既不自动获得不规则时间能力，也不能保证旋转等变。原始 Mamba 的选择性 SSM 和 Mamba-2 的 SSD 是实现依据；下面的物理时间、向量和 query 读出是本项目新增设计，不是论文已证明的分子动力学结论。[Mamba](https://arxiv.org/abs/2312.00752)、[Mamba-2](https://arxiv.org/abs/2405.21060)。

## 源码替换表

| 当前位置 | 当前问题/职责 | 最终替换 |
|---|---|---|
| `molvid/time.py::PhysicalTimeEmbedding`、`pairwise_time_features` | 各模块独立投影；100 ps 归一化上的 Fourier 项周期重复，线性/log 项仍可区分时间，不能称整个 embedding 混叠 | 一份共享的 `PhysicalClock` 特征定义与 `ClockMamba`，输出 observed/query/decoder clock 条件；按用途保留小型输出投影 |
| `codec/history.py::TemporalDifferenceUpdate`、R4 Haar、detail/time gates | H4→1、H8→2 个 atom memory token；短历史压缩、差分和 M 有重叠 | `MambaHistoryEncoder`：level 与 motion 两支的 observed-only 选择性扫描；移除 Haar/detail 手工压缩 |
| `codec/motion_context.py::*` | 把序列降为相邻差分的加权汇总，再重复注入各层 | motion 纳入同一个 history 状态；最终不保留独立 M encoder/injector |
| `codec/learned_memory.py::PhysicalScaleAnalysis`、8-slot attention | 5 个手工尺度×4 个中心+recent/level，共25个分析 token再压8槽；先 residue mean 后时间压缩不可逆 | 复用 raw-history/chemical mapping/轻量空间编码；用真正有序的 Mamba 扫描替代 quadrature+slot attention；B 原后端仅留可复现实验入口 |
| `dit/frame.py` history cross-attention 与 time bias/decay | 每层重复读取 memory；多种时间特征叠加 | observed state 的 horizon-aware SSM readout；不得把8个无序slot直接当成8个时间步 |
| `dit/blocks.py::FrameDiTBlock.temporal` | Q8/Q12 同原子双向 attention | `BidirectionalEquivariantMamba`，扫描当前 flow state 的 future 帧 |
| `codec/decoder.py::TrajectoryDecoderBlock.temporal` | 第二套 temporal attention，可能补偿而非消除错误生成 latent | 对 observed+generated latent 做两层双向等变 Mamba；保留 covalent/FFN/head |
| spatial encoder / spatial attention / G / covalent messages | 空间关系，不是时间序列 | 保留；不能沿任意 atom 排序扫描替代空间等变网络 |
| frozen teacher / statistics / RF source & loss | target 表示与生成目标 | 首轮保持；否则无法识别 Mamba 的独立贡献 |

最终数据流：

```text
observed coordinates + topology + actual timestamps
       ├─ frozen teacher anchors ───────────────┐
       └─ light spatial features               │
          → causal equivariant history Mamba   │
          → level state + motion state        │
          → query-horizon readout              │
                                               ↓
source latent + query clock + flow s → spatial/G + bidirectional future Mamba
                                               ↓
                           observed + generated future latent
                           → bidirectional decoder Mamba + covalent/head
                           → coordinates [Q,N,3]
```

训练 future teacher latent只用于 RF target、oracle/near 分支与 loss。history states 的接口不接受它。decoder 的 oracle 分支可接触 clean future，生成分支只接触 sampled future；两者没有共享可变 recurrent cache。

## 核心接口与数值定义

新核心放 `molvid/temporal/`，不建立平行模型框架。`EquivariantSelectiveScan.forward(h, v, physical_time_ps, frame_mask, system_id, initial_state=None, reverse=False)` 输入 `h[T,N,Cs]`、`v[T,N,3,Cv]`、clock/mask `[B,T]`，输出同形 features 和显式 state。history 可在 residue group 轴 R 使用同一算子；atom/group 均是 batch 轴，绝不沿 packed atoms 扫描。

定义归一化实际间隔 `d_i=(t_i-t_(i-1))/100ps`。选择性速率来自旋转不变内容 `g_i=softplus(W[h_i, ||v_i||, clock_i]+b)`，有效间隔 `δ_i=d_i*g_i`。两者分别记录；不能直接把 Mamba 学到的 `dt` 标为 ps。初始化衰减时间常数覆盖10、100、1000、10000 ps，可学习，但无需另堆四条独立网络。

对于稳定的对角状态矩阵 `A=-exp(A_log)`：

```text
a_i = exp(A * δ_i)
b_i = expm1(A * δ_i) / A         # A→0 时使用连续极限 δ_i
S_i = a_i * S_(i-1) + b_i * B_i * u_i
y_i = C_i * S_i + D * u_i
```

这是建议的物理间隔 ZOH 版本，**不冒称与上游 Mamba/Mamba-2 的输入离散化逐位相同**。若映射到上游 fused kernel，需要显式重标定输入项，先对照上述 CUDA/FP32 reference 的 forward/backward，再测性能。不能跳过公式差异直接替换 kernel。

首帧的时间间隔不是伪造的100 ps：用独立、可学习的 anchor initialization 写入初始状态，`δ=0`。真实单帧仍能提供静态 level，motion 标为 unavailable/zero。跨大间隔衰减允许饱和；不可为了稳定而把所有长间隔裁成同一 dt。指数、`expm1`、状态累加使用 FP32，线性投影可 BF16。

向量状态 `S_v[N,3,Cv,Dstate]` 的 A/B/C/门控系数都由标量、不变量和真实时间产生，并对 xyz 三轴共享。向量只做无偏置通道线性、标量门控、求和与 SO(3) norm；禁止对 xyz 分量独立 SiLU、带偏置卷积或不共享的三套 Mamba。标量支可用标准 Mamba 风格门控。local conv 先用 `d_conv=1`，避免把不规则帧索引邻域冒充固定物理时间；`d_conv=4` 作为单独消融。

padding 必须在投影前以有效值替代，不能用 `0*NaN` 消毒。无效帧既不写入状态，也不推进 SSM/conv state，输出严格为零。不同 packed systems、分子、轨迹、replica、rollout draw 的状态完全隔离。正向/反向扫描使用对应的正间隔；反向边界在最后一个**有效**帧，不在padding尾部。

## history 与 query 的具体读出

第一阶段先从 teacher 的 observed h/v 序列扫描，隔离时间架构变化。第二阶段才切换 B 已有的 raw-history/light encoder 输入，并保持同一个 paired raw record/anchor/future schedule。

level支处理绝对 observed features；motion支处理相对 observed anchor 的变化和带实际 `dt` 的增量。对恒定历史，motion输入、state和readout严格为零；level不应被一起抹掉。对于 residue pooling，保留每个 atom 的最后 observed teacher特征与一个局部残差支；不能假定 pooled memory 能恢复组内相互抵消的侧链运动。ligand identity 应由数据边界显式提供，不能由“非蛋白 block”猜测。

history结束后产生固定的 `S_obs`，每个 query horizon `τ=t_query-t_last_obs>0` 从它**独立**读出，例如 `C(query,h_ref)*exp(A*τ*g_query/100ps)*S_obs + skip`。query之间的依赖交给future双向Mamba，不在history读出中把第一个query写回state。这样加一个中间query不会改变其它query的历史条件。保留快/慢state通道和recent observed skip；不只输出一个平均向量，也不把slots伪装成时间序列。

当未来 query grid 固定时，observed history与clock可每个prefix准备一次，并供K个sample和全部ODE求值共享；训练沿本次合并的condition-gradient boundary回传。与当前noisy future或flow `s`有关的state每次向量场求值重算，严禁跨Euler步复用。双向future与decoder分支可读取同一时刻的全部**生成/噪声**future latent，不读取真实future坐标。

## 配置、迁移与删除规则

显式新增 `temporal_backend: attention|mamba_physical_v1`，以及可审计的阶段开关 `history / history_readout / future / decoder / clock`；首个完整候选才全开。初值 `state_dim=32`、scalar expand=2、vector expand=1、history depth=2、future沿原4个block、decoder depth=2，最大history测试129帧。所有尺寸记入model/DiT/decoder/checkpoint contract，推理loader与模型检查器必须共同支持。

复制兼容的teacher、stats、spatial/G、identity embedding、flow-s MLP、FFN、coordinate head。attention→Mamba、Haar/slot→SSM无精确权重映射，报告 `loaded / initialized / retired / unexpected / shape_mismatch`，重置新分支optimizer与scheduler；标为changed-architecture warm start，不叫exact resume。

不要把attention与Mamba长期并行相加。为了迁移可临时蒸馏 observed条件与flow输出，然后移除被替代的attention/M模块。零初始化新residual只能保证起点，不自动保证后续表示等价；history替换没有天然恒等映射。新分支连续几步要验证门控与内部参数均获得非零梯度，防止双重零初始化锁死。

当前 `torch-ito` 为 PyTorch 2.5.1+cu121，2026-10-04 探测 `mamba_ssm` 未安装。首轮可以在现有环境实现明确标为reference的CUDA PyTorch selective scan；它不是fused Mamba性能结果。后续在独立环境固定版本构建Mamba-2 kernel，不升级B/C正在使用的环境。[官方Mamba-2接口](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba2.py)。

## 实验顺序和验收

| 顺序 | 成对实验 | 回答的问题 |
|---|---|---|
| 0 | 合并A1 vs 原A1，固定真实checkpoint和输入 | 合并本身是否改变生成/梯度 |
| 1 | A1原结构 vs 只换history+统一motion状态；4 blocks不变 | 时序表示是否确实是瓶颈 |
| 2 | 阶段1 vs 再换query readout和future mixer | 生成过程的时间相互作用贡献 |
| 3 | 阶段2 vs 再换decoder/clock；形成全物理时间Mamba | decoder时间混合是否改善或压制运动 |
| 4 | 全Mamba vs 参数量匹配attention；相同曝光，另报GPU小时 | 收益来自架构还是容量/算力 |
| 5 | 胜出结构 × A1 / 重标定generated几何目标 | 时间结构与目标设计的交互，避免一次同时更改全部因素 |

每轮先128步数值/峰值显存profile，再固定同一批2k步paired pilot；有趋势的前两臂跑至少两个training seeds、至少3个generation seeds。保持parent/data/atom-frame exposure/source RNG/optimizer预算一致。不得直接拿A的一轮与D的500步比较，也不能把B不同history/source schedule的原始loss当胜负。

数值验收：FP32/BF16有限值和梯度；SO(3) rotation与translation行为；packed system和atom permutation隔离；padding任意值不影响有效输出；time shift不变性；ps↔ns显式单位换算等价；不规则100/200/400与未训练300 ps；恒定history的motion精确零；full scan与chunk/cache等价；未来坐标mutation不改变condition或sample；同模型exact resume复现；flow-s cache污染反例；最长history/最大atom真实batch显存。

科学验收以同一validation协议下的per-system配对结果为准：bond RMSE、vdW clash/severity、residue RMSF MAE、system RMSF spread/correlation、MSD曲线，补ACF/VACF与torsion/contact分布。先检查时间序列长度能否支持统计量，不能从Q8/Q12短片段声称恢复slow kinetics。然后才扩展rollout，检查漂移、手性/键角、碰撞累积和checkpoint曝光差异。

阶段promotion建议预先冻结为：主运动误差相对A1至少改善10%，同时bond/clash/severity任一不恶化超过5%；geometry主导版本反向对称设置guardrail，并报告所有体系/seed而非挑最优。置信区间跨零或只有一个seed时标为探索性，不称胜出。长history测吞吐/峰值显存；Q8/Q12上Mamba可能更慢，不能用语言模型论文的速度替代本机实测。

## 超过PVB/BioKinema的必要对照

这项结构改造不是最终对标协议。统一原子集合、Å/ps、observed prefix、预测长度、轨迹数、采样预算和对齐方式；若baseline只支持单帧，分别报告共同H1任务与各自最佳合法条件，不能让它默默少看history。公开权重若用过本项目valid systems，须单列重叠；sealed test保持封存直到方法和协议冻结。

PVB仅以用户指定的 `/data4/users/sihao/workspace/PVB_origin`（`c08e5e3`）为原始源码基准，不使用深度改动的 `workspace/PVB`。PVB的静态数据作用于同一生成桥接训练路线；D的decoder-only T=1 pilot不是该预训练的替代。[PVB官方仓库](https://github.com/yaledeus/PVB)。BioKinema的发布流程包括预计算Pairformer表征及分层forecast/interpolation；其不同checkpoint训练数据和动力学目标不同。当前源码base config的`alpha_acf=0`，不能仅凭存在ACF代码就说所有发布模型都启用该损失。[BioKinema训练文档](https://github.com/IDEA-XL/BioKinema/blob/main/docs/data_and_training.md)。

因此需要并行推进生成latent→几何流形的约束、真正作用于生成路径的结构预训练，以及matched baseline评估。Mamba负责其中的时间表示问题；不能承诺替换它就超过两个baseline。
