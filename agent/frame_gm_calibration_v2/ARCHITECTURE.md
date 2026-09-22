# 实现规格：G、M、实际采样训练

## 数据流与 shape

记 B 为体系数，N 为 packed atom 数，H 为观测帧数，Q 为有效查询帧数（可 padding），C=128。

- observed h `[H,N,C]`；v `[H,N,3,C]`；coordinates `[H,N,3]`。
- query time `[B,Q]`；frame mask `[B,Q]`；atom-to-system `[N]`。
- future noisy h/v、flow velocity、target h/v 的 frame/atom/channel shape 相同。
- DiT 内部 scalar 256、vector `[Q,N,3,128]`，4 blocks、8 heads。
- history R4 memory 保留；新增 motion context 为 per-atom scalar/vector，不占用 future token。
- decoder 输出 `[Q,N,3]`。padding 不参与任何 graph、attention、loss、特征和统计。

目标 teacher 冻结且 eval。history encoder、DiT、decoder 继续联合训练。目标 teacher 的 observed 编码、sample_origin、拓扑缓存均须不依赖真实 future。P2/P3 必须复验 target mutation，而不能只沿用旧测试名。

## G：局部拓扑交互与 vector→scalar 信息

第一版在第 2、第 4 个 DiT block 的 spatial 后、future temporal 前各增加一层局部消息。复用当前 block spatial attention，不把全原子 attention 改成 N²。

1. 图为去重的共价 1-hop 与 1–3 两跳邻接，双向、无跨体系边。缓存静态拓扑；高阶聚合用邻居数归一化。
2. edge scalar 包含拓扑跳数、原子类型，以及 observed 末帧的距离 RBF。观测参考距离是条件，不能作为所有未来距离的硬约束。
3. 当前 noisy hidden 的向量，在 SO(3) normalization 之前读出 `log1p(sum_xyz(v²))`、稀疏边上对应通道内积/差分幅度，送入 scalar message/门控。先 FP32 累加，随后可转回工作 dtype；不要从归一化后近常数 norm 假造信息。
4. vector 消息只用 invariant scalar 权重混合向量通道、v_j−v_i 和 observed 相对方向。不能混合 xyz 轴或添加独立向量 bias。
5. residual 采用单侧零初始化：输出分支普通初始化、外层 gate 为零，或反过来；禁止两处同时为零导致无法启动。验证第 1 步 gate、第 2–3 步消息参数确有梯度。
6. 不在每层完整解码坐标、不动态重建非共价 radius graph、不加入新几何 encoder。未来候选几何读出本轮先不加，避免 G 内出现隐含的第三组变量。

G=off 走原 spatial 路径。实现结果必须能回答“提升来自哪些新增计算、增加多少参数/显存/耗时”。

## M：体系相关 motion context 与真实时间交互

### Observed-only motion context

从已有 observed teacher h/v 计算相邻差分和相对末帧差分，不再运行一次 geometry encoder。保留差分均值与幅度/平方项，避免往返运动在求均值后抵消。

可读实现：每 atom 对 valid history increments 做学习权重聚合；权重可依赖真实 Δt、history span、observed scalar。输出包括 scalar 变化摘要和等变 vector 摘要。所有幅度信息在归一化前保留；不做 per-trajectory 单位方差归一化。

将内容与时间通过 zero-preserving 分支相结合：scalar 使用 `F(motion, state, time)-F(0,state,time)`，vector 使用 invariant gate/channel mixing 作用于 motion vector。重复静态 history 的 motion summary 必须为零；这不要求未来随机采样完全不动。

context 在一次采样中算一次并在每个 DiT block 使用（AdaLN/gate/残差均可，需显式模块）。训练重用时保留 autograd graph，不能为了 cache 而 detach history。未来 noise/target 不进入此 context。

### 时间语义

分别提供：

- `query_horizon_ps = query_time - last_observed_time`；
- `query_delta_ps`，首个 query 相对 last observed，后续为真实相邻 query 差；
- `history_span_ps` 及 observed 相邻间隔。

flow_time 仍是生成过程的 s∈[0,1]，独立编码。时间平移不改变结果。

使用 signed-log/linear 的多尺度特征；固定尺度默认 `[10, 100, 1000, 10000] ps`，避免仅依赖在 100 ps 整数倍上重复的 Fourier 项。原 PE 路径可保留作为兼容底座，新增路径以 residual 引入并在 load report 解释。M=off 完整使用旧定义。

future temporal attention 与 history cross-attention 在已有 learned content/time bias 之外，增加每 head 的非负衰减参数：

`bias_ij = learned_residual(time_features) - softplus(lambda_head) * (abs(t_i-t_j)/100ps)^0.5`。

β 固定 0.5，本轮不扫参。history memory 的代表时间和跨度显式记录。只有一个 history key 的 H4/R4，时间 softmax bias 不能独自提供选择能力，因此 motion context 的 value/门控路径是必需项。

时间不直接加到将被 Haar 差分的静态几何内容中。保持现有 zero-preserving history detail；detail 不除以 Δt，不宣称四帧表示慢模态。

## P3：真正的源分布采样路径

J0 为常规联合续训；J1 在相同 base loss 上新增低频实际采样及分布校准。二者从同一选中 P2 parent 开始。

训练采样的默认设置：每 8 个全局成功 update 启用一次；每个选中 history 生成 K=2 个独立样本；每个样本用 4 步 Euler 从 `repeat(last_observed_latent)+N(0,I)` 完整走到 s=1。用常规步进核实现，但不能调用包裹 no_grad 的 inference wrapper。

采样起点、noise 及条件完全不含 target；真实未来只在 loss 中读取。整个 4 步对 history/DiT/decoder 保留梯度，可使用 activation checkpointing。首版不截断梯度，也不把 target-interpolated endpoint 冒充实际采样。记录 training sampler 4 steps，主评估仍为 16 steps，并检查其差异。

一次 DDP forward 内显式完成所需路径并一次 backward；不要绕过 DDP wrapper 在 trainer 中偷调未同步的模块。控制所有 rank 同步进入 sampled branch；局部无有效样本的 rank 用有效计数与零贡献参与同样的 collective。独立 sampler RNG 不改变主 flow/noise RNG 和 dataloader 顺序。

若 profile 显示 K=2/4-step 的全 batch 超内存：按确定性 sample-id 子集做 sampled loss，其余保留主 loss；所有 rank 使用明确相同的选择规则，记录真实参与数。不要减少某一臂主训练 exposure 或改成 CPU。采样频率/子集大小在 pilot 后冻结。

### 分布特征 φ

首版实现三组可微特征，各组独立按 train-only 固定尺度归一化，组间等权：

1. 每 residue/block 的 RMSF profile：使用一致 observed-reference 对齐；记录 Kabsch 稳定策略和梯度边界。如 detached rotation，明确为近似梯度，不伪装成完整对齐梯度。
2. 基于同一参考的位移平方：查询 horizon 上的 MSD 曲线与有效相邻增量平方。仅比较同 grid、同 history 的生成与真值；matched-view 训练另以公共物理时刻评估。
3. 固定局部 pairs 的内部距离增量及相邻增量乘积/相关特征。pair 选择仅依赖静态拓扑/observed，避免未来 contact 选边泄漏。稀疏边，避免 N²。

特征允许因序列太短、缺 valid pairs 或数据的结构性缺失标注不可用，不可用组不填 0 冒充物理值。loss 的 eligibility 仅由观测/查询 mask、拓扑及缺失标注确定，不能根据预测或目标的运动幅度、方差决定，否则可能改变被学习的分布。预测或目标零运动时 RMSF=0 是合法值，仍须计入；不能通过坍缩让困难样本退出 loss。训练首版用未归一化增量乘积/协方差特征，避免方差为零时相关系数未定义。训练集恒定特征的固定尺度处理在calibration阶段解析，不逐样本择优删除。样本按可用组归一，rank/sample 的有效计数必须正确。

对每一个条件单独计算 energy score，禁止把不同体系混成一个 ensemble：

同一条件下的全部 X–Y、X–X 项共用固定的特征坐标、mask、尺度和组权重；不能每对样本重选特征。目标或预测出现非有限值应作为数据/数值失败报告，不能自适应删除这些困难特征后继续报正常loss。

`ES = mean_k d(phi(X_k),phi(Y)) - [1/(2*K*(K-1))] * sum_{k!=l} d(phi(X_k),phi(X_l))`。

`d` 用固定特征尺度后的 RMS 欧氏距离（sqrt(mean(square)+eps)，eps 只为数值稳定）。K=2 时第二项为 `0.5*d(phi(X1),phi(X2))`。禁止包含对角项后仍沿用无偏估计系数；禁止对随机样本做 paired coordinate MSE。

用 fixed training batches 拟合特征尺度；不按每条轨迹自身标准差缩放，不接触 valid/test。主 flow loss 保留，ES 只校准所选物理特征。

### 实际样本上的几何与权重

实际样本可增加 covalent bond 的局部合理性项：目标使用 observed bond 长度或 train-only 化学类型统计，不能读取实际未来逐帧 bond 作为模型条件。其他 angle/clash 首版评估，不把所有几何指标都加入 loss。

在 8 个固定 train calibration batches 上，将“actual-sample geometry + ES”对 DiT 的梯度范数分别定到主 flow 参考的 0.05（合计目标约 0.1），保存原始量和 resolved 系数。该比例指启用 sampled branch 的 update，不自动乘以 8 补偿稀疏频率。沿用现有梯度校准思路，不扫权重，不用 valid 调 λ。

保留 parent 已解析的 generated/clean/near bond 权重，P2 各臂不能各自重新校准。P3 J0 同样完成新的训练配方 warmup/调度与相同主数据曝光，只有 sampled auxiliary 不启用。

若出现 bond keep/release：所有显式 bond loss（含新增实际样本 bond）统一置零；ES 内部运动统计不是 bond length penalty，保留。局部拓扑图保留。两臂同 parent、同初始化 optimizer policy、同数据/RNG/步数。

## CUDA 数值验证：按改动选择

- G：rotation/translation 等变、无跨体系边、padding 不影响、单侧零初始化启动。
- M：physical-time shift 不变、零运动 summary/detail 为零、query/last 时间定义、相同几何不同时间可通过新路径传导梯度。
- P3：实际源采样 target mutation；sample loss 的梯度到 history/DiT/decoder，teacher 无梯度且 hash 不变；K=2 ES 符号/系数；非启用步 RNG 无意扰动检查；有/无有效样本 DDP 计数。
- checkpoint：新增模块明确 load report；严格 resume 还原 RNG/data cursor/aux sample cadence。

不得通过 monkeypatch 绕过 teacher freeze、fake source、CPU fallback 或静默 strict=False 获得通过。
