# 汇合后的两臂实验：设计记录，本轮不执行

## 先交付、再整合

用户审阅 A 的诊断、B 的等价性与真实速度证据后，另行授权一次整合。届时先把新 evaluator 接入新 backend，加载同一个旧 R4 checkpoint，在相同 per-sample seed 和输入上做交叉复核，再开展训练。A/B 本轮不得替用户自动 merge 或开始第三阶段。

## 科学问题

同样的 R4 codec、history、模型容量、噪声幅度和训练预算下，将 source 从零均值 Gaussian 改为以 observed state 为中心，是否改善几何/连续性而不把运动压成静止？两个 arm 都有 H4/H8 history；“从头生成”指未来 latent 从纯噪声开始，不是去掉历史条件，也不是只有一边从随机权重训练。

两边都从同一初始化 state_dict 的独立副本训练。先在模型/adapter 构造之前固定随机种子；只在 trainer 构造后设 seed 不足以保证权重一致。不要给 conditional arm 额外 history 特征、更多层或不同 noise scale。

## 初版 prior 和 RF 定义

用原始 codec 空间的最后完整观测 token 的 state_h/state_v 复制到未来；未来 detail_h/detail_v 设原始零。这是 block-state persistence，不等于最后观测原子坐标；state 是聚合结果，零 detail 只保证块内静止，不保证与上一块末帧连续。

先由 A 验证解码几何/边界。如果这个基线明显不合理，暂停该 prior 的训练设计，请用户选择新基线。允许 A 诊断“重复最后观测坐标并经过同一 codec 编码”的替代基线，但不能默默用它替换实验 prior。

设 N 为冻结的原始 train statistics 归一化，z 为 N(z_future_raw)，m 为 N(mu_raw)。注意：原始 detail=0 在有 scalar mean subtraction 时通常不是归一化 detail=0。

- Gaussian-source：`z0 = epsilon`。
- Conditional-source：`z0 = m + epsilon`。
- 两边相同 `epsilon ~ N(0,I)`，相同四 field 权重和原有归一化，不重新拟合 residual 方差。
- 两边使用 `z_tau=(1-tau)*z0+tau*z`，velocity target=`z-z0`，仅 future 上算 loss，observed 原样 clamp。
- 主模型输入都保留相同的 z_tau 表示、history、时间和 metadata；conditional arm 只改变 source/path/target，不另加 prior-only 条件通道。

等价 residual 表述：`r=z-m`，从 epsilon 生成 r 后再加 m。但若把网络输入也改成 r_tau，必须明确其等价性/条件信息，不能把多处参数化改变合并为“只换 prior”。

零 residual 的回加/归一化必须能恢复先验；future mutation 不改变 m；所有 xyz 通道保持 SO(3) 规则。conditional mean shift 没有自动减小噪声方差，不预设一定改善多样性或消除 OOD。

## 初轮规模与选择

R4 主比较，两种 source，一对 seed，48 个 train 体系、原有 H4/H8。4500 steps 可作为与 pilot 对齐的首个检查点，不是永久训练上限或“已收敛”声明；加速后按吞吐和学习曲线决定是否给两边相同增量，预算在看比较结果前一起冻结。

固定 validation 噪声/样本，与训练 step 无关；RF loss 仅用于同一 arm 的诊断/检查点参考，不跨两个不同 source target 直接排名。保存最后 checkpoint 和 best-validation checkpoint，周期性小子集生成评测用于判断 RF 改善能否转化为坐标/分布改善；不得只根据最好看的单个采样 seed 汇报。

共同评估：几何与 bond/contact、边界、RMSF/位移分布/ACF、采样 diversity、forecast error-vs-lag、吞吐。persistence 是几何与误差基线，不是要求随机样本必须逐点超过它。短 clip 不支持长期自由能/转移时间声称。

R1 不进入 DiT；R2 仅在诊断明确指向 R4 块内运动信息缺失时做有针对性的补充。不预先安排 R1/R2/R4 × 两 prior × 多 seed 的全矩阵。
