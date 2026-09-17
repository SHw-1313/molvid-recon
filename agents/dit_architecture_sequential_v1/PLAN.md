# 四轮顺序改进：实现设计

## 0. 目标与边界

本轮回答四个窄问题：幅度是否更容易被网络使用；局部几何是否能由额外监督改善；四帧块边界能否通过跨块 refinement 改善；带误差 history 训练能否减缓短 rollout 的额外退化。

这是顺序筛选，不是四因素全因子实验。后面的增益依赖已经保留的前项；不能从结果推出四个模块独立、可加或所有组合的最佳性。R4 的 state/detail 是 Haar 系数及压缩 bank，不命名为已被验证的慢/快动力学。

历史证据来源：`027659fe872a6c869b14ecce5b799356bd695694` 的 `outputs/dit_source_reassessment_v2/20260914_r4_source_reassessment_v2_neibu/report.md`、`interpretation.json`。旧 conditional H8/16-step valid bond RMSE 约 0.693 Å，oracle 约 0.0122 Å；块间/块内位移比约 1.833，MD 约 0.985；generated-prefix 三段 bond 约 0.692→0.954→1.203 Å。以上只是污染模型的描述，必须在干净起点复核。

## 1. 冻结的实验底座

| 项目 | 本轮设置 |
|---|---|
| 数据 | 原 64-system manifest：48 train、8 valid、8 test 封存 |
| Manifest | `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000` |
| 结构与时间 | 有序 16 帧；真实 100 ps；H4/H8 交替；R4 四个时间 token |
| 编码 | 冻结 TorchMDNet、centered vector stem、deterministic state/detail codec |
| Codec 权重 | 原 T1 `full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`；SHA256 `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`；查明并记录实际绝对路径 |
| Statistics | 原 48-system 拟合资产；文件 SHA256 `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`；不重新拟合 |
| DiT | 4 layers，scalar 256 / vector 128，8 heads，FFN multiplier 4，factorized_v2 |
| Source | conditional，`repeat_last_coordinate_encode`，sigma=1；条件只由已观测坐标构造 |
| 基础目标 | 四个 future latent fields 等权 RF MSE；无 KL/VQ |
| 优化 | AdamW，lr=2e-4，weight_decay=.01，grad_clip=1；bf16，TF32=false；采用与核实 C48 相同的既有调度 |
| 生成 | 主比较固定 Euler 16 步；同 clip/H/draw 共用 eps |

如果实际 C48 配方不同，先记录差异并决定是否可作为同一底座；不偷偷改本表或把不匹配模型称为 C48。若必须从头训练基线，使用本表，目标约 20,000 成功更新；按完整吞吐和总预算在运行前冻结可行共同暴露量，不把欠训练基线当成熟结果。

公共代码可新增：

- `scripts/run_dit_architecture_sequential_v1.py`：阶段入口、独立构造、resume、预算、结果登记。
- `configs/dit_architecture_sequential_v1/`：baseline 和各轮对照/候选配置。
- `evaluation/dit_architecture_sequential_v1.py`：调用已核对 A 指标，修复独立加载/必要 CUDA 路径，保留旧评估可复现。
- `tests/test_dit_architecture_sequential_cuda.py`：只收录本文要求的关键 CUDA 检查。

文件名可按项目既有约定调整，但必须有实际入口和精确运行命令，不能只交付配置示意。评估通过新的独立 loader 调用旧计算函数；不要直接用会共享 adapter 的多模型入口。

## 2. 每轮继承与训练量

称已接受模型为 P。第 1/2/4 轮同时派生 `control(P)` 和 `candidate(P)`：共享初始权重数值，不共享 storage；复制各自 optimizer/scaler 状态；新轮使用同一登记的 batch/H/tau/eps 序列，corruption 使用另一个 RNG。两臂读取同一批数据，累计相同有效 future atom-frame tokens；保存共同 token 点。

优先目标每臂约 5,000 成功更新的追加暴露量。先以真实端到端吞吐折算，运行前冻结每轮可行预算；少于约 2,000 更新或曲线仍剧烈变化时，只形成 pilot 证据，不机械宣布架构无效。不同轮可以有不同预算，同一轮不能让更快的对照多训练。计数不能把 AMP 跳步算成功更新。

评估主 checkpoint 使用共同训练预算末端，不跨不同 loss 用各自最低 validation loss 任意挑选。额外保留训练曲线和 best checkpoint 供诊断，但不据此更换主比较。

候选被拒绝时，下一轮父模型是本轮已经等量续训的 control。候选被接受时选 candidate。无论哪种，继续使用其真实 optimizer 状态，记录谱系。第 3 轮冻结 P 的 DiT 与 adapter，仅新增 refiner，因此不额外推进 DiT 更新数。

## 3. 第 1 轮：将归一化前的幅度交给 scalar FFN

### 机制与实现

现有 `ScalarVectorFFN` 拼接的 vector norm 来自 `ffn_adaln` 之后；`SO3ChannelNorm` 已把每个 vector channel 在 xyz 上归一化。该特征较难表示输入原始幅度。残差 vector 仍保留幅度，所以不能说现架构完全丢失幅度。

唯一开关：`ffn_norm_source = post_adaln | pre_adaln`。默认 post 路径与旧行为一致。

- 在每个 block 完成 spatial/temporal residual 后，`ffn_adaln(h,v,condition)` 调用前保存 `v_pre`。
- FFN 的 vector 分支仍用现有归一化后的 `v_norm`；只把 scalar 拼接项改为 `sqrt(sum_xyz(v_pre.float()**2)+1e-6)`。
- 使用原来 Dv 个 norm 输入槽，不增宽 scalar_in，不增加参数，不去掉 SO3ChannelNorm，不改 attention，也不同时加 log、clip 或新的幅度归一化。
- reference `module/molecular_dit.py` 与 `module/dit_backend_v2.py::factorized_block_forward` 同步传入此可选特征。训练、采样、checkpoint 配置都读同一开关。
- norm 在 FP32 累积，再 cast 回 scalar dtype；不混 xyz 轴。

### 只做必要验证

关闭开关旧输出/梯度兼容；开启时 reference/v2 输出及梯度一致；固定 h，给 `v_pre` 乘 0.5/1/2，输入 norm 确实变化且旋转不变。测试 FFN 局部或非零门控的已训练 block，不要求 AdaLN-zero 初始化的整块残差立刻产生非零变化。

真实 clip 一步反传有限，冻结权重不变，然后进入 paired continuation。主问题是逐体系 RMSF 幅度误差是否下降，同时 bond/contact 不恶化；不能只看全局 RMSF 均值靠近 MD。

## 4. 第 2 轮：先增加几何监督

### 唯一改动：RF + future bond 辅助项

不加 message passing，不解冻 codec，不做硬投影。由本轮一次 RF forward 的预测速度直接构造近数据端 endpoint：

\[
z_\tau=(1-\tau)z_{source}+\tau z_{data},\qquad
\hat z_{data}=z_\tau+(1-\tau)\hat u_\theta(z_\tau,\tau).
\]

这是训练时与该 MD target 耦合的 endpoint 估计，不是从随机 source 完整生成的样本。

- 只对 `tau >= .75` 的样本启用几何项；RF 的原 tau 分布不变。batch 无 eligible sample 时几何 loss 为可追踪的零，不能 NaN。
- 对四 field 正确反归一化、恢复 mask/metadata、应用 observation clamp，再调用冻结 decoder。**冻结参数不等于 no_grad：必须保留 endpoint→decode→bond→DiT 的梯度。** 不经过当前仅供推理的 inference_mode/detach 包装。
- 使用已登记的合法拓扑 bond，固定 edge index，只在 valid future 帧和合法 atom/edge 上比较预测 bond length 与训练 target 对应帧长度。每样本先平均，再样本等权。不能从 future 坐标新推断一套邻接充当条件。
- `L_bond = mean(((length_pred-length_target)/1 Å)^2)`；`L = L_RF + lambda_bond * L_bond`。本轮不同时加 RMSD、angle、velocity、acceleration loss。
- 先在固定训练校准 clips 上测两项对同一组 DiT/adapter 参数的梯度范数，取中位数比，选使辅助梯度约为 RF 梯度 10% 的 lambda，限制在 `[0.01, 1]`；仅做一次，不使用 validation 选系数。校准的实际值和 ratio 记录后冻结。梯度为零/不有限时先修实现，不能靠加大 lambda 掩盖。
- 没有有效 bond 的样本记清适用性，不把缺失填零后宣称改善；主比较必须有足够相同的有效样本。

### 检查与判断

CUDA 检查几何 loss 梯度实际到达 DiT，codec 梯度/权重不变；observed/padding 不贡献损失；刚体变换下 bond loss 一致；零辅助权重保持旧 RF 更新。

然后对照 RF 与 RF+bond 同量训练。真正的判断来自 16 步噪声采样后的 bond/contact；不能用辅助 endpoint loss 降了代替生成改善。若靠接近静止来改善 bond，而 RMSF/位移分布明显退化，不保留。

如果几何监督仍无效，记录支持 topology-aware decoder 的证据，但本次不自动加一个 spatial network；那是下一次独立实验。

## 5. 第 3 轮：独立 temporal refiner，隔离跨块连接的收益

### 为什么本轮不能照搬 RF 续训

refiner 位于 latent RF 输出和冻结 decode 之后，RF loss 不依赖 refiner 参数。必须给 refiner 坐标域监督。为不把新监督和跨块连接混在一起，本轮 **DiT、adapter、codec 全冻结**，训练两份参数相同的 refiner：

| 模型 | 可训练部分 | 相邻帧连接 |
|---|---|---|
| P（原父模型） | 无，本轮只评估 | 原 decode，无 refiner |
| L（control） | 小 refiner | t±1，仅允许在同一四帧块内 |
| T（candidate） | 同样的小 refiner | t±1，可跨四帧块边界 |

L/T 初始化权重数值、监督、输入、预算完全一致，唯一差别是邻接 mask。P 用来显示额外 refinement 的总收益；T vs L 才能归因于跨块连接。L 不是“no temporal” codec，本轮不重新进行 codec/no-temporal 选型。

### 最小结构

新增 `module/trajectory_temporal_refiner.py`，独立保存权重，不覆写 `module/state_detail_codec_v2.py` 的参数或语义。

输入使用 `StateDetailDecoderOutput` 已有的 `h [16,N,C]`、`v [16,N,3,C]`、`x_hat [16,N,3]`、真实 time_ps、frame/atom/observation mask 与样本 id。

- 每原子仅连相邻有效帧 t±1；不跨样本/原子，不添加原子空间边。T 允许 3↔4、7↔8、11↔12，L 在这些边置 mask=false。observed→future 边可用；输出只更新 future。
- 用 invariant scalar 特征形成门控：当前 h、邻帧 h 差、`||v_j-v_t||`、真实 dt、观察标志。小 MLP hidden=64、两层、SiLU 即可，不加大 attention。
- 残差向量由邻帧差分 `v_j-v_t` 经无 bias、只混 channel 的线性投影，并乘 scalar gate 后聚合到 xyz；可按有效邻居数归一化。输出 `x_refined=x_hat+future_mask*delta_x`。
- 最终 vector 投影零初始化，使首次输出严格等于原 decode。零差分 v 必须给出零 correction；禁止独立 xyz bias 或额外 raw-coordinate anchor bypass。
- scalar 计算/投影可向量化；向量沿 xyz 共享权重。对称邻接允许读取本次共同生成的未来帧，但不能读取真实未知 future；这仍是统一 clip 生成后的 refinement，不把 tokenizer 改成自回归。
- 观察输出精确保留提供的观测坐标；仅对 future 施加 refinement。对 refiner 开关前后使用相同原有 history 输出约定，不能把 clamp 收益算到 refiner 上。

### 如何训练两个 refiner

先由冻结父模型、仅用 train clips 生成共同训练对：按固定 seed，在 `tau ~ Uniform(.75,.95)` 的 target-coupled RF 插值上取 endpoint，经原 decode 得到 h/v/x。reference 是同一个训练 clip 的干净 MD future。L/T 共用相同输入和 target；缓存 key 包含父模型、codec/statistics、sample/H/tau/eps hash。

不要把自由随机生成的一条轨迹与任意 MD reference 强配成逐帧监督。这里 endpoint 的噪声与 target 是耦合的；这也是该后训练与自由生成存在分布差异的原因。

两臂只训练 refiner，统一：

\[
L_{ref}=L_{coord}+0.1L_{bond}+0.1L_{motion},
\]

其中 `L_coord` 为同一 gauge 下 valid future 坐标 MSE / (1 Å)²；`L_bond` 同第 2 轮；`L_motion` 为与 target 的相邻位移差 MSE，使用 `(x[t+1]-x[t])*100 ps/dt`，尺度也为 1 Å。至少一端 future 的合法帧对才贡献 motion loss；已观测端使用同一真实输入坐标，不能让 codec 的历史重建误差进入条件。

这是额外的 refiner 后训练目标，不悄悄替换 DiT 的 RF 目标。L/T 使用相同 AdamW 配方、同 batch、同训练对数；optimizer 全新且彼此独立。最多缓存一个预先固定、有界的训练子集，可流式重算冻结 endpoint，禁止无限缓存整库。

### 验证和选择

CUDA 检查零初始化 identity、零运动零 correction、SE(3)、mask、L/T 唯一连接差别、参数/梯度隔离；修改别的样本、padding 或真实 future payload 不能影响本样本推理输入。只检查 observation 输出不变，不错误要求未知帧之间严格因果。

在同一组 **真正自由生成** 的 latent/坐标上运行 P/L/T。保存 raw/refined 双版本；主指标为逐体系块间/块内位移比相对 MD 的误差，同时看两类位移绝对量、RMSF、bond、contact 和样本多样性，防止统一平滑。

T 必须同时胜过 L，并相对 P 没有关键退化，才能声称跨块连接有价值。若只有 L 有明确收益，可保留 L，结论写为“局部 refinement 有效，跨块连接未获支持”。二者都不可靠就保留 P。下一轮冻结本轮被保留的 refiner，原 codec 永远不被改写。

## 6. 第 4 轮：用轻度有误差 history 训练

### 唯一改动：conditioning corruption

从已接受父模型派生两份同量续训臂。control 仍全部 clean history；candidate 每个 clip 独立选择 50% clean、25% 每坐标轴标准差 0.02 Å、25% 0.05 Å 的各向同性观测噪声。只污染 observed 坐标，valid future target 保持干净；噪声 seed 独立于 RF tau/eps。

- 保留 16 帧、H4/H8、原真实 dt、source sigma 和原 loss。本轮不把 RF source sigma 改小。
- 构建独立的 `condition_view` 和 `target_view`，不得原地污染共享 batch/缓存。观测 latent、repeat-last source center、输入 origin 都由对应 condition_view 产生。
- target future 与 corruption 后的观测 origin 使用同一坐标 gauge；合法地重表达干净 future，不能使输入与 target 相差一个意外平移。
- sampler 的 observed clamp 使用实际提供的有误差 history；不能 condition 用脏历史、clamp/anchor 却偷用干净历史。损失仍只覆盖有效 future；第 2 轮的 bond 项若保留则继续同样使用干净 future 监督。
- 若有 refiner则冻结其权重，control/candidate 相同。DiT 几何辅助项继续使用原 frozen decode 上的定义，不因新增 refiner 隐式改 loss。
- history noise 在线产生；不能命中仅以 sample_id 缓存的 clean observation/source center。cache key 必须区分 corruption seed/level，或对这部分关闭缓存。共享冻结 clean target latent 时要处理 origin 一致性。
- 默认不做在线自生成 history，不反传完整 rollout，不给 future 额外噪声。这个实验检验轻度条件误差鲁棒性，不能等同于已解决真实 generated-prefix 的全部结构化误差。
- corruption 只在训练启用。正式推理接收实际提供的 clean/generated history，不再人为追加这份训练噪声；测试不同噪声等级属于另行标注的鲁棒性诊断。

### 检查和判断

CUDA 检查 sigma=0 与 control 完全一致；只变 observed 条件、future 目标物理坐标不变；输入来源/缓存/observation clamp 一致；随机噪声的数值旋转等变测试要同时旋转所用 epsilon，不能旋转结构后重新抽一份噪声再要求逐元素相等。

先做 clean-history 单段生成，再做 8 true + 3×8 generated 的短 rollout，比较逐段 true-prefix 与 generated-prefix。关键不是第 3 段绝对 RMSD，而是 generated-prefix 相对 true-prefix 的额外 bond/contact 退化是否减少，且 clean 单段不明显变差。累计 32 帧只是短诊断，不能宣布长时动力学成立。

## 7. 预算、输出和可复现性

profile 包含数据/H2D、冻结 encoder/source、decode、反传、优化器、生成评估，不以纯 DiT forward 推算。至少预留总额 25% 给生成评估和恢复。若需要从头训练 C48，先从同一总预算扣除；不能在第 1 轮花光资源。

各轮输出保留 config、父与子权重 hash、精确启动/恢复命令、预算、successful updates、有效 token、wall/GPU time、峰值显存、曲线、逐体系指标、decision。训练与评估各自 RNG；model/adapter/refiner load contract 缺项时报清楚错误，不做静默 fallback。

启动下一轮依据 EVALUATION 中的配对结果自动保留/回退；失败代码通过 flag/config 保留为可复现实验，不删除证据。预算不足保存恢复点，并明确哪些轮只有代码、哪些有 CUDA 证据、哪些已有科学比较。

最后复评最初干净 baseline 与最终组合的同一固定生成集合，报告总收益、参数/延迟增量及每轮成本。任何总收益同时包含顺序续训；模块归因以各轮匹配对照为准，不把最终模型与更少训练的起点之差全部归因于四项架构。
