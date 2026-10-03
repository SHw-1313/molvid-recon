# D2：静态结构进入共享主干，与历史条件多帧动力学联合训练

2026-10-04 用户澄清：D 的目标是**静态结构预训练共享主干**，不是只增加 decoder 重建任务。旧 D 保留为对照，不再被称作该目标的完整实现。本方案接入 ABCD 合并树，与 `mamba_temporal_v1.md` 的接口一致；设计完成，D2尚未实现/提交训练。

## 核心决定

使用一个共享的空间/生成主干、两种有明确语义的任务。动态任务预测 `p(X_future | X_history, t_history, t_query, chemistry)`；静态任务学习 `p(X_clean | X_corrupted, chemistry)` 的结构生成/去噪桥。两者共享flow的 `s` 和空间表征，不把单结构冒充时间序列。

静态任务至少必须更新 DiT 的scalar/vector输入输出投影、spatial attention、G、FFN及共享decoder。history编码、物理clock、跨历史读出、future时序混合器只在动态任务训练。这样静态数据教主干分辨“什么结构合理”，动态轨迹教模型“结构怎样随真实时间变化”。只改任务embedding或decoder，仍然不算实现。

当前A1可先做D2的联合continued pretraining验证。真正的“静态预训练→动态微调”需另从同一初始化启动成对实验，不能把从已训练A1追加几百步叫作从头预训练。第一轮保持frozen target teacher和latent statistics，避免同时改变target空间；共享主干指trainable DiT/spatial/decoder，不包括自动解冻这个target teacher。

原始PVB对照固定为用户指定的 `PVB_origin`（commit `c08e5e3`），不是已改动的 `PVB`。其 `module/model.py::dyVAE._train` 将pretrain/md的target分别设为clean结构/MD future，但共同经过坐标interpolant与同一decode。D2借鉴的是“静态训练真正作用于生成主干”，首轮仍保留molvid latent RF，并非宣称等价复现PVB；若继续存在latent→geometry差距，再独立消融坐标感知主干，不能把该变化混入首次接入实验。

## 两种任务如何经过同一个主干

| 项目 | 动态多帧+历史 | 静态单结构 |
|---|---|---|
| 观测输入 | 真实过去坐标、chemistry、ps时钟 | `x_corrupt`、chemistry；没有物理时间标签 |
| 目标 | frozen teacher的真实future h/v，Q8/Q12 | frozen teacher的clean结构h/v，Q1 |
| source | last-observed normalized latent + isotropic noise | corrupted-structure normalized latent + isotropic noise |
| RF训练 | `z_s=(1-s)z0+s*z_target`，预测velocity | 同样的RF形式与h/v归一化 |
| history/Mamba | 观测历史编码和query horizon读出 | 跳过；不复制结构构造H4/H8 |
| 空间主干 | shared projections、spatial attention、G、FFN | **同一套参数，必须有非零梯度** |
| future temporal | Q帧双向混合、真实dt | 跳过时序混合，不给它零运动监督 |
| decoder | observed+generated，共享空间/坐标头和动态temporal | corrupted context+Q1 generated，共享空间/坐标头，跳过temporal |
| 损失 | flow、已有几何、sampled-source、多帧统计 | flow+generated结构几何；无RMSF/MSD/ACF/velocity标签 |

物理 `dt=0` 不是静态任务的替代标签。新增 `task_kind=dynamic|static` 与独立 `StructureCondition` / `StructureQuery`；静态clock为not-applicable，调用路径不经过 `QuerySpec` 的正future间隔校验。两任务可共享几何context的显式字段，但不能添加一个把static伪装成future的任意100 ps。

static decoder实际处理的是一个generated latent位置；corrupted结构只提供几何context/origin/reference，不将它与generated结构拼成伪T2再做时间混合。结构condition可以复用teacher与拓扑，不能复用带“过去→未来”含义的接口后偷偷填占位时间。

各block按任务跳过history cross、M以及future temporal，仅执行相同的spatial/G/FFN。Mamba版也是同样分工：static训练共享内容/空间部分，不更新SSM时间常数、历史状态和时间读出。Q1经过普通Mamba仍可能训练A/dt/conv参数，因此必须显式跳过，不仅依靠长度1。

参数边界按实际对象而非模糊的“主干”名字列清：`history_encoder`、M encoder/injector、`query_time`、history/temporal attention及其专属AdaLN/time-bias/decay、decoder `time_embedding`/temporal/gates归入temporal；adapter/reference投影、identity、flow-s、spatial/FFN及其AdaLN、G、decoder local/FFN/head归入shared。decoder被local/FFN复用的norm也是shared，不能因为temporal也调用它就整体冻结。static不直接更新专属时间参数，不代表shared表示变化后dynamic输出不受影响——后者必须由配对验证和梯度冲突监控约束。

## 防止又退化成容易的重建旁路

旧D是 `corrupted coordinates → frozen teacher → decoder`；新D必须经过 `source → shared flow backbone → generated latent → shared decoder`。clean teacher只供target/loss，不从clean latent直接喂给static生成decoder。不把clean坐标、clean距离/RBF、clean邻居图、clean centroid或target-derived参考方向放进condition。参考origin、reference geometry、半径邻居来自corrupted输入；共价拓扑/atom identity可以来自chemistry。

首版静态corruption使用显式的Å单位混合：少量轻扰动用于局部精修，中等扰动用于主训练，少量较强扰动用于覆盖生成偏离。候选sigma/axis为0.1/0.3/0.8 Å，概率0.25/0.50/0.25；这些是待train-only profile冻结的初值，不是实验结论。不得只重复旧D的clean/0.05 Å近恒等映射。随后单独增加保持键长的torsion/local rigid perturbation，区分随机笛卡尔噪声与真实生成错误。

source噪声尺度与动态A1保持同一normalized latent约定，corruption RNG与RF/source RNG分开。target与corrupted context按同一个**corrupted observed origin**编码，避免改变坐标规范。记录static target h/v的分位数、norm、相对MD statistics的尺度；若外部PDB严重超出统计范围，先缩小化学/尺寸域或另做representation研究，不偷偷重估normalizer破坏A1权重。

静态loss：

```text
L_S = L_flow_static
    + λ_coord * L_coord(generated_endpoint, x_clean)
    + λ_bond  * L_bond(generated_endpoint)
    + λ_angle * L_angle(generated_endpoint)
    + λ_clash * L_clash(generated_endpoint)
```

`generated_endpoint`第一层监督可用target-interpolated endpoint，但不对DiT detach；必须另外在冻结cadence上用actual-source短rollout验证/训练几何，使训练见到真正source路径的偏差。只对单步target-interpolated endpoint优化，不足以解决A已暴露的train/sample gap。先用K1/Euler4或8的static短rollout作profile，明确报告额外cost；不要将其叫作真实MD轨迹。

flow内部保留h/v各0.5的既有归一化，以便首轮归因。新几何项在train calibration上按**shared-trunk梯度**标定；旧D的gamma=`0.000173...`只针对decoder，不可照搬。coordinate、bond、angle、clash分别记录raw/weighted值与active次数，不能把它们都隐藏在一个gamma里。手性/二面角和非局部接触是随后应补的结构约束，不能声称bond+clash已覆盖全部化学。

## 平衡策略：同一次optimizer step合并，动态决定时间学习

首选每个global update处理一批dynamic和一小批static，顺序forward/backward以节省显存，最后**一次**AdamW step。dynamic sampler、history distribution、LR schedule和成功更新计数保持与A1对照相同。static有独立sampler/cursor/RNG，checkpoint一并保存。不要在每4个dynamic update后额外做一个共享AdamW step：不同moment更新次数、weight decay和scheduler语义会把任务作用与优化器变化混在一起。

静态batch先设 `max_static_atoms ≈ max_dynamic_atom_frames/8`（A1的40,000对应约5,000 static atoms），按profile调整到额外wall/GPU成本约20–30%。这只是compute起点，不能把frame数当loss倍数。dynamic每体系先平均future有效帧与原子，static每结构先平均有效原子，最后各自按体系/结构等权；Q12不能天然获得静态12倍权重，大分子也不能因原子多占据全部梯度。

当前 `FrameRectifiedFlowObjective::_sample_equal_frame_mse` 与 joint coordinate loss 已按体系平均；首轮继承这些动态归一化，不额外改成全局atom-frame平均，也不套用旧四字段RF的全局mask归一化。新static loss要独立处理有效structure计数；任何其它动态loss归一化调整都需单独记录与成对对照。

参数划分为 `θ_shared`、`θ_temporal`、极小的 `θ_task`。设动态和静态在shared上的梯度为 `g_D,g_S`：

```text
g_shared   = g_D + λ_S * g_S
g_temporal = ∂L_D / ∂θ_temporal       # static分支必须为None；joint step仍按dynamic更新
λ_S        = ρ * median(||g_D||) / median(||g_S||)
```

推荐第一轮 `ρ=0.25`，两个探索臂 `ρ=0.10,0.50`。这表示static在shared上约为dynamic的四分之一梯度范数，不是25%的样本，也不是loss数值的25%。标定用至少16个冻结train batch pair，覆盖atom-size/lag/history桶，先冻结λ再跑paired pilot；`λ_S`是主干真实梯度的标定结果，不沿用decoder-only gamma。

static-only阶段temporal参数不进optimizer，避免动量/weight decay造成无监督更新；joint阶段保留dynamic产生的temporal梯度，不能在static backward前误清空它们。隔离检查应分别测dynamic/static的梯度贡献，而不是要求joint总梯度在temporal上为None。

每128个global updates记录各shared block的norm比与cosine、history/temporal梯度和实际参数更新范数，校验固定λ是否仍在目标附近。先不引入自适应loss controller，使比较可解释。若长期范数比超过0.5或static与dynamic梯度反向比例过高，按预先规则停在checkpoint、用降低ρ的**新臂**比较，不在一个臂中静默改目标。

只做全shared norm仍可能掩盖DiT与decoder不平衡，因此分别报告 `dit.spatial/G/ffn` 与 `decoder.spatial/head`。若static只在decoder产生大梯度、主干梯度近零，直接判为实现失败。若global grad clipping频繁压低temporal更新，也不能把它误判成history无用。首轮保留现有clip策略并记录clip系数；需要改变时另做成对实验。

梯度冲突不能用norm比解决。若cosine持续负值，优先降低static权重/收窄扰动域；后续可单独消融“不允许static反向抵消dynamic”的非对称投影：`g_S -= min(0,<g_S,g_D>)/(||g_D||²+eps)*g_D`。它只是优化器梯度空间约束，不能保证AdamW实际更新或验证性能不退步，故不作为首版隐藏默认值。

如果显存/吞吐迫使static只在每4步执行一次，必须明确区分平均贡献和单次冲击：维持同一平均ρ时static-active step的λ要乘4，瞬时梯度可能过大。优先每步小batch；cadence版本固定较低active-step比，报告真实平均贡献，不笼统称作同一个“4:1配比”。

## 数据采样与预训练阶段

静态PDB按sequence/structure cluster、蛋白/配体、atom-size分层，先抽cluster再抽结构，限制同cluster/同结构重复。去掉水/离子、缺失原子与化学标签不兼容记录的规则需与dynamic一致；不能在评估时才补原子。独立记录unique structures、clusters、supervised atoms、累计exposure和重复率。

旧D仅做exact-ID exclusion且曾获准放弃near-homology控制。该豁免不能被宣称为已过滤；新主干预训练的正式对比应补train/valid的homology/structure-cluster控制，sealed test仅使用既有允许的ID清单排除，不能打开其坐标筛方法。internal-MD静态对照与外部PDB匹配atom-size、chemotype、static atoms与重复次数，减少“数据多”和“覆盖面广”的混淆。

推荐两个明确阶段：

1. **当前A1的D2桥接实验**：从同一个A1 checkpoint、相同reset策略做dynamic-only与joint D+S；不用static-only覆盖已学动力学。先128步profile，再冻结2k步paired pilot。这里回答“共享主干能否吸收静态信号”。
2. **真正预训练对照**：共同初始化teacher/stats与spatial/G/FFN/decoder后，一臂先static预训练共享参数，另一臂无static预训练，分别记录相同dynamic曝光和相同总compute两种比较。history/time参数在static-only阶段保持初始化，不进入optimizer。再进行dynamic主导joint训练，static replay梯度比从0.5降至0.25，并在预先固定的过渡长度内执行。这里回答“结构预训练是否提高样本效率与最终动力学”。

训练static-only到验证denoising/geometry不再改善或预先冻结预算，而非无限PDB epoch；不能用额外static训练量掩盖dynamic曝光不匹配。计算充足时扩大的是独立结构/cluster和重复seed，不优先堆叠更多小loss。

## 最小可解释实验矩阵

| 臂 | static数据 | static梯度到哪里 | 作用 |
|---|---|---|---|
| D0 | 无 | 无 | 相同dynamic曝光的A1延续 |
| D1 | 外部PDB | decoder-only，复现旧D | 检验主干接入本身的价值 |
| D2-I | internal MD单帧 | shared DiT/spatial/G/FFN+decoder | 区分任务/额外训练作用 |
| D2-E | cluster控制的外部PDB | 同D2-I | 检验外部结构多样性的迁移 |

首先固定ρ=0.25和4-block架构比较四臂，再只给D2-E做ρ的两侧小消融。胜出候选至少2个training seeds、3个generation seeds。static atoms/数据曝光尽量匹配；D0另补相同GPU小时的dynamic-only对照，以区分额外compute。当前C8不用等待、也不加入首轮；Mamba替换与D2共享预训练分开首轮，最后做2×2交互（原时间/Mamba × 无static/shared static），防止两个大改动相互掩盖。

## 验收与实现清单

实现新增 `molvid/training/static_backbone.py`、`StructureCondition/Query`，在现有 `FrameDiT`、`TrajectoryDecoder` 内显式按任务选择模块；保留shared权重对象，不复制一份static backbone。training orchestration负责两个microbatch、单次optimizer step、独立RNG/cursor和完整load报告。配置显式记录task、参数组、noise分布、gradient calibration、static数据/hash、每步预算和replay计划。

必要检查：static loss到每组shared参数有有限非零梯度；temporal/history/teacher无梯度且权重不变；动态正常更新time/history；clean target mutation不改变static source/condition（给定相同corrupted输入），只改变loss；无未来数据泄露；没有静态伪时钟；小batch mixed update与等价accumulation一致；single optimizer step和strict resume复现两套RNG/cursor；static微批不能改变dynamic下一个batch/噪声；FP32/BF16和最大真实batch通过。

static指标必须同时比较raw corrupted input、旧decoder-only、shared-backbone输出，要求真正net denoising，并报告bond/angle/clash/chirality及不同sigma表现。dynamic采用A的同一ordinary/fixed-history完整validation协议，至少bond/clash/severity、RMSF MAE/spread/correlation、MSD，并补足有效序列上的ACF/接触/二面角。探索性promotion建议主几何指标改善至少15%、motion/bond各不恶化超过5%，同时看per-system与seed分布；阈值应在训练前冻结。

只有静态MSE下降、共享主干梯度接近零、dynamic运动进一步收缩，或输出仍差于raw corrupted input，都不能算D2成功。对PVB/BioKinema的优势必须在统一数据/原子/时间/采样协议下实测；预训练设计本身不构成超越证据。
