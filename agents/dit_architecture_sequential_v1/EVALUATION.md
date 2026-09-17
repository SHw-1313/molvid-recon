# 最小充分的验证与保留规则

## 1. 一次登记，后续复用

固定原 48 train / 8 valid / sealed test。科学选择只看 train/valid，不读取 test payload。首次公共选择登记后不得因结果更换体系或窗口。

- 快速生成集：固定 8 个 train systems、全部 8 个 valid systems，各用 1 个预先选定 clip，valid 沿用 R1/window30；H4/H8、Euler 16、4 draws。按当前实际 manifest 核实 R1 的 replica 含义，不把它当 compression ratio=1。
- 最终验证集：原 72 valid clips（8 systems×3 replicas×windows 0/30/61），H4/H8、16 steps、draw0；候选和对应 control 都跑相同集合。快速集合上的 4 draws 保留用于多样性和随机误差诊断。
- 固定 eps 的 seed 只由 sample/H/draw 和全局 generation seed 组成，不包含 arm、round、checkpoint step 或 batch 位置。
- 训练配对 seed：`20260914 + round_id`，初始化、training、validation、history corruption 使用分离 generator。checkpoint 保存完整状态；只设全局 seed 不能证明配对成立。
- 每个模型完整独立加载 checkpoint+adapter+refiner（如有）。保存加载后权重 hash，评估另一模型不能改变已加载模型。

## 2. 验证顺序与停止范围

每轮只运行下面四步；公共隔离测试首次通过后，不因无关 loss 改动重复整套，只验证新 loader/resume 等受影响部分。

1. 完成代码和配置，再跑 PLAN 指定的 targeted CUDA 数值检查。
2. 沿用 3 体系/9 轨迹做真实 clip 一步反传与短生成 smoke。它只判断是否可运行，不用于排名。已有缓存可只读复用，验证来源和 gauge。
3. 跑直接受影响的 adapter/mask/backend/checkpoint 等回归，数值测试必须 CUDA。不要全仓 pytest；不重跑 codec T1、不重选 geometry。
4. 正式配对训练、固定生成评估、写 decision，然后下一轮。

任何 `.cpu()` 仅可出现在离开数值路径后的最终标量/数组序列化；Kabsch、bond/contact、位移、RMSF、采样、refiner、rollout 计算留在 CUDA。可复用已有指标定义并移植必需的计算路径，记录容差内的数值一致性；不能为“全 CUDA”改变指标语义或重写整套工具。

只在遇到明确数值风险时扩大检查。不要把数百个 metadata 测试、全 dt 扫描或新的 benchmark 变成科学训练前置条件。

## 3. 所有轮次的共同指标

主结果按 draw→clip→system 聚合，体系等权；H4/H8 分开。报告真实单位、有效样本数和缺失原因；不把 batch-row 均值伪称逐体系均值。

| 指标 | 解释与边界 |
|---|---|
| future bond RMSE（Å）、contact F1 | 局部几何及相互作用；不要只看平均 bond，把逐帧也存下 |
| future aligned RMSD（Å） | 与一条 reference 的路径偏差诊断，随机生成不要求逐条复制；不作为唯一胜负标准 |
| 每体系/原子 RMSF prediction、MD、MAE、correlation | 不能由 pooled RMSF ratio≈1 断言体系差异学对；近常数 correlation 返回 null |
| 块内位移、块间位移和 ratio | 按有效 future transition 分类；observed→future 单列；比较绝对量及相对 MD 的误差，不追求边界跳变=0 |
| future-only sample diversity | 同一条件不同 draws 的 pairwise aligned RMSD，排除历史；多样性可能是噪声，结合几何/MD 分布解释 |
| 实际推理延迟、训练吞吐、显存 | 分开列 DiT、source/decode、refiner 成本；不只报 FLOPs |

RMSF 的参考和预测使用同样的已核对对齐规则、同一 future 区间。位移/速度须保留物理时间，不对每一帧任意独立对齐后声称得到了原始动力学。沿用 A 现有明确坐标 convention，记录任何必要修正并对所有臂重算。

静止 repeat-last 与 oracle decode 在每个固定集合只计算一次复用。它们是尺度参考，不是“动态生成一定要打败的逐路径 oracle”。不加 best-of-N 作为主指标。

## 4. 各轮主要判断

定义逐体系误差：

- `E_amp`：预测 future RMSF 与 MD 的逐原子绝对误差，先原子平均，再体系等权；同时报告各体系平均 RMSF 的差异/标准差，防止统一运动幅度。
- `E_boundary`：每体系 `abs(log(r_pred/r_MD))`，r=块间位移/块内位移。只在两者均有合法非零分母时适用；近静止体系改报绝对位移差，不制造大 ratio。
- `E_roll_bond`：第 2、3 段 `max(0, bond_generated_prefix - bond_true_prefix)` 的均值；同时报告不截断的两组原始值、contact 变化和误差曲线。control 已接近零时不计算夸张的百分比收益。

| 轮次 | 主变量与主指标 | 必须排除的假改善 |
|---|---|---|
| 1 | pre vs post norm；E_amp | 均值碰巧接近，但体系差异/相关仍塌缩；几何被破坏 |
| 2 | RF+bond vs RF；future bond RMSE | 训练 endpoint 更好，自由生成没变；通过不运动降低 bond |
| 3 | T vs L，再分别对 P；E_boundary | 统一平滑使块间/块内都过小；收益仅来自任何 refiner 后训练 |
| 4 | noisy vs clean training history；E_roll_bond | true-prefix 也变坏而“差值缩小”；clean 一段退化；只是几何投影隐藏了误差 |

第 4 轮短 rollout：固定 8 valid systems、每个 1 条可覆盖 32 帧的连续原轨迹片段、2 draws；H8，首 8 真帧，连续生成 3 段各 8 帧；同初始条件和逐段 eps。true-prefix 对照在同一绝对时间使用真实前 8 帧；generated-prefix 只使用自己最近生成的 8 帧。两者后续 context 可以不同，其他 sampler 规则相同。第一段两者必须一致，否则先修对齐/随机数/缓存。

refiner 若被采用，每段先 refine 后用实际输出作为下一段 history，同时保留 raw/refined 指标。不得下一段暗中喂 pre-refine latent、真历史或额外 reference。

第 1/2 轮不跑完整 rollout；第 3 轮最多在候选有单段收益后做 4-system/1-draw 的短检查，不能用它声称第四项已经完成。第 4 轮才做上述完整短诊断。

## 5. 保留、回退和不确定

以下是本轮**筛选用默认阈值**，不代表统计显著性或论文成功标准。执行前可以因明确指标量纲/样本适用性登记修改，不能看本轮结果后移动标准。

- 主指标相对同轮 control 改善约 ≥10%，在至少 5/8 个 valid systems 同向，且 H4/H8 没有明显相反趋势：可考虑 `KEEP`。rollout 主判断适用 H8，另检查 H4 clean 单段。
- 共同保护项：bond 不恶化 >5%，contact F1 不下降 >0.01，E_amp 不恶化 >10%；检查绝对位移、逐体系幅度和多样性有无明显静止化/噪声化。接近零的基准使用实际测量误差/绝对差解释，不依赖不稳定相对百分比。
- 候选自身主指标良好但另一项存在可信代价时写 `TRADEOFF`，不自动叠入下一轮；默认沿用简单 control，并完整记录候选。不能通过删除 guardrail 使它获胜。
- 无改善或明确退化：`REJECT`，继续使用等量续训的 control。
- 训练不足、seed 方差、少数体系相反、可适用样本不足：`INCONCLUSIVE`。仅在预留预算内双方共同续训一次；仍不清楚保留 control，不宣布机制已被否定。
- 第 3 轮只有 L 好时可 `KEEP_LOCAL_ONLY`；T 必须相对 L 有明确收益，才能 `KEEP_CROSS_BLOCK`。两者不超过 P 则 `KEEP_PARENT`。

快速集合上有希望才进入 72-clip 验证。快速集合已经明确失败时保留失败证据即可，不继续把昂贵评估当仪式。最终 72-clip 结果若推翻初筛，则按失败/不确定处理；不能只保留初筛漂亮图。

默认不为每轮跑三份独立训练 seed；4 draws 只是采样变化，不是训练重复。筛选结束后，最终组合及关键 matched control 若需论文证据，另安排独立 seed 和充分训练，这不隐含在本轮预算中。

## 6. 每轮最小交付

`decision.json` 至少有 round、parent/code/config/data/stats hashes、唯一变量、initialization hash、checkpoint/adapter/refiner hash、successful_updates、有效 token、GPU-hours、primary/guardrail 按体系指标、status、selected_checkpoint、remaining_risk。

每轮短 report 加三类图即可：生成质量/运动的逐体系配对图；必要的逐帧或块边界曲线；训练/成本曲线。第 4 轮增加 true/generated-prefix 分段 bond/contact 图。图可以由保存的紧凑数值重新生成，不必提交大预测文件。

`HANDOFF.md` 清晰区分“代码完成”“CUDA 通过”“科学比较完成”，列未跑事项和原因。报告不得把本地文档自查说成已完成模型测试，也不得把一次 48-system pilot 称为可扩展性证据。
