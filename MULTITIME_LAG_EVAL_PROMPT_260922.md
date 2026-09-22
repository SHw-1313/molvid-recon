# 混合时间 lag 补充评估执行 prompt

请直接完成下面的评估实现、CUDA 检查、推理和结果汇总。任务是评价现有 checkpoint，不是重新训练或更改模型架构。完成必要的脚本适配后继续运行，不要只交计划，也不要先跑全量 pytest。

## 1. 要回答的问题

当前模型已完成 100/200/400 ps 混合训练，但尚缺该 checkpoint 的 held-out 生成结果。本次回答：

1. 同一个模型在训练见过的三个 lag 上，能否同时保持几何与运动统计？
2. 未训练过的 300 ps 上，是否出现明显的插值失败？
3. 正确的时间条件是否比错误时间标签更有帮助，还是模型近似忽略时间？
4. 运动幅度在体系之间、同一体系的原子之间是否仍被拉平？这种问题是否依赖 lag？

不要把训练 loss、有限梯度或“输出随时间标签变化”单独当成成功证据。本轮仍是 ATLAS 同源轨迹不同抽帧间隔，不覆盖跨数据集、1 ns 外推或长期 rollout。

## 2. 工作范围与读取顺序

在当前 molvid-recon 工作目录执行，无须为本任务另建 worktree。先确认当前路径、分支、未提交修改和正在使用的 GPU；不覆盖他人修改，不停止其他作业。若该工作目录已有其他 session 同时编辑评估文件，使用本任务独立的新脚本避免冲突。

先读：

- 根 `AGENTS.md`、`HANDOFF.md`。
- `agent/frame_joint_v1/HANDOFF.md` 中多时间训练与完成状态。
- `configs/frame_joint_v1_multitime_260921.yaml`。
- `tools/evaluate_frame_joint_rmsd_diagnosis.py`、`tools/summarize_frame_joint_rmsd_diagnosis.py`。
- `tools/build_frame_joint_multitime.py`、`molvid/data/preprocess.py` 中的数据身份、时间戳和抽帧处理。
- `molvid/generation.py` 的 `sample_frame_joint`；按需读 `molvid/evaluation/geometry.py`、`runner.py`、`motion.py`。
- `results_archive/frame_joint_v1_rmsd_diagnosis_260922/report.md`，复用固定 valid-quick 体系和既有指标定义。

已核对的参考代码版本为 `10b78e0045f5c96668fa45d2cbe006a635c7a12c`；若当前版本更新，记录实际版本并适配当前入口，不回退分支。

所有 Python、测试、模型推理和绘图命令均通过 `enter-container`，在 `torch-ito` 环境执行。模型及张量几何/运动评估使用 CUDA；文件解析、原始数据读取、最终序列化和绘图可使用 CPU。禁止把模型或整套几何计算悄悄搬到 CPU。

允许增加小型评估脚本、指标和派生 valid 数据，不修改训练目标、模型结构、checkpoint、训练集或原有统计量。无需新增一套 agent 文件或改根 AGENTS。不要启动训练、安装依赖、创建提交或 push。

## 3. 固定模型与推理设置

主模型：

```text
runs/frame_joint_v1_multitime_260921/frame_joint_step_00022272.pt
SHA256: b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4
```

通过当前加载入口加载 checkpoint 自带的架构和归一化统计，codec 路径与 SHA 从该配置/训练记录解析。不得套用 100 ps 单时间模型的 latent statistics，也不得重新在 valid 上拟合 statistics。若路径因挂载发生变化，可以定位同 SHA 文件并记录实际路径。

主评估设置：

- `model.eval()`、无梯度推理。
- Euler 16 步，沿用既有推理精度设置，记录实际精度。
- H4 与 H8 分开；总窗口仍为 16 帧，因此未来分别为 12 与 8 帧。
- generated 使用 seeds `[0, 1, 2]`；同一对照共享对应 seed。
- persistence 与 clean-latent decoder oracle 每个窗口、每个 H 只算一次，不按 seed 重复。
- 默认单张空闲 GPU；有两张空闲 GPU 时可按窗口分片后合并，不使用或中断已占用的 GPU。

本次只评估上述混合模型。已有 100 ps 模型可作为报告中的历史背景，但不同训练步数、暴露数据和统计量的模型不能被称为公平的单时间/多时间消融。无需额外启动单时间模型推理。

## 4. 构建配对 valid 评估视图

沿用最近诊断中的 **8 个 valid 体系 × 3 条 replica × 每条 2 个窗口**，每个窗口构建 100/200/300/400 ps 四个视图。以现有实际清单为准，不凭记忆重新选择体系；若实际数量不足，完整报告缺失项，不从 train/test 补齐。

重要：当前 multitime valid store 只有 300 ps。需要在相同 valid 体系的原始 ATLAS 轨迹上补建其他 lag 的小型评估视图，不能使用 train 中的 100/200/400 ps 数据充当 held-out。

要求：

1. 按原始 XTC 时间戳与 manifest 核对 native 间隔。已知原生 10 ps，对应 stride 10/20/30/40；真实抽帧，不能只改时间标签或插值伪造细帧。
2. 以同一原始 trajectory 的同一预测起点 `t0`（最后观测帧）配对所有 lag/H 视图。每个视图的历史为 `t0-(H-1)*dt ... t0`，未来为 `t0+dt ... t0+(16-H)*dt`。
3. 保留原有体系/replica 清单。若原有窗口的 t0 不支持最长视图，根据轨迹长度、各视图共同有效范围和固定规则选择替代 t0；在看生成结果前写入评估清单。两窗口尽量不重叠，无法做到则标注。不能按运动大小或结果好坏选窗口。
4. 记录原始体系、replica、绝对 t0、原始帧索引、原始时间戳、模型相对时间、lag、H、未来帧数、历史和未来物理跨度。不同 lag 的同号 `_w` 不代表同一物理窗口。
5. 原始 native dt=10 ps 与 sampled dt=100/200/300/400 ps 使用不同字段。不要沿用旧诊断中把 sampled dt 标成 native dt 的命名。
6. 只在新的评估输出目录写派生数据/索引，不重写已有 multitime manifest、train/valid store，不打开 test。

不要直接照搬旧脚本的 sample ID 正则：新 ID 中含 `_dt_300ps_` 等片段，旧的末尾 `_R数字` 解析可能失败或把 lag 混进 system 名。优先使用 record 的显式体系/replica 信息，所有结果键必须包含真实 system、replica、anchor、dt、H、path、seed，防止覆盖与错误聚合。

## 5. 必做评估 A：各 lag 的真实生成

对四个 lag、两个 H、上述固定窗口分别运行：

- **persistence**：复制最后观测帧，仅用于运动/坐标基线。
- **clean-latent decoder oracle**：用真实未来编码检验该时间间隔下的解码上限；明确标为使用未来信息的诊断。
- **generated**：仅用观测历史、拓扑、查询时间，从现有随机源正常生成未来。不能把真实未来 latent 或任何未来几何派生特征传入生成路径。

至少输出以下指标，按 dt/H 分开，不先混成总均值：

| 目的 | 指标与解释 |
| --- | --- |
| 几何合理性 | bond RMSE；aligned RMSD、dRMSD 作为与一条参考路径的偏差诊断 |
| 原子运动分布 | future RMSF 的生成值/MD 值、MAE、原子间 Pearson/Spearman 相关；保存逐原子 RMSF |
| 体系运动差异 | 每体系平均 RMSF，跨体系相关与离散程度；画 generated–MD 散点 |
| 幅度拉平 | 各体系 RMSF 比值；跨体系 std(predicted RMSF)/std(MD RMSF)；体系内逐原子 RMSF 的同类离散程度比值 |
| 时间相关运动 | 下面定义的物理 lag 位移统计；已有 ACF 同时输出生成/MD/误差及其确切物理 lag |
| 构象统计 | 复用已有 contact occupancy MAE，不增加一轮新阈值扫描 |

零分母、常量序列、帧数不足导致的无定义值，输出 null、原因和有效计数，不能写成 0。persistence 的 RMSF=0 是正确结果，但相关性/ACF 可能无定义。

RMSF 沿用声明的 align_mask 和已有“对齐到本视图首个观测帧”定义；预测与 MD 共享该观测参考。新增的跨视图位移统计以共同最后观测帧 t0 为对齐参考，记录口径。不能通过逐帧对齐到真实未来来改善生成运动统计。保留原有 aligned RMSD 的独立指标定义，二者不要混用。

为了避免只看平均 RMSF，保存每体系/原子的值。少量高柔性体系是否被压低、低柔性体系是否被抬高，要直接由散点与数值回答。

### 时间指标的具体口径

至少增加/输出物理 lag 的均方位移：

```text
MSD(delta_ps) = mean over valid atom/time pairs of
               ||x_aligned(t + delta_ps) - x_aligned(t)||^2
```

单位为 Å²，不除以 delta_ps，不把它称作瞬时速度。分别输出生成和 MD、误差、参与配对数量。

- 原生视图内：报告 `dt`、`2*dt`、`3*dt`，只用有效未来帧配对。
- 另输出从最后观测 t0 出发的平方位移曲线，横轴为距离最后观测的真实 ps。H4 下 t0+1200 ps 是四种视图共有的查询点，可比较该端点统计；明确历史跨度仍不同，不当成纯时间条件消融。
- H8 的 100 ps 视图只预测到 t0+800 ps，四个 bucket 没有共同的非零查询时刻。不要声称它也有 1200 ps 公共端点，更不要插值伪造；只在确实共有的时刻做成对比较。未来区间内部的位移配对与 t0 到未来的位移是两种统计，分别命名。
- 现有 `dynamic_acf_metrics` 是对对齐后的帧间平均速度做 lag-1 相关。它的 lag-1 在四个 bucket 中分别代表 100/200/300/400 ps，而且速度本身也经过不同时间宽度的平均。可复用为各 bucket 对照 MD 的辅助指标，不能把这些数值直接当成同一物理量横比。现有 dynamic_correlation 也不能替代生成轨迹自身的 ACF。

这里不拟合 implied timescale、不做 CK 检验、不用 8/12 个未来帧宣称恢复了长时程动力学。路径 RMSD、velocity RMSE 不能单独裁决随机生成优劣。

## 6. 必做评估 B：模型是否真正利用时间条件

复用 A 的 H4、200/300/400 ps 全部窗口，不增加训练，做 **真实时钟 vs 错误的 100 ps 时钟** 对照。

- 坐标历史、拓扑、真实目标、帧数、checkpoint、采样 seed 完全相同。
- true-clock 为 A 中的正常结果，直接复用。
- wrong-clock 仅把模型可见的全部历史/未来物理时间统一改成相邻 100 ps，仍以最后观测为共同时间原点。同步更新时间派生字段，history encoder、DiT、decoder 都接收同一个错误时钟；不修改 flow time。
- 独立保留原始真实时钟和原始目标，所有评分使用真实物理时间。不能把评估器的时钟一起改错。
- 以相同 seed 进行配对；时间干预前后 tensor shape 相同。记录初始噪声是否一致，不能只假定一致。

报告两类结果：

1. 输出/运动特征改变多少，证明条件敏感性。
2. true-clock 是否使 RMSF、物理 lag MSD、ACF 更接近对应真实 MD，同时几何是否恶化。这才是有用的时间条件证据。

不要要求每个体系所有指标都改善，也不要预设“lag 越大 RMSF 必须越大”。若真/错时钟差异小，结论是当前小样本未观察到时间条件收益；历史坐标本身也可提供间隔线索，不能直接判定时间模块完全失效。若差异大但正确时钟不更准确，结论是敏感但未显示正确利用。

本轮不追加固定历史、任意不规则查询序列的大矩阵。那会同时引入新的查询分布，留待上述结果需要时再做。

## 7. 公平比较、聚合与计算量

16 帧的总跨度随 dt 改变：100/200/300/400 ps 对应 1.5/3/4.5/6 ns。各 bucket 首先对照自身 MD，而不是用 RMSF 随 dt 增大的趋势证明成功。

H4/H8 的历史长度和未来跨度同时不同，不根据整段平均分数直接断言 H8 更好。需要比较时使用共同预测时刻，并展示各自历史跨度。

聚合先对采样 seeds、窗口、replica 取平均，再对 system 等权，dt/H 保持独立。不得把大量原子、相关帧、重复 seeds 作为独立体系。报告 seed 波动和各体系分布；样本小就明确是初步证据，不新增人为通过阈值。

完整主生成矩阵为 8×3×2×4×2×3 = 1152 次，wrong-clock 为 8×3×2×3×1×3 = 432 次；oracle/persistence 不随 seed 重跑。实际清单不足时按真实数目报告。先完成 seed 0 的所有 bucket 和对照，再补 seeds 1/2；中断可以按输出记录续跑，复用已完成结果。

使用一个实际大体系做 CUDA 运行估时，打印总预计耗时，避免重复加载模型/重复编码固定历史等明显浪费。预计耗时异常时先定位评估实现，不自行扩大 GPU 占用或改变采样步数。资源不足时保存可恢复状态和已完成结果，明确缺失，不能用不足的 seeds 冒充完成。

## 8. 最小正确性检查与执行顺序

先实现适配，再进行以下有针对性的 CUDA 检查，随后直接完成正式 valid 推理：

1. 一条 300 ps 样本、H4/H8：checkpoint 加载、时间/单位/shape 正确，结果有限，观测前缀保持一致。
2. 对一条固定样本，固定 seed，修改模板中的未观测未来坐标及未来几何占位字段；生成结果应在声明的数值容差内不变。真实目标只用于评分/oracle。复用已有有效泄露检查入口即可。
3. 检查 dt 标签不会污染 system/replica 分组，结果键无冲突；wrong-clock 的模型输入时钟确实改变，而评估真实时钟不变。

不需要跑全量 pytest、不需要重新测训练反向传播、不需要性能扫参或 Euler 步数扫描。无架构修改时不要重复此前已通过的整套 SE(3)/训练检查。

完成 A/B 后不要因为结果差而自行修改模型或追加训练。本任务的产物是可据以决定下一步的数据。

## 9. 输出与报告

建议新增一个清晰入口 `tools/evaluate_frame_joint_multitime.py`，复用现有加载、生成、指标实现；必要的新通用指标放 `molvid/evaluation/` 对应模块，不复制整套模型，不给模型 forward 加评估特例。辅助汇总代码只在有需要时拆出。

大文件输出到 `runs/frame_joint_v1_multitime_eval_260922/`。保留逐样本记录、可恢复完成状态及足以重算指标的生成坐标/真实帧索引，避免把所有坐标长驻 GPU 或一次性堆满内存。

把便于审阅的小文件写入 `results_archive/frame_joint_v1_multitime_eval_260922/`：

- `report.md`：主结论、证据、限制、下一步最小动作。
- `protocol.json`：代码版本、checkpoint/codec SHA、statistics hash、数据身份与配对窗口、time/mask/单位、seed、Euler 步数、真实执行命令。
- `summary.json`、`per_system.csv`、`time_condition_ablation.csv`，以及小型逐原子 RMSF 文件或其明确索引。
- 三张图即可：各 lag 几何和运动误差；各 lag 的逐体系 RMSF 散点；true/wrong-clock 的运动误差配对比较。物理时间曲线可作为补充，不堆无关图。

对被 gitignore 的大结果，报告写明真实位置和生成命令；小型汇总与图保持可被用户上传审阅，不强制加入巨大坐标文件。更新本任务结果说明即可，避免覆盖其他 session 的 HANDOFF。

报告逐项回答：

1. 已见三个 lag 是否都可用，还是某个 lag 明显失效？
2. 300 ps 相比相邻 200/400 ps 的生成—MD 误差是否出现异常？不能要求 300 ps 的原始运动幅度必须介于两者之间。
3. 正确时钟是否比错误时钟提供实际收益？
4. RMSF 拉平发生在所有 lag 还是特定 lag，是跨体系还是体系内原子层面？
5. 若某 lag 生成变差，clean oracle 是否也变差？oracle 正常而 generated 差，优先指向生成路径；oracle 也差，则需要检查解码/时间条件/该视图数据。但不能仅凭此唯一归因。

最终分别标记“混合 lag 已见间隔生成”“300 ps 插值”“正确使用时间条件”“保留运动幅度差异”为：有初步支持 / 未观察到支持 / 证据不足，并引用对应表格数值。不要把四者压成一个通过/不通过。

完成后用简短回复给出结果目录、最重要数值与最值得采取的一项后续动作，不自动开始该后续动作。
