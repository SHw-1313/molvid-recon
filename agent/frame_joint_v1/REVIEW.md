# Frame Joint v1 独立实现审阅

审阅日期：2026-09-20。

本文件整理本次独立审阅结论，不表示实现者已经修复下列问题。修复由写代码 session 执行；本次审阅不修改科学阈值。

## 结论与审阅范围

**目前不能按现有配置进入 48 体系长训练。** 三个直接阻断项是：BF16 decoder 报错、decoder 局部拓扑分支无法学习、DDP 对阈值筛选后的辅助损失归一化错误。成对 continuation 另有恢复缺陷，应在启动需要可恢复性的双臂实验前修复。

审阅基准：

- HEAD：`93d191fd8c88c2355944c1cf8bf05e8519ddb598`。
- 实际对象：上述 HEAD **加审阅时的未提交工作区**。Frame Joint v1 主体尚未进入该 commit，不能将结论归给该 commit 单独包含的代码。
- 范围包括 `molvid/`、`configs/`、`tools/`、本任务目录、`docs/frame_joint_v1.md` 以及根 AGENTS/HANDOFF/TASKS。审阅开始和结束时核对了范围内 101 个文件的 SHA-256，内容未变化，未混入审阅期间的新 diff。
- 已读取根 AGENTS、任务 PLAN、ARCHITECTURE、CHECKPOINTS、TRAINING_EVAL、READABILITY 和最新 HANDOFF，以及相关迁移记录和历史实验边界。
- 审阅过程未修改文件、启动训练或执行 optimizer 更新。随后仅按用户要求创建本报告。短 CUDA 检查通过 `enter-container` / `conda activate torch-ito` 执行；未打开封存 test。

以下区分：**代码事实**是可从当前实现直接确认的机制；**实验事实**来自已有结果文件或本次短 CUDA 检查；**推测／未知**不作为已观察到的科学结论。

## 阻断问题

### R1 · P1：BF16 decoder forward 报 dtype 错误

**位置：** [molvid/codec/decoder.py](../../molvid/codec/decoder.py)，`CovalentMessage.forward`，审阅时第 71–77 行。

**代码事实：** `scalar_message` 经 autocast 成为 BF16，`torch.zeros_like(h)` 创建的累加器仍为 FP32，随后 `scalar.index_add_` 的输入类型不匹配。

**本次实验事实：** 使用 corrected step-12 checkpoint 和真实训练样本，在 CUDA 上执行 decoder 的 BF16 autocast forward，复现：

```text
RuntimeError: index_add_(): self (Float) and source (BFloat16) must have the same scalar type
```

**影响：** 48、bond_keep、bond_release 配置均选择 BF16；已有 FP32 smoke/profile 不能证明这些配置可运行。

**最小修复方向：** 明确消息与累加器的精度边界，保持必要的几何累加为 FP32，再验证实际训练配置使用的 autocast 路径。不能仅以旧 FP32 检查代替。

### R2 · P1：decoder 共价键消息分支被双重零初始化锁死

**位置：** [molvid/codec/decoder.py](../../molvid/codec/decoder.py)，`CovalentMessage.__init__` 第 42–46 行、`TrajectoryDecoderBlock.__init__` 第 94–95 行和 `forward` 第 113–115 行。

**代码事实：** 局部消息输出层为零，外层 `tanh(local_gate)` 也为零。因此消息参数梯度被 gate 乘零，gate 梯度又被消息输出乘零；该分支无法靠正常梯度更新自行启动。

**本次实验事实：** corrected step-12 checkpoint 的两层 local gate 最大绝对值均为 **0**。真实 clean reconstruction backward 后，两层全部 local 分支梯度最大绝对值均为 **0**，但没有缺失梯度张量。

**影响：** 声明的 decoder 局部拓扑交互实际上不起作用。“decoder 组件有非零梯度”或“没有参数缺失梯度”不足以发现该缺陷。

**最小修复方向：** 只保留一处零初始化，使另一侧能够提供启动梯度；修复后的训练应明确区别于现有失活分支 checkpoint，不能把旧结果视为已验证了有效的局部拓扑交互。

### R3 · P1：DDP 辅助损失不等价于同一 global batch 的单卡计算

**位置：**

- [molvid/training/joint.py](../../molvid/training/joint.py)：`_sample_equal_coordinate_mse` 第 98–100 行、`FrameJointTrainer._ddp_scale` 和 `train_step` 第 439 行。
- [molvid/losses/geometry.py](../../molvid/losses/geometry.py)：`future_bond_distance_loss` 第 130–134 行。
- [tools/check_frame_joint_ddp.py](../../tools/check_frame_joint_ddp.py)：`_fixed_step`。

**代码事实：** generated/near loss 先按各 rank 的 applicable samples 求均值，最终却统一按整批样本数修正。各项 loss 的分母不同，不能共享同一个整批缩放因子。

例如两卡各一个样本，只有一张卡满足阈值：单卡 global-batch 辅助损失为 `L`，当前 DDP 梯度对应 `L/2`。

**已有实验事实：** DDP 检查固定所有样本 `s=0.95`，单卡／双卡参数更新最大差为 `1.1920928955078125e-07`。该结果有效，但未覆盖各 rank 阈值筛选后有效样本数不同的情况。

**影响：** 有效 bond/near 权重随 rank 分配而改变，影响校准和成对实验解释。

**最小修复方向：** 各 loss 项分别保留有效样本计数，按全局 applicable count 归一化。针对性检查应包含不同 rank 上阈值资格不一致的 global batch。

### R4 · P1：continuation checkpoint 不能正常普通 resume

**位置：** [molvid/cli/train_frame_joint.py](../../molvid/cli/train_frame_joint.py)，`main` 第 373–380 行；[molvid/training/joint.py](../../molvid/training/joint.py)，`FrameJointTrainer.__init__`、`contracts`、`load_checkpoint`。

**代码事实：**

- keep 配置的 `inherit_parent: true` 在 `--resume` 时先要求 `--continuation-parent`，而这两个参数互斥。
- 即使绕过该处，新 trainer 的 `continuation_parent=None` 仍与已保存 child 契约中的 parent 信息冲突；普通加载在恢复该身份之前就比较契约。

**影响：** 两臂首次分叉可以通过现有检查，但不能据此宣称两臂中断后可恢复。此项阻断需要可恢复性的成对续训，不是普通 tiny 首次运行的失败。

**最小修复方向：** 普通恢复从 child checkpoint 恢复 resolved loss 和 parent 身份；只有首次分叉执行 parent 继承逻辑。

## 其他具体问题及适用边界

### R5 · P2：“严格恢复”没有覆盖完整训练调度

**位置：** [molvid/cli/train_frame_joint.py](../../molvid/cli/train_frame_joint.py)，`_scheduled_rates`、`_stage_at` 和 `history_order` 处理；[molvid/training/joint.py](../../molvid/training/joint.py)，`FrameJointTrainer.contracts`。

**代码事实：** LR schedule 在 CLI 根据当前 YAML 重算，checkpoint 的 `scheduler_state` 为 `None`。契约没有保存完整 stage 长度、逐阶段 LR 和 `history_order`。改变这些值可以通过现有检查，却改变恢复后的训练。

**影响：** 相同配置下已有的精确恢复结果仍有效，但不能扩大为完整调度语义受到严格保护。

**最小修复方向：** 保存并验证完整 resolved schedule 与 H 调度。此项不应与 R1–R3 的直接启动阻断混为一谈。

### R6 · P2：无定义的运动指标仍可能显示为好分数

**位置：** [molvid/evaluation/geometry.py](../../molvid/evaluation/geometry.py)，`_safe_correlation` 第 490–497 行、`dynamic_acf_metrics` 第 749–760 行；新 tiny evaluator 直接调用这些指标。

**代码事实：** 两个零方差信号的 correlation 可以返回 `1`；没有足够动态样本时，dynamic 指标返回零误差。

**影响：** 静态或过短窗口可能被显示为有效的良好分数，违背本轮“不可用必须明确标记”的评估要求。没有证据表明当前 tiny 汇总已经触发这些退化情况。

**最小修复方向：** 在新评估结果中标记 `available/reason`，从聚合中排除无效项，不修改科学阈值。

## 八项审阅结论

| 审阅重点 | 结论 |
| --- | --- |
| 1. teacher、统计与初始化 | 指定 `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df` artifact 严格核对 key/shape/dtype 后，提取 Haar 前 TorchMD＋stem；本次加载也通过 SHA 校验。新统计从 train 流拟合，独立 schema，不使用旧四字段统计。teacher 未发现随机补齐。新 DiT、history temporal 和 decoder 新层在文档中明确声明；provenance 仅覆盖 teacher，尚缺全模型逐项初始化报告。 |
| 2. 条件、目标、source、endpoint、坐标 | `prepare_frame_joint_batch → source/interpolation → FrameJointModel → velocity → endpoint → inverse statistics → decoder` 可追踪。原点由首帧定义，未发现未来均值或真实未来坐标进入条件。已有 future-mutation 检查为零差异。公开 `sample_frame_joint` 仍接受完整 `ClipBatch template`；当前不读取其未来坐标，但尚未达到计划要求的类型层隔离。 |
| 3. 历史压缩与未来生成 | 已分离。Haar 只用于历史完整四帧组；旧 inverse 模块仍挂在 history 内，但冻结且未调用。未来为两字段逐帧 h/v，无未来 ratio、四帧整除或固定 T=16 约束。每个 Euler step 联合更新全部 query，future attention 非因果。teacher 的四帧 chunk 是编码批次设置。 |
| 4. 梯度、optimizer、DDP | history 在模型 forward 内执行，未被 batch preparation 的 no-grad 切断。coordinate head 已设为可训练并进入 optimizer。DDP 包装整个 composite model。实际缺陷是 R2 的局部分支失活和 R3 的辅助项归一化，不是遗漏整个 decoder/history。 |
| 5. 时间、静态 detail、xyz | `time_ps` 与 `flow_time` 分开。静态 detail 有零保持设计，已有检查最大值为 0。vector 线性层作用于最后通道轴，无 vector bias；归一化收缩 xyz，未发现普通 Linear 混合 xyz。已有 flow 旋转检查发生在零输出初始化阶段，证明范围有限。 |
| 6. bond release 与成对续训 | 三处显式 bond 权重统一归零；已有 forced-s 检查显示与删除 bond 项的梯度差为 0。保留拓扑不等于 bond loss 未关，但当前 local topology 分支另有 R2。双臂配置的数据、H 顺序、低 LR schedule 相同；已有单步证据确认共同 parent、moments、cursor、RNG。尚无长程两臂结果，且 R4 待修。 |
| 7. 科学成功判定与可比性 | 当前新路径没有按 bond 改善自动选优，HANDOFF 明确 tiny 未收敛。RMSF target 来自真实 MD。H4/H8 分别评估 12/8 个未来帧，可以分别报告，不能用于“更多历史更好”的归因；尚无相同末八帧比较。tiny 协议独立标识，未发现 quick/full 混合。无效运动指标另见 R6。 |
| 8. 可读性 | model、history、DiT、decoder 主体职责清楚，没有新 registry 或多层模型 wrapper。具体混杂集中在 CLI：同时拥有阶段状态机、LR 计算、loss 继承/校准、cursor 恢复和训练循环，而 trainer 拥有另一半 checkpoint 契约。这与 R4/R5 的恢复遗漏直接相关。 |

可读性上的最小整理方向是：把训练运行状态和契约集中到 training 层，让 CLI 构造并调用。判断依据是职责分裂与恢复数据流，不是文件行数；无需重建架构。

## 证据及其边界

### 已有结果

- [CUDA 检查](../../runs/frame_joint_v1_cuda_260920/cuda_checks.json)：teacher h/v 差约 `6.68e-6` / `2.38e-5`；重复静态 detail 为 0；future mutation 为 0；forced generated-bond 对 DiT 梯度范数约 `0.39210`；near-endpoint 对 DiT 梯度为 0；release 与删除显式 bond 项的梯度差为 0。
- [DDP 检查](../../runs/frame_joint_v1_cuda_260920/ddp_check.json)：固定全体 `s=0.95` 的单卡／双卡更新最大差约 `1.19e-7`，model/optimizer 恢复差为 0，每 rank generator 相同。其适用边界见 R3。
- [continuation 检查](../../runs/frame_joint_v1_cuda_260920/continuation_check.json)：共同 parent 起始 model/optimizer 差为 0，cursor 与起止 generator 相同，单步 `flow_time=0.3394112288951874`；release 三处 weighted bond 为 0。该步未触发 generated/near 阈值，也没有验证 child checkpoint 的普通恢复。
- [最大体系 profile](../../runs/frame_joint_v1_cuda_260920/profile_max_system.json)：3542 atoms、T16、H8，全流程峰值分配约 36.51 GB。HANDOFF 明确该路径使用 FP32，不能视为 BF16 验证。
- [最终源码 tiny 评估](../../runs/frame_joint_v1_tiny_corrected_260920/evaluation_current/tiny_evaluation.json)：3 systems / 9 trajectories，每轨迹一个固定 valid window，seed 0，Euler 16，H4/H8 分别报告。仅为 12-update pipeline smoke，不构成科学收敛或 production parity 证据。

### 本次针对性短检查

使用 GPU 0、`enter-container` / `torch-ito`，设置 `PYTHONDONTWRITEBYTECODE=1`，加载：

```text
runs/frame_joint_v1_tiny_corrected_260920/frame_joint_step_00000012.pt
SHA-256: c7e18d55a7d5e8bebf29c2dbee31470bcd9dfe5ec73003bb03919ef522622c7e
```

读取原 tiny train store 的首个真实样本，H8；执行一次 clean decoder FP32 forward/backward，检查两层 local 分支，再执行一次 BF16 decoder forward。未执行 optimizer step，未保存 checkpoint，未创建测试文件，未重跑完整套件。

结果：两层 local gate 与 local gradient 的最大绝对值均为 0；两层均无缺失梯度张量。BF16 forward 抛出 R1 所列 dtype 错误。本次输出保留在审阅会话中，没有另行生成结果文件。

## 尚未交付及不能据此推断的内容

- 完整下一阶段评估尚未接通：训练 CLI 不执行 epoch validation；[现有 evaluator](../../tools/evaluate_frame_joint_tiny.py) 硬性限定 3 systems / 9 trajectories，不能直接承担 48 体系 quick/full 协议。
- 48 statistics 尚未拟合、48/192 长训练和长程成对 continuation 尚未执行，是 HANDOFF 已声明的状态，不应伪装为已有证据。
- 现有检查支持其实际覆盖的路径，不能扩大为 BF16 可运行、每个分支均有有效梯度、任意 rank 筛选下 DDP 等价或 continuation child 可恢复。
- **未知：** 上述缺陷对最终运动质量的影响幅度。本次没有据 tiny 结果推断架构成败，也没有把“几何更好但运动更差”判定为成功。

当前交付应描述为“部分工程链路已验证”，不能描述为“48/192 训练与评估已就绪”。

## 修复回应（2026-09-20）

本节由实现 session 在保留以上独立审阅原文的前提下追加。结论来自逐项重新阅读当前代码和针对性复现；没有把审阅意见直接当作已证实事实。本次只做必要修复和短检查，没有启动 48/192 正式训练。

### R1 · BF16 decoder dtype

- **是否成立：成立。** autocast 下消息 MLP 输出可能为 BF16，而 residual 累加器沿用 FP32 `h/v`；原 `index_add_` 的确要求相同 dtype。
- **修改位置：** [molvid/codec/decoder.py](../../molvid/codec/decoder.py) `CovalentMessage.forward`。scalar、vector-neighbor 和 direction 消息在 `index_add_` 前显式转换到对应 residual dtype；距离范数仍显式用 FP32 计算。
- **验证结果：** [cuda_checks.json](../../runs/frame_joint_v1_review_fixes_260920/cuda_checks.json) 使用真实 codec、真实 887-atom/T16 train sample 和与 48 配置相同的 256/128/depth-4 模型规格；BF16 decoder forward finite，并完成 3 个 BF16 joint optimizer step。此前的 dtype 异常未再出现。
- **尚未解决：** 这不是 3542-atom 最大体系的 BF16 吞吐/显存 profile，也不是长训练稳定性证据；二者仍留给 `TRAIN_PROMPT` 阶段。

### R2 · decoder local 分支双重零初始化

- **是否成立：成立。** message 输出层与 outer local gate 同时为零时，两侧梯度相互锁死。
- **修改位置：** [molvid/codec/decoder.py](../../molvid/codec/decoder.py) `CovalentMessage.__init__` 删除 message 输出侧的零初始化，只保留 `TrajectoryDecoderBlock.local_*_gate` 的零初始化。这样 fresh model 第一步先更新 gate，随后 message 参数可得到梯度；没有引入额外兼容路径。
- **验证结果：** 同一 [CUDA 检查](../../runs/frame_joint_v1_review_fixes_260920/cuda_checks.json) 中，两层第一步 local-gate 梯度范数为 `0.00538812 / 0.00529391`，step 后 gate 最大绝对值为 `4.99997e-5 / 4.99997e-5`；第二步 local-message 梯度范数为 `3.85737e-6 / 3.73313e-6`。三步后 history/DiT/decoder 分别有 `14/204/79` 个参数得到 finite nonzero gradient。
- **尚未解决：** `frame_joint_v1_tiny_corrected_260920` 的 step-12 checkpoint 是本修复前生成的，保存了零 message 输出与零 gate，不能证明修复后 local topology 有效，也不能作为后续科学训练 parent。本次没有擅自重跑 tiny 训练；该旧 checkpoint 仅继续作为推理输出和恢复机制检查夹具。

### R3 · DDP 阈值辅助项归一化

- **是否成立：成立。** flow、clean、generated 和 near 各项的 applicable sample count 可以不同，不能共享整批 sample 缩放因子。
- **修改位置：** [molvid/losses/geometry.py](../../molvid/losses/geometry.py) 的 `FutureBondLoss.applicable_count`；[molvid/training/joint.py](../../molvid/training/joint.py) 的 `JointStepLoss.applicable_samples`、`_sample_equal_coordinate_mse`、`frame_joint_loss` 和 `FrameJointTrainer._distributed_total`。每项分别 all-reduce 自己的 count，并按 `world_size * local_count / global_count` 缩放该 rank 的 local mean。
- **验证结果：** [ddp_check.json](../../runs/frame_joint_v1_review_fixes_260920/ddp_check.json) 固定同一 global batch 的 flow time 为 `[0.95, 0.50]`，使 generated/near 资格在两个 rank 间不同。单卡 global-batch 与两卡更新最大差 `2.593151293694973e-08`，低于 `2e-6`；checkpoint model/optimizer 恢复差均为 0，per-rank generator 完全相同。
- **尚未解决：** 该检查覆盖两样本、两卡和现有阈值组合，不代替长训中的损失分布监控。

### R4 · continuation child 普通 resume

- **是否成立：成立。** 原 CLI 在读取 resume checkpoint 的 resolved loss 之前检查 `inherit_parent`，且新 trainer 在契约比较前没有 child 的 parent identity。
- **修改位置：** [molvid/cli/train_frame_joint.py](../../molvid/cli/train_frame_joint.py) 先 preview resume payload，再以 checkpoint loss 覆盖 `inherit_parent` 配置，并把 checkpoint 的 `continuation_parent` 交给 trainer；[molvid/training/joint.py](../../molvid/training/joint.py) 构造器显式接收并保存该身份。
- **验证结果：** [continuation_check.json](../../runs/frame_joint_v1_review_fixes_260920/continuation_check.json) 的 child checkpoint SHA 为 `5979faf368cf159ed96703bbaadd50059c1d83102f04d36fb6329ed8d2c86f77`；trainer 级 ordinary load 的 model/optimizer 最大差均为 0，generator 和 parent identity 完全相同。随后真实 `train_frame_joint` CLI 从 child step 1 普通 resume 到 step 2，输出 `runs/frame_joint_v1_review_child_cli_resume_260920_v2/frame_joint_step_00000002.pt`（SHA-256 `3a40dfdfaf5e214ee9233cb841605eb1b176641a6a08c0afa1d06363df254b53`），保留 parent SHA/step/stage 和 resolved parent loss。
- **尚未解决：** 这是一个更新的恢复验证，不是 keep/release 长程双臂实验。作为机械检查来源的旧 tiny parent 仍受 R2 所述限制。

### R5 · 完整 schedule/H 恢复契约

- **是否成立：成立。** 仅保存 step 而允许 YAML 改变 stage 长度、base LR 或 H 顺序，会改变后续训练语义。
- **修改位置：** [molvid/cli/train_frame_joint.py](../../molvid/cli/train_frame_joint.py) 构造 versioned resolved schedule（各 stage 名称、updates、三组 LR、`history_order`、LR 公式版本）；[molvid/training/joint.py](../../molvid/training/joint.py) 把它加入 checkpoint contracts。
- **验证结果：** [continuation_check.json](../../runs/frame_joint_v1_review_fixes_260920/continuation_check.json) 记录 `schedule_mismatch_rejected=true`；把 child 的 `[4,8]` 改成 `[8,4]` 后 strict load 被拒绝。R4 的真实 CLI resume 同时验证了未改变 schedule 时可以继续到 step 2。
- **尚未解决：** 审阅前 checkpoint 没有 schedule contract，当前代码不会为其提供普通 resume 兼容层；这符合本仓库“不保留旧兼容路径”的约束。旧 tiny 仍可做 inference/evaluation，但不是当前训练 parent。

### R6 · 无定义运动指标

- **是否成立：成立。** 零方差 correlation 返回 1、无足够动态样本返回零误差都会把 unavailable 伪装成好分数。
- **修改位置：** [molvid/evaluation/geometry.py](../../molvid/evaluation/geometry.py) 用 `_defined_correlation` 返回 nullable value/reason；RMSF correlation 与 dynamic/ACF 分别输出 `available/reason` 和有效样本数。[tools/evaluate_frame_joint_tiny.py](../../tools/evaluate_frame_joint_tiny.py) 升级为 `molvid.frame_joint.tiny_eval.v2`，保留 row-level availability，聚合时排除 `None` 并写出 `available_counts`。
- **验证结果：** [tests/test_geometry.py](../../tests/test_geometry.py) 分别覆盖 3 帧零方差和 2 帧过短窗口：前者返回 `zero_variance`，后者返回 `insufficient_frames`，不再产生零误差/相关系数。4 个 retained test 文件合计 `20 passed in 7.74s`。旧 tiny checkpoint 的 3-system/9-trajectory evaluator-only 重跑生成 [v2 JSON](../../runs/frame_joint_v1_review_fixes_260920/tiny_evaluation_schema_v2/tiny_evaluation.json)（SHA-256 `ce594800c036a03966206e5c713c675454efd48a130bb9e0c0a88b933060fa53`），18 个 H4/H8 rows 均含 availability；本批真实窗口的相关项都可用，因此汇总数值未改变。
- **尚未解决：** tiny evaluator 仍只服务原 3/9 协议；48 quick/full evaluator 和 rollout 属于 `TRAIN_PROMPT`，本次未扩写。

## 未作为 bug 修复的审阅意见

R1–R6 没有被判定为不成立。以下是对表格和可读性段落中非编号意见的边界核实；它们没有被偷偷并入本轮 bug patch。

1. **“完整 `ClipBatch template` 等于未来标签泄漏”这一解释不成立。** [molvid/generation.py](../../molvid/generation.py) 的 `_observed_frame_batch` 先切出 observed frames，再用显式 `prefix_coordinates` 覆盖坐标；`QuerySpec` 只从 template 读取 future `time_ps/frame_mask`。CUDA 检查把 template future `x` 加 123、future `bpos` 减 91 后，生成最大差仍为 0。严格的 inference-only 输入类型仍未实现；这会改变公开生成 API，作为单独的类型/API 设计项保留，不能把当前 mutation 证据扩大成“类型隔离已完成”。
2. **“必须现在把 CLI 状态机整体迁入 training 层”不是 R4/R5 的必要修复。** 本轮在既有浅层文件中集中补齐 resume 数据流和 schedule contract，已由 child ordinary resume 与 mismatch rejection 覆盖。进一步移动 stage/LR/cursor orchestration 会是结构调整，未在 bug 修复中擅自实施。
3. **同一末 8 帧的 H4/H8 对照只在声称“更多历史更好”时必需。** 当前 tiny 仅分别报告 H4 的 12 个 future frames 和 H8 的 8 个 future frames，没有作该因果结论，因此不伪造额外比较。
4. **全模型逐参数初始化报告、48 quick/full evaluator、epoch validation** 都是后续证据/产品能力，不是 R1–R6 的最小代码修复。target teacher 的严格 provenance 仍在；其余内容继续明确列为未交付，不改写成已完成。

## 本次验证总览与边界

- 所有 Python/CUDA/pytest 命令均经 `enter-container`、`conda activate torch-ito` 执行。
- targeted py_compile 通过；retained targeted tests 为 `20 passed`；`git diff --check` 通过。
- 新证据集中在 `runs/frame_joint_v1_review_fixes_260920/`，另有一次真实 CLI child resume 输出目录 `runs/frame_joint_v1_review_child_cli_resume_260920_v2/`。
- 没有启动 48/192、frozen-decoder、bond keep/release 或 multi-time 正式训练；没有把审阅前 tiny 结果改称为修复后科学结果。
