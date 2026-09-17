# R4 DiT：四项改进逐项实现、逐项验证

这是一个新任务。按 **1 幅度信息 → 2 几何监督 → 3 跨块 temporal refiner → 4 有误差 history** 的顺序工作。每轮实现、检查、训练、生成评估、做出保留/回退决定后，才实现下一轮。不要一次接上四项再训练，也不要修复前先跑全仓 pytest。

## 先读什么

在目标 worktree 中依次读取根 `AGENTS.md`、本文件，以及：

1. `agents/dit_architecture_sequential_v1/AGENTS.md`
2. `agents/dit_architecture_sequential_v1/PLAN.md`
3. `agents/dit_architecture_sequential_v1/EVALUATION.md`

旧 phase 文件只提供相关历史合同；不要重新启动旧任务。先输出一段简短复述，明确四轮各自唯一的比较变量、当前从哪个干净模型开始，随后继续工作，不等待口头确认。

## 工作区：独立于正在运行的数据/容量实验

- 原目录 `/data4/users/sihao/workspace/molvid-dit-factorized-backend-v2`、分支 `perf/dit-factorized-backend-v2` 只读。
- 数据/容量实验 `/data4/users/sihao/workspace/molvid-dit-capacity-data-v1`、分支 `exp/dit-capacity-data-v1` 只读。不要改它的训练代码、配置、进程或 checkpoint。
- 本轮目标目录 `/data4/users/sihao/workspace/molvid-dit-architecture-sequential-v1`，分支 `exp/dit-architecture-sequential-v1`。
- 已核对的 A 结果提交为 `027659fe872a6c869b14ecce5b799356bd695694`。先检查提交、worktree 和分支是否存在。缺少对象时只读 fetch 指定 molvid 远端并核对 SHA；不自动采用远端新 HEAD。
- 仅在目标均未被占用时执行：

```bash
git -C /data4/users/sihao/workspace/molvid-dit-factorized-backend-v2 worktree add -b exp/dit-architecture-sequential-v1 /data4/users/sihao/workspace/molvid-dit-architecture-sequential-v1 027659fe872a6c869b14ecce5b799356bd695694
```

目标已是本任务 worktree 时核实后续做；属于别的工作则加独立后缀并记录，不能覆盖或强制清理。将本文件和 `agents/dit_architecture_sequential_v1/` 放入目标目录。数据、冻结权重通过绝对路径只读引用；实际 Python 源码从本目标目录加载。

只在新 worktree 的根 `AGENTS.md` 追加本次阶段说明：本任务允许顺序实现、必要修复、CUDA 检查、预算内的 train/valid 实验及本地提交；允许新增独立 decoder 后处理 refiner，但不改冻结 codec 权重、编码语义、latent statistics。该说明取代旧阶段的任务目录、任务顺序和停止状态，不删除历史记录。不要更新其他 worktree 的 AGENTS。

## 第零步：建立可信起点

1. 检查参数隔离修复。每个模型独立 adapter、optimizer、scaler、RNG；禁止旧 `context.adapter` 在多个训练模型之间共享。优先只移入 B 已提交、已检查的必要隔离修复，记录确切提交和 diff；不合并整个 B 分支。B 尚未提供时，在本目录完成同等最小修复，不改 B。
2. **评估加载也要独立。** A 的历史 loader 可能沿用共享 adapter；评估新 checkpoint 时不能再用同一个可训练 adapter 轮流装载多模型。使用独立实例或逐模型完整构造/评估/释放，确保采样使用当前模型的 adapter。
3. 优先只读复用 B 已完成、证明参数隔离的 `C48`：conditional source、48 train systems、4-layer DiT。核对 checkpoint 的模型/adapter/optimizer、source、数据、统计量、代码 provenance 和冻结权重。不要读取正在写入的 checkpoint。不能根据文件名猜测其训练已结束。
4. 若没有可复用 C48，按 PLAN 中的同一配方训练一个干净基线；这不阻止先完成公共入口和隔离检查。**旧 shared-adapter 污染模型仅供解释旧现象，不可作为本轮训练起点。**
5. 新基线重新做紧凑生成评估。A 报告中的幅度趋同、块边界跳变、rollout bond 累积都是待复核假设，不是新模型必然有的缺陷。

## 四轮执行摘要

| 轮次 | 改什么 | 对照如何成立 |
|---|---|---|
| 1 | FFN 的 scalar 分支读取归一化前 vector norm | 同一父 checkpoint；只切换 norm 来源；同量续训 |
| 2 | RF 加可微 decode 的 future bond loss | 同一父 checkpoint；只切换几何 loss；同量续训 |
| 3 | decoder 后的小型跨块 temporal refiner | 冻结 DiT/codec；同参数、同数据、同 loss 的块内/跨块 refiner；另保留无 refiner 基线 |
| 4 | 训练时部分已观测 history 加小误差 | 同一父 checkpoint；只切换 history corruption；future target 保持干净；同量续训 |

第 2 轮先只做几何监督，不同时加 topology message passing。第 3 轮必须有自己的坐标监督：latent RF loss 无法训练位于其后的 refiner。第 4 轮先做可控 history corruption，不同时引入在线自生成前缀、长 rollout BPTT 或新的采样器。

## 运行与预算

- 全部 Python、测试、模型、训练、评估经项目实际支持的 `enter-container` / `conda activate torch-ito`。先检查本机入口，不编造容器命令。科学数值路径 CUDA；CPU 只用于 metadata、I/O、最终结果汇总。不得把 CUDA unavailable/skip 记作通过。
- 默认只使用一张已确认空闲卡；有第二张明确空闲卡时可并行同轮两个训练臂。总占卡不能与 B 或其他任务冲突；不启用 DDP，不抢卡，不终止进程。
- 默认本任务总额 **48 GPU-hours、72 小时 wall time**，含基线（若必要）、四轮训练、评估和恢复。执行前可按用户另行指定预算替换；看结果后不得只给领先臂追加预算。先 profile 后登记各轮共同 token 预算，并为后续轮次和评估留额度。
- 顺序始终是：实现本轮 → 针对性 CUDA 正确性 → 真实 clip smoke → 直接受影响回归 → 同轮训练 → 固定生成评估 → 决策。首次真实 smoke 使用先前的 3 体系/9 轨迹；科学比较使用固定 48 train / 8 valid。
- 达到本轮有效比较后停止可选测试，不跑全仓回归、Euler 步数网格、R1/R2/R4 重选、完整 test 或长 rollout。
- 本轮通过的改动保留；明确失败的候选隔离保留代码和证据，下一轮从共同预算下的对照 checkpoint 继续。不能从训练更少的旧父 checkpoint 回退后假装训练量相同。
- `INCONCLUSIVE` 只在预留预算内给双方共同延长一次；仍不清楚则保留简单对照并继续下一轮，不能无限调参。环境/数据/正确性真阻塞时记录原因；不绕过权限或造结果。

## 文件与完成状态

新输出：`outputs/dit_architecture_sequential_v1/<run_id>/`，按 `baseline/round1/round2/round3/round4/` 分目录。新 runner/config/测试按 PLAN 实现；历史输出和训练入口保持可复现。

每轮更新 `HANDOFF.md` 与 `decision.json`：父模型 hash、唯一变量、两臂实际 token/更新数/训练成本、指标、保留者和理由。检查点包含模型、adapter、refiner（如有）、配置、优化器、RNG、sampler 进度及冻结资产 hash。

完成后给一份逐轮报告，说明四个假设的支持程度、最终保留组合、相对原始干净基线的总变化和尚未解决的问题。小代码和文本证据可以本地提交；不自动 push、merge、开 PR，也不提交大权重/数据/预测二进制。

最终状态：`SEQUENTIAL_V1_COMPLETE`、`SEQUENTIAL_V1_PARTIAL_BUDGET` 或 `SEQUENTIAL_V1_BLOCKED`。不要在旧阶段的 review stop 停下，也不要宣称小规模顺序筛选已经证明 scaling law。
