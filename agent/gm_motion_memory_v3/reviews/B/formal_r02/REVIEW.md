# B learned-memory formal pre-launch follow-up r02

**结论：PASS。可以在固定提交 `5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6` 上，使用原冻结配置先启动 M-short、完成后再启动 M-long。**

r01 的三个 P1 均已关闭：严格恢复可加载同一 runner 生成的 checkpoint 并精确延续；32 GPUh 正式预算由双臂共享、跨 invocation 持久累计并在 update 前后执行 checkpoint-safe guard；raw XTC sidecar 在 dataset 初始化时逐轨迹绑定冻结 manifest 的路径、SHA、帧数和 offset invariants。未发现新的 P0/P1 发射阻断。

## 固定身份与审阅边界

- reviewed commit：`5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6`
- base review/failing source：`e10148d49068da91bde37aca808b10f56c200cc4`
- safety evidence：`runs/learned_memory_v1/formal_r02/safety_evidence_5ec8726.json`
- safety evidence SHA-256：`1487f838f7264ef64287951d0f4af0c40f5754581df70066b0c9c09af5b69d90`
- resume equivalence SHA-256：`bba7910bc3e7e493738c5acb5ace2dc1672c997ba803ba043968f6ca989908ce`
- 原 r01 review SHA-256：`1f109fb461c40da10a45227989c26fe3b95167b016b1630c72d453507aad0e8c`
- XTC offset manifest SHA-256：`21c3dc05271b7c018278b3e4e6972b29987a4b513bc21ac122adfaf9ceda0e3f`
- formal M-short config：`7d103a90dbd80dfd013c5953c9aaae8982eb5a2c32b778c2057850ea939f608d`
- formal M-long config：`5a18ff4210e600c40d1f69121785bd37ae5407aaca50f2f03ca440d895282321`
- `frozen_configs.json`：`7917b420ddb3d1ffd83a2a1dd9b6e551d4e206dd81757620b4d519a2f7d5f875`
- `budget_decision.json`：`9db1e69ec2ee53cc747bdc22dec7c4cef11efa59b05432b4cc09ac94c8f97136`
- A1 parent checkpoint：`52d922f88ec50f08ebe889837ec7ee2cc2ae2a5c7ced1dd46b2909a917aa14a9`

HEAD 与 reviewed commit 一致。`e10148d..5ec8726` 的变更仅包含 r01 审阅文档、raw sidecar 安全校验、formal budget/runner 修复和针对性测试；未修改 learned-memory 架构、loss、采样目标或冻结科学配置。提交范围 `git diff --check` 通过。当前未提交的 `runtime_paths.json` 和工作区协调文档不作为模型源。正式输出父目录 `runs/learned_memory_v1/formal_r02/configs/formal` 在本次审阅时不存在，因此隔离的 resume/budget 验证未污染正式账本或输出。

本 reviewer 未使用 GPU3，也未运行新的 CUDA 模型任务。提供的固定提交测试结果为 **13 passed**。我在 `enter-container` / `torch-ito`、`CUDA_VISIBLE_DEVICES=` 下独立复跑 `tests/test_learned_memory_formal_safety.py`，结果 **3 passed in 3.63s**；并独立实例化 train/valid dataset，重新完成全部 600 条 sidecar 校验。

## r01 P1 关闭情况

### 1. 严格 resume：关闭

`molvid/training/learned_memory_formal.py:58-90` 现在从 immutable A1 parent 重建唯一的 model-only warm-start contract，并在构造 `FrameJointTrainer` 时传入该 contract（`303-316`）。fresh 分支执行实际 warm start 后再次要求 trainer 中的 contract 与预重建值相等；resume 分支因此在严格 checkpoint contract 比较前已经具有同一个非空 `continuation_parent`。原来 fresh 保存非空、resume 期望 `None` 的冲突已消失。

真实 runner 的两步对照证据通过：从 step1 checkpoint 恢复执行 step2 后，以下 checkpoint section 全部 exact equal：

- model state、optimizer state、scheduler state、scaler state；
- contracts、parameter names、step、cursor、sampler state、extra state；
- global RNG、training generator、sampled generator；
- 下一更新的 batch、flow time、LR、loss、梯度范数、raw/weighted 科学指标；
- teacher calls `[2, 12]`，prepared memory 次数 `1`。

reference/resumed 整体 checkpoint 文件 SHA 不同是序列化文件身份差异；上列训练/科学状态逐 section 精确相等。source step1 checkpoint SHA 为 `337ec272d83d57957377c076b745eeefc467236101fc2717ca38770a4ce14bd3`，reference/resumed final SHA 分别为 `d6e2378223b78039ff78c67d1a76dde990e26b6ab6ee9e7887e3b39a325a5222` 和 `e04ac1092e376c5b6b1360eb9dcf7955fe7c07da3c819874724a78868bbd7937`。

### 2. 双臂累计预算与越界保护：关闭

`molvid/training/formal_budget.py` 新增一个由文件锁保护的 paired ledger：

- pair contract 对两份正式配置去除 `arm` 和 `output_root` 后绑定其余完整配置；两臂原配置除这两个字段外完全一致。
- ledger 同时记录 arm、config SHA、code commit、resume checkpoint SHA、计划/成功更新数、PID/host、开始/结束时间和失败状态。
- 同一时刻只允许一个 active invocation；同主机死亡进程会转成 `abandoned` 并计费，异主机未确认的 active invocation fail closed。
- M-long 只有在 M-short 以 `successful_updates == planned_updates` 完整结束后才能启动；任一 arm 已完整完成后不能重复启动。
- resume checkpoint 的 budget state 必须匹配 pair contract、arm、config SHA、32 GPUh cap 和旧 invocation，并且 durable ledger 已计费用时必须覆盖 checkpoint 中记录的累计用时。checkpoint 本身仍由 CLI SHA 和严格训练合同校验。
- runner 在每次 update 前和后检查累计用时。达到 cap 时先保存包含 cursor、sampler、exposure、三组 RNG 和 budget state 的 checkpoint，再抛出 `FormalBudgetExceeded`；pair status 变为 `budget_exhausted`，以后两个 arm 都不能继续启动。

端到端隔离验证将旧累计时间预置到距离 32 GPUh 仅 5 秒。固定 runner 完成初始化后在 **0 updates** 处保存 step0 checkpoint并退出，证明没有跨 cap 再执行一个训练更新：

- checkpoint SHA：`4e9019e9cbb5c5fc8f6fbdf2a4f0adee0e3f61aba0596ddf7c45ca4495a45602`
- local ledger SHA：`ad5995b1a52aec88f5bdbb49d2d7b4b82eb00c9894851152dd523698815c5f33`
- paired ledger SHA：`333e942260cf448e85b7a1fe9fbf81593663af78d797d3f09f648af73319420b`
- final status：`budget_exhausted`
- 累计：`32.0053243434` GPUh；其中本 invocation `0.0067130933` GPUh

超过阈值的约 19 秒来自从 invocation 起点计费的初始化和 checkpoint-safe 检查间隔，已完整入账；状态封死整个 pair。正常运行的正式成本预测仍为两臂合计 `23.4933260892` GPUh，距 32 GPUh cap 有约 8.51 GPUh 余量。评估稳态外推约 `6.8753` GPUh，低于独立的 8 GPUh evaluation cap。

我另行读取了 step0 checkpoint，确认 `step=0`、`successful_optimizer_updates=0`、cursor 为 epoch0/batch0，continuation parent 为冻结 A1，且 checkpoint 内含同一个 pair contract/config/invocation/cap 的 `formal_budget_state`。

### 3. XTC sidecar 字节与路径绑定：关闭

`RawMemoryDataset` 现在要求 offset manifest schema、raw root、source manifest root、split 和 record 数量正确；按 resolved XTC path 建唯一 record map，并为当前 split 的每条物理轨迹校验：

- manifest record 只含冻结字段且 XTC/path/split 一致；
- sidecar 路径与 offset root 下的预期 system/XTC 名称一致；
- XTC size/mtime 与 manifest record 一致；
- 实际 sidecar SHA-256 与 record 的 `sidecar_sha256` 一致；
- sidecar 字段、int64 一维 offsets、manifest frame count、最低所需帧数、first offset=0、严格单调、4-byte alignment、末 offset 小于 XTC size；
- sidecar 内嵌 XTC size/mtime 与 record 一致。

经验证的具体 sidecar path 被保存并传给实际 raw reader，不再由每次读取时脱离 manifest 重新猜测。uniform-shift 测试首先在未更新 record SHA 时被 SHA 校验拒绝；即使测试同步更新 record SHA，仍因 first-offset invariant 被拒绝。

独立 CPU 复核结果：train `576` trajectories / `29376` anchors / data hash `a62c60ba76a86093fa563bec8549d831e66db17d8be86ea574c12814675ce3ae`；valid `24` trajectories / `384` anchors / data hash `7704a2634ece890f80f9ffb83927d2df261e84a66c02de39392f66a7b01ae843`。共 600 条 sidecar 均通过固定 manifest 下的逐文件检查。

## Severity

### P0

无。

### P1

无。r01 的三个 P1 均关闭。

### P2

保留 r01 已记录的非训练阻断项：当前 evaluator 能完成固定 raw-valid 网格、逐体系聚合、future-leak 检查和可回放产物，但 B-specific paired bootstrap CI、short/long selection、figures 与最终 `report.md`/`selection.json` 仍需在正式训练后、发布科学结论前完成。这不阻断本次 formal training 发射。

### P3

paired ledger 在 dataset/codec/model 初始化完成后才创建，但用 invocation 起始 wall time 回填计费；正常启动和进入 ledger 后的异常均累计完整时间。若进程在 ledger 创建前被硬杀，该极早期初始化时间不会进入此 ledger，需要由外部作业计费记录补齐。以现有约 8.51 GPUh 正式预算余量看，这不是发射阻断；后续可把 durable invocation reservation 提前到昂贵初始化之前。

## 发射条件与判定

**PASS / APPROVED FOR FORMAL LAUNCH**，仅适用于以下固定组合：

1. source 必须保持 `5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6`；任何代码提交变化都需要重新审阅。
2. 使用上述原 SHA 的 formal M-short/M-long 配置，不改变一 epoch、9741 updates/arm、paired schedule、数据、teacher、A1 parent 或 loss。
3. 从干净的正式 output parent 启动 M-short；保留其 `paired_formal_cost_ledger.json`，完整完成后才启动 M-long。
4. 中断后必须使用同一 arm checkpoint 的精确 `--resume` 与 `--resume-sha256`，不得删除、重写或绕过 paired ledger。
5. 若 pair status 变为 `budget_exhausted`，停止整个 pair，不把单臂或不等曝光结果作为正式 short/long 比较。

在这些条件下，固定 formal M-short/M-long 可以启动。
