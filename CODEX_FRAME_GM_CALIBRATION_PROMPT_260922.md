# 主实现 session：Frame GM Calibration v2

你负责把已确认的前三阶段完成到训练和评估。不要只返回计划，也不要在 pilot 后要求用户再次授权预算内训练。

## 授权与工作边界

用户确认的流程是：实现功能和 pilot → 新建独立无聊天上下文的审阅 session → 产生审阅文件 → 主实现 session 按文件修复 → 训练 → 各项指标评估。P1 可以不另开 reviewer。用户已授权这一完整流程及预算内的本地作业。

旧 `CODEX_FRAME_JOINT_PROMPT_260920.md`、旧 HANDOFF 中“tiny 后暂停/等 TRAIN_PROMPT”是此前任务的阶段授权，不是本任务的停止点。根 AGENTS 中“长训练需明确授权”由本指令满足。保留容器、数据保密、外部 worktree、依赖和推送规则。

只允许一个模型代码写入者。reviewer 可由你启动，但必须是新的、无本会话继承的 session；不能把本会话 resume/fork 给它。它只审阅，不改代码、不训练。文件交接由你完成。

## 首次执行顺序

1. 读当前 repo 的 `AGENTS.md`、`README.md`、根 `TASKS.md`/`HANDOFF.md` 和 `docs/frame_joint_v1.md`；对旧状态按时间区分，不把旧“未训练”覆盖新实验记录。
2. 读本包 `agent/frame_gm_calibration_v2/PLAN.md`，随后按其顺序读设计、实验和审阅协议。所有必要上下文已写在文件中，无需旧聊天。
3. 记录 HEAD、分支、worktree 状态、原有运行和资源归属。参考基线为 `f49cf27efa60a0cca172248c49913d70ff1516a5`；若本机有新提交，核对相关差异并保留，不 reset 回基线。
4. 建立/复用专用实现 worktree；把任务包文件加入工作树。只合并 `ROOT_AGENTS_APPEND.md` 中长期适用的短规则，不复制整个计划进根 AGENTS。
5. 核验 parent checkpoint、codec、统计文件和 raw ATLAS 路径。宿主机 repo 旧记录是 `/data4/users/sihao/workspace/molvid_recon`，容器旧记录是 `/workspace/molvid_recon`；必须核实真实映射。
6. 所有 Python、模型、CUDA 测试、绘图在 `enter-container` 后的 `torch-ito` 中执行。不得假设 `enter-container bash -lc ...` 一定受支持；查本机现有入口。shell/git/Codex launcher 可在它们所在的宿主环境执行。
7. 先实现 P1 的必要修复，再定向检查和重算；之后实现 P2。不要先启动全套 pytest，也不要把旧审阅问题搁置去跑回归。

## 必须完成的阶段

- P1：完成指标修复、固定历史/跨度视图、latent 诊断。新旧 metrics 分版本保存，原始 archive 不改写。
- P2：实现 G 与 M，完成 B0/G/M/GM CUDA smoke 和小 pilot，提交候选；启动独立 reviewer；修复与复核之后自动进行 192 体系的四臂训练和评估。
- P3：按预先冻结的规则选择一个配置（允许 B0），实现真正从源分布出发的可微采样训练及物理特征 energy score；两臂 pilot、独立审阅与修复后自动正式训练和评估。满足条件时执行短 bond keep/release。
- 完成每阶段后更新 TASKS/HANDOFF、实际命令、配置哈希、checkpoint SHA、资源使用和结构化结果。

P1 发现的源噪声尺度问题本轮只诊断；P2/P3 不擅自改源噪声、teacher、latent 统计或主 flow 定义。若这造成结果不理想，报告证据，不能默默进入未批准的第四阶段。

## 连续推进规则

- 顺序：实现/修复 → CUDA 定向验证 → pilot → 独立审阅 → 修复/复核 → 正式训练 → 评估。
- reviewer 的普通风格建议不阻挡正式训练。只把影响泄漏、梯度、指标、恢复、运行可靠性或实验有效性的可复现问题列为阻塞。
- 收到 `FIX_REQUIRED`：先按 issue ID 修复，再跑相关验证，写 `FIX_RESPONSE.md`；影响科学语义/梯度/数据的修复发新 reviewer 复核，不能自行宣告独立审阅通过。
- reviewer 不可用时，做不依赖审阅的准备并保留任务状态；明确说明环境障碍。不能把“用户尽量不停”解释成跳过独立审阅。
- 所有训练进入可恢复作业，保存 PID/job ID、日志、checkpoint 节奏、退出码和预算账本。不要只启动后台进程就宣布完成；继续监测并执行后续评估。会话必须中断时写出精确恢复状态。
- 检查点恢复只续未完成工作，不重复训练已完成的臂。相同目标的严格 resume 与改变架构/数据/loss 的 warm start 分开。
- 不自动安装/升级依赖、不抢其他人的 GPU、不 kill 既有作业、不推送、不改外部参考仓库。缺少权限不得修改策略绕过。

## 资源与实验范围

默认上限同时 4 GPU；P2 96 GPU-hours，P3 96 GPU-hours，P1/pilot/eval 32 GPU-hours。profile 后，在开始正式比较前把实际 exposure 预算固定到 `resolved_experiment.json`。不得跑完一臂看到效果后才给它额外训练。

实测显存允许时四臂单卡并行；否则按两卡/四卡等量批处理。模型/批次不为某一臂单独缩水。将静态与动态 GPU 编号映射记录到 runtime 文件，禁止把旧记录的 1/3/4/7 当作当前授权空卡。

实现细节可自主解决，记录理由与影响；只有越过三阶段范围、关键文件不可获得或资源/权限真正不足时才需要用户介入。最终报告真实结果、负结果和剩余不确定性。
