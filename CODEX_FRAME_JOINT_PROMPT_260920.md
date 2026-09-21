# Molvid Frame Joint v1：实现执行文本

本任务基于用户已接受的方案：历史压缩、未来逐帧联合 flow、整段时间 decoder、历史编码器—DiT—decoder 联合训练；增加同 checkpoint 的 bond-on / bond-off 续训对照。不要重新发起 geometry encoder 或 R1/R2/R4 选型。

审阅基线：`SHw-1313/molvid-recon@93d191fd8c88c2355944c1cf8bf05e8519ddb598`。开始时记录实际 HEAD 和工作区状态；若已有后续提交，保留它们并核对相关差异，不 reset 到该基线。

## 现在执行什么

完成实现、针对性 CUDA 检查和原 3 体系/9 轨迹的小规模试运行；准备可直接执行的训练配置与命令。先修复/实现，再跑检查，不要收到本文件后先启动全量 pytest。

1. 阅读当前仓库 `AGENTS.md`、`README.md`、`TASKS.md`、`HANDOFF.md`，以及 `docs/migration.md` 中与本任务有关的结论。
2. 按 `agent/frame_joint_v1/PLAN.md` 的顺序阅读任务文件及对应源码。
3. 将 `agent/frame_joint_v1/ROOT_AGENTS_APPEND.md` 中的长期规则合并到根 `AGENTS.md`；保留原规则，不全文替换，不把本轮实验参数塞进去。
4. 建议使用 `feat/frame-joint-v1`。若已经在专用开发分支，沿用即可；不要自动增加第二个 worktree，也不要改用户其他 worktree。
5. 依次完成 `agent/frame_joint_v1/TASKS.md`。阶段提交服务于可读 diff，不要求每一小改动重训。
6. 实现完成后自审 `READABILITY.md`，运行必要的 CUDA 数值检查、真实 checkpoint 特征提取检查和小数据试运行。
7. 更新当前根 `TASKS.md` / `HANDOFF.md` 的摘要及本任务 HANDOFF，写出实际可复制的 train/eval 命令、配置、设备、预算、恢复方法。

本实现指令允许针对性测试和 tiny 试运行，总计最多 2 GPU-hours；不包含 48/192 体系长训练。完成具体实现和 tiny 结果后交付。用户发送本包的 `TRAIN_PROMPT.md` 即构成长训练指令，届时按预算继续，不重复索取同一项授权。不要现在把未执行长训练写成结果。

## 不能遗漏的科学语义

- H4/H8 是总 16 帧中的已观测帧；未来分别 12/8 帧。模型接口不写死 16。
- 历史压缩与未来生成表示独立。未来只有逐帧 h/v，不生成 state/detail，不走未来 inverse Haar。
- 目标编码器从指定旧 codec 提取 TorchMD + coordinate stem；固定目标空间，训练新的历史 temporal、DiT、decoder。
- 物理时间 `time_ps/delta_time_ps` 与 `flow_time` 分开；detail 不除以时间间隔。
- 生成器只接收已观测结构、拓扑、条件与未来查询时间；真实未来仅用于监督/评估。
- bond-off 关闭所有显式 bond 损失，包括 clean/near-endpoint 重建分支。共价键图保留。
- 一条 bond-off 续训曲线不够：必须有相同 parent、数据、噪声和训练长度的 bond-on 续训对照。
- CUDA 是模型数值执行后端。文件 I/O、元数据和必须的 CPU RNG 状态处理允许在 CPU，禁止静默 CPU 模型回退。

## 工作方式

推荐主实现 `gpt-5.6-sol` / `xhigh`。独立审阅推荐 `gpt-6-astra` / `high`；不可用时 Sol / xhigh。训练执行可以用 Luna / high，但不得自行重写架构、loss 或 split。

默认单个写代码 session；本文件不要求启动子 agent。需要独立审阅时由用户另开只读 session。保持一个代码写入者，避免训练期间修改已启动 run 的源码。

模型、损失和数据范围已有决定。实现细节可自行解决并记入 HANDOFF；遇到真实缺失的 checkpoint/data/GPU，完成所有不依赖它的工作，准确报告缺少的具体路径与证据。不要伪造 SHA、已运行指标或实验通过状态。
