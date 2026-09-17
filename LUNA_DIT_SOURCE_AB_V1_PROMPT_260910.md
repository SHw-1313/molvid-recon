# Luna 执行提示：在 B 整合诊断、修复验收并完成 R4 source 两臂实验

## 任务与授权

用户已选择：直接沿用 B 分支，选择性引入 A 的诊断能力，不完整合并两个分支。本提示作为下一阶段执行授权：实现、必要检查、source 中心验证、限定规模的 R4 两臂训练、评估、报告与本地提交均在范围内。通过前置检查后继续运行，不再引用旧 A/B 文档的“仅诊断/仅性能、不训练、等待审阅”来中止本阶段。

本次仅在执行本提示的 B session 中工作，不启动额外编码 agent。若另一个 worker 正在写同一 B 工作树，先协调所有权；不要并发写同一组文件。

## 首先读取，再实现；不要先 pytest

按顺序读取：

1. 当前根 `AGENTS.md`，了解环境与历史约束。
2. 本文件。
3. `agents/dit_source_ab_v1/AGENTS.md`。
4. `PLAN.md`、`FIXES.md`、`EXPERIMENT.md`。
5. `ACCEPTANCE.md`、`RUNBOOK.md`、`DECISIONS.md`。
6. `TASKS.md`、`HANDOFF.md`、`ROOT_AGENTS_APPEND.md`。

记录 git 状态、HEAD、分支和已有用户修改。确认后，将 `ROOT_AGENTS_APPEND.md` 的内容只追加一次到 B 的根 `AGENTS.md`。不要替换根文件，不改 A 工作树或历史阶段文件。

执行目录预期为 `/data4/users/sihao/workspace/molvid-dit-factorized-backend-v2`，分支为 `perf/dit-factorized-backend-v2`。以实际容器挂载为准，记录映射。

## 固定来源

- B 审阅基线：`557a9042cb545750706e37040b26fc000e39b9dc`。
- A 诊断来源：`da5bdbb8e18b5b54884bd9b5f1be13c13e70fb51`，源代码实现提交为 `a25e6bf66214a82adf9a56d492a85a28207ef248`。
- 只引入 A 的 `evaluation/dit_diagnostics.py`、`scripts/evaluate_dit_pilot_diagnostics.py`、`config/dit_pilot_diagnostics_v1.yaml`、`tests/test_dit_diagnostics.py` 及核实确需的依赖。
- 不整体 merge/cherry-pick A，不导入其根 AGENTS，不导入整个 outputs。A 结果和旧 checkpoint 只读引用。
- 本地已有 Git 对象时直接从固定提交导出指定文件；缺对象才允许对已指定 molvid 远端做必要只读 fetch。不要为本任务安装包或下载其他仓库/数据。
- 如果 B 已在基线上追加了提交，先检查差异并保留用户工作，不 reset 回基线。

## 执行顺序

1. 整合 A 诊断到 B，显式接通 `factorized_v2`；修复 `FIXES.md` 中的判断与报告问题。
2. 必要 CUDA 数值检查、真实采样/解码 smoke，再运行直接受影响的回归子集。只修失败关联项，不全仓重跑。
3. 用固定8个 validation clips 检查 source 中心；同时实测训练循环和缓存代价。
4. 固定一个共同 source 中心和两臂预算，写入运行合同；两边从同一随机初始化的独立副本开始训练。
5. 到共同终点后完成同预算生成评测，更新 HANDOFF 与小型证据，做本地提交后停止。

不要在训练中同时添加几何 loss、改 field 权重、缩小 conditional 噪声、增加网络层、重训 decoder 或改时间间隔。

## 范围与停止

- 当前两臂训练仅 R4；R2 用于后端检查，不启动 R2 科学训练。R1/pooling/no-temporal 不进入新实验。
- 原48 train/8 valid 体系，16帧，100 ps，H4/H8。test payload 不打开。
- 使用现有 `enter-container` 与 `torch-ito`。模型、source 计算、loss、采样、解码及科学指标都在 CUDA；元数据、文件读写和汇总可用 CPU。
- 使用1–2张确认空闲的 GPU；两张可每臂一卡，一张可按阶段顺序轮流运行两臂。不增加 DDP，不抢占/终止他人作业。
- 两臂科学训练加评估的共同资源上限、预算选择和中断处理见 EXPERIMENT。不能因为某臂好看就只延长该臂。
- 实际正确性失败、缺冻结输入、两个中心都无法通过实现检查时，不启动正式训练；继续完成可做的修复/文档，准确报告阻塞。不以结果“不够好”伪装成实现失败。
- 允许本地逐文件提交；不自动 push、PR、merge/rebase 或清理他人文件。

最终状态必须是 `SOURCE_AB_V1_COMPLETE_FOR_REVIEW`、`SOURCE_AB_V1_PARTIAL_BUDGET` 或 `SOURCE_AB_V1_BLOCKED`，附真实证据。不得把计划中的检查填成 PASS。
