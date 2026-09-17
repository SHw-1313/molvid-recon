# 执行交接

Status: NOT_STARTED

此文件是计划模板，没有模型测试/训练结果。交付包自检不等于CUDA验收。

## 工作区与环境

- 实际worktree/container路径：
- branch/start HEAD/final HEAD：
- 保留的用户修改：
- A来源与导入文件hash：
- enter-container/torch-ito命令、Python/PyTorch/CUDA：
- GPU UUID、精度、TF32、任务PID：

## 输入与修复

- manifest/codec/旧DiT/statistics文件和semantic hash：
- F1–F7实际修复：
- 实际CUDA检查、真实smoke、相关回归命令/退出码/证据：
- 当前backend在train/eval/resume中的确认：
- 未完成项和影响：

## Source与预算

- repeat/block-state两种中心的模板、静止性、边界结果：
- source_center选择与理由：
- cache模式/大小/prep时间、直接与缓存batch等价结果：
- 实际runner wall p50/p90、编码/中心/模型/评估成本：
- budget.json、共同终点、预留评估成本：
- init_hash、schedule_hash、RNG合同：

## 训练

| arm | actual optimizer updates | common comparison step | selected RF step | train/valid趋势 | GPU-hours |
|---|---:|---:|---:|---|---:|
| gaussian | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |
| conditional | 待执行 | 待执行 | 待执行 | 待执行 | 待执行 |

- 中断/恢复、代码语义变化：
- 两臂初始权重、tau/noise/批次顺序一致性：
- 是否仍有改善、是否有平台证据：

## 生成评估与判断

- 同步固定checkpoint、H/L、sample/draw/steps、聚合协议：
- bond/contact/occupancy、边界、路径RMSD、RMSF/位移/ACF/diversity：
- 每体系paired差异：
- 几何改善是否伴随运动塌缩：
- 是否支持conditional source、仍未知：

## 最终交付

- 完整本地输出根：
- 小型evidence路径：
- 精确复现命令：
- 实际local commits：
- 最终状态：
- 下一阶段建议（仅建议，不自动实现）：

不推送远端、不启动R2训练或后续架构实验。
