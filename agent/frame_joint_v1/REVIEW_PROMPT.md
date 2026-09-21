# 独立审阅 session 执行文本

推荐模型：`gpt-6-astra` / high；不可用时 `gpt-5.6-sol` / xhigh。此 session 只读，不与实现 session 同时改文件，不启动训练或创建新架构。

请审阅当前 Frame Joint v1 的具体实现，而不是重复规划。先读取根 AGENTS、任务 PLAN、ARCHITECTURE、CHECKPOINTS、TRAINING_EVAL、READABILITY 和最新 HANDOFF；记录审阅的具体 commit。工作区有实现者尚未提交的改动时，明确审阅范围，避免把不同时间的 diff 混在一起。

集中回答：

1. teacher 是否从指定 SHA 的旧 codec 提取 Haar 前 TorchMD+stem？新统计是否来自 train，是否错误混入旧四字段 statistics？是否存在未报告的随机初始化？
2. 输入条件、目标、source、flow endpoint、decoder 输出能否沿一条清楚的调用链读懂？是否有真实未来或未来定义的坐标参考进入生成条件？
3. 历史压缩和未来逐帧生成是否真正分离？是否还藏着 inverse Haar、固定未来 ratio、硬编码 16 或未来块约束？未来帧是否联合更新？
4. history temporal 的梯度有没有被旧 batch no_grad 切断？decoder 是否进 optimizer？DDP 是否覆盖所有训练模块？
5. 物理时间和 flow_time 是否混用？重复静态帧是否产生假 detail？vector 的 xyz 约定是否被普通 Linear/归一化破坏？
6. bond_release 是否关闭 generated/clean/near-endpoint 三处显式 bond loss？拓扑保留是否被误报为 bond 未关？两臂 parent、optimizer、scheduler、数据和随机顺序是否一致？
7. 是否仍把“几何更好但运动更差”自动判为成功？RMSF 是否相对真实 MD，quick/full 与 H4/H8 预测长度是否混比？
8. 新架构是否破坏重构后的可读性：CLI 是否塞计算、trainer 是否庞杂、类型/配置是否散落、是否新造 registry/多层兼容 wrapper？

阅读现有 CUDA 检查证据；只在一个具体疑点需要验证且环境/权限允许时跑针对性短检查，不默认重跑完整套件。不要创建仅匹配实现细节的测试。

输出顺序：
- 能否进入下一阶段，以及真正阻断它的代码问题；
- 每个问题给文件/符号、机制、对结果的影响、最小修复方向；
- 可读性问题只指出具体职责混杂/不可追踪数据流，不用任意行数规则；
- 代码事实、实验事实、推测分别标清。

不把一般建议写成阻塞，不自行替用户扩大任务，不修改科学阈值。修复由写代码 session 执行。
