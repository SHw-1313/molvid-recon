# B — 验收

## 必须成立的功能

- 独立B worktree和branch，源快照明确；A分支/文件/进程未修改。
- 模型width/depth/head/参数量、标量qk、同权重scalar/vector mixing、mean pooling、per-atom temporal/FFN保持。
- 旧checkpoint可严格核验加载；需要backend contract迁移时仅允许explicit等价迁移，其他hash/key/shape错误仍拒绝。
- 无逐time×逐block pooling/broadcast循环或CUDA小索引tensor工厂；不可避免metadata工作在batch准备阶段，forward/ODE复用。
- mask、全mask、ragged、sample/block编号冲突、缓存失效均有CUDA测试。
- FP32前向/梯度/单步更新与BF16检查有实际误差证据；非零attention gate参与测试。
- SO(3)、sample isolation和clamp检查通过；不得用新架构改语义换速度。
- 训练每batch恰好一次codec encode；原codec/statistics不变。
- 主路径无CPU模型回退；无CUDA时结果BLOCKED，不以skipped test声称完成。

## 实测证据

- 同输入/同权重/同设备的reference与optimized；warmup不入steady计时；trace不污染吞吐。
- R4 small/median/large、R2 median；不同case使用train-only元数据确定。
- forward/backward/optimizer/encoder/host wait分开；CUDA peak在warmup后reset。
- model-only/end-to-end分开；tokens按原子帧与latent时间帧区分。
- 报告实际SDPA kernel；使用math backend时明确，不声称Flash生效。
- 一个真实R4 end-to-end smoke与R2加载/forward证据；checkpoint缺失时准确列出。
- R2:R4倍率实测，不要求1:1。无加速也交付负结果，不凭向量化代码外观宣称scalable。

## 验证顺序

实现 → CUDA单元等价 → 实clip smoke → 受影响回归 → 短profile。现有完整pytest不是前置。元数据CPUtests单独列出；CUDA test skips不计数值通过。

Profile只用可丢弃模型/optimizer，默认合计200短步。不得修改生产checkpoint或训练预算，不启动完整4500-step run。

完成交付小型evidence、复现命令、commit与已知性能局限；状态 `B_BACKEND_READY_FOR_REVIEW` 后停止。若等价或实测没完成，状态 `B_INCOMPLETE_FOR_REVIEW` 并逐项列明。
