# 本阶段工作规则

当前任务是 `dit_source_ab_v1`，唯一写入工作树为现有 B：`perf/dit-factorized-backend-v2`。

本目录文件与根提示定义本阶段范围。旧 A/B 阶段的禁止科学训练、文件所有权分割、最终等待状态和旧步骤上限属于历史范围，不限制本阶段明确授权的修改。保留旧文件及旧输出，不把历史记录改写成新实验。

## 不变量

- TorchMD 几何 encoder、R4 deterministic state/detail codec、坐标 decoder 和生产 statistics 保持冻结。
- state/detail 是现有 Haar-like 特征分解，不称为慢/快模态或匀速坐标残差。
- 固定16帧 tokenizer，H4/H8 observation mask，不改变分块、不新增18帧。
- 两臂都有同样 history；未来都可更新，只有 observed token 被 clamp。
- 两臂只改变 future source 的均值及相应 RF path/target，噪声每系数标准差均为1。
- 保持 scalar/vector 四 fields、模型容量、SO(3) 通道规则、原始归一化与四 field 等权 MSE。
- 不加入 per-atom 坐标旁路、几何 loss、VAE/KL/VQ、AF3/MSA、静态混训或长 rollout。

## 环境与验证

所有 Python 命令在真实 enter-container/torch-ito 内。数值工作使用 CUDA，无 CUDA 时不改成 CPU 通过。CPU 读取拓扑/元数据、缓存序列化、最后日志标量转换正常；不得让坐标/latent 在每个模型层或 flow step 中往返 CPU。

实现优先；按 ACCEPTANCE 做针对性检查、真实 smoke、相关回归。不能因未运行全仓 pytest 宣称任务未完成，也不能将 CUDA skip 算作通过。

本阶段不额外启动 agent。运行多个 GPU 进程不等于多个编码 session；两臂的输出、RNG 和 optimizer 分开管理。

## 记录

PLAN/FIXES/EXPERIMENT/ACCEPTANCE 是准备好的执行合同，常规实现细节可在 DECISIONS 追加说明。没有证据不能修改实验定义来迁就结果。

TASKS 逐项记录真实状态；HANDOFF 写实际路径、commit、输入 hash、命令、设备、耗时、结果与阻塞。大缓存/权重/轨迹不提交；小型报告可放本目录 `evidence/`。
