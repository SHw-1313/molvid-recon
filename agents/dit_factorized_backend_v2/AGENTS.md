# Session B instructions

本目录约束当前backend重构，覆盖父级旧codec任务/CPU-first/结束状态；不解除环境、权限、frozen artifacts保护。

完整读取根Session B prompt、共享CONTRACT/NEXT_STAGE和本目录PLAN/ACCEPTANCE/DECISIONS/TASKS/HANDOFF。先做隔离worktree与reference，再实现、CUDA验证、短profile。

只在 `perf/dit-factorized-backend-v2` 独立工作树写入；A的pilot分支、evaluation源码、旧codec、RF objective与生产数据只读。只允许数值等价加速，不改变attention的空间粒度、vector语义、模型宽深、noise source、loss或history。

Python必须在enter-container+torch-ito。新数值tests在CUDA执行，不以skip/CPU fallback代替。输出到自己的新目录。短optimizer profile使用模型/optimizer一次性副本，不覆盖原checkpoint。

向量化优先于强求FlashAttention；正确性和真实分段计时必须有。完成后本地提交并停止，不push/merge，不训练conditional prior，不升级到4500-step实验。
