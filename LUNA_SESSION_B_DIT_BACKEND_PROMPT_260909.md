# Session B — 独立 worktree 的 factorized DiT backend v2

用户已经授权实施 backend 加速和限定 profile。先读协议、实现，再进行等价性与性能验证。不要重新跑完整 pytest 后才开始修改。

## 建立隔离工作树

起点固定为已审阅 commit `a22f60c4ffd1f502ab300352a00eafe841b1f8d2`；不是 A 此刻随提交移动的 HEAD。

只读检查 A 的仓库：

```bash
git -C /data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1 rev-parse --show-toplevel
git -C /data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1 worktree list
git -C /data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1 cat-file -t a22f60c4ffd1f502ab300352a00eafe841b1f8d2
git -C /data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1 branch --list perf/dit-factorized-backend-v2
```

仅当分支和目标目录都不存在、commit 存在时执行：

```bash
git -C /data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1 worktree add -b perf/dit-factorized-backend-v2 /data4/users/sihao/workspace/molvid-dit-factorized-backend-v2 a22f60c4ffd1f502ab300352a00eafe841b1f8d2
```

若分支/目录已有内容，核对是否是本任务的可恢复工作树；不删除、不强制 checkout、不新建替代名字掩盖冲突。缺起点 commit 时可按现有配置执行一次正常 Git fetch；权限/网络审批阻断就报告，不换通道绕过。不要修改 A 的 HEAD、工作文件或正在运行的进程。

从解压包复制本 prompt、`agents/dit_parallel_v2/` 和 `agents/dit_factorized_backend_v2/` 到 B 根目录。目标已有文件时逐项核对，不覆盖进行中的任务记录。A 的二进制 checkpoint 不会随 Git worktree 自动出现：通过参数只读访问原路径，不复制几十 GB 数据，不修改 A 的统计缓存。

## 阅读与阶段切换

完整读取 B 的根 `AGENTS.md`、`agents/AGENTS.md`、本 prompt、共享三个文件、B 目录中的 AGENTS/PLAN/ACCEPTANCE/DECISIONS/TASKS/HANDOFF，以及旧 pilot HANDOFF。

在 B 根 `AGENTS.md` 末尾追加 ROOT_AGENTS_APPEND 的共享说明，并记录本 session 为 B。不要修改 A 根 AGENTS；两边的追加在将来合并时可人工保留一份共享段和各自证据。

## 工作与停止点

1. 留存冻结 reference backend 与参数/数据 contract，完成 B001。
2. 实现批量 pooling/broadcast、attention 路径、固定拓扑预处理及重复 encode 修复。
3. 先 CUDA forward/gradient/mask 等价测试，再真实 clip 短 smoke，再受影响的回归，最后短 profile。测量同一批输入、同一初始化/权重，不能以两个不同 checkpoint 宣称新旧数值等价。
4. 只允许 PLAN 的短正确性/性能 optimizer 步，使用一次性副本；不续训原 best/latest，不启动 4500-step 科学实验，不训练 conditional prior。
5. 更新 B 的任务记录，提交明确范围的代码、小型 profile 结果和 ROOT_AGENTS 追加到 B 本地分支。
6. 状态为 `B_BACKEND_READY_FOR_REVIEW`，若等价性或性能证据不成立则明确标记对应未完成项。停止，不 push、不自动 merge/cherry-pick A。

所有 Python、测试、profile 都通过 `enter-container` 的 `torch-ito`，模型数值运算必须在 CUDA。不要安装依赖或为强行使用 FlashAttention 改变 attention 数学语义。SGD/Adam、loss、latent schema、时间条件、空间交互结构保持不变。
