# Molvid static-reference research workspace

唯一管理入口是 /data4/users/sihao/workspace/molvid_recon/AGENTS.md、HANDOFF.md、TASKS.md。恢复或规则更新后须重读三文件。本工作区旧HANDOFF/TASKS仅继承历史，不是PM状态。

2026-10-22已授权本独立静态实验：corruption0.2Å与source rho1.139794Å分开；PDB/ATLAS/MISATO单帧50/25/25；query_post共同随机初始化，两臂原路径与reference h/v concat，各8192日程、4096/8192保存；必要CUDA及同0.2恢复probe。独立总12A100GPUh，全部失败/验证/训练/生成计入。原INPAINT任务继续且其依赖冻结，不触碰外部进程。只用enter-container与torch-ito，不改依赖、不新平台、不push。具体交付仅 tasks/static-ref-20261022/README.md。
