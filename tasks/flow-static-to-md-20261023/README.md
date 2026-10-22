# 新静态父模型转MD与静态回放

2026-10-23 三臂均已真实开始512更新。科学冻结5d560df：old_static在GPU4（容器lane1245214），new_static在GPU6（1245215），new_static_replay在GPU7（1245216），各allocator32GiB。首轮已见有限loss/grad和相同数据/时钟/progress；首32保存后自动恢复128与512。科学文件冻结，其他作业未停止。

共同query_post/groupQK/atomV架构10,009,356参数，固定source rho=1.1397942064485225Å、原FM与数据。旧父为CAPACITY query_post/static4096；新父为STATIC-REF control8192（无concat、无learned source）。三臂seed71、lr1e-4、warmup128、clip1、同512个动态更新；按step交替H4Q12/H8Q8，共5120动态目标帧。PDB旧父到新父同时改变了静态扰动、快照数据和训练曝光，父模型差异不能单归因某一个因素。

回放臂每4个动态更新，在同一优化步追加0.25×独立归一Q1 FM，之后统一梯度clip/update；另外两臂只用动态FM。额外128次Q1按PDB/ATLAS/MISATO=64/32/32，观测扰动0.2Å，与source rho分开。回放的数据、标准normal及所有默认CPU/CUDA/Python/NumPy随机流独立保存/恢复，不改变动态曝光；不以少学动态交换几何。不同父模型/回放导致权重、loss和优化器轨迹不同，不要求它们逐位相等。

三项CUDA检查通过：默认关闭回放兼容、真实Q1有限非零梯度、含dropout的动态RNG与保存恢复匹配。第一次单参数测试错误地要求共同clip后权重必不同，已改为检查额外的裁剪前梯度及加权损失，trainer未改；失败计入成本。真实parent严格加载无缺失参数，原生与任务adapter同35NFE采样最大差0.0001472Å，低于既定2e-4容差；adapter仅复用冻结的原采样计算。检查总32.765438GPU秒，记录于run/auxiliary_execution.json与checks/。

有限执行链为首32保存→恢复128并保存→恢复512→128/512各12个同0.2输入静态恢复probe→六DEV×两seed×joint/Q12/coarse_fine三协议。每臂36条、共108新Q50；Q12固定5×12保留60、主评前50/175NFE，joint35NFE，pure-clamp CF70NFE，不输入真实未来。恢复probe共72Q1，输入复用原12DEV构象，不能冒充独立PDB外部泛化。

独立总57600GPU秒=16 A100 GPUh：三有限lane各18000秒、辅助3600秒；含所有检查失败、训练、probe和生成。进程实际时间逐段减剩余，失败保留已有产物、不自动重试。共享卡吞吐不作为架构速度对比。实际run为runs/flow_static_to_md_20261023，CPU几何/RMSF/幅度/ACF/G/D/接缝与原播放器紧随有限GPU链，缺项显式记录。各臂配置与run_arm.py为复现入口；check_pair.py在32/128/512核数据、RNG及曝光。

中央molvid_recon/AGENTS.md、HANDOFF.md、TASKS.md仍是唯一管理状态；本README仅本实验交付入口。目前没有训练后科学结果。

CPU尾部已真实启动，容器PID1245760，等待本任务有限lane后自动最终配平、恢复/common/G/D/接缝/播放器；失败及缺项保留，不回填其他权重。
