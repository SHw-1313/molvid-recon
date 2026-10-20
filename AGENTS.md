# Flow capacity positions — 2026-10-20

唯一权威管理入口为 /data4/users/sihao/workspace/molvid_recon/AGENTS.md、HANDOFF.md、TASKS.md；开始/恢复/更新时重读，旧本地HANDOFF/TASKS仅历史。不新建PM状态或调度/门禁系统。

用户已授权六个容量位置扩增＋一个共享匹配控制，PM总上限96 A100 GPUh（含验证/训练/84条Q50/失败），最多六卡并发，每卡独立臂，不做DDP。一个工作区、七配置run。group-QK/原子V/legacy Collective/固定rho/原RF目标；不混noise、GroupV、pairmsg或新loss。静态及动态训练按同数据/曝光与初始化规则；训练ready全集与原始磁盘全集区分，继承TRAIN/test隔离及MISATO参考片段5Å过滤。无NAS下载/新外部数据/长轨迹授权。

所有Python/测试/模型通过enter-container、torch-ito，不安装或升级环境，不改旧工作区/权重，不占外部进程卡。科学源在正式运行前冻结；阶段检查通过后连续完成，不逐阶段请示。工程结果只写tasks/flow-capacity-20261020/README.md及同run证据；PM三文件由PM维护。不push。
