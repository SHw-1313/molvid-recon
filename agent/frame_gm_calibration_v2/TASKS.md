# 执行清单

初始状态：任务包已生成，以下实现/运行均未执行。

- [ ] 建立/核对专用worktree，记录base HEAD及existing jobs；映射宿主/容器路径。
- [ ] 核验parent、codec、statistics、192系统数据及raw ATLAS。
- [ ] 建立预算账本；确认至多4张实际属于本任务的可用卡。
- [ ] P1修复metrics，对已有坐标重算全部派生字段，保存schema v3。
- [ ] P1构造fixed-history/fixed-horizon视图并验证native-time、mask与split。
- [ ] P1补latent motion/noise/decoder sensitivity诊断并写结论。
- [ ] P2实现G/M开关，off兼容、history-only、zero-preserving、稀疏拓扑。
- [ ] P2实现warm-start/derived-view normalization契约；保留strict resume。
- [ ] P2定向CUDA检查、四臂128-update pilot、真实最大体系profile。
- [ ] P2冻结exposure、配置、sampler清单及自动选择规则。
- [ ] P2提交candidate；独立review r01 → FIX_RESPONSE → 修复/必要复核。
- [ ] P2获审阅代码上自动跑四臂训练，监测/恢复，逐臂评估。
- [ ] P2汇总三目标结果，产生selection.json，选择P3 parent（可为B0）。
- [ ] P3实现可微实际源采样、ES/物理特征、aux RNG/cadence/计数。
- [ ] P3定向CUDA/DDP/恢复检查和双臂64-update pilot。
- [ ] P3冻结配方与budget；提交candidate；独立review → 修复/必要复核。
- [ ] P3自动完成J0/J1训练与同协议评估。
- [ ] 判断是否触发bond keep/release；若触发同parent短分叉并评估。
- [ ] 汇总实际架构数据流、三类指标、成本、负结果与后续建议。
- [ ] 更新根TASKS/HANDOFF摘要和本任务HANDOFF；归档小型证据。

每项填写完成日期、code commit、结果路径；未运行不打勾。完成P1/P2不是交付停止点。只有全部范围完成或真实环境/预算阻塞时才结束主任务。
