# 执行清单

最终状态：P1--P3 全部完成；可选 bond 分叉按冻结判据明确不触发。所有
Python、测试、训练和评估均在专用 worktree
`/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2` 中通过
`enter-container` / `torch-ito` 执行。

- [x] 建立专用 worktree，记录 base HEAD、作业和容器映射。（2026-09-22；
  base `f49cf27e`；`agent/frame_gm_calibration_v2/experiment_spec.json`）
- [x] 核验 parent、codec、statistics、192 系统数据及 raw ATLAS。
  （2026-09-22；parent `b809c988...`，codec `ba10c441...`；
  `agent/frame_gm_calibration_v2/DATA_TRAIN_EVAL.md`）
- [x] 建立预算账本并限制本任务并发卡数不超过 4。
  （2026-09-23；`runs/frame_gm_calibration_v2/P2/evidence/resolved_experiment.json`，
  `runs/frame_gm_calibration_v2/P3/r01/evidence/resolved_experiment.json`）
- [x] P1 修复 metrics，对已有坐标重算全部派生字段并保存 schema v3。
  （2026-09-22；`339e4bc`；`runs/frame_gm_calibration_v2/P1/metrics_v3/`）
- [x] P1 构造 fixed-history/fixed-horizon 视图并验证 native time、mask 与
  split。（2026-09-22；`339e4bc`；manifest SHA `900fbd41...`）
- [x] P1 补齐 latent motion/noise/decoder sensitivity 诊断。
  （2026-09-22；`0895fa7`；`runs/frame_gm_calibration_v2/P1/latent_scale/`）
- [x] P2 实现 G/M 开关、off 兼容、history-only、zero-preserving 和稀疏拓扑。
  （2026-09-22；`f6ec672`；`molvid/dit/local_geometry.py`，
  `molvid/codec/motion_context.py`）
- [x] P2 实现 warm-start / derived-view normalization 契约并保留 strict
  resume。（2026-09-22；`3dda7ab`--`c6d1f6e`）
- [x] P2 完成定向 CUDA 检查、四臂 128-update pilot 和真实最大体系 profile。
  （2026-09-22；`c6d1f6e`；`runs/frame_gm_calibration_v2/P2/{logs,profile}/`）
- [x] P2 冻结 exposure、配置、sampler 清单及自动选择规则。
  （2026-09-22；candidate `c6d1f6e`；`runs/frame_gm_calibration_v2/P2/evidence/`）
- [x] P2 独立审阅、修复和复核闭环。
  （2026-09-23；r02 正式训练准入 PASS，r07 selector-path 复核 PASS；
  `agent/frame_gm_calibration_v2/reviews/P2/`）
- [x] P2 在获审阅代码上完成 B0/G/M/GM 四臂正式训练和两族评估。
  （2026-09-23；21,208 updates/arm；
  `runs/frame_gm_calibration_v2/P2/formal/run_manifest.json`）
- [x] P2 汇总三目标并自动选择 P3 parent。
  （2026-09-23；B0；`ca4e6d6`；`runs/frame_gm_calibration_v2/P2/selection.json`）
- [x] P3 实现可微实际源采样、energy score、物理特征及独立 aux RNG /
  cadence / 计数。（2026-09-23；`0b8be9f`--`f6f33a9`）
- [x] P3 完成定向 CUDA、DDP、continuation/resume 和双臂 64-update pilot。
  （2026-09-23；42 项定向测试；
  `runs/frame_gm_calibration_v2/P3/r01/evidence/pilot_summary.json`）
- [x] P3 冻结配方和预算，并完成独立审阅修复闭环。
  （2026-09-23；candidate `f6f33a9`；r01 FIX_REQUIRED → r02 PASS；
  `agent/frame_gm_calibration_v2/reviews/P3/`）
- [x] P3 自动完成 J0/J1 两臂训练与同协议 Euler16 评估。
  （2026-09-24；21,208 updates/arm；
  `runs/frame_gm_calibration_v2/P3/formal/formal_summary.json`）
- [x] 执行冻结的三向 CI 判据并决定是否追加 bond keep/release。
  （2026-09-24；判定不触发；
  `runs/frame_gm_calibration_v2/P3/formal/bond_fork_decision.json`）
- [x] 汇总实际架构数据流、三类指标、成本、负结果与后续建议。
  （2026-09-24；`docs/frame_gm_calibration_v2.md`）
- [x] 更新根 TASKS/HANDOFF 和本任务 HANDOFF，归档独立审阅小型证据。
  （2026-09-24；本交接提交）

未打开 sealed test；没有第四阶段，也没有修改 source noise、teacher、
latent statistics 或主 flow 定义。
