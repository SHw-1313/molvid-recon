# B learned history memory：GPU0 pilot 报告

2026-09-28，用户明确指定 GPU0 后已完成 CUDA smoke、243 次真实数据更新和生成／autograd 检查。正式 M-short/M-long 尚未启动，等待 A 已评估 recipe；以下不是运动质量或长历史收益结论。

## 实现与数据流

raw observed `[Th,N,3] Å` → 两层共享共价局部原子 encoder（64 scalar / 32 vector）→ chemical residue tokens → 真实 ps 多尺度分析与近期路径 → 每 token 8 个 learned slots（128/64）→ 256/128 投影与局部 gather → 四层 DiT+G → per-frame future h/v → decoder。旧 R4 history/M 不在新 backend 实例化；时间衰减权重兼容保留。warm start 实测 loaded387、initialized37、retired94；不恢复旧 optimizer/cursor。

完整 teacher 只编码两个 observed geometry anchors `[-800,0]ps`；训练另编码 Q12 target。future 复用 observed reference origin，不参与 observed memory。旧 whole-clip teacher 路径在新 backend 退出。新增组件纳入 optimizer；训练 prepared memory 每 prefix 一次且保留梯度。

## 实际数据、模型和 loss 账本

冻结192 train / 8 valid划分；只读 train 的1bkp_A（2307 atoms）、1by2_A（886）、1cpq_A（920）各3条replica，另有中位3gwi_A（1374）、最大4yal_A（4975）的R1用于profile。sealed test未打开。raw native10ps，indices0..1760、anchor1280，真实past12.8ns/future4.8ns。198项短长配对检查通过；短长共用anchor/future和相同两个geometry anchors。

pilot243更新覆盖 span800/3200/12800ps × history dt100/200/400ps × query lag100/200/400ps × 9条轨迹。history dt与query lag独立；300ps未用于训练。这里只审计所列raw范围，不代表全576条train轨迹。

模型共16,122,904参数，15,561,680可训练。AdamW，history1e-4、DiT2e-4、decoder5e-5；BF16，teacher frozen，确定性CUDA/TF32关闭。实际base-only loss：flow1（h/v各0.5）、clean coordinate1、clean bond0.1、near coordinate0.1、near bond0.01、generated bond0.051707249134778976；near s≥0.9且detach，generated s≥0.75且保留flow梯度。没有自拟A辅助loss。pilot checkpoint不作为formal parent。

teacher/parent/statistics/raw bank均核对SHA；详见[data_card.json](data_card.json)、[loss_manifest.json](loss_manifest.json)和实际run目录的resolved_experiment、code_identity、load_report。

## CUDA 结果

实际训练 source `52617d23f79f743fd585e9fe9e143ca3e2768199`，目录 `runs/learned_memory_v1/cuda_pilot_r03/`。243更新全部完成，训练343.72秒，含启动/检查0.09765GPUh；有效46,201 atom-frames/s；峰值24.705GiB。此为用户授权共享GPU0条件下测量，不是专用卡基准。所有新增路径梯度有限且组件级非零；teacher权重保持不变。每prefix teacher实测 observed2+target12。

`cuda_smoke_r03.log`：6 passed（FP32/BF16、梯度、变换/掩码等targeted检查）。先前失败与修复见[CUDA_FIXES.md](CUDA_FIXES.md)，失败成本也保留。

`execution_r01/checks.json`，source `3a484c88252ab8a6b953dd3616d8b130e183bd2d`：16步生成 memory1、teacher2、DiT16；修改隐藏future后生成完全一致。训练双采样各4步，memory1、DiT9，新增encoder/slots/readout梯度均非零，teacher无梯度。该检查无optimizer更新，使用坐标平方数值探针，不是科学loss或评估指标。

中位/最大体系、冷/热重复及mixed-history-dt profile另存`cuda_pilot_r03_profile/`，完成状态以该目录train_manifest为准。

## 审阅、代码身份与限制

旧r01发现冗余key偏置，已修复；r02当时受GPU环境阻塞。当前fresh r03对固定`3a484c8`及实际CUDA证据独立审阅，原文保存在`../reviews/B/r03/REVIEW.md`；旧静态审阅不替代当前审阅。

A公共release固定吸收`eec7645`与`5a745960f673d5397b883c2b129f39cf57593856`，本地后者`0a556d7`。common_release不是recipe。正式短长对照、逐体系generated运动/multi-lag、置信区间、逐残基RMSF/MSD均未运行；不能用pilot训练loss代替这些结果。M-index仅配置/pilot入口，没有第三个formal。没有验证AF3融合、复合物泛化或长rollout。

## Profile 完成与累计预算

中位/最大体系、short/long、history dt100/400、cold/warm和混合dt packed batch的18次profile更新全部通过（固定`3a484c8`）。profile阶段562.26秒、含启动0.15835GPUh；整体6,214 atom-frames/s包含重型profiler及trace导出开销，不能与无profiler训练吞吐直接比较。逐条件峰值/耗时见[profile_summary.json](profile_summary.json)。当前条件最大峰值约53.43GiB，混合batch约27.13GiB。

原train_metrics.profile_ranges的key_averages CPU/device汇总字段不可信，已明确废弃；从完整trace的CPU user_annotation区间重建backward/teacher wall ms。其为host wall（含launch/wait），不是GPU kernel累计耗时。module_ms来自CUDA events，包含重叠的inclusive范围，不能相加。

所有自动cost ledger累计0.263742GPUh，另加smoke与小探针0.05GPUh保守余量，总账0.313742/48GPUh（pilot预算8GPUh）。GPU0无B任务继续运行；共享授权未更改A的资源归属。正式工作按[FORMAL_HANDOFF.md](FORMAL_HANDOFF.md)衔接。

独立fresh r03最终结论：**PASS，仅限实现与bounded pilot**，无P0/P1/P2阻断；P3计时问题已按上述后处理关闭。见[独立报告](../reviews/B/r03/REVIEW.md)与[处理记录](../reviews/B/r03/FIX_RESPONSE.md)。Euler8／activation-checkpoint辅助路径与完整A recipe尚未验证，留待正式集成后的targeted检查。
