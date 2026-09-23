# Frame GM Calibration v2 交接

status: COMPLETE

P1--P3 已全部完成。可选 bond keep/release 分叉经过冻结的三向系统配对
置信区间判据后明确不触发，因此不是待办项。

## 环境与代码

- host repo：`/data4/users/sihao/workspace/molvid_recon`
- 专用 worktree：`/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2`
- container repo：`/workspace/molvid-recon-gm-calibration-v2`
- branch / base：`feat/frame-gm-calibration-v2` /
  `f49cf27efa60a0cca172248c49913d70ff1516a5`
- P3 implementation candidate：`f6f33a95a08627e5c779ca099d075679e9bed5ac`
- environment：`enter-container` 后 `conda activate torch-ito`
- 正式训练实际用卡：P2 为 physical GPU 5/6/7/0；P3 J0/J1 为
  physical GPU 1/5。用户已允许使用这些空闲卡。当前无本任务 GPU 作业。
- 原始 Frame Joint parent SHA：`b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`
- codec SHA：`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`
- statistics hash：`7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`
- 训练为 192 systems / 576 physical trajectories；sealed test 从未打开。

## 阶段状态

| Stage | 状态 | candidate / review | 主要结果路径 |
|---|---|---|---|
| P1 | COMPLETE | `339e4bc`, `0895fa7`；无需独立审阅 | `runs/frame_gm_calibration_v2/P1/` |
| P2 | COMPLETE | `c6d1f6e`；r02 PASS；selector r07 PASS | `runs/frame_gm_calibration_v2/P2/{formal,selection.json,report.md}` |
| P3 | COMPLETE | `f6f33a9`；r01 FIX_REQUIRED 已修；r02 PASS | `runs/frame_gm_calibration_v2/P3/{formal,formal_metrics_fixed,formal_metrics_legacy}/` |

## 正式产物

P2 四臂均完成 21,208 successful updates（2 effective epochs），主 exposure
和 sampler manifest 完全一致。最终 checkpoint SHA：

| Arm | SHA-256 |
|---|---|
| B0 | `ff8121aa47a7c1885b4ad59ece586dcc90a3d8a12c67d71fb767f586c6b36258` |
| G | `e3ba2dfcf89b13f76b3633b1730f56cf8bd7359680051384f4931fe85893d8e6` |
| M | `6c7ad59594a4a806e29c4e5e6a1de30dbcf1ad435cc67ab647ca553123ae0091` |
| GM | `aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540` |

自动选择文件 `runs/frame_gm_calibration_v2/P2/selection.json` 的 SHA 为
`198216cac78afae06fc6da4672f91db42b568414c4504c5932009784d88e993b`；
没有候选通过主指标与 safety 联合门，因此选择 B0。

P3 J0/J1 同样各完成 21,208 successful updates。两臂主数据、主 RNG 和
sampler 完全配对；J1 在 2,651 个 cadence step 启用 sampled auxiliary。

| Arm | config SHA-256 | checkpoint SHA-256 |
|---|---|---|
| J0 | `e9cb92ba0235ccb61455e58249e50c3db51f09452a2293909682aa909897a8dc` | `60e14c3e7cc59c8e4c1a2ff38512df829db7418446cc82bd3ed3587722696f5e` |
| J1 | `b05ef564d9f0c778fc31cccd3a0cbec042fc675044ff1fe37426a1f1d17e6a3f` | `8859a499be7d06f31ba33fb9e71febf5019cebb9ced1947d0c277326da0c66e1` |

formal summary SHA：
`65e89f93364cbb9933c41d8e67a0580c578edfdc2d08e68db9c01676335dd661`。
bond decision SHA：
`fc1fd8f159641c693b78d4b9df36025080cef4e408b0ef61b848b43b709c5d0f`。

## 审阅闭环

- P2：r01 提出正式准入问题，修复后 r02 对 candidate `c6d1f6e` 给出
  PASS；后续 selector artifact path 修复由 fresh r07 再审并 PASS。
- P3：r01 仅因 formal budget 证据未按 cadence 分解给出 FIX_REQUIRED；
  加入 ordinary/sampled 分项 profile 和预算公式后，fresh no-context r02
  给出 PASS。r02 REVIEW SHA：
  `3ed00db19f9d6f13dda16f2c37d3f366f00831b3de1b747492bb84308f78b76d`。
- 审阅材料在 `agent/frame_gm_calibration_v2/reviews/P2/` 和
  `agent/frame_gm_calibration_v2/reviews/P3/`；没有把 FIX_REQUIRED
  误写成 approved。

## 实际预算与验证

- P2 formal：run manifest 采用冻结 profile 记账为 31.581 GPU-hours /
  96；四臂均完成。
- P3 formal：按两臂实际开始/结束时间合计 14.565 GPU-hours
  （J0 6.569，J1 7.996）/ 96；冻结 profile 上界为 16.850。
- P2 八组正式 generation evaluation 日志累计 6.130 GPU-hours；P3 四组
  累计 3.479 GPU-hours。pilot/profile/metric recompute 日志未统一记录
  elapsed 字段，因此不伪造一个精确总数；已记录部分和冻结预算均低于各自
  32/96 小时限额。
- P3 通过 42 项定向测试；target mutation、BF16、DDP pre-Adam gradient、
  strict resume、continuation、独立 aux RNG 和 cadence 均有证据。
- 正式评估均为 observed-only、Euler16、seeds 0/1/2；legacy 2,352 rows /
  arm，fixed-history 1,392 rows / arm，全部完整并重算 schema-v3 metrics。

## 科学结论

P1 显示单位高斯 source noise 的 normalized coefficient RMS 显著大于训练
latent motion：h 约 11.31 对 3.40--3.68，v 约 19.59 对 1.70--2.13；
equal-norm v 扰动对坐标的影响也远大于 h。该结果是 future-informed
train-only 诊断，不是生成质量结果；本任务未据此改 source/noise。

P2 中 G 改善 bond 但恶化 fixed-history MSD 与 clash；M 改善 residue RMSF
但恶化 bond 与 clash；GM 改善 bond 与 residue RMSF，但 clash 明确恶化。
因此无非基线臂获得联合优势，P3 从 B0 开始。

P3 中 J1 相对 J0：

- bond RMSE：`0.4130 → 0.3312 Å`，差 `-0.08184`，95% CI
  `[-0.08707, -0.07705]`；
- fixed-history MSD curve MAE：`2.6178 → 3.1162 Å²`，差
  `+0.49850`，CI `[0.42882, 0.57977]`；
- residue RMSF MAE：`0.4120 → 0.4316 Å`，差 `+0.01967`，CI
  `[-0.02448, 0.07637]`。

前两项分别明确改善和恶化，但 RMSF CI 跨零。冻结规则要求几何改善且
RMSF、时间两组都明确恶化才启动 bond 分叉，所以决定为
`do not run a bond keep/release fork`。

## 后续使用与限制

默认不要把 J1 当作无条件优于 B0/J0 的生产 checkpoint；它展示的是
geometry--motion trade-off。若后续获得新授权，优先在独立 validation
协议上研究 source prior 尺度和 sampled physical objective 的冲突，再决定
是否做新的配方，不应事后改本轮 gate。本轮只有八个 valid systems，
没有 sealed-test、真实生产 checkpoint/data 对照，也没有 production parity
结论。

任务已完成，不存在需要恢复的训练命令。复现实验的精确已运行命令保存在
`runs/frame_gm_calibration_v2/P2/evidence/commands.sh` 和
`runs/frame_gm_calibration_v2/P3/r01/evidence/commands.sh`；运行前仍须进入
本 worktree 的 container，并激活 `torch-ito`。
