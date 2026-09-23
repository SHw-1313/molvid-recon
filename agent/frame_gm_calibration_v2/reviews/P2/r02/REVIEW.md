# 独立审阅报告

REVIEW_STAGE: P2  
REVIEWED_COMMIT: c6d1f6eaa34861a0a408dfc60a1385b2cca833ed  
VERDICT: PASS

- reviewer/session：全新独立只读 reviewer；固定 detached worktree `/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2-review-P2-r02`
- base commit：`f6ec672c7d8582772c9056cac0d116c1eddf1ffc`
- 阅读内容：根 `AGENTS.md`、全部 P2 规格与审阅协议、r01 `REVIEW.md`、r02 `FIX_RESPONSE.md`、`review_request.json`，以及 R1–R4 修复 diff、四份 formal config 和请求绑定的 pilot/profile/resume evidence
- 审阅环境限制：无读取阻碍；未修改文件、未启动训练、未运行全量 pytest。仅在 `enter-container`/`torch-ito` 中独立运行了 R2–R4 定向契约测试，17 项通过。审阅前后固定 worktree 均干净。

## 结论

r01 的 R1、R2、R3、R4 均已关闭，修复未引入明显回归。候选代码、四臂 pilot、最大样本 profile、seed-0 leakage 检查、派生数据身份和 strict-resume 曝光账本形成了可复查的一致证据链。

可以按下列四份已核验、已冻结配置开始 P2 正式四臂训练：

- B0：`configs/frame_gm_p2_formal_B0_260923.yaml`，SHA-256 `9b243c2fb1477459ea2461816283c49fc57e476aa1132bf7e6d2b42287cb359d`
- G：`configs/frame_gm_p2_formal_G_260923.yaml`，SHA-256 `b201c095efa4531825bbc61168315296bfca318924e8584b22e1adb5fbfeb4e2`
- M：`configs/frame_gm_p2_formal_M_260923.yaml`，SHA-256 `895337e06aa8a235bc0816ec39f2e445b3dbe8dc9dca697c4a28a417b0e83502`
- GM：`configs/frame_gm_p2_formal_GM_260923.yaml`，SHA-256 `45cc0f1950ad4591ec44b8c4bffe1593cd95e92825fab888e87cab55b5f431c5`

四份配置除预期的 G/M 开关和输出目录外保持对称；均从同一 parent 做 model-only warm start，重新建立 optimizer、scheduler、cursor 和 RNG。

## 必须修复的问题

无。

## r01 问题关闭情况

### R1 — Pilot 与候选提交绑定：已关闭

- 四臂 `target_encoder_provenance.json` 均记录完整候选 SHA `c6d1f6eaa34861a0a408dfc60a1385b2cca833ed`，文件 SHA 均为 `921037ce2d8841019e64b89794a8bdc4685716fe41cc5184fb9492365c2dc8ff`。
- hash-bound `pilot_summary.json` 同时绑定 candidate、provenance、checkpoint、resolved config、sampler、exposure、warm-start report 和 metrics。
- 四臂 checkpoint SHA 已独立重算并匹配 summary：
  - B0 `6e0f93ba618535bf64ff1e5bc740f5b571c6cd0cf39f6c6a5248a06d68ede3c9`
  - G `5a9bb3b94fa5c1a4fa68dfc20f3a91f28bc21734234ca61e825da268409ec53f`
  - M `49ee34d919374da61c629f2834fc02359f29c918bb243adaf779be3817244768`
  - GM `ab1547468538ca2c848a431b41886756fe818fd616529c8b3100d6ac9ef682de`
- 每臂 checkpoint 均为 128 个成功 update；parent、loss contract、teacher state hash、采样 schedule、sample-ID hash和曝光完全一致。
- G/M/GM 所有新增外层 residual gate 均已离开零点；optimizer state 数分别为 355、400、424。
- 四臂最大真实样本 profile 均使用正式全量配置及同一 4975-atom、16-slot 样本；GM 峰值显存 `56,563,855,872` bytes。

### R2 — 严格配置解析：已关闭

- 根配置以及 data、derived item、codec、statistics、model、loss、training、stage、learning-rates 均有显式 allowlist。
- G/M、loss 和 training 布尔字段只接受真实 boolean。
- 验证在 CUDA 初始化和数据打开之前执行。
- `JointLossConfig.resolve` 独立拒绝未知 loss 键及字符串布尔值。
- reviewer 独立运行 `tests/test_frame_joint_contracts.py`：17 项全部通过；四份 formal config 均通过严格解析。

### R3 — 派生 store 身份：已关闭

- `_verified_derived_identity` 同时比较当前 `index.txt`、manifest 声明和配置声明的 SHA，并核对 manifest/config/current record count。
- 加载端进一步校验整体 `expected_data_hash`。
- reviewer 独立重算：
  - manifest SHA `863d368d8b6937dd7c5a5efce0ccfc4a2857b77b7aaf1de007ade3a40ae0cc61`
  - index SHA `3a1be208b90f77bf78452cb6e7402b14f9066b8b0208ce6e250852133d250045`
  - index 记录数 `6912`
  - derived data hash `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1`
- manifest 明确为 `split=train`、`test_opened=false`，仅含 100/200/400 ps，valid query 数为 12/6/3。

### R4 — Strict resume 累计曝光：已关闭

- 每个 rank 的 trajectory、bucket、view/history、valid atom-frame 和成功更新计数均写入 checkpoint。
- strict resume 要求 exposure schema 和成功更新数匹配；旧 checkpoint 缺少账本时会明确失败。
- 最终 `exposure.json` 通过跨 rank gather 汇总，schema 为 `molvid.frame_joint.exposure.v2`。
- reviewer 独立读取连续两步和“一步后 strict resume”两个最终 checkpoint，确认：
  - cursor 相等；
  - rank exposure state 相等；
  - step 和 successful update 均为 2；
  - 最终 `exposure.json` 字节哈希相同；
  - sample IDs、bucket/history 计数及 `valid_atom_frames=51632` 一致。

## 非阻塞建议

正式结果产生后、执行 P3 parent 自动选择前，仍需按冻结规则补齐 system-level paired bootstrap 和 `selection.json` 工具。该事项不阻塞四臂正式训练启动。

## 已核验的关键路径

- request 所列四个 artifact SHA-256 全部精确匹配：
  - `pilot_summary.json`：`da9b2e5f8229852343d6d0b7c3dc4a88ad6359624ab10e3353e0d3909d93b73c`
  - `resolved_experiment.json`：`25c002e5995a6251b341eb0d13ff642e47bbebf3c7a8f54da6e8a9cbcb68e971`
  - `commands.sh`：`f1e35f4e57a3fa509229ebab4c9eae0496d26bd6491495acf18e061a2856013d`
  - `resume_comparison.json`：`3cce79694ba313fee8638112526b21d17387acb47ca91520c56c9ff338dbb504`
- 正式 parent SHA 独立重算为 `b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`。
- 四臂各 128 update，loss/gradient 全有限，共同曝光 `valid_atom_frames=3658849`。
- warm start 均加载 386 个旧 tensor；B0 初始化 0 个新增 tensor，G/M/GM 分别初始化 26/69/95 个；均未恢复 optimizer、scheduler、cursor 或 RNG。
- legacy/fixed-history 共八组 Euler16 seed-0 检查均满足 finite、prefix exact 和 future leakage PASS；future mutation 最大差异均为 0。
- profile 使用 A100 80GB；预计四臂两 effective epochs 合计约 `31.58 GPU-hours`，低于 P2 的 96 GPU-hours 上限。
- 当前不存在 `P2/formal/` 输出目录，正式训练尚未开始。
- pilot evaluation protocol 仅指向 `valid` stores；数据 manifest、profile、summary 和 request 均记录 `test_opened=false`，sealed test 未打开。

## 主session下一步

从获审阅的 `c6d1f6eaa34861a0a408dfc60a1385b2cca833ed` 代码快照，使用上述四份精确 SHA 的 formal config 启动 B0/G/M/GM 正式训练。启动时继续记录实际 GPU 映射、配置副本与哈希、run manifest 和累计 GPU-hours；不要热修改运行中的代码或配置。
