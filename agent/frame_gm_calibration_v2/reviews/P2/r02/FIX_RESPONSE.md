# P2 r01 修复响应

候选提交：`c6d1f6eaa34861a0a408dfc60a1385b2cca833ed`

正式四臂训练尚未开始；sealed test 未打开。本轮仅修复 r01 阻断项并从最终干净候选重跑 pilot/profile/seed-0 检查。

## R1 — pilot 与候选提交绑定

- 在提交 `3dda7abc500c295618be0e2b031c53ae301ad679` 修复 R2–R4 后重跑时，严格 data hash 暴露旧 profile 工具从 pilot 子集切换全量数据却保留 pilot hash 的问题；该批输出完整保存在 `P2/candidate_3dda7abc500c2956/`，不作为 r02 放行证据。
- 最终提交 `c6d1f6eaa34861a0a408dfc60a1385b2cca833ed` 要求 profile 使用正式全量配置，并让 summary 直接校验/记录 candidate SHA、四臂 target provenance 及关键 artifact 哈希。
- 从最终提交重跑 B0/G/M/GM：每臂 128 个成功更新，loss/gradient 全有限，`valid_atom_frames=3658849`，相同 sampler schedule hash `2f4e32f605bad62138ae3e0676ac6cba296769ae23c2a42b5ed4daed44039024` 与 selected-ID hash `263d9ad1b5f120d51aef56a8c08bee7a8a70497ace379b859f1d9c89c1170e88`。
- 四份 `target_encoder_provenance.json` 都直接记录最终 candidate，且 SHA-256 均为 `921037ce2d8841019e64b89794a8bdc4685716fe41cc5184fb9492365c2dc8ff`。
- checkpoint SHA-256：B0 `6e0f93ba618535bf64ff1e5bc740f5b571c6cd0cf39f6c6a5248a06d68ede3c9`；G `5a9bb3b94fa5c1a4fa68dfc20f3a91f28bc21734234ca61e825da268409ec53f`；M `49ee34d919374da61c629f2834fc02359f29c918bb243adaf779be3817244768`；GM `ab1547468538ca2c848a431b41886756fe818fd616529c8b3100d6ac9ef682de`。
- G/M/GM 的全部新 residual gates 均已离开零点；optimizer state 数分别为 G 355、M 400、GM 424。最大 gate 绝对值按臂为 G `0.0076653`、M `0.0047746`、GM `0.0081035`。
- 四个全量最大样本 profile 均成功，样本统一为 `atlas_4yal_A_R3_dt_400ps_wf000061`（4975 atoms、16 slots、79600 effective atom-frames）；GM 峰值 `56563855872` bytes。
- legacy/fixed-history 共 8 个 seed-0 Euler16 输出的 finite/prefix/future-leakage/wrong-clock 检查全部 PASS，future mutation 最大差异均为 0。

## R2 — 配置未知键与类型

- `molvid/cli/train_frame_joint.py::_validate_frame_joint_config` 对 root、data、derived item、codec、statistics、model、loss、training、stage、learning-rates 建立显式 allowlist 和逐字段类型/数值校验；在 `_distributed_device` 与 `_open_data` 之前调用。
- model/training/loss 布尔字段只接受真实 boolean；不再使用字符串真值转换。
- `JointLossConfig.resolve` 也拒绝未知 loss key 与字符串 boolean，避免 checkpoint/parent 合同绕开入口验证。
- 新增 `tests/test_frame_joint_contracts.py`；目标测试总计 49 项通过。所有 18 份现有 Frame Joint/P2 配置也逐份通过严格解析。

## R3 — 派生 store 身份

- `_verified_derived_identity` 同时要求 manifest 与配置中的 `store_index_sha256` 等于当前 `index.txt`，并核对两处 `record_count`。
- pilot/formal 八份配置显式绑定 index `3a1be208b90f77bf78452cb6e7402b14f9066b8b0208ce6e250852133d250045`、6912 records，以及各自 expected derived data hash。
- 真实 pilot/full 打开结果分别为 `a6a6f9f242d10f215b854e391971ae863f5b988bf3fbcbf74bf17f009aa9cfee` 和 `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1`。单测覆盖 index 篡改与 record-count 不一致的启动前失败。

## R4 — strict resume 累计 exposure

- 每个 rank 的 trajectory/bucket/view-history/valid-atom-frame 累计账本随 checkpoint 保存；strict resume 要求 schema 与 `successful_updates` 完全匹配后恢复。旧 checkpoint 缺账本时不再把后缀伪装为全程统计。
- 最终输出通过 `all_gather_object` 汇总所有 rank，并写 `molvid.frame_joint.exposure.v2`。
- 最终 candidate 上实际运行连续两步与“一步 checkpoint → strict resume → 第二步”。最终 exposure、cursor、selected-ID hash、global schedule hash、逐步 sample IDs 和 checkpoint rank exposure state 六项完全相等；证据为 `resume_comparison.json` 及两套保留运行目录。

## 非阻塞项

- r02 `resolved_experiment.json` 的 generated-bond 系数直接采用 parent contract 的 `0.051707249134778976`。
- r01 指出的两个 EOF 多余空行已移除。
- paired system bootstrap/selection 工具将在正式结果产生前实现；本轮尚无正式结果可选择，冻结规则未改变。

## 证据

- `runs/frame_gm_calibration_v2/P2/r02/evidence/pilot_summary.json`
- `runs/frame_gm_calibration_v2/P2/r02/evidence/resolved_experiment.json`
- `runs/frame_gm_calibration_v2/P2/r02/evidence/resume_comparison.json`
- `runs/frame_gm_calibration_v2/P2/r02/evidence/commands.sh`

