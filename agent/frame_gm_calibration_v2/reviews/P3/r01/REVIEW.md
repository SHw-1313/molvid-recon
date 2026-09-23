# 独立审阅报告

REVIEW_STAGE: P3  
REVIEWED_COMMIT: f6f33a95a08627e5c779ca099d075679e9bed5ac  
VERDICT: FIX_REQUIRED

- reviewer/session：fresh independent read-only reviewer，`p3_independent_review_r01`
- base commit：`68abde6e1dc9196983cc5fd408e69118e60df41b`
- 阅读的规格版本与 pilot artifact：候选 commit 内 `PLAN.md`、`ARCHITECTURE.md`、`DATA_TRAIN_EVAL.md`、`REVIEW_PROMPT.md`、`REVIEW_PROTOCOL.md`；request SHA-256 `0610cf6212f5c591bbf1fd68c72dd59fc0560aed8b95fa8b6a3d82de8c8cde2e`
- 审阅环境限制：只读审阅；未训练、未运行全量 pytest，也未自行重跑已提供的 CUDA 检查。detached worktree HEAD 正确且最终仍干净。request 所列八个 artifact 均可读且 SHA-256 全部匹配。

## 结论

P3 的模型和训练实现满足主要科学边界：两个独立 source draw 确实从 observed-only conditional source 经过完整四步可微 Euler、共享 DiT/decoder；不存在调用 inference `no_grad` sampler 或 target interpolation 捷径。K=2 energy score、结构 mask、固定 train-only scales、独立 RNG、successful-update cadence、strict resume、DDP global applicable counts、校准、pilot 对称性和 bond keep/release 语义均有可复查实现与证据。

但目前不能开始正式训练。hash-bound `resolved_experiment.json` 的预算输入与同一 request 绑定的最终 largest-sample profile 明确矛盾，因此 `within_budget=true` 尚不能作为 96 GPU-hour gate。除下述 R1 外，没有发现其他可复现的 P3 代码或科学路径 blocker。

## 必须修复的问题

### R1 — 正式预算使用的 step time 与最终候选 profile 不一致

- 优先级：P1（正式运行预算 gate 不可审计）
- 定位：
  - `runs/frame_gm_calibration_v2/P3/r01/evidence/resolved_experiment.json:68`
  - `runs/frame_gm_calibration_v2/P3/r01/profile/largest_sample.json:99`
  - `runs/frame_gm_calibration_v2/P3/r01/evidence/commands.sh:10`
- 证据与触发条件：
  - SHA-verified final profile `c58448d6372d4fefe16e1795e467b5487ff378024dda203092fff987badb5056` 记录 largest sampled update 为 `6.8627921268343925 s`。
  - SHA-verified `resolved_experiment.json` 却把 `profile_step_seconds_largest_sampled_update` 写成 `4.364293091930449 s`，并据此计算两臂两 epoch 为 `51.42107105203387 GPU-hours`。
  - `commands.sh` 明确将前述 `6.862792... s` 文件标为 final candidate-bound profile，因此两个数不能同时代表最终 largest sampled update。
  - 若沿用当前“每个 update 使用该 profile 时间”的保守公式，最终 profile 会给出约 `80.85 GPU-hours` 主训练；再加当前 `19.2` reserve 约为 `100.05 GPU-hours`，超过 96。实际 cadence-weighted 成本可能明显更低，但现有 artifact 没有给出该公式或所需 ordinary-step profile，故不能据此确认预算。
- 影响：正式两臂训练和条件 bond fork reserve 的预算授权缺少有效依据，可能在错误的 `within_budget=true` 下启动。
- 最小修复建议：
  1. 使用最终候选的真实 profile 重建预算；明确区分 J0 ordinary update、J1 ordinary update、每八步一次的 sampled update。
  2. 写出包含 `21208` updates/arm、cadence=8、GPU 数和可选两支各 0.25 epoch 的公式。
  3. 若缺 ordinary largest-sample 时间，只需补一个同候选、同最大样本、sampled branch 关闭的短 profile；无需训练或全量测试。
  4. 根据重算结果保留两 epoch、调整 fork reserve，或统一缩减 epoch；随后更新 resolved artifact、hash 和 review request。
- 最小复验：核验新 artifact SHA；独立重算 GPU-hours；确认两份 formal config 的 update 数和 reserve 总和不超过 96。若不改模型代码，只需定向 r02 复核预算证据，无需重跑 pilot 或全量 pytest。

## 非阻塞建议

- Calibration 文件内容和 hash 自洽，八个 train batch、四个固定尺度及两项 0.05 DiT gradient ratio 均可复核。但 calibration 输出的 `target_encoder_provenance.json` 记录 code commit `ca4e6d6f4fb2d9932908555dd7705d9a356b985e`，且最终 `commands.sh` 没列出 calibration 命令。候选之后与 calibration 数值相关的唯一实现差异是 Kabsch FP32 solve 的 autocast guard；calibration 本身未进入训练 autocast，因此未将其提升为结果 blocker。建议在最终 clearance 中记录准确命令和这段代码等价关系，或候选绑定地重跑短 calibration。
- 可选 bond fork 尚未触发，因此无需现在创建分叉配置；若正式 CI 触发，应把两个 0.25-epoch config 和 hash 纳入同一 run manifest。

## 已核验的关键路径

- 实际 source-distribution 采样：
  - `molvid/flow/source.py:274`、`:299` 从最后 observed normalized h/v 加独立单位高斯噪声。
  - `molvid/model.py:209` 对每个 K=2 source 执行四次 `s=0,0.25,0.5,0.75` DiT Euler 更新，再 inverse statistics 和 decoder；该路径无 `no_grad`。
  - activation checkpoint 使用 non-reentrant 模式；Frame Joint 明确固定 dropout=0。largest-sample CUDA evidence 中 history/DiT/decoder 梯度均非零，teacher 梯度为零且 state hash 不变。
- Future-label 边界：
  - `prepare_frame_joint_batch` 将 observed context、query、source center、future target 分开。
  - target-mutation artifact 在真实 CUDA 样本上将 future `x/bpos` 扰动约 50 Å 后，observed coordinates、origin、source 和两个 sampled outputs 均保持 exact zero difference；两 source draw 明确不同。
- Energy score：
  - `molvid/losses/distribution.py:310` 独立按 condition 计算；K=2 data term为两个距离均值，diversity 正确为 `0.5*d(X1,X2)`，没有 paired coordinate MSE。
  - RMSF、MSD/相邻位移平方、静态拓扑局部距离增量和增量乘积使用同一 observed reference、query mask、固定坐标和固定 train-only scales；结构性缺 pair 时整组缺失而非填零；零运动仍 eligible。
- RNG、cadence 与 resume：
  - main flow generator 和 sampled generator 分开，sampled branch 只消费后者。
  - cadence 由 restored `successful_optimizer_updates` 决定；逐 rank sampled generator state、cursor、global RNG、optimizer 和 model 均进入 checkpoint。
  - interrupted/resumed J1 与 uninterrupted J1 的 checkpoint 和 metrics stream exact；sampled active steps 为 8、16、…、64。
- DDP：
  - `FrameJointTrainer._distributed_total` 对每个 component 使用 local applicable count / global applicable count 修正 DDP gradient averaging。
  - 两 rank检查覆盖各 rank阈值 eligibility 不同和 sampled loss；component 最大差 `1.19e-7`，pre-Adam gradient max absolute `3.77e-6`、relative L2 `3.26e-6`，均在冻结阈值内。
- Calibration 与 pilot：
  - 八个 train-only batch 的 feature scales 和 energy/bond gradient calibration hash 匹配；resolved ratios 均为约 `0.05`。
  - J0/J1 使用同 B0 parent `ff8121aa47a7c1885b4ad59ece586dcc90a3d8a12c67d71fb767f586c6b36258`，相同 warm-start/reset optimizer policy；64 步 sampler manifest、exposure、sample IDs、H、bucket 和 main flow-time stream 相同。
- Formal evaluation gate：
  - `tools/summarize_frame_gm_p3_formal.py` 要求完整 21208-row metrics、冻结 cadence、同 parent/config/exposure/sampler。
  - `tools/compare_frame_gm_p3.py` 绑定 checkpoint 和 formal metric provenance，要求 Euler16、seeds `[0,1,2]`、legacy/fixed-history paired views、八体系 paired bootstrap，以及预先冻结的三方向 bond-fork trigger。
- Bond keep/release：
  - `bond_enabled=false` 会统一清零 generated/clean/near bond 权重及 sampled observed-bond 权重；energy score 的内部运动特征保留，拓扑图保留。
  - continuation 检查确认 keep/release 从相同 model、optimizer、cursor、main RNG 和 sampled RNG 起步；release 的四个显式 bond 项均为零，sampled energy 保持相同。
- P2 selector-only interleaved commits未被当作 P3 实现；仅检查了 P3 comparison 对已审阅 `load_arm_values` 接口的调用，未发现影响 P3 的新 interaction blocker。

request 所列 artifact SHA 核验结果：

- `pilot_summary.json`：`ed663fb336eed2c4f2b7a80869a75e3b51118b8cc25f278891529bd0b914c33d`
- `resolved_experiment.json`：`9cd7f4a18ccb314731dba9da02a2517317fcfc023f79c8c17521b0ac271331fc`
- `commands.sh`：`b9a9dbcb1290968fdc5125729f893b8b8994f228d1ed954138ff8dcc4b51af56`
- `runtime_gpu_mapping.json`：`2140b7ec3dcfab81ab9a501992925c7646e11d5aa25e892ce9211a14e124b400`
- `largest_sample.json`：`c58448d6372d4fefe16e1795e467b5487ff378024dda203092fff987badb5056`
- `target_mutation.json`：`b3e01c4ada88834ef1fedcf5aeeaf7363083c1a004d734da164503d0b0255432`
- `ddp_check.json`：`7d4c509bf047b11994a63ca1f3d96ccd125ad6cb96414bef50fa352f3d82da17`
- `continuation_check.json`：`59f20d6692472e482d4fcada78afc56fb0a8fbb3074a214823a9bd769f1de1b9`

## 主session下一步

先修复 R1 并进行定向 r02 预算复核；在此之前不要启动正式训练。

已审阅但暂未获准启动的 formal configs 是：

- `configs/frame_gm_p3_formal_J0_260923.yaml`，SHA-256 `e9cb92ba0235ccb61455e58249e50c3db51f09452a2293909682aa909897a8dc`
- `configs/frame_gm_p3_formal_J1_260923.yaml`，SHA-256 `b05ef564d9f0c778fc31cccd3a0cbec042fc675044ff1fe37426a1f1d17e6a3f`
