# Frame Joint v1 交接

状态：实现、独立审阅 R1--R6 修复、针对性 CUDA 检查、1/2 GPU 等价/恢复检查和
原 3 体系/9 轨迹审阅前 tiny 已完成。没有重跑修复后 tiny；48/192 长训练未启动，
等待本目录 `TRAIN_PROMPT.md`。

## 仓库与环境

- repo：`/data4/users/sihao/workspace/molvid_recon`
- branch / HEAD：`refactor/flat-layout` / `93d191fd8c88c2355944c1cf8bf05e8519ddb598`
- 容器：`enter-container`，环境 `conda activate torch-ito`
- 实测 GPU：NVIDIA A100-SXM4-80GB；单卡检查/训练用 GPU 0，DDP 检查用 GPU 0/1。
- 初始用户 `.gitignore` 改动和未跟踪设计文件均保留；未 reset、未 push、未修改外部参考 worktree。

## 产物与数据身份

- target codec：
  `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`
- verified SHA-256：
  `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`
- checkpoint artifact step：39721。历史 result 的 45844 是旧 run 最终记录，不是本 artifact step。
- 提取：`frame_encoder` + centered coordinate stem，均冻结；当前 teacher 每四帧分块执行以控制显存，和旧 codec 全 16 帧输出的最大误差为 h
  `6.68e-6`、v `2.38e-5`。teacher state hash：
  `9962114f5842304311f1504224304d8a11758ffb2c22b958d18ecbc0fe8919fd`。
- tiny：train 441 / valid 117 clips；3 systems、9 trajectories、T=16、100 ps；
  train windows 0--48，valid windows 49--61；atoms 887/1503/2499。
- tiny direct-store data hash：
  `140cbf1fab701ba40686eff60b9a8a055e309cf755d5df448fdddfd9b09d1788`。
- 48/8 frozen manifest：
  `/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`；
  manifest SHA `88325925b339ef34a421d48b5439bfc869fb1003b9c59e3d7eeac45b62da3d2e`；
  train 8928 / valid 1488，48/8 systems，test 未用于本任务。
- 192 candidate：
  `/data4/users/sihao/workspace/molvid-dit-capacity-data-v1/outputs/dit_capacity_data_v1/data_manifest_20260914`；
  manifest SHA `fdb925f3d9845a9ce83af3c4de21e908bc9415056cdffbe575378c3c2e38fb86`；
  train192 35712 / valid 1488，materialization 明确 `test_opened=false`。
- tiny frame statistics：
  `runs/frame_joint_v1_tiny_260920/frame_statistics.pt`；file SHA
  `7427bdd20f658ce02e44823fd63de96bbedb4225311dbc1f6c172136604c4be2`；
  statistics hash `8440b35301c5691b68b50fddfcc69096dd92f53207a5dca3719756e4459a7672`。
  第一次 80k stats batch 因两个 2499-atom clips 同批而 OOM；改为 40k 后覆盖全部 441 train clips。

## 实现摘要

- `molvid/latent/types.py`, `latent/statistics.py`, `time.py`：逐帧 h/v 类型、条件/查询边界、训练集统计和独立物理时间。
- `molvid/codec/frame.py`, `codec/history.py`, `codec/decoder.py`：冻结分块 teacher、zero-preserving R4 history、两层联合 decoder。coordinate head warm copy 后在 optimizer 构造前设为可训练。
- `molvid/dit/frame.py`, `dit/blocks.py`：history cross → spatial → 双向 future temporal → FFN；未来不使用 state/detail 或 inverse Haar。
- `molvid/flow/*`, `generation.py`, `model.py`：逐帧 RF、observed-only source/Euler 生成和显式 composite forward。
- `molvid/training/batches.py`, `training/joint.py`：teacher-only no-grad、sample-equal loss、统一 bond switch、16-batch λb 校准、stages、严格 resume 和 changed-objective continuation。
- `molvid/data/sampling.py`, `cli/train_frame_joint.py`：每 trajectory 无放回 cap、合法 oversize singleton、完整 DDP model、rank cursor/RNG、5% warmup + cosine。
- `configs/frame_joint_v1*.yaml`：tiny、48 two-GPU pilot、bond keep/release continuation。48 sampler 实测每 effective epoch 2716 global batches；two-GPU 为 1358 synchronized updates。
- `tools/frame_joint_cuda_check.py`, `check_frame_joint_ddp.py`,
  `check_frame_joint_continuation.py`, `profile_frame_joint.py`,
  `evaluate_frame_joint_tiny.py`：可重复验证入口。
- 人类审阅入口：`docs/frame_joint_v1.md`；`tools/inspect_model.py --kind frame_joint`。

真实 inspector：15,554,664 total parameters，14,862,240 trainable；history
429,088、DiT 13,429,120、decoder 1,004,032，teacher 0。H8/887-atom
forward 的所有三组 trainable component 都有非零梯度，且无 trainable parameter 缺梯度。

## CUDA 与恢复检查

证据目录：`runs/frame_joint_v1_cuda_260920/`。

- teacher/rotation/origin：h/v identity 差 `6.68e-6` / `2.38e-5`；旋转误差
  `5.72e-6` / `3.05e-5`；origin translation check 为 0；teacher 三步后 state 未变。
- history：重复静态 detail 最大值 0；T=1 通过；不规则 time/mask finite。
- future mutation：替换 evaluator 真实未来标签后生成最大差 0。
- forced `s=0.95`：generated-bond raw `0.60396`，DiT gradient norm
  `0.39210`；near endpoint 对 DiT gradient norm 为 0。
- bond release：generated/clean/near 三个 weighted bond 均为 0；与物理删除这些项的总梯度最大差 0。
- 同一两样本 global batch：单卡 loss `85.11075`；启用阶段所需的 DDP unused-parameter
  检测后，双卡参数 update 与单卡最大差 `1.19e-7`（阈值 `2e-6`）。双卡 checkpoint
  的 model、AdamW moments 和 per-rank generator resume 最大差均为 0。证据：
  `ddp_check.json`。
- 独立双卡 stage smoke 完成 3 个 decoder-warmup 更新并跨入第 1 个 flow-start 更新，
  未发生 reducer hang；checkpoint：
  `runs/frame_joint_v1_ddp_stage_smoke_final_260920/frame_joint_step_00000004.pt`，SHA-256
  `d9bad8da09001ca8432b4d1add7569e75c2a61ca4dfe2caceeb2aeda8892e549`。
- 审阅前普通单卡 resume：从 corrected step 9 重跑到 step 12，与当时 uninterrupted
  checkpoint 的 model/optimizer 最大差 0，cursor/generator 相同。旧 checkpoint 没有
  新 schedule contract，当前代码不会兼容恢复它。
- 审阅前 paired continuation：同 parent 起始 model/optimizer 最大差 0，cursor、起止 generator、
  H 和 flow-time (`0.33941123`) 相同；release 三个显式 bond weighted loss 全 0。
  证据：`continuation_check.json`。该单步的 s 未过 generated/near 阈值；它们的非零梯度归因由上面的 forced-s 检查覆盖。
- 初始实现的 18 个 retained targeted tests 通过；审阅修复增加零方差/过短动态指标
  回归后，同一组 `test_data_sampling.py`、`test_geometry.py`、`test_imports.py`、
  `test_runtime_config.py` 当前为 `20 passed in 7.74s`。

48-system 真最大 train sample 为 `atlas_4k2m_A_R1_w000000`，3542 atoms、T16、H8。
四帧 teacher chunk 后的 warm profile：teacher prepare 1.166 s / 14.34 GB peak；history
0.072 s；DiT 0.097 s；decoder 0.046 s；完整 joint forward 0.495 s / 36.13 GB；
loss 0.076 s；backward 0.480 s / 本次全流程最高 36.51 GB；optimizer 0.076 s。
各段独立同步计时，不能相加解释为重叠 pipeline 时间。证据：`profile_max_system.json`。

## 审阅修复证据

新证据目录：`runs/frame_joint_v1_review_fixes_260920/`。

- 真实 887-atom/T16 sample、完整 256/128/depth-4 规格完成 BF16 decoder forward
  和 3 个 joint updates；第一步两层 local gate 梯度范数为 `0.005388/0.005294`，
  第二步 message 梯度范数为 `3.857e-6/3.733e-6`。
- threshold-asymmetric DDP 使用 global flow time `[0.95, 0.50]`；单卡/双卡更新
  最大差 `2.593e-8`，model/optimizer/generator strict resume 通过。
- continuation child checkpoint SHA
  `5979faf368cf159ed96703bbaadd50059c1d83102f04d36fb6329ed8d2c86f77`；
  trainer resume 差为 0，parent identity 保持，改变 H 顺序会被 contract 拒绝。
  真实 CLI 从 child step 1 恢复到 step 2 的 checkpoint SHA 为
  `3a40dfdfaf5e214ee9233cb841605eb1b176641a6a08c0afa1d06363df254b53`。
- 3/9 tiny evaluator-only v2 重跑在
  `runs/frame_joint_v1_review_fixes_260920/tiny_evaluation_schema_v2/`；JSON SHA
  `ce594800c036a03966206e5c713c675454efd48a130bb9e0c0a88b933060fa53`。
  row-level availability 和聚合 available counts 已写出；本批窗口均有效，数值未变。

完整逐项判定及不实施的 API/架构建议见 `REVIEW.md` 的“修复回应”。

## 审阅前 corrected tiny

- config：`configs/frame_joint_v1_tiny.yaml`
- run：`runs/frame_joint_v1_tiny_corrected_260920`
- final checkpoint：`frame_joint_step_00000012.pt`
- SHA-256：`c7e18d55a7d5e8bebf29c2dbee31470bcd9dfe5ec73003bb03919ef522622c7e`
- 3 decoder-warmup + 3 flow-start + 6 joint successful updates；H4/H8 交替；全部
  9 trajectories 有曝光。final loss 184.5595，其中 flow 1.03725、clean coordinate
  183.4373、clean bond raw 0.84984。tiny 随机 joint steps 未抽到 generated/near
  threshold，故使用 forced-s CUDA 检查补足该路径。
- `runs/frame_joint_v1_tiny_260920` 和 `frame_joint_v1_tiny_final_260920` 是发现
  coordinate-head optimizer 缺口前的保留证据，不能用作 parent 或正式 tiny 结果。
- 上述 corrected step-12 自身也早于 decoder local-message 初始化修复；它可以继续
  做固定 inference/evaluation artifact，但不能作为新训练 parent 或 local topology
  已生效的证据。本轮按要求没有启动新的 tiny 训练。

固定评估：每条原 valid trajectory 的首个 window（共 9），H4/H8，Euler 16，seed 0，
无 best-of-N，按 replica 后 system 等权。最终源码重跑的真实坐标保存在
`evaluation_current/tiny_coordinates.npz`（SHA-256
`f6affcd9bc8bd33e186aaaf5b6c2875f489fbdbbf30553cc6a7ba2fac6bed847`），完整指标在
`evaluation_current/tiny_evaluation.json`（SHA-256
`8f896f1084c706ca516d3cd52817221c20b669739ca2320c1e6c98b270d6988c`）。原
`evaluation/` 是 teacher 分块优化前的保留结果；最终源码对其首条 H4 坐标最大差
`5.01e-6 Å`，表中汇总值在所示精度不变。

| H | 路径 | aligned RMSD Å | bond RMSE Å | RMSF abs err Å | RMSF corr | velocity RMSE | dynamic corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | generated | 16.9560 | 5.6484 | 3.8778 | 0.0708 | 0.04011 | -0.00252 |
| 8 | generated | 16.9726 | 5.6572 | 3.7795 | 0.0532 | 0.04009 | -0.00215 |
| 4 | clean-latent decoder oracle | 16.2225 | 0.8581 | 0.4642 | 0.5708 | 0.00775 | -0.37408 |
| 8 | clean-latent decoder oracle | 16.2653 | 0.8595 | 0.4587 | 0.5297 | 0.00776 | -0.36198 |

最终源码重跑的生成-only CUDA time 23.789 s；7.567 future frames/s、
12,331 future atom-frames/s。
这只是 12-update pipeline smoke：生成和 clean-oracle 均未科学收敛，不能声称 production parity。

## 可复制命令

以下均在 `enter-container` / `torch-ito` 中执行。

```bash
# tiny train
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m molvid.cli.train_frame_joint \
  --config configs/frame_joint_v1_tiny.yaml --checkpoint-every 3

# tiny eval
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
python tools/evaluate_frame_joint_tiny.py \
  --checkpoint runs/frame_joint_v1_tiny_corrected_260920/frame_joint_step_00000012.pt \
  --checkpoint-sha256 c7e18d55a7d5e8bebf29c2dbee31470bcd9dfe5ec73003bb03919ef522622c7e \
  --codec /data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt \
  --codec-sha256 ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df \
  --valid-store /data4/users/sihao/workspace/PVB/outputs/atlas_selected_trajectories/clip_store/valid \
  --output runs/frame_joint_v1_tiny_corrected_260920/evaluation_current \
  --steps 16 --seed 0

# 48 statistics then documented two-GPU pilot; do not run before TRAIN_PROMPT
CUDA_VISIBLE_DEVICES=0 python -m molvid.cli.train_frame_joint \
  --config configs/frame_joint_v1.yaml --fit-statistics-only
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m molvid.cli.train_frame_joint --config configs/frame_joint_v1.yaml \
  --checkpoint-every 1358

# paired continuation, both commands use the exact same future parent/SHA
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m molvid.cli.train_frame_joint --config configs/frame_joint_v1_bond_keep.yaml \
  --continuation-parent PARENT.pt --continuation-parent-sha256 PARENT_SHA \
  --checkpoint-every 1358
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m molvid.cli.train_frame_joint --config configs/frame_joint_v1_bond_release.yaml \
  --continuation-parent PARENT.pt --continuation-parent-sha256 PARENT_SHA \
  --checkpoint-every 1358
```

## 未执行与限制

- 未启动 48 pilot、frozen-decoder 对照、bond 两臂长 continuation、192 主训练或
  multi-time；没有长训 PID/job。
- 48 的 generated-bond 16-batch gradient calibration 已实现，但尚未运行；实际 λb
  必须由长训 flow-start parent 现场生成，不能用 tiny 的 0.01 代替。
- 48 statistics 尚未拟合，因此 `configs/frame_joint_v1.yaml` 中 stats SHA/hash 留空，
  先运行 `--fit-statistics-only`；训练 loader 仍校验 data/codec provenance。
- 192 配置/实际 batch schedule 仍应在 TRAIN_PROMPT 阶段按 192 manifest 和实测吞吐解析，
  不能把 48 配置伪装成 192 已就绪结果。
- 当前 tiny 和最大体系 profile 使用 FP32 correctness 路径；BF16 长训吞吐尚未实测。

## 训练记录（2026-09-21）

- 工作树 commit 记录为 `93d191fd8c88c2355944c1cf8bf05e8519ddb598`，工作树含本轮实现修改，未 push。
- 48 pilot 命令使用 `configs/frame_joint_v1_48_pilot_260920.yaml`，2 GPU、13222 updates，
  终点 `runs/frame_joint_v1_48_pilot_260920/frame_joint_step_00013222.pt`，
  SHA256 `e615b36a600d56b69e004305695fa804509ecb5591d2dcb10643a63d698759fb`。
  statistics SHA256 `2a0093ba1e85188b69b1bb0e3f450d9e278b3ab5ed71ee48de8431d5f8c9abfa`；
  calibrated generated-bond weight `0.0690296888`，16 calibration batches，全部训练指标 finite。
- frozen-decoder control 从 joint-start parent（5075 updates，SHA256
  `a00e9a3a14a8664bca14c2d47bf861732cb581e6d5573fd9cc3cc05b0c6dd8a8`）fork，2716 updates；
  终点 SHA256 `84ad9aa03cf41ab50d53df0feefe390a0ed5f70f2fbbc6fe9f643657988d93e0`。
- bond keep/release 都从同一个 pilot parent fork，各 6790 updates。keep 终点 SHA256
  `505ecd2d0a820d53a503ee1e7a5a58982811d0fd308c155562ddc6cbf00f32b5`；
  release 终点 SHA256 `33fc22e08b85f2f420aecb2b4f76602d95abe55b9638d6c64a858056ca52fa62`。
  release 日志逐条显示三个 bond weighted 项为 0。
- pilot、frozen-control、keep、release 均在原 3-system/9-trajectory valid tiny 上完成 H4/H8、
  16-step evaluation；结果保存在各自 `evaluation_tiny/`，动态/RMSF availability 使用 schema v2。
  tiny 结果只作固定协议对照，不声称 production parity。
- 192 capacity manifest 的 `train192`/valid frozen views 已加入严格校验；statistics SHA256
  `3ce928ad96ee79096872b719f3dc07a64c0d5c0322835ada7318ef21805d1c85`，
  statistics hash `5f8e85833a89acd4a09b8d715b00824ad644b9654fb97075e394dbd7dd149fe7`。
  2-GPU sampler 测得 5313 updates/effective epoch；new-B 主臂已启动，旧 A 无可审计实现/起点，
  不伪造 A 结果。multi-time manifest（100/200/400 train、300 valid）本轮未找到，未启动。
  目前 new-B 已完成 warmup+flow-start，运行到 joint step 18378/123199；warmup checkpoint
  `frame_joint_step_00015939.pt` SHA256 `1bd3eb117cc25707643b6a695cf57066b21e2defb669b53d25c3d66c2a8017f0`，
  进程仍在运行，未宣称主臂完成。
- `tools/evaluate_frame_joint_tiny.py` 现支持 `--windows-per-trajectory` 与期望 system/trajectory
  数；原 tiny 默认保持不变。48 valid quick 已按 8 systems/24 trajectories、2 windows/trajectory、
  H4/H8、16 Euler steps、seed0 完成，结果写入各 run 的 `evaluation_valid_quick/`。生成均值
  （aligned RMSD / bond RMSE / contact F1 / velocity RMSE / dynamic correlation / RMSF AE）：
  pilot H4 `2.313609/0.543506/0.765578/0.011841/0.000939/0.266333`，H8 `2.228858/0.597614/0.754301/0.012056/0.000199/0.203083`；
  keep H4 `2.349169/0.489324/0.752703/0.012019/0.000591/0.217967`，H8 `2.276970/0.536420/0.740900/0.012177/0.000153/0.187108`；
  release H4 `2.405827/0.569225/0.737704/0.012466/0.000296/0.206515`，H8 `2.350840/0.621700/0.724954/0.012648/-0.000103/0.215500`；
  frozen-control H4 `2.501975/1.192724/0.673716/0.013413/0.001561/0.255362`，H8 `2.406604/1.271273/0.659028/0.013522/-0.001105/0.293681`。
  这些是短预算点的同批系统对照，不作收敛或因果赢家结论。


## 多时间训练启动记录（2026-09-21，用户更正后）

此前“multi-time 未找到数据、未启动”的判断已撤回：原始 ATLAS 数据可读，且已从原始轨迹补齐 200/400 ps 训练 bucket。数据构建使用 `/data1/repo/BioKinema/data_atlas`，沿用既有 100 ps `train192` payload；原生帧间隔为 10 ps，故 `dt_100ps` 使用 stride 10、`dt_200ps` 使用 stride 20、`dt_400ps` 使用 stride 40；valid 使用 300 ps、stride 30。没有打开 test。

- 数据根：`runs/frame_joint_v1_multitime_data_260921`
- train：100/200/400 ps 分别 35,712 / 17,856 / 8,640 clips；合计 62,208；valid 300 ps 480 clips。
- 数据 hash：`54bbdc198373ae9f477bf5803643aab77f960fe4498da8db1fcf9c5bfdd770c0`；manifest SHA `5664cd9fc9f1a328427e5732f2ce10751fa4dc9e7295ac28d97236b2d05e4233`。
- 采样：同物理 trajectory 跨 dt bucket 共享 cap，每 trajectory 每 epoch 24 clips，bucket 权重各 1/3；4 GPU 每 effective epoch 2,659 synchronized updates。阶段为 warmup 7,977、flow-start 1,000、joint 13,295 updates。
- frame statistics：由 4 个完整索引 shard 并行冻结 codec 拟合后精确按 atom-frame count 合并；file SHA `117dbcb674f9932c40216c4e2c3ee69153dacc3d3efebcca6e67c876a3b6bee9`，statistics hash `7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`，atom-frame count 1,661,186,880。
- config：`configs/frame_joint_v1_multitime_260921.yaml`。
- 正式命令（实际已启动）：
  `CUDA_VISIBLE_DEVICES=1,3,4,7 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. torchrun --standalone --nproc-per-node=4 -m molvid.cli.train_frame_joint --config configs/frame_joint_v1_multitime_260921.yaml --checkpoint-every 2659`
- 输出：`runs/frame_joint_v1_multitime_260921`；torchrun PID `260241`，4 个 worker 已存活。预检成功，正式训练已写入 `train_metrics.jsonl`，初始 step 62，sample 中已出现 `dt_200ps`，loss/grad 均 finite。
- 同时仍在运行的 192 新 B 使用 GPU 0/2；GPU 5/6 为其他既有作业，未触碰。当前本机 8 张卡均有作业，没有空卡可再启动独立训练。192 旧 A 没有可审计的合法起点/实现，因此未伪造启动；100/200/400 本轮按批准的 multi-time 混合配置作为一个 4-GPU run 启动，而不是偷偷改成三套不同 bond policy。


## 多时间完成状态补记（2026-09-21）

multi-time 4-GPU run 已正常完成全部 22,272 updates（warmup 7,977、flow-start 1,000、joint 13,295），`train_metrics.jsonl` 22,272 行且 loss/flow/grad 全部 finite。最终 checkpoint：`runs/frame_joint_v1_multitime_260921/frame_joint_step_00022272.pt`，SHA-256 `b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`。采样记录中 dt100/dt200/dt400 sample counts 为 9,552/9,222/10,409；bond calibration weight `0.05170724987983704`。当前尚未对该 checkpoint 执行 held-out 生成评估，因此只能总结训练收敛曲线和 bucket/exposure，不能把训练 loss 当作运动学结果。


## 192B 运行状态快照（2026-09-21）

192 new-B 仍由 torchrun PID `253811` 在 GPU 0/2 正常运行，无 NaN/Inf/OOM。当前约 step `66670/123199`（54.1%），处于 joint 阶段（joint 已完成 49,731/106,260 updates）；最新已保存 checkpoint 为 step `63756`，SHA-256 `facc7715c695e8d787d31ac37e0606e43eedb7497c91284826196e91651c53d6`。最近 5,000 steps 平均 loss `0.1211`、flow `0.1198`、clean-coordinate `6.97e-4`、grad norm `1.91`。此前 checkpoint 间隔约 1 h 42 min/5,313 updates，剩余时间按当前吞吐粗估约 18 h，仅作运行安排不作完成承诺；尚未做 held-out eval。
