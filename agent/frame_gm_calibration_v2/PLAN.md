# 已确认的三阶段计划

## 阅读次序

1. 本文件；`ARCHITECTURE.md`；`DATA_TRAIN_EVAL.md`。
2. `REVIEW_PROTOCOL.md`；`TASKS.md`；`experiment_spec.json`。
3. 对照源码：`molvid/latent/types.py`、`codec/frame.py`、`codec/history.py`、`dit/frame.py`、`dit/blocks.py`、`dit/backend.py`、`codec/decoder.py`、`model.py`。
4. 再读 `flow/objective.py`、`flow/source.py`、`flow/sampling.py`、`training/batches.py`、`training/joint.py`、`cli/train_frame_joint.py`、`evaluation/motion.py`、`time.py`。
5. 评估入口：`tools/evaluate_frame_joint_multitime.py`、`repair_frame_joint_multitime_rmsf.py`、`build_frame_joint_multitime_eval_views.py`、`summarize_frame_joint_multitime.py`。保留旧工具历史行为，新的主逻辑收拢到可读模块。
6. 参考 `configs/frame_joint_v1_multitime_260921.yaml` 与 `agent/frame_joint_v1/HANDOFF.md` 最新追加段落；旧 HANDOFF 中早期状态已被后续记录更新。

## 当前真实起点

- repo：SHw-1313/molvid-recon；审阅 commit `f49cf27efa60a0cca172248c49913d70ff1516a5`。
- active package：根 `molvid/`；任务目录惯例是单数 `agent/`。
- Frame Joint v1：冻结逐帧 target teacher；训练 history encoder、DiT、decoder；H4/H8 在 16 帧内；history R4，future per-frame h/v 联合生成。
- DiT：scalar 256 / vector 128 / depth 4 / heads 8。target h/v 通道 128。
- parent：`runs/frame_joint_v1_multitime_260921/frame_joint_step_00022272.pt`。
- parent SHA：`b809c988e5257b722c84d64219123721764db424d5a0b2ba0e51bf9067c11cb4`。
- codec SHA：`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`。
- statistics hash：`7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`。
- train：192 体系，100/200/400 ps；300 ps 是未训练 lag。已存在混合时间训练，不能当成从未加入 time embedding。
- 已有 decoder 联合训练。但真实源分布全程采样并反传尚不是当前 joint loss 的路径。

## 阶段与交付

| 阶段 | 主要工作 | 独立审阅 | 完成后动作 |
|---|---|---|---|
| P1 | 指标/对齐/时间视图、latent 诊断 | 简单任务无需 | 写诊断；继续 P2 |
| P2 | G/M 四臂实现、pilot | pilot 后审阅；修复后必要复核 | 四臂正式训练与评估，选择 P3 起点 |
| P3 | 实际生成 latent + 分布校准、双臂 pilot | pilot 后审阅；修复后必要复核 | 双臂训练评估；按条件补 bond 分叉；交付 |

科学负结果不是软件错误。若 G/M 没有胜出，P3 可在 B0 上继续验证训练分布问题；不因此强行改 decoder 参数化或增加噪声先验。

## 可读代码组织

复用：FrameLatentBatch / ObservedContext / QuerySpec、frozen teacher、normalizer、history R4、当前 joint future DiT、decoder、训练与 checkpoint 框架。

建议新增的窄职责模块（名称可合理微调，避免空壳层）：

- `molvid/dit/local_geometry.py`：稀疏拓扑消息和原始向量不变量。
- `molvid/codec/motion_context.py`：仅由 observed history 构造的运动摘要。
- `molvid/losses/motion_distribution.py`：特征提取、有效 mask、energy score。
- `molvid/training/sampled_trajectory.py`：小步数可微源采样；可放 flow/sampling 中复用公共步进核。
- `molvid/evaluation/temporal.py`：物理时间与公共对齐参照的动力学指标。

修改已有 model / trainer / time / batch / sampler / evaluator 接口；不要复制一整套 model_v2。B0/G/M/GM 用显式配置控制两项功能，公共逻辑只写一处。P3 的训练专用采样保留显式 autograd 边界，禁止在 CLI 里实现模型算法。

不做：geometry encoder sweep、未来块 latent/inverse Haar、VAE、条件/相关源噪声、内坐标 decoder、长 rollout、AF3、静态大数据或 scaling。P1 可诊断这些动机，但实现留到下一次授权。
