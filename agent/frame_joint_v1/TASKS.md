# 实现与验证清单

初始状态：此文件是待执行计划，不表示任何代码已经实现或训练通过。

## P0：读代码与解析产物

- [x] 记录实际 HEAD、分支、工作区与 CUDA/容器入口；保留用户未提交修改。
- [x] 定位指定 codec，校验完整 SHA，记录 Haar 前 encoder/stem 提取路径。
- [x] 定位原 3/9、48/8、候选 192 的实际 manifest；test 保持封存。
- [x] 按 ROOT_AGENTS_APPEND 合并长期规则；列出本轮改动文件的职责。

## P1：最小闭合数据流

- [x] FrameLatent / ObservedContext / QuerySpec、frozen teacher 与新 statistics。
- [x] 真实时间模块、zero-preserving history encoder。
- [x] 新 adapter、history-conditioned Frame DiT、联合 decoder。
- [x] 逐帧 flow/source/Euler 采样，observed-only 生成入口。
- [x] 核心 API 支持可变 H/Q；16 只是当前数据配置。T=1 静态适用路径可用。

## P2：训练与可恢复实验

- [x] 明确冻结和可训练参数，history 不在旧 no_grad 准备器内。
- [x] 实现预热/flow起步/joint 的短而清楚的阶段控制，不重写为庞大实验框架。
- [x] loss 分量与统一 bond 开关；release 时全部显式 bond 权重为零。
- [x] joint checkpoint、普通 resume、同 parent 成对 continuation；源 codec 仅 warm start。
- [x] DDP 覆盖 history/DiT/decoder，rank 数据/RNG/保存正确。
- [x] actual CLI 配置与启动命令；`experiment_spec.json` 不能直接伪装成已可执行配置。

## P3：先必要 CUDA 检查，再 tiny

只检查本次改动的真实风险，不先恢复旧全量 pytest 或追求测试数量。

- [x] CUDA frozen teacher 特征与指定旧 codec Haar 前输出一致；teacher 无梯度且不更新。
- [x] 几何旋转/平移与 vector xyz 约定；改变 t 不破坏等变性。
- [x] 重复静态轨迹在多 dt 下 history detail 为零；不规则时间 mask 无 NaN。
- [x] 推理 future-mutation：相同 observed/query/noise，替换 evaluator 的真实未来标签不改变生成。不要要求训练 z_s 在改变 target 后仍相同。
- [x] 小 forward/backward：正确模块得到梯度；zero-init 残差允许前几步部分上游梯度为零，不能误判整套 history 永远可训练。
- [x] 两臂 bond switch 梯度核验；同 parent 单步 continuation 能复现数据、H、s、noise。
- [x] 同一小 global batch 的 1/2 GPU 更新与 resume 检查；只在有第二张授权 GPU 时执行双卡项，否则如实待执行，不能宣称多卡可用。
- [x] 最大体系 warm CUDA step profile：encoder、DiT、decoder、backward、optimizer、显存；不相加重叠计时，不只测空 forward。
- [x] 原 3/9 tiny 重建和生成；预算见执行入口。输出真实坐标、几何、运动与吞吐摘要。

## P4：可读性与交付

- [x] 核对 READABILITY；CLI 无模型业务、trainer 无数据解析大段、无万能 ratio 分支。
- [x] 更新 inspect_model 与 `docs/frame_joint_v1.md`，真实参数/shape/梯度说明。
- [x] 保存 resolved 配置、启动/恢复/eval 命令、路径/hash、预算与当前未完成项。
- [x] 自审 diff，更新 HANDOFF。独立 review 可使用 REVIEW_PROMPT；先修科学正确性问题，再启动长训。
- [x] 逐项核实独立 REVIEW R1--R6；完成必要修复及 BF16、local-gradient、
  threshold-asymmetric DDP、child CLI resume/schedule rejection、不可用运动指标检查。
- [x] 保留审阅前 tiny 作为 inference/evaluation artifact，但明确禁止把它当作
  修复后 local topology 证据或新训练 parent；未擅自重跑训练。

## P5：收到 TRAIN_PROMPT 后

- [ ] 48 pilot 与短 frozen-decoder 对照；已有 tiny/检查不重复全跑。
- [ ] 从固定 joint parent 运行配对 bond_keep / bond_release；保留 parent+两臂完整评估。
- [ ] 192 主训练 A/B；是否先于 D 由已完成 parent 与资源安排决定，不改变 D 的数据/loss 对照语义。
- [ ] 完成已定义评估与短 rollout；记录 tradeoff，不自动把运动变小列为成功。
- [x] multi-time 100/200/400 ps 混合训练已启动；长训完成与评估仍待执行，不扩充成参数扫参。
