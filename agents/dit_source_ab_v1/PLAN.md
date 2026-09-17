# 实现计划

## 本轮问题

已有证据指向 state 生成及其解码后的几何失真：R4 深诊断中 generated-state/generated-detail 的 bond RMSE 约2.28 Å，oracle-state/generated-detail 约0.39 Å，generated-state/oracle-detail 仍约2.23 Å。

B 后端已经大幅减少计算开销。本轮在保持相同模型的前提下，比较 future 的 Gaussian source 与 observation-centered source；通过更多有效训练和共同生成评估判断改善。

## 文件与改动边界

| 文件 | 本轮改动 |
|---|---|
| `evaluation/dit_diagnostics.py` | 从 A 引入，修正聚合、适用性和扰动统计；支持新 source 的评估 |
| `scripts/evaluate_dit_pilot_diagnostics.py` | 从 A 引入，显式 backend、按 draw/steps 保存、启用缺失基线 |
| `config/dit_pilot_diagnostics_v1.yaml`、`tests/test_dit_diagnostics.py` | 从 A 引入，必要适配；旧输出不覆盖 |
| `scripts/profile_dit_backend.py`、`tests/test_dit_backend_v2.py` | 真正的 parity/finite 断言、相同 RNG、真实 wall timing |
| `module/dit_backend_v2.py`、`module/molecular_dit.py` | 仅必要 backend 正确性修复；不新增网络层或改变 attention 语义 |
| `module/latent_flow_source.py`（新） | observed-only 中心构造、source 合同、Gaussian/conditional 两模式 |
| `module/latent_rectified_flow.py` | 可选 source 输入/策略，共享训练与 sampler 路径；默认 Gaussian 语义保持 |
| `trainer/dit_trainer.py` | 受控接入 source 和新 phase 步数上限、checkpoint/resume 合同；旧 phase 限制保持 |
| `data/dit_latent_cache.py`（新） | 有界缓存与原批次顺序还原；先测代价，不要求全库缓存 |
| `scripts/run_dit_source_ab.py`（新） | 新阶段 CLI、两臂调度、预算/恢复/评测，复用现有数据 loader |
| `config/dit_source_ab_v1.yaml`（新） | 本轮固定合同与实测预算入口 |
| `tests/test_dit_source_ab.py`（新） | source/mask/归一化、CUDA、checkpoint/resume、cache 关键检查 |
| 根 `AGENTS.md` | 只追加本轮阶段说明 |

优先复用现有函数，不复制整个 trainer/evaluator。若需要小型 helper，可在上述新模块内实现并记录。旧 codec、encoder、decoder、生产 statistics、旧 codec evaluator 默认不改；需要新指标时放在新诊断层，不重写历史指标结果。

## P0：工作区与输入

确认现有 B 分支，记录 HEAD 和工作区差异；选择性导入 A 的四个文件，记录源 SHA 与文件 hash。不得覆盖与本轮无关的用户修改。

冻结输入见 EXPERIMENT。缺文件时先按已知 worktree/container 映射查找。确认缺失后报告具体文件，不能替换成随机 codec 或另一训练 seed。

## P1：整合与修复

依照 FIXES 修正已知问题，并为新 runner 接通 `execution_backend=factorized_v2`。reference 仅用于等价检查，旧 CLI 默认行为可以保留，但新阶段的默认和 resolved config 必须是 v2。

metadata 准备与 source 坐标计算分开：`codec.prepare_batch(cpu_batch)` 只注册拓扑；GPU 上生成的 template coordinates 应直接替换 CUDA batch 的 x 后编码，不能像 A 的旧 `_encode_coordinates` 一样将整段 GPU 坐标搬回 CPU 再上传。

每批 target codec encode 至多一次；source 中心额外编码必须单列计时，按 sample/H 复用，不在每个 Euler step 重算。

## P2：必要验证

按 ACCEPTANCE 执行。数值检查通过后，用同一个旧 R4 checkpoint、相同 sample/H/seed/16步，比较 reference 和 v2 的最终 raw latent、解码坐标和科学指标；同时确认 observed clamp。

这些是可丢弃的验证，不把 smoke 权重、optimizer、缓存顺序带入科学训练。

## P3：source 中心、缓存和共同预算

先实现 EXPERIMENT 中两种中心的 observed-only 构造，用固定8 clips 比较。优先 `repeat_last_coordinate_encode`；实现正确性条件通过后自动采用，不等待额外许可。

记录两种中心的模板重建、静止性、相对最后观测坐标的边界、对真实未来的指标。选择不得用未来 RMSD 的较低者决定。

测量编码、中心准备、缓存、H2D、模型训练与生成评估的真实 wall time。采用 EXPERIMENT 的预算公式，先冻结共同终点再训练。若缓存不划算或容量不足，可使用明确的在线路径；记录决定，不为缓存单独扩成新工程。

## P4：两臂训练

新 runner 允许从同一 init state_dict 开始两条独立训练进程。可一臂一卡或同卡分阶段轮流执行；显式验证初始权重、batch schedule、H schedule、tau/noise 计划一致。

训练范围仅 R4/48 train/H4/H8，最多一对 seed、20000 steps/arm。4500步不是停止门槛；通过验证后继续到冻结的共同终点。EXPERIMENT 定义时间和资源上限。

科学上效果差时仍按共同预算运行并如实汇报；发生 NaN/OOM/错误 source/数据泄漏等实现故障时保存状态并修复，不能继续产生无效结果。改变 batch size、优化配方或 source 定义必须建立新 experiment ID；不拼接不兼容训练曲线。

## P5：汇总与交付

对共同固定步数 checkpoint 做主要比较；各臂 best-RF checkpoint 可另附，不能拿不等训练量的 best 点作唯一结论。

交付内容：执行合同、修复检查、source 选择依据、profile、缓存代价、训练曲线、四 field loss、相同预算生成指标、逐体系差异、耗时与后续判断。保留 raw rows 便于重聚合。

只在 `agents/dit_source_ab_v1/evidence/` 提交小型 JSON/Markdown；大数据在本轮独立 outputs。更新 TASKS/HANDOFF/DECISIONS，本地提交后停止。后续几何监督、局部原子交互、R2训练或多时间尺度实验不在本轮自动执行。
