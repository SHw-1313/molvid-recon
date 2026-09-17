# 已确认问题与最小修复

以下来自 A `da5bdbb` 和 B `557a904` 的代码审阅。先在当前 B 核对，若已有等价修复就记录并复用，不重复改动。

## F1：真实 checkpoint 的 PASS 必须由断言产生

`scripts/profile_dit_backend.py::_equivalence` 当前计算 forward/gradient/update 误差，却直接写 PASS。改为逐 field、loss、逐参数 gradient/update 的有限性与容差断言；任一失败写 FAIL、返回非零退出码，不输出成功状态。

- FP32 禁 TF32：使用元素级 `abs(a-b) <= 2e-5 + 2e-4*abs(reference)`，同时报告 max_abs/relative_L2。
- 不把 max_abs 单独与 atol 比较后误判：B 的 R2 max_abs=2.77e-5 本身不能证明违反上述组合容差。
- 梯度缺失模式必须一致，不能只对两边都非 None 的参数做交集比较。
- 比较真实 RF loss、所有参与参数梯度和同 optimizer 单步更新。optimizer 的初始状态、权重、输入、tau/noise 相同。
- BF16 初始容差 atol=2e-2、rtol=5e-2。真实权重存在非零 attention modulation；不能只用 AdaLN-zero 初始模型证明等价。
- smoke 的 finite/clamp 也必须真正断言，不能只写布尔字段然后无条件 PASS。

已通过的小型 CUDA 测试可复用；补真实 R2/R4 median cases，不重跑整轮科学训练。

## F2：性能口径

- `_run_profile` 的 seed 当前含 backend 名称，删除该差异；两种 backend 使用同 weights、optimizer 起点和 tau/noise 序列。
- median case 补一次相同 RNG 的 reference/v2 对比，5 warmup+20 measured 即可。reference/v2 串行，避免另一模型占显存；不重做所有历史 profile。
- 将旧 `median_compute + one_time_encode` 标为估算，新增实际多 batch wall timing。外层 perf_counter 加必要 CUDA 同步，包含数据读取、metadata/layout、H2D、target encode、source 准备、forward/backward/optimizer 与日志。
- 预载 compute-only 和端到端分别报告；参数哈希用真实 tensor 字节 hash，不能把 key/shape hash 称为权重 hash。
- 记录至少一个2–3步短 profiler trace，确认 grouping/pooling/attention/FFN/encoder/同步的真实耗时。trace 时间不计 steady throughput；没有 trace 不填算子占比。
- small/median/large 目前按总 N 选，未覆盖大分子长尾。报告这个范围，本轮不扩展成 scaling benchmark。

## F3：source 判断与缺失基线

`_summary_report` 的 `reference_only_not_deployable` 当前是固定文字，不是计算结果。将其拆成：是否可只用 observed 构造、模板几何、边界、是否作为本轮 source 中心；附对应指标。

删除“DiT 比 persistence 差，所以 persistence 不能作 source”这一推理。persistence 不是有效动力学生成器，与它能否当 source 均值是两个问题。

`coordinate_repeat_reencoded` 虽有实现，但 main 禁用且 deep 未调用。启用在固定8个 deep clips 上的该基线，并额外在 source_check 比较模板本身。保留 `block_state_zero_detail` 对照。

## F4：深诊断保存和分组

- 保存所有生成 per-sample/per-H/per-steps/per-draw rows；原 deep 只聚合后丢弃了生成行。
- 同一个 sample/H/draw 的8/16步必须从同一 eps 开始。按 steps 单独汇总，不把8/16步合成一个主性能数字。
- perturbation 按 scope、scale、H 分开，field-swap 按 method、H、steps、draw 分开。
- 先在唯一 sample 内平均 draws，再平均 samples，再平均 systems；返回 unique_sample_count、draw_count、row_count，不能把64 rows写作64个独立样本。
- 保留每体系结果；主比较不取 best-of-N。
- 原始 JSON/输出不覆盖，修正结果写新 run_id 和 diagnostics schema/version。

## F5：零运动与指标适用性

A 复用了旧 codec evaluator 的 `_safe_correlation`，使零速度自相关出现1。新诊断层对零/近零方差信号返回 null 和 applicability_reason；阈值注明单位与 FP32 精度，不能用加 epsilon 后的任意数值冒充有效 ACF。

- RMSF 的实际值0仍是有效结果；RMSF Pearson 在退化信号上可能无定义，应分开判断。
- near-zero 可采用固定坐标波动 RMS <=1e-5 Å 的数值阈值，速度按真实 dt 换算，冻结在协议中，不看结果后调整。
- `dynamic_correlation` 是生成与目标速度逐点相关；另列 velocity-lag1 ACF prediction/target，不混称。
- 旧 `contact_occupancy_mae` 实际是逐帧 contact disagreement 比例。保留 legacy 名称说明，并新增真正的 occupancy MAE：每个非共价 pair 先跨有效未来帧平均 contact indicator，再对预测/目标 occupancy 的绝对差做 pair 平均。相同 cutoff=4.5 Å，排除共价 bond，时间/mask口径显式记录。
- 无 torsion index 时 torsion 指标为 null，不将旧0当作扭转准确证据。现有 clash cutoff 如沿用则注明，不作为本轮主要选择依据。
- 上述实现位于新诊断层，历史 codec evaluator 的默认结果保持可复现。

## F6：噪声幅度和 latent 统计

`perturb_normalized_oracle` 的 actual_noise_rms 分母当前只计算有效 token 数，漏掉 C/3C。使用完整 broadcast 后有效元素数，或先 masked_select 再 RMS。scale=0.1 的每系数扰动 RMS 应约0.1，不是标量场约1.13/向量场约1.96。

已有扰动生成数学操作不因此变更；修的是幅度报告。复用能重建的旧证据，只重跑缺失必要行。

latent 范数统计新增 future-only 版本，避免干净 observed fields 稀释生成误差；向量统计用旋转不变 norm/Gram，不减 xyz 均值，不把单通道 RMS 当作严格流形距离。

## F7：新实验接线

- 新 trainer、evaluator、resume 和 sampler 全部明确 `factorized_v2`。旧模型默认 reference 不算新实验接通成功。
- 为 `source_ab_v1` 定义明确的最多20000步 phase 分支；保留旧 probe<=100、旧 t1_pilot<=5000 的限制。不要仅删除所有步数验证。
- 禁止通过 `trainer.flow` 临时替换但不写 checkpoint 合同的方式实现新任务。source_mode、center_kind、sigma、normalization_hash、init_hash、schedule、预算必须可恢复且经过兼容检查。
- Gaussian 模式与旧 RF 目标/采样默认路径一致；新 schema 不能被旧 pilot loader 静默解释成 Gaussian checkpoint。
