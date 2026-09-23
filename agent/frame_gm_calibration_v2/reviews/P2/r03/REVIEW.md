# P2 r03 独立审阅：P3 parent 自动选择工具

REVIEW_STAGE: P2
REVIEWED_HEAD: fc2fef7fec505fd77d61eb43ef8f787e68131fa5
VERDICT: FAIL

- 审阅对象：当前未提交的 `tools/select_frame_gm_p2_parent.py` 与 `tests/test_frame_gm_selection.py`
- 文件身份：工具 SHA-256 `0074471be93d0d4a607ee89c02c769b1cb575b96a4022ee7397b265dba220270`；测试 SHA-256 `0e93209799ef0a94b264339dba4e1739a4928d52ee5e098304158b4564629062`
- 依据：`DATA_TRAIN_EVAL.md` 中冻结的 `frame_gm_p2_system_paired_bootstrap.v1` 规则、r02 已核验的 `resolved_experiment.json`，以及现有 metrics-v3/run-manifest/protocol schema
- 动态检查：按仓库要求在 `enter-container`、`conda activate torch-ito` 中运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`，结果为 `3 passed in 0.14s`
- 审阅边界：未修改工具或测试代码，未运行训练、评估或全量 pytest；本文件是本轮唯一交付写入。

## 结论

系统顶层的配对 percentile bootstrap、三项主误差方向、三项安全误差方向、候选门和主要 tie-break 算法本身实现正确：

- 四臂使用同一组 bootstrap system indices，按 system 有放回重采样；CI 为 candidate-minus-B0 的 2.5/97.5 percentile。
- bond RMSE 与 residue RMSF MAE 对全部 formal views 求体系内均值，fixed-history MSD error 只对 fixed views 求体系内均值；随后体系等权。
- 安全量使用较低为优：clash rate、`abs(1 - system RMSF Pearson)`、`abs(log(prediction spread / target spread))`；仅当安全量差值 CI 下界大于 0 时判定明确恶化，方向正确。
- 候选要求至少一项主误差 CI 上界小于 0，其他主误差和安全误差均无明确恶化；多候选按三项主误差平均名次、再按 profile step time 选择，方向正确。

但当前输入验证不足以保证这些统计量来自已核验的同一批 formal run、同一 checkpoint 和完整唯一的视图网格。工具可以在关键 artifact 被替换、评估 checkpoint 不匹配，或重复视图掩盖缺失视图时继续选择 P3 parent。因此当前版本不能用于正式 P2 选择。

## 必须修复的问题

### R1 — 指标、评估 protocol 与 formal checkpoint 没有形成 fail-closed provenance 链

- 优先级：P0
- 定位：`tools/select_frame_gm_p2_parent.py:92-107`（`_validate_metrics_run`），`326-351`（`main` 的输入装配）
- 触发条件与证据：
  - `_validate_metrics_run` 只检查 metrics schema、两个布尔声明、总行数和文件存在；没有检查 `run_manifest.json` 的 schema，也没有用其中的 `metrics_sha256`、`per_system_sha256`、`output_rows_sha256` 验证当前文件。
  - metrics-v3 已提供 `source_protocol_sha256`，run manifest 已提供 `source_run`，评估 `protocol.json` 已提供 checkpoint SHA、paired manifest/store、steps、seeds 和 IDs；选择工具没有读取或交叉核对这些字段。
  - `main` 只按约定文件名找到 formal checkpoint 并计算 SHA；没有证明 legacy/fixed 指标正是由该 checkpoint 生成。把其他 checkpoint 的 metrics 目录放到对应 arm 路径，或在指标生成后改写 `per_system.csv`，当前实现仍会继续。
  - `resolved_experiment.json` 只检查 rule version；没有验证完整冻结 rule 字段、resolved artifact SHA、candidate commit、formal run manifest、四份 config SHA、arm 完成状态或 sealed-test 状态。相同 version 下被改写的 rule 会被原样写进输出，但实际仍执行硬编码规则，形成自相矛盾的 provenance。
- 影响：可能用错误代码、错误臂、错误 checkpoint、被修改指标或未经核验数据选出 P3 parent；输出中即使记录新算的文件 SHA，也不能补回选择前缺失的身份验证。
- 最小修复：
  1. 验证 metric run-manifest schema，并逐项重算/比较 metrics、per-system、output rows 的 SHA；验证 metrics 的固定 aggregation、source row count 和 source protocol SHA。
  2. 打开每个 source evaluation protocol，要求其 checkpoint SHA 等于对应 formal checkpoint 当前 SHA；legacy/fixed 同臂必须同 checkpoint，四臂的 paired manifest/store、Euler16、seeds `[0,1,2]` 和视图协议必须符合冻结方案。
  3. 验证 formal run manifest 与 resolved experiment 的 candidate commit、parent、四份 config SHA、每臂成功完成状态、expected step 和 sealed-test=false；同时核对每臂 target provenance/resolved config 的代码与配置身份。
  4. 对完整 selection-rule contract 做精确校验，并把 resolved experiment、formal run manifest、metric manifests 和 evaluation protocols 的 SHA 写入 `selection.json`。
- 最小复验：新增集成测试分别篡改 `per_system.csv`、metrics、run manifest hash、source protocol checkpoint SHA、formal candidate/config SHA 和 test-opened 标志；每种情况都必须在 bootstrap 前失败。正确的最小四臂 fixture 应成功并在输出中保存完整 provenance 哈希链。

### R2 — 只检查 64/32 行和每体系 12/4 行，重复视图可以替代缺失视图

- 优先级：P1
- 定位：`tools/select_frame_gm_p2_parent.py:110-159`，尤其 `116-137` 的 coverage 检查与 `139-153` 的均值
- 触发条件与证据：
  - 当前只检查 generated/true 行总数，以及每个 system 的总行数。某体系重复两次 `(100 ps, H4)` 并缺少 `(400 ps, H8)` 时，legacy 仍是 8 行；fixed 同样可以用重复 lag 替代缺失 lag。
  - 没有要求 key `(system, lag_ps, history_frames)` 唯一，也没有核对 legacy 精确网格 `100/200/300/400 × H4/H8` 和 fixed 精确网格 `100/200/300/400 × H4`。
  - 没有检查 `replica_count == 3`。缺 replica 的 per-system 行仍会被等权纳入。
  - RMSF target 只在十二个视图先求 system 均值后跨臂 `allclose`。不同臂逐视图 target 被交换或替换、但体系均值相同，会被误认为配对。
- 影响：三项主误差的 view 权重、体系配对和 RMSF 安全门可能基于不同条件；重复简单视图可掩盖缺失的困难视图并改变候选资格或平均名次。
- 最小修复：逐体系验证精确且唯一的 view-key 集合、`replica_count == 3`、legacy/fixed system 集合完全相同；在聚合前按 `(system, view)` 跨臂核对 target RMSF，或通过已校验的共同 paired-view protocol/manifest 保证目标身份。不要只依赖最终计数或体系均值。
- 最小复验：加入重复一个 view 并删除另一个、错误 H、错误 lag、`replica_count=2`、legacy/fixed system 不同，以及逐视图 target 交换但体系均值不变的测试；均须失败。

### R3 — Profile cost 未验证为有限正数，冻结 tie-break 可被 NaN/Inf/负值破坏

- 优先级：P1
- 定位：`tools/select_frame_gm_p2_parent.py:300-307`、`332-347`
- 触发条件与证据：`main` 只检查 cost mapping 的 key 集合，然后直接 `float`；`select_from_system_values` 也直接索引。JSON 中的 `NaN`/`Infinity` 会被 Python 接受，负数也会通过。多候选平均名次相同时，这些值会让 tuple 排序产生非物理或不稳定结果；输出还可能包含非标准 JSON `NaN`。
- 影响：冻结的“平均名次相同则选择 GPU 成本低者”无法保证按真实、有限的 profile 成本执行。
- 最小修复：在进入 bootstrap 前要求四臂 cost key 精确、每个值为有限正数，并把所用 profile artifact/hash 与 resolved experiment 绑定。若平均名次和已冻结 cost 都完全相同，应 fail closed 或使用事先写入 rule contract 的明确最终 tie-break，而不是未声明的 arm 字母序。
- 最小复验：覆盖 NaN、Inf、负数、缺臂、额外臂、两个合格候选同平均名次但不同 cost，以及平均名次和 cost 都相同的情况。

### R4 — 三个测试未覆盖输入加载、Pearson 安全门和 tie-break 的关键错误路径

- 优先级：P1
- 定位：`tests/test_frame_gm_selection.py:25-68`
- 证据：现有测试仅覆盖一项清晰主误差改善、一项主误差恶化、clash 阻断和 spread-ratio 阻断；它们全部直接调用已聚合数组入口，不经过 `load_arm_values` 或 `main`。
- 影响：R1–R3 中的 artifact 篡改、视图缺失/重复、逐视图 target 不配对和非法 profile cost 均不会被发现；`abs(1-Pearson)` 的方向、候选多选与 rank/cost tie-break 也没有回归保护。
- 最小修复：在修复 R1–R3 时加入小型文件级 fixture 测试，并补：Pearson 变差阻断、非安全改善不阻断、主误差 CI 边界、无候选回退 B0、多个候选平均名次及 cost tie-break、system 顺序/非有限值/不足 bootstrap 的 fail-closed 测试。
- 最小复验：仅运行 `tests/test_frame_gm_selection.py` 即应覆盖上述契约；无需全量 pytest 或真实 formal 数据。

## 非阻塞观察

- `_ci` 对非有限 bootstrap replicate 设有至少 95% 且不少于 100 个有效 replicate 的阈值；对于退化 Pearson/spread 情况会 fail closed，这一方向合理。
- `sources` 在 `load_arm_values` 的 `123-124` 行构建后未使用，仅是可清理的死变量，不影响选择结论。
- 当前 resolved profile 的四臂 step time 本身为有限且互异的正数，因此 R3 的非法 cost 分支不会由已核验 r02 artifact 自发触发；仍需在工具入口防止 artifact 损坏或误接线。

## 已核验为正确的统计路径

- `select_from_system_values` 要求四臂和 system 顺序一致，并拒绝所有必需数组中的非有限值。
- 四臂共用一次生成的 bootstrap indices，保证 candidate-minus-B0 是 system-level paired bootstrap，而不是独立抽样。
- 三项主误差均以 system 为最终等权单位；fixed MSD 没有混入 legacy views。
- Pearson 和 spread ratio 在每个 bootstrap replicate 上重新计算，而不是错误地 bootstrap 一个预先固定的全局标量。
- improvement、main worsening、safety worsening 的 CI 方向与冻结的 lower-is-better 定义一致。
- 平均名次使用并列平均 rank；在合法且互异的 profile cost 下，候选与 cost tie-break 顺序符合冻结规则。

## 主 session 下一步

先修复 R1–R3，并按 R4 增加最小文件级和统计边界测试；保留现有 bootstrap 与门控核心。定向测试通过后再发起一次只读复核。当前版本不要用于生成正式 `selection.json`，也不要据此启动 P3。
