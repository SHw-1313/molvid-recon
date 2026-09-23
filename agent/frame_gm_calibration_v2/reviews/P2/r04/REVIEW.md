# P2 r04 定向独立复核：P3 parent 选择器

REVIEW_STAGE: P2
REVIEWED_COMMIT: 7c8d5e7fca99bd5af77ef581124c6b675d5185f0
VERDICT: FAIL

- 复核范围：r03 `R1`–`R4`、`agent/frame_gm_calibration_v2/reviews/P2/r03/FIX_RESPONSE.md`、`tools/select_frame_gm_p2_parent.py`、`tests/test_frame_gm_selection.py`
- 隔离说明：工作树另有未提交的 `molvid/losses/distribution.py` 与 `tests/test_frame_gm_distribution.py`，属于 P3 并行开发；本轮未读取其实现、未运行其测试，也未将其计入结论。
- 动态检查：按仓库要求在 `enter-container`、`conda activate torch-ito` 中运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`，结果 `11 passed in 0.18s`。
- 修改边界：未修改选择器、测试或 P3 文件；本报告是本轮唯一写入。

## 结论

r03 的统计核心仍然正确，R3 已关闭；R1、R2 和 R4 仅部分关闭。当前实现新增了 substantial fail-closed 校验，但仍存在两个可以让不配对或错误身份的正式结果通过的具体路径：

1. formal 运行的实际 `resolved_config.json` 没有与已审阅 config 完整绑定，paired store/manifest 的当前身份及 sealed-test 状态也没有实际核验；
2. legacy H4 与 fixed-history H4 的 RMSF target 使用相同字典键，合并时 fixed target 覆盖 legacy target，导致部分逐视图跨臂配对检查失效。

因此当前版本仍不能用于最终 P2 `selection.json` 或启动 P3。

## r03 问题关闭状态

| r03 issue | 状态 | 复核结论 |
|---|---|---|
| R1 provenance fail-closed | PARTIAL | metrics 文件哈希、source protocol 和 checkpoint 链已补；formal 实际 config 与 paired data/test 身份仍未闭合 |
| R2 精确视图与 target 配对 | PARTIAL | 精确网格、唯一键、三 replicas 已补；legacy/fixed target 键碰撞使 legacy H4 配对未覆盖 |
| R3 profile cost/tie-break | CLOSED | 四臂 exact key、有限正数和最终同值 fail-closed 均正确 |
| R4 测试覆盖 | PARTIAL | 新增 8 个测试案例方向正确，但未覆盖 formal provenance 主函数及跨 family target 碰撞 |

## 剩余 blockers

### R1a — Formal manifest 声明的 config 未绑定到实际运行的完整 resolved config

- 优先级：P0
- 定位：`tools/select_frame_gm_p2_parent.py:345-450`，尤其 `381-411`
- 证据与触发条件：
  - 代码重算并核对了 manifest 声明的 config 文件 SHA，但对实际运行生成的 `resolved_config.json` 只检查 schema、`total_updates`、parent SHA 和 data hash。
  - 没有比较实际 resolved config 中的 `model.geometry_enabled` / `motion_enabled`、loss contract、训练 seed、LR、precision、sampler weights、history order、预算等与对应审阅 config 的完整内容。
  - 因而可以把 B0 配置实际跑出的目录标为 G：manifest 仍声明并哈希正确的 G config，而实际 `resolved_config.json` 只要 parent/data/updates 相同就会通过；checkpoint 和后续 metrics 也会自洽地绑定到这个错误臂。
- 影响：四臂身份或训练配方可能错接，候选名次和最终 P3 parent 失去意义。
- 最小修复：使用仓库配置加载器读取 `config_path`，逐 section 比较 `resolved_config.json` 的原始 config 部分；至少要求 data、codec、statistics、model、loss、training 完全等于已哈希 config，仅允许明确的运行时 `resolved` 附加区。另应比较四臂 sampler schedule/selected-ID hash 与最终 exposure，确保对称 formal exposure，而不是只检查更新数。
- 最小复验：构造最小 formal fixture，分别修改实际 resolved model arm 开关、loss、seed/LR、sampler schedule 或 exposure；`_validate_formal_provenance` 必须在读取 metrics 前失败。正确四臂 fixture 应通过。

### R1b — Paired store/manifest 只比较声明，未核对当前数据身份或 sealed-test

- 优先级：P0
- 定位：`tools/select_frame_gm_p2_parent.py:142-177`、`599-602`
- 证据与触发条件：
  - `_validate_metrics_run` 仅要求 metric manifest 的 paired-store 路径等于 evaluation protocol 的路径，并在四臂间比较 protocol 中的声明对象。
  - metrics-v3 已记录 `paired_store_index_sha256`，protocol 也记录 `paired_store.index_sha256`，但两者没有比较；当前 store 的 `index.txt` 也未重算。
  - protocol 给出的 paired-manifest path/SHA 没有对当前文件重算，manifest 内容也未打开检查 `test_opened=false`。四臂若一致地指向 test store 或同一份被替换的 manifest，现有 cross-arm equality 仍会通过，输出却无条件写 `test_opened: false`。
- 影响：正式选择可能使用与评估协议不同的数据内容，或误用 sealed test，而仍产出表面完整的 provenance 和 P3 parent。
- 最小修复：要求 metrics 的 store-index SHA、protocol 的 store-index SHA 与当前 `index.txt` SHA 三者相等；重算 paired-manifest SHA 并核对 protocol，读取 manifest 确认 valid 身份和 `test_opened=false`。对 legacy/fixed 各自验证预期 view family，并把实际 index/manifest SHA 写入输出 provenance。
- 最小复验：分别替换 store index、改写 paired manifest、让 metrics/protocol index SHA 不同、以及令 manifest 指向 test/opened-test；全部必须 fail closed。

### R2 — 合并 target map 时 fixed H4 覆盖 legacy H4，逐视图配对检查不完整

- 优先级：P1
- 定位：`tools/select_frame_gm_p2_parent.py:181-215`、`268-285`，以及 `591-598`
- 证据与触发条件：
  - `_validate_view_grid` 对两种 family 都返回 `(system, lag, history)` 键。
  - legacy 中存在 H4 的 100/200/300/400 ps，fixed 中也存在相同四组键。
  - `"rmsf_target_by_view": {**legacy_targets, **fixed_targets}` 会由后者静默覆盖前者；最终跨臂检查只看到 fixed H4 和 legacy H8，共 8 个键，而不是全部 12 个 formal views。
  - 只改变某一臂 legacy H4 target、保持 fixed H4 与体系均值检查可抵消时，当前逐视图检查不会发现错配。
- 影响：legacy H4 的目标数据可以跨臂不同，破坏 system/view 配对和 safety metric 的可比性。
- 最小修复：把 view family 纳入键，例如 `(family, system, lag, history)`；或分别保存/比较 `legacy_targets` 和 `fixed_targets`，禁止用字典展开合并具有重叠键的两个 family。
- 最小复验：四臂 fixture 中仅改变 G 的一个 legacy-H4 target，同时保持 fixed-H4 不变，并用另一 legacy view 抵消体系均值；必须明确失败。另断言每臂 target identity 恰有 12 个带 family 的键。

### R4 — 定向测试仍未覆盖上述主路径

- 优先级：P1
- 定位：`tests/test_frame_gm_selection.py:1-227`
- 证据：
  - 新测试已覆盖 Pearson safety、rank/cost、非法 cost、view 重复、replica count、per-system CSV 篡改和 evaluation checkpoint 错接，均通过。
  - 没有任何测试调用 `_validate_formal_provenance`，因此 candidate/config/status/profile/exposure/sampler 的实现错误无回归保护。
  - 没有测试 `load_arm_values` 后四臂逐 family target 配对，因此 R2 的覆盖 bug 未被发现。
  - metric fixture 没有覆盖 paired store index、paired manifest SHA 或 sealed-test manifest。
- 影响：R1a、R1b 和 R2 可在全部 11 个 selection 测试通过时继续存在，当前通过数不能作为正式选择放行证据。
- 最小修复：增加一个最小端到端 selection fixture，至少贯穿 formal provenance → legacy/fixed metric provenance → view target pairing → selection；同时保留针对每个篡改点的单一失败测试。
- 最小复验：只需运行扩展后的 `tests/test_frame_gm_selection.py`；不需要 P3 distribution 测试、真实 formal checkpoint 或全量 pytest。

## 已关闭与已核验项

- `FROZEN_RULE` 与 r02 resolved rule 做完整结构相等比较，不再只检查版本。
- metrics-v3 manifest schema、aggregation、row count，以及 rows/metrics/per-system SHA 会在 bootstrap 前校验。
- source evaluation protocol SHA、formal checkpoint SHA、Euler16、seeds `[0,1,2]`、三条 evaluation paths 和行数会校验。
- 每体系 legacy `100/200/300/400 × H4/H8` 与 fixed `100/200/300/400 × H4` 网格要求精确唯一，且 `replica_count == 3`。
- 四臂 system 顺序、必需数值有限性、system-paired bootstrap、CI 和安全门方向保持正确。
- Profile costs 要求 exact four-arm mapping、有限且严格为正；多候选先按平均名次、再按成本，二者完全相同则明确失败。R3 已关闭。
- 未提交 P3 distribution 文件未进入测试或判断。

## 主 session 下一步

修复 R1a、R1b、R2，并补齐 R4 的文件级集成测试后再做一次定向只读复核。在此之前不要运行正式 parent selection，也不要据此启动 P3。
