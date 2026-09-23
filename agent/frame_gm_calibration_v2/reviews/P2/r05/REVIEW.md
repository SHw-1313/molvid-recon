# P2 r05 定向独立复核：selector r04 修复

REVIEW_STAGE: P2
REVIEWED_COMMIT: 3af219547755afc906d7a836aa903875536cb10e
VERDICT: FAIL

- 复核范围：`agent/frame_gm_calibration_v2/reviews/P2/r04/REVIEW.md`、`agent/frame_gm_calibration_v2/reviews/P2/r04/FIX_RESPONSE.md`、`tools/select_frame_gm_p2_parent.py`、`tests/test_frame_gm_selection.py`。
- 工作树边界：复核开始时被审 selector/test 相对 HEAD 无 diff；未提交的 P3 model/trainer/distribution 改动全部排除，未运行 P3 distribution 测试。复核期间并发出现的 `molvid/cli/train_frame_joint.py` 工作树改动同样不计入本结论。
- 动态检查：按仓库要求进入 `enter-container` 并激活 `torch-ito`，运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`，结果 `13 passed in 0.24s`。
- 修改边界：未修改实现、测试或任何 P3 文件；本报告是本轮唯一写入。

## 结论

r04 的三个实现缺陷已经关闭：formal runtime config 与 hash-bound reviewed config 完整绑定，四臂 exposure/sampler 要求相同；paired store/index/manifest/sealed-test 链已实际核验；view family 已进入 target key，legacy 8 views 与 fixed 4 views 不再碰撞。

但是 r04 的 R4 要求并未完成。新增测试分别覆盖了若干 helper，却没有贯穿 `load_arm_values()` 和 `main()` 的四臂 legacy/fixed 主路径，也没有执行 r04 指定的“只修改某一臂 legacy-H4 target、fixed-H4 保持不变，必须 fail closed”回归案例。因此当前测试通过不能证明 R2 修复实际接入最终选择入口，r05 仍为 FAIL。

## r04 问题关闭状态

| r04 issue | 状态 | 复核结论 |
|---|---|---|
| R1a formal config + exposure/sampler | CLOSED | runtime raw config（仅排除运行时 `resolved`）与 reviewed config 精确相等；四臂完整 exposure/sampler 对象必须一致 |
| R1b paired data provenance | CLOSED | metric/protocol/current index 三方 SHA、paired-manifest SHA、`test_opened=false` 与 legacy/fixed family contract 均核验并写入 provenance |
| R2 family-key 12 views | CLOSED | target key 为 `(family, system, lag, history)`；每体系 8 legacy + 4 fixed 键保持独立 |
| R4 主路径测试 | OPEN | helper 测试通过，但未覆盖 `load_arm_values → main` 的四臂跨 family target 配对路径 |

## 剩余 blocker

### R4 — 缺少 r04 要求的文件级主路径与跨臂 target 篡改回归测试

- 优先级：P1
- 定位：`tests/test_frame_gm_selection.py:142-158,174-309,414-433`；未覆盖的生产路径为 `tools/select_frame_gm_p2_parent.py:268-334,645-677`。
- 证据：
  - 测试文件没有导入或调用 `load_arm_values`、`main`。
  - `_metric_fixture` 只构造 legacy provenance，且 `per_system.csv` 仅有 `system` 列；没有可由 `load_arm_values` 消费的 legacy/fixed 视图，也没有四臂目录。
  - `test_view_grid_requires_unique_exact_keys_and_three_replicas` 只断言两个 helper 返回的字典合并后有 12 个键；legacy 与 fixed 的 target 值实际相同，且测试没有进入 `main()` 第 660–667 行的跨臂 target equality 检查。
  - formal provenance 测试修改 exposure，但没有单独修改一臂 sampler manifest；虽然实现的对象相等比较会覆盖该路径，FIX_RESPONSE 所称 sampler artifact 测试仍不完整。
- 影响：`load_arm_values()` 的 family 合并或 `main()` 的四臂 target 比较若在后续回归中断线，现有 13 个测试仍可全部通过；这正是 r04 R2 缺陷所在的集成边界。
- 最小修复：增加最小四臂文件级 fixture，包含可通过校验的 legacy 8-view 与 fixed 4-view `per_system.csv`、metric/protocol/store/manifest provenance 以及 formal provenance；调用 `main()`（或至少实际调用 `load_arm_values()` 后执行同一跨臂检查）。先证明正确 fixture 可选择，再仅修改 G 的一个 legacy-H4 target、保持 fixed-H4 不变并同步相关文件哈希，断言明确报 `per-view RMSF targets are not paired`。另加一个单臂 sampler-manifest 差异用例。
- 复验：仅运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`；无需运行 P3 distribution 测试或全量 pytest。

## 已核验的实现修复

- `tools/select_frame_gm_p2_parent.py:433-480` 对 reviewed config 文件做三方 SHA 绑定，使用 `load_config` 读取完整 config，并把 runtime `resolved_config.json` 除 `resolved` 外的全部内容做精确比较；同时检查完整 exposure/sampler 对象跨四臂相同。
- `tools/select_frame_gm_p2_parent.py:167-225` 重算当前 `index.txt` 与 paired manifest SHA，绑定 metric/protocol/current store，拒绝 `test_opened != false`，并区分 legacy/fixed family contract；真实 identity 被保留在 selection provenance。
- `tools/select_frame_gm_p2_parent.py:230-265,289-327` 把 family 纳入 target key；每体系 12 个 view 不再因 H4 重键而覆盖。
- 被审 selector 与 selection 测试在交付回查时仍与 commit `3af2195` 一致；并发 P3 工作树改动未纳入审阅。

## 下一步

只需补齐上述 R4 主路径回归测试后再做一次定向复核；无需重做已关闭的 R1a、R1b、R2，也无需扩大到 P3 或全架构审阅。
