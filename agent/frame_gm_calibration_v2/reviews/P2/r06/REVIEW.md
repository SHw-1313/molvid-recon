# P2 r06 最终定向复核：selector R4 主路径测试

REVIEW_STAGE: P2
REVIEWED_COMMIT: 170ce389e4f39b13b8c90d6a825cdb46394da97f
VERDICT: PASS

- 复核范围：`agent/frame_gm_calibration_v2/reviews/P2/r05/REVIEW.md`、`agent/frame_gm_calibration_v2/reviews/P2/r05/FIX_RESPONSE.md`，以及提交 `170ce38` 对 `tests/test_frame_gm_selection.py` 新增的 R4 fixture/test。
- 提交边界：相对 `3af219547755afc906d7a836aa903875536cb10e`，production selector 无改动；本轮只复核 R5 唯一剩余的测试覆盖 blocker。r05 已关闭的 R1a、R1b、R2 未重审。
- 工作树隔离：被审 selector/test 相对 HEAD 无 diff。所有未提交 P3 model/trainer/distribution/contract-test 改动均排除，未读取其 diff、未运行其测试、未纳入结论。
- 动态检查：按仓库要求进入 `enter-container`、激活 `torch-ito`，运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`，结果 `14 passed in 0.33s`。
- 修改边界：未修改实现、测试或 P3 文件；本报告是本轮唯一写入。

## 结论

r05 的唯一 blocker R4 已关闭。新增文件级 fixture 确实让 production `main()` 贯穿 formal provenance、四臂 `load_arm_values()`、legacy/fixed metric provenance、12-view target pairing、bootstrap 与 selection 输出；正常 fixture 成功选择 B0，指定的单臂 legacy-H4 target 篡改和 sampler-manifest 篡改均 fail closed。

本轮无剩余 blocker，提交 `170ce389e4f39b13b8c90d6a825cdb46394da97f` 的 P2 selector 定向修复与测试交接通过最终复核。

## 验收证据

### 1. 四臂 legacy + fixed production passing 路径

- `tests/test_frame_gm_selection.py:447-559` 为每臂构造 hash-bound metric/protocol/store/manifest 文件链，并生成每体系 legacy `4 lags × H4/H8 = 8` views 与 fixed `4 lags × H4 = 4` views。
- `tests/test_frame_gm_selection.py:562-588` 对 B0/G/M/GM 四臂同时构造 legacy/fixed artifacts，并把各自 checkpoint SHA 写入 evaluation protocol。
- `tests/test_frame_gm_selection.py:595-610` 直接调用 production `selection_main()`，而非复制 selector 逻辑；调用成功、写出 selection，并断言 `selected_arm == "B0"`。因此实际覆盖 `main → _validate_formal_provenance → load_arm_values → cross-arm pairing → select_from_system_values → output`。

### 2. 仅 G legacy-H4 target 篡改 fail closed

- `tests/test_frame_gm_selection.py:517-525` 的 override 仅在 `family == "legacy"`、`system_0`、100 ps、H4 时改变 target；fixed-H4 文件由独立的 fixed 调用生成且未修改。
- `tests/test_frame_gm_selection.py:612-626` 只对 G 启用该 override，并调用相同 production entry point；明确断言 `ValueError: per-view RMSF targets are not paired for G`。
- fixture 在写完被修改的 CSV 后重算 `per_system_sha256`，所以失败来自跨臂 family-key target identity，而不是陈旧文件哈希。

### 3. 单臂 sampler manifest 篡改 fail closed

- `tests/test_frame_gm_selection.py:438-444` 只修改 GM 的 `selected_sample_ids_hash`，保留其他三臂不变；`_validate_formal_provenance` 明确以 `formal sampler schedule or exposure differs` 拒绝。

## r05 blocker 状态

| issue | 状态 | 结论 |
|---|---|---|
| R4 production main/load_arm_values passing path | CLOSED | 四臂完整 artifact fixture 直接执行 production main 并成功写出 B0 selection |
| R4 G legacy-H4-only target mismatch | CLOSED | fixed-H4 保持不变，production main 在 per-view 跨臂比较处 fail closed |
| R4 single-arm sampler mismatch | CLOSED | GM sampler manifest 单独改变后 formal provenance fail closed |

## 放行结论

在本次限定范围内，P2 selector 的 r04/r05 blockers 已全部关闭。可使用已复核的 production selector 和已核验的 P2 artifacts 执行正式 parent selection；本结论不涵盖并发、未提交的 P3 改动。
