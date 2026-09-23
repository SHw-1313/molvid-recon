# P2 r07 独立定向复核：selector paired-store 路径同一性

REVIEW_STAGE: P2
REVIEWED_COMMIT: 039202e4ed6bc497d3f2cb70e59a3cd15c4c3e4f
VERDICT: PASS

- reviewer/session：新建无实现会话上下文的独立只读 r07 selector reviewer
- base commit：`1e242c775639d1662293d55f2636b571e89faac3`
- 阅读的规格版本与 artifact：完整阅读 `review_request.json`、主任务 prompt、`REVIEW_PROMPT.md`、根 `AGENTS.md`、`PLAN.md`、`ARCHITECTURE.md`、`DATA_TRAIN_EVAL.md`、review template；审阅候选提交中的 selector/test 及实际 P2 formal provenance、B0/G/M legacy+fixed metric artifacts
- 审阅环境限制：GM legacy evaluation 仍在运行，未生成完整 metric artifact，故本轮不执行四臂最终 selector；该项已被 request 明确排除，不影响本次代码修复结论

## 结论

候选提交正确修复了实际 fixed-history artifact 的相对/绝对路径别名问题，且没有放宽为“只要内容 hash 一样即可”。selector 仅在 metric manifest 与 evaluation protocol 的 store 路径经 `Path.resolve()` 后一致时放行；解析后不同的 store 仍明确抛错。后续 store `index.txt` 和 paired manifest 的当前文件 SHA-256 重算与多方绑定保持不变。

定向测试、实际 B0/G/M 双类 metric loader 以及实际 formal provenance 链全部通过。本轮无阻塞问题；可继续完成 GM legacy evaluation，待四臂 metric artifacts 齐备后使用已复核 selector 生成 P2 `selection.json`。

## 必须修复的问题

无。

## 非阻塞建议

无。

## 已核验的关键路径

- 在隔离 detached worktree 核对 `HEAD == 039202e4ed6bc497d3f2cb70e59a3cd15c4c3e4f`，工作树干净；审阅 `1e242c775639d1662293d55f2636b571e89faac3..039202e4ed6bc497d3f2cb70e59a3cd15c4c3e4f` 只改动 selector 的 store-path 同一性比较和对应回归测试。
- `tools/select_frame_gm_p2_parent.py:165-169`：metric/protocol 路径各自 `resolve()` 后比较；同一对象的相对/绝对/符号链接拼写可通过，不同 resolved path 仍触发 `metric/evaluation paired store differs`。
- `tools/select_frame_gm_p2_parent.py:170-187`：仍从 protocol 指定 store 读取现有 `index.txt`，重算 SHA-256 并同时要求 metric summary/protocol 匹配；paired manifest 仍需存在、SHA-256 匹配且 `test_opened == false`。
- `tools/select_frame_gm_p2_parent.py:397-521`：formal config 三方 SHA、runtime reviewed-config 精确相等、checkpoint SHA、code commit、data hash、四臂 exposure/sampler 对称性和 profile 绑定未被修复触及。实际 `P2/r02/evidence/resolved_experiment.json` + `P2/formal` 校验成功，绑定四臂 checkpoint/profile；formal manifest SHA-256 为 `80b1f6709b0b20fabc20637f02404c8894d557794a021eb91191f1058fc20031`。
- 容器 `torch-ito` 中从隔离候选 worktree 运行 `PYTHONPATH=. pytest -q tests/test_frame_gm_selection.py`：`15 passed in 0.34s`。新回归测试用相对于当前 repo cwd 的 protocol store 路径与绝对 metric store 路径贯穿 production validator。
- 以 candidate worktree 模块而非实现 worktree 源码，对实际 `formal_metrics_legacy/{B0,G,M}` + `formal_metrics_fixed/{B0,G,M}` 调用完整 `load_arm_values`：三臂均成功读取 8 systems，并通过 checkpoint、protocol、store index、paired manifest 和 metric-file hashes。
- 实际 B0 fixed artifact 正是本次故障形态：metric manifest 记录绝对 `/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views/clip_store/valid`，evaluation protocol 记录相对 `runs/frame_gm_calibration_v2/P1/fixed_history_views/clip_store/valid`；两者解析为同一 store，且两份 artifact 声明的 index SHA 都是 `c8b91ad7a430de907ab4695f87bb0eb97f234efb2db618b6b9aa246f58ccb7d5`。

## 主session下一步

继续监测已授权的 GM legacy evaluation；完成后重算 GM legacy metrics v3，然后从 repository root 运行提交 `039202e4ed6bc497d3f2cb70e59a3cd15c4c3e4f` 中的 production selector，使用已核验的 `P2/r02/evidence/resolved_experiment.json`、`P2/formal`、`formal_metrics_legacy`和 `formal_metrics_fixed` 产生最终 `selection.json`。
