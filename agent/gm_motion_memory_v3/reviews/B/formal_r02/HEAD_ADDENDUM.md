# Formal r02 HEAD identity addendum

**结论：PASS。formal launch 批准延伸到 HEAD `0af49b1f893a851af951f86394fe94a43f506179`。**

原独立审阅批准的实现提交是 `5ec8726950aa2be04d8ed7d23d837f7f00c3e8d6`。对 `5ec8726..0af49b1` 的固定提交差异核查结果只有：

- `A agent/gm_motion_memory_v3/reviews/B/formal_r02/REVIEW.md`
- `M agent/gm_motion_memory_v3/runtime_paths.json`

后一项只记录 reviewed/candidate commit、正式 artifact 路径和 PASS 状态；前一项是 SHA-256 为 `5272dd29a51e8e852733700f6af4f868e806fcdaa5d330bf5b3f85b3aa89dba5` 的既有审阅报告。没有模型、训练、数据、runner、工具、配置或测试代码差异。

源码 tree object 在两个提交间逐项相同：

- `molvid/`：`b7b7b3690dbf0cec1de4d928831b82ec0af2dd60`
- `tools/`：`7b6633dcf0a155c733497ca4d51f3130280415d9`
- `tests/`：`1836924b576e7f715352883368db0980536e9b70`

关键修复 blob 也逐字节相同：`molvid/training/learned_memory_formal.py`、`molvid/training/formal_budget.py`、`molvid/data/learned_memory_formal.py`、`tests/test_learned_memory_formal_safety.py` 的 SHA-256 在两提交间分别保持 `a3753463...`、`fa1a62b3...`、`e9d756ba...`、`ce6ec851...`。

冻结科学 artifact SHA 仍为：M-short `7d103a90...`、M-long `5a18ff42...`、`frozen_configs.json` `7917b420...`、`budget_decision.json` `9db1e69e...`。因此 runner 在 HEAD 上由 `_commit()` 记录 `0af49b1f893a851af951f86394fe94a43f506179` 仅更新真实运行 provenance；其执行代码与已批准的 `5ec8726` 完全一致。

本 addendum 取代原报告中“source 必须保持 `5ec8726`”这一单一身份限制：允许的运行 HEAD 是纯文档后代 `0af49b1f893a851af951f86394fe94a43f506179`，其余配置、顺序、resume、paired ledger 与 budget-exhaustion 条件全部保持原 formal_r02 PASS 报告不变。若 HEAD 再发生任何源文件差异，批准失效并需重新审阅。
