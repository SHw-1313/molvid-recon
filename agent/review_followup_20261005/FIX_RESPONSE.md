# 工程审阅修复响应

本文件响应 2026-10-04 的独立审阅，不冒充新的独立审阅批准。
代码在 `fix/review-followup-20261005`，原复现/审阅文件不变。

| 项目 | 实现 | 验证与边界 |
| --- | --- | --- |
| E1：配体静默丢失 | `molvid/data/chemistry.py` 统一选择器，显式角色按 topology-unique residue.index 定义；逐原子 source→model 审计；未知组分显式失败 | LIG/ATP/MOL/UNK/OTHER 同组分选择测试；支持 protein/ligand/solvent/ion/exclude；非重原子选择策略不变，不推断未知配体化学 |
| E1 adapter | `iter_misato_records`/`preprocess_misato` 暴露角色与预期配体集合；CLI `--residue-roles` 核验对应 PDB SHA | 实际小 PDB→trajectory record 回归；SHA 不匹配失败；不改已有缓存 |
| E2：跨链 resSeq 冲突 | CA anchor 按实际 residue 对象/唯一 index 求取；不再用 resSeq 字典跨链复用 | 双链同 resSeq 回归；缺失/多重 CA 报错 |
| E3：同形异图 | `StaticTopologyMetadata` 内容 SHA 区分 schema contract 与实际身份；含 atom identity、类型、组分、分段与键图 | 同 shape 错类型 decoder 拒绝；同边数异图 cache 不复用；张量原地修改触发失效错误 |
| E4：混合 H 的 batch-min | `HistoryEncoder` 按样本有效前缀压缩再打包；同 H 全有效路径保留原数学运算 | H8/H4、H7/H5、H3/H8 独立/混批/换序/脏 padding、梯度测试；不是全模型任意输入生产验收 |
| E5：padding 参与 dt | 物理时间取前一个 valid timestamp；decoder 先清理 invalid feature/clock | padding nan/极值不影响输出；紧凑与带 padding 输出一致；物理时间/flow time 仍分离 |
| 运行身份 | `molvid/provenance.py` 新运行专用只创建清单，代码 SHA、commit、dirty patch 与 inputs SHA | N1/N2/N3/N4 各运行清单独立；不存在用最终 HEAD 重标早期运行 |
| 冗余实现 | 保留 S 作为本分支唯一共享静态 continuation 实现；U/K 共用 predictive_data/evaluation | 不重复移植 D2 runner，不复制 feature cache，不默认启用新训练目标 |

语义摘要缓存通过 tensor version 检测正常原地更新；不要用 `.data` 绕过 PyTorch 版本机制修改拓扑。推理张量无版本计数时重新计算摘要。新输入未提供 atom identity 时仍依靠类型/图内容，不能凭该模式声称已识别所有同构原子置换。

角色 manifest 示例（只演示结构，SHA 必须来自实际 PDB）：

```json
{
  "SYSTEM_ID": {
    "topology_sha256": "actual-pdb-sha256",
    "residue_roles": {"153": "ligand", "154": "ion"},
    "expected_ligand_residues": [153]
  }
}
```

索引为 MDTraj topology 的 `residue.index`，不是 PDB resSeq 或 residue name。
提供 manifest 时，CLI 在创建输出 store 前检查所有所选系统的条目与 SHA；默认模式保留原 MOL/UNK 适配，未知 residue 进入明确错误/skip 日志，`--strict` 可使其立即失败。

## 证据和复现入口

全部 Python/模型/测试均在 `enter-container` → `conda activate torch-ito` 中执行。
容器工作树 `/workspace/molvid-review-followup-20261005`。

```bash
python -m pytest -q tests/test_data_preprocess.py tests/test_review_contracts.py tests/test_history_controls.py tests/test_reference_qualification.py tests/test_review_diagnostic_metrics.py
python -m tools.run_review_diagnostic --help
python -m tools.run_history_controls --help
python -m tools.review_reference_qualification --help
```

科学运行必须使用新 output 目录；不能覆盖这次结果。
`runs/review_20261005/` 包含失败的初次 fixture 日志、修复后日志、旧新实际推理输出、所有 head checkpoint/optimizer/statistics、逐体系/逐 seed 指标及输入 SHA。
原运行输入绝对路径列于各 `run_identity.json`，不用拷贝大原始数据或父 checkpoint。

关键代码提交：`6a91450`（E1–E5）、`ffbf7d6`（N1/N2/N3 与旧同 H 精确路径）、`acd50a7`（N4）、`a0e869e`（solver32）、`a29f6a9`（角色 adapter/聚合）、`9ee19f6`（密度控制）、`de33a5a`（密度结果身份检查）。
