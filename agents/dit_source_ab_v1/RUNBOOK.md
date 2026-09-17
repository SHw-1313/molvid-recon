# 执行手册

## 环境

先检查实际 `enter-container` 用法，在容器内进入 B 的映射目录并激活 `torch-ito`。本文件中的 Python 命令均指该环境内执行，不是宿主机命令。

记录 Python/PyTorch/CUDA、GPU UUID、当前进程及可用显存。使用物理 GPU 到逻辑 cuda:0 的明确映射。不要把某个历史 GPU 编号当作今天空闲的证据。

## 文件整合

由固定 A SHA 导出 PLAN 指定四个文件。允许在本地Git对象缺失时对指定molvid远端作只读fetch；不移动分支，不整体cherry-pick。

如果目标文件已存在，先比较是否已经整合；不要覆盖用户本地修改。记录最终来源和补丁。根AGENTS按ROOT_AGENTS_APPEND追加，旧阶段文件不覆盖。

## 新 CLI 合同

以下是本轮需要实现的接口，不宣称当前仓库已经存在。完成实现后先检查 `--help`，再按顺序执行：

```bash
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage preflight
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage verify --device cuda:0
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage source_check --device cuda:0
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage prepare --device cuda:0
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage train --arm both --device cuda:0
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage evaluate --device cuda:0
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage summarize
```

- preflight：仅元数据/hash/路径/可用环境检查，不触发训练或test数据读取。
- verify：组织ACCEPTANCE所需 targeted CUDA检查、真实smoke、相关回归，不全仓pytest。
- source_check：固定8个clips的中心验证，写source_decision.json。
- prepare：实测新runner/cache/评估代价，确定缓存模式并写budget.json和共享initialization。只做可丢弃profile，不把其权重作为科学初始化。
- train：读取已通过、hash吻合的阶段证据和冻结合同；运行到共同预算终点。两臂都从shared初始化开始。
- evaluate：固定checkpoint/seed/样本，复用已完成相同行，生成缺失终点评估。
- summarize：读取完整raw rows汇总，不自行标记未执行阶段为通过。

另实现 `--stage all`，顺序执行上述阶段，阶段证据失败即停止后续科学步骤。主运行命令可以使用：

```bash
python scripts/run_dit_source_ab.py --config config/dit_source_ab_v1.yaml --stage all --device cuda:0
```

提供 `--run-id`、`--output-root`、`--manifest-root`、`--pilot-root`、显式codec/statistics路径覆盖项，真实挂载差异可以配置，不改源码常量。覆盖后必须重新校验输入hash。

两卡模式：可实现 `--devices cuda:0,cuda:1` 为每臂单GPU进程。父进程先完成一次prepare并发布只读合同；独立子进程使用 `--stage train --arm gaussian/conditional`。禁止两个进程同时写budget/init/同一checkpoint目录。单卡 `--arm both` 在共同检查点轮流推进。

## Resume

提供 `--resume`，按已有run合同恢复；不要自动覆盖同名run。阶段产物通过输入/source/config/代码语义hash校验后可复用。

修改实现以修复确定性bug后，明确哪些阶段失效。仅报告文字变动不要求重训；source/loss语义改变需要新experiment ID。任何恢复都不能消耗不同的训练RNG或跳过未成功optimizer更新。

## 输出

输出根为 `outputs/dit_source_ab_v1/<run_id>/`，建议包含：

- config_resolved.yaml、inputs.json、verification.json、source_decision.json。
- backend_profile.json、runner_profile.json、cache_manifest.json（未启用则写disabled及理由）。
- budget.json、experiment_contract.json、shared_init.pt。
- gaussian/、conditional/：各自 checkpoints、train_history.jsonl、validation_history.jsonl、generation_metrics.jsonl。
- source_check_rows.jsonl、deep_generation_rows.jsonl、comparison.json、report.md。

每个checkpoint记录actual_optimizer_updates和step；selected_checkpoint_step单列，不再复用含混的completed_steps。

## 可持续过夜执行

通过既有作业管理方式启动并记录PID/日志/输出根，能恢复即可。不把进程启动当成任务完成；stage all应自行等待两臂和必要评估结束。

若某臂因可恢复中断落后，优先恢复该臂到共同检查点；另一臂不要继续消耗额外预算制造不等比较。GPU不可用时等待或报告，不杀他人进程。

## 收尾

更新TASKS/HANDOFF/DECISIONS；小型comparison/report和证据索引复制到本阶段evidence目录，显式逐文件stage并本地commit。不git add -A，不force-add outputs，不push。

最终用户可直接看到：修复了什么、source怎么选、训练量/是否仍改善、两臂生成质量差异、剩余问题。完整结果优先于大量通过计数。
