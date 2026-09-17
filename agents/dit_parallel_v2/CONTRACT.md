# 共享协议：并行、所有权、科学边界

## 1. 两条 lane 与汇合

A 是现有 pilot 分支上的独立诊断实现；B 是从 a22f60c 固定快照建立的独立 backend worktree。两者可以同时进行，不等待对方。A 的评测实现与 B 的运算重排应独立；合并后再做交叉验收。

本轮不执行 Gaussian-source / conditional-source 两臂训练。用户审阅两边交付后才进入 NEXT_STAGE。

## 2. 文件所有权

| 文件/目录 | A | B |
|---|---|---|
| `evaluation/dit_diagnostics.py`（新增） | 写 | 只读；本轮不需要 |
| `scripts/evaluate_dit_pilot_diagnostics.py`（新增） | 写 | 只读；本轮不需要 |
| `tests/test_dit_diagnostics.py`（新增） | 写 | 不改 |
| `config/dit_pilot_diagnostics_v1.yaml`（新增） | 写 | 不改 |
| `module/molecular_dit.py` | 只读 | 写 |
| `module/dit_backend_v2.py`（可新增） | 不改 | 写 |
| `module/state_detail_latent_adapter.py` | 只读 | 仅派生布局缓存/host metadata plumbing，不改四 field schema/归一化 |
| `trainer/dit_trainer.py` | 只读 | 限 backend 选择、计时/非语义性能调整 |
| `scripts/run_state_detail_dit_pilot.py` | 只读 | 限重复 encode、backend 选择/报告兼容 |
| `scripts/profile_dit_backend.py`（新增） | 不改 | 写 |
| `tests/test_dit_backend_v2.py`、`tests/reference/dit_backend_v1_reference.py`（新增） | 不改 | 写 |
| `tests/test_dit_pilot_runner.py`、`tests/test_molecular_dit.py` | 不改 | 仅本次直接受影响部分 |
| `config/dit_backend_profile_v2.yaml`（新增） | 不改 | 写 |
| `agents/dit_pilot_diagnostics_v1/` | 写记录 | 不改 |
| `agents/dit_factorized_backend_v2/` | 不改 | 写记录 |
| 根 `AGENTS.md` | 只在 A 树追加阶段说明 | 只在 B 树追加阶段说明 |
| 旧 codec、geometry、RF objective、旧 evaluator、旧实验记录 | 只读 | 只读 |

如实现必须触及名单外共享代码，先在 HANDOFF 说明具体原因；优先使用新适配层，确实需要扩大语义变更时请用户决定。不要让两个 session 私下交叉写入。

## 3. 数据和 checkpoint

- 基础 manifest：`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`。
- 48 train / 8 valid / 8 test；本轮只读 train/valid，不打开 test clip store。过去 codec T1 已经评测过 test，这不等于 DiT 的最终完全未见测试；如有正式泛化声称，应另行讨论新 holdout。本轮不要重新读取旧 test payload。
- Pilot checkpoint 根：`/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1/outputs/dit_state_detail_pilot_v1/full_20260908_S4500`。
- R2：`ratio2_state_detail/seed20260907_S4500/best_validation.pt`。
- R4：`ratio4_state_detail/seed20260907_S4500/best_validation.pt`。
- Codec 结果/权重的真实路径来自 pilot summary 中 `codec` 字段；按已配置挂载映射解析，不凭空替换路径。R2 codec checkpoint SHA256 为 `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9`；R4 为 `ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`。
- 不把新 seed 的更优 codec 偷换进现有 pilot 诊断/新旧 backend 等价比较。
- 使用 checkpoint 自带/对应的原始 FP32 statistics，不重估生产归一化，不转 BF16 后重新哈希。报告统计值与 provenance，不把计算 dtype 差异误认成数据来源改变。
- 输入文件只读；所有输出进自己新目录。不要覆盖旧 pilot_summary、latest_metrics、checkpoint 或 shared statistics。

## 4. 环境、GPU 和执行效率

所有 Python 命令进入真实 `enter-container` 工作流后激活 `torch-ito`。容器内确认 `pwd`、实际 Git 顶层、`sys.executable`、PyTorch/CUDA、GPU UUID。模型输入/权重/中间激活/解码/科学指标在 CUDA；CPU 可以读取元数据、整理报告、序列化，不允许把数值路径悄悄搬去 CPU。旧 CPU 元数据单测不是 CUDA evidence。

先实现，再 CUDA targeted checks → 一批真实数据 smoke → 相关回归 → bounded evaluation/profile。全仓 pytest 不作为实现前置要求，也不必每个小修改都重跑。skip 的 CUDA 测试不算通过。不得杀进程、抢占 GPU 或安装/升级包。

为两边避免抢同一卡，可使用 `/data4/users/sihao/workspace/.molvid_gpu_claims/GPU-UUID` 的原子 `mkdir`；UUID 由当前 nvidia-smi 查出并验证为安全单一路径片段。写入自己 session/worktree/pid/time 的 owner 文件。占用目录不能替代 nvidia-smi 进程检查；现有占用不自行删除，自己的任务结束才删除自己创建且 owner 匹配的空标记目录。只读诊断没有空闲 GPU 时，继续文档/实现，不强行启动。

B 的精确吞吐测量在专属空闲 GPU 上串行测 reference/optimized；A 的模型评测在另一张卡。共享 I/O 与 CPU 争用应记录；compute-only 使用预载输入，不能把一次并发文件扫描当作模型吞吐差异。

## 5. 证据和结论

来自用户已上传 `t1_comparison.json` 的同一主seed、best checkpoint确定性重建结果（历史证据，不在本轮重跑）：

| Codec | 重建 future RMSD / Å | 重建 future dRMSD / Å | 本轮角色 |
|---|---:|---:|---|
| R1 state/detail | 0.0210834 | 0.0179108 | 无时间下采样参照 |
| R2 state/detail | 0.0228804 | 0.0223210 | 更密时间token的诊断/性能对照 |
| R4 state/detail | 0.0214491 | 0.0184857 | 工程主线 |
| R4 matched pooling | 0.0435352 | 0.0525751 | 历史负对照，不继续 |

这组证据支持R4接近R1重建、优于当前R2重建，但不证明任意数据规模/生成任务上R4都最好。不要把codec重建中的future字段解释为forecast；encoder当时看到了完整clip。

- R4 是当前工程主线；R1 是重建参照，并非数学保证的生成上限。R2 只参加当前 checkpoint 诊断/性能对照，不启动完整训练。
- `K_R1=16`、`K_R2=8`、`K_R4=4`；scalar 信息容量为 R1 `16C`、R2 `16C`、R4 `8C`，vector 场同理另计。R4 相比 R1 减半 coefficient volume，不可笼统称所有 latent 数据量都缩小四倍。
- R4 pooling 不再加入。
- 旧 pilot 的 RF loss 下降证明学到了 RF 目标，不保证有效 MD；R2/R4 不同 latent 分布的 loss 数值不等于共同物理误差。
- Codec oracle 约 0.02 Å、generated 约 3 Å 表明生成链路差距很大，但不能单凭此证明 decoder 无责任或生成 latent 已经 OOD。
- Stochastic generation 的一条轨迹与 reference velocity 逐点相关接近零，不足以宣判动力学分布未学会；要分开 pathwise 与 ACF/RMSF/分布证据。
- 不使用仅 RMSD 胜过 persistence 作为生成成功门槛；合理随机样本可能不如条件均值的点对点 RMSD，但几何、非零运动、分布与时间统计仍必须可信。

## 6. 提交与报告

允许本地 commits，不自动 push、PR、merge/rebase 或 cherry-pick；不得 `git add -A`。明确逐文件 stage。结果只提交小 JSON/Markdown 摘要与配置；大 tensor、缓存、checkpoint 和旧日志保持本地。不要 force-add 整个 outputs。

用 TASKS 表示真实完成度，HANDOFF 记录实际命令、输入哈希、CUDA device/dtype、输出路径、关键指标及阻塞。不要把规划中的测试填为 PASS。父级旧 agent 文档、历史“等待”状态不重写成新任务。
