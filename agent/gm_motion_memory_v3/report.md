# A 线：GM + actual-source 几何与 motion-only ES 实现与实验报告

状态：**formal_completed**。日期：2026-10-03。实现与正式训练所用代码提交
`af60c05aa02ce46872edbad0711833317773436e`，独立 review 固定同一提交并 PASS；
base 为 `cc74a385577e57c64c9623111e0db86e54e48770`。运行环境为
`enter-container` / `torch-ito` / CUDA，正式任务使用
`GPU-b6b9107a-2a73-2d74-1dc0-1cbe16ae86d0`。sealed test 未读取。

## 本实验回答什么

在同一个 P2-GM parent、相同架构、基础 loss、训练曝光和 optimizer 对称重置下，
比较 actual-source sampled clash、motion-only ensemble-statistic（ES）以及 sampled
bond band 对生成几何与运动幅度的影响。A0 是本轮公平 baseline；A1 加 clash，A2 再加
RMSF/displacement，A3 再加 bond band。G/M 是所有四臂共有的架构模块，没有专属 loss。
旧 J0/J1 从 B0 起步且关闭 G/M，只作历史背景，不是本轮对照或初始化。

## 数据

| 项目 | 实际值 / 身份 |
|---|---|
| train | 192 systems / 576 physical trajectories / 69,120 可用 clips；data hash `8b972d…ed0b1` |
| valid | 8 systems / 24 trajectories；ordinary 与 fixed-history 都用同一组系统 |
| raw 时间 | native dt 10 ps；physical time 使用 ps，与 flow time `s` 分开 |
| ordinary | train buckets 100/200/400 ps；H4/Q12 与 H8/Q8；每臂 H4 5,200、H8 5,168 clips |
| fixed-history | observed `[-300,-200,-100,0] ps`，future 到 1,200 ps；每臂 3,456 clips |
| 每臂曝光 | 1 effective epoch，13,824 clips，576 条轨迹各 24；10,606 updates；340,312,590 valid atom-frames |
| family / lag 权重 | ordinary 75%、fixed-history 25%；ordinary 三个 lag 各 3,456 clips，fixed 三个 lag 各 1,152 |
| ordinary valid | 384 generation records / 1,920 metric rows |
| fixed-history valid | 192 generation records / 960 metric rows |
| sealed test | **false** |

训练 data card、完整 system IDs 与 split/hash 在
[data_card.json](artifacts/A/metadata/data_card.json)。normalization source hash 为
`54bbdc…770c0`；statistics 文件 SHA256 为
`117dbcb674f9932c40216c4e2c3ee69153dacc3d3efebcca6e67c876a3b6bee9`。
验证只有 8 个体系，所以以下 CI 是探索性 paired-system bootstrap，不作统计确证。

## 模型与初始化

| 实验全名 | 架构 / 新目标 | trainable | parent checkpoint SHA256 |
|---|---|---|---|
| A0 GM + complete base | G blocks 2/4，M on，DiT 4 blocks | history/M/DiT/decoder | `aa3b8992…cd540` |
| A1 GM + base + sampled clash | 与 A0 相同；加 actual-source clash | 同上 | 同上 |
| A2 GM + base + clash + motion ES | 与 A1 相同；加 RMSF/displacement | 同上 | 同上 |
| A3 GM + base + clash + motion ES + band | 与 A2 相同；加 sampled bond band | 同上 | 同上 |

模型共 19,600,857 参数，其中 18,908,433 可训练；DiT scalar/vector width
256/128、4 blocks、8 heads。teacher/geometry encoder 来自
`inputs/codec_best.pt`（SHA256 `ba10c44…e6df`），全程 frozen/eval。

数据流为：observed frames → frozen teacher `h [H,N,128]`,
`v [H,N,3,128]` → trainable history encoder 与 observed-only M condition；source 为
最后 observed normalized latent 的重复加独立单位高斯 → DiT 在 query physical times 上预测
flow → decoder 输出 future coordinates `[Q,N,3]`。`clock/mask [B,T]`，packed systems 由
`abid/atom_ptr`、真实共价拓扑和 frame/loss masks 隔离。训练 target future 只进入 RF target、
loss 与评估；生成接口只接收 observed context 和 query times。

四臂分别重新载入同一个 parent：每臂 481 个 tensor loaded，`initialized=[]`、
`unexpected=[]`；optimizer、scheduler、cursor、main RNG 和 sampled RNG 都重置，未恢复 parent
optimizer，也未从 pilot 或前一臂继续。checkpoint 身份如下。

| arm | config SHA256 | final checkpoint SHA256 |
|---|---|---|
| A0 | `2d15e38c…64da4` | `f459f1d3…23bc7` |
| A1 | `15d55300…1cce` | `52d922f8…14a9` |
| A2 | `db842c3a…b00e` | `3d10ddd9…c65` |
| A3 | `b41ba050…a867` | `efafdd61…5711` |

## Loss 账本与训练

所有数值来自冻结 recipe/checkpoint contract，不使用隐藏默认值。G/M 没有各自 loss。

| loss 实际字段 | 输入分支 / target 来源 | 权重 | 生效条件 / cadence | 梯度边界 | 与 A0 差异 |
|---|---|---:|---|---|---|
| `flow` (`h`,`v`) | target-interpolated normalized latent / frozen teacher target − independent source | 1（0.5/0.5） | 每步 | DiT/history/M；teacher frozen | 全臂相同 |
| `clean_coordinate` | clean teacher latent 解码 / true future coordinates | 1 | 每步 | decoder；teacher/target 不回传 | 全臂相同 |
| `clean_bond` | clean 解码 / true future bond lengths | 0.1 | 每步 | decoder | 全臂相同 |
| `near_coordinate` | detached single-step endpoint 解码 / true future | 0.1 | flow time `s>=0.9` | endpoint 对 DiT detach，仅 decoder | 全臂相同 |
| `near_bond` | detached single-step endpoint / true future bonds | 0.01 | `s>=0.9` | 仅 decoder | 全臂相同 |
| `generated_bond` | **target-interpolated 单步 endpoint** / future bonds；不是 actual-source sample | 0.051707249134778976 | `s>=0.75` | DiT/history/M/decoder | 全臂保留 |
| `sampled_clash` | K=2 actual-source trajectories / frozen train-only vdW+topology thresholds | 0.0010638914903412656 | A1–A3；每 stratum 约 1/8 | DiT/history/M/decoder；neighbor indices detach、distance 可导 | A1 起新增 |
| `sampled_rmsf` | K=2 observed-reference aligned residue RMSF / detached future RMSF | 0.0037894484312085723 | A2/A3；ordinary Q≥8；994 updates | 同上；FP32 Kabsch rotation detach | A2 起新增 |
| `sampled_displacement` | K=2 MSD 与相邻平方位移（Å²）/ detached future features | 0.002193553424286401 | A2/A3；所有 family Q≥1；1,327 updates | 同上；Kabsch detach | A2 起新增 |
| `sampled_bond_band` | K=2 actual-source bonds / last-observed center + frozen train widths | 0.005290786076012187 | A3；1,327 updates | DiT/history/M/decoder | 仅 A3 |

RMSF 与 displacement feature scales 分别为 1.2989751036755146 Å 和
6.790559953979887 Å²。actual-source 采用 K=2、Euler=8，执行 schedule 在
`(view, lag, H)` 内分层取整约 1/8；A0 也共享同一 schedule，从而 main flow RNG 不因
aux 开关改变。clash 排除真实 1-hop/1–3（含二硫键），1–4 用独立阈值；距离按有效帧/原子数
归一化。新 ES 不含旧 internal-distance/bond-increment 组。

训练用单进程单 GPU、BF16、动态 packed batch（每更新不超过 40,000 atom-frames），无
DDP/gradient accumulation；AdamW，weight decay 0.01（norm/bias 0），grad clip 1。
history/M、DiT、decoder LR 分别为 `1e-4 / 2e-4 / 5e-5`，5% warmup 后 cosine 到
0.1 倍。profile 预测四臂跑 2 epochs 为 93.43 GPU·h，超过正式训练 48 GPU·h 子预算，
因此在正式运行前对所有臂共同冻结为 1 epoch，而不是事后改单臂曝光。

训练日志的 raw/weighted loss 均值如下。base 范围按所有 10,606 updates 计算；aux
按其非零 active updates 计算。

| 项目 | raw mean | weighted mean |
|---|---:|---:|
| flow（四臂范围） | 0.117032–0.117241 | 同 raw |
| clean coordinate | 0.0008760–0.0008780 | 同 raw |
| clean bond | 0.00121180–0.00121193 | 0.000121180–0.000121193 |
| near coordinate | 0.0006935–0.0007025 | 0.00006935–0.00007025 |
| near bond | 0.0007843–0.0008052 | 0.000007843–0.000008052 |
| generated bond | 0.004339–0.004490 | 0.0002244–0.0002322 |
| A1 sampled clash | 0.029835 | 0.00003174 |
| A2 clash / RMSF / displacement | 0.040619 / 0.269086 / 0.289671 | 0.00004321 / 0.00101969 / 0.00063541 |
| A3 clash / RMSF / displacement / band | 0.037945 / 0.274047 / 0.294486 / 0.045408 | 0.00004037 / 0.00103849 / 0.00064597 / 0.00024024 |

精确值见
[training_loss_summary.json](../../runs/gm_motion_memory_v3/A/r21/final_analysis/training_loss_summary.json)、
[loss_manifest.json](artifacts/A/loss_manifest.json)、
[train_manifest.json](artifacts/A/train_manifest.json) 和冻结 recipe
`runs/gm_motion_memory_v3/A/r04/resolved_auxiliary.json`（SHA256 `625d11…e6f3`）。

## 结果

评估用 FP32、Euler=16、generation seeds 0/1/2；每臂只有一个共同 training seed
20260928，不是 best-of-N。主聚合为 family 内 lag grid 等权、ordinary H4/H8 等权、
system 等权；paired bootstrap 以 system 为单位、10,000 draws。所有指标均为真实 generated
路径；oracle 与 persistence 保留在 evaluator `bucket_summary` 中，仅作校验，不参与选择。

下表 `↓` 越低越好，Pearson `↑` 越高越好；spread 是 generated/MD system-mean RMSF
标准差之比，理想值 1，但必须与 MAE、相关性和逐体系方向一起解读。

| family / metric | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| ordinary bond RMSE Å ↓ | 0.293627 | **0.291442** | 0.314318 | 0.292752 |
| ordinary clashes / 1000 atoms ↓ | 152.359 | **146.159** | 187.297 | 164.642 |
| ordinary vdW severity Å²/atom ↓ | 0.046088 | **0.043731** | 0.059501 | 0.051985 |
| ordinary residue RMSF MAE Å ↓ | 0.344480 | 0.348561 | **0.309478** | 0.311098 |
| ordinary system Pearson ↑ | **0.896964** | 0.897853 | 0.885242 | 0.884950 |
| ordinary spread | 0.291876 | 0.285463 | **0.373607** | 0.370106 |
| fixed bond RMSE Å ↓ | 0.280791 | 0.278858 | 0.298328 | **0.277491** |
| fixed clashes / 1000 atoms ↓ | 125.087 | **119.906** | 154.524 | 132.709 |
| fixed vdW severity Å²/atom ↓ | 0.033227 | **0.031271** | 0.044085 | 0.037089 |
| fixed MSD-curve MAE Å² ↓ | 2.897830 | 2.895127 | **2.530343** | 2.588668 |

A1 相对 A0 的 paired-system 结果：ordinary clash `−6.200`
（95% bootstrap CI `[-8.355, -4.485]`，−4.07%），severity `−0.002357 Å²/atom`
（`[-0.003456, -0.001397]`，−5.11%）；ordinary RMSF MAE `+0.004081 Å`
（`[0.001264, 0.007648]`，+1.18%），fixed MSD MAE `−0.002703 Å²`
（`[-0.017337, 0.009256]`，−0.09%）。因此 A1 通过 5% motion guardrail，但远未达到
预设约 30% clash 理想改善。

### 逐体系方向

表中均为 A1 − A0；clash、severity、RMSF MAE、MSD MAE 的负值表示改善。

| system | ordinary clash/1000 Δ | severity Δ Å²/atom | ordinary RMSF MAE Δ Å | fixed MSD MAE Δ Å² |
|---|---:|---:|---:|---:|
| atlas_2ejn_B | −7.550 | −0.003514 | +0.000044 | −0.044360 |
| atlas_3f0o_B | −4.218 | −0.001298 | +0.001661 | −0.000153 |
| atlas_3hdj_B | −4.224 | −0.000934 | +0.001896 | +0.004782 |
| atlas_3l4h_A | −4.931 | −0.001549 | −0.000604 | +0.020804 |
| atlas_3rv1_B | −12.786 | −0.005243 | +0.014460 | −0.015243 |
| atlas_5ef9_A | −7.863 | −0.003928 | +0.004090 | +0.012622 |
| atlas_5ii7_A | −3.610 | −0.001045 | +0.008230 | −0.009705 |
| atlas_6xb3_H | −4.418 | −0.001344 | +0.002868 | +0.009630 |

完整四臂逐体系表在
[per_system.csv](../../runs/gm_motion_memory_v3/A/r21/final_analysis/per_system.csv)；完整 paired CI
和每体系 delta 在
[selection.json](../../runs/gm_motion_memory_v3/A/r21/final_analysis/selection.json)。

### 逐残基 RMSF、体系幅度与 MSD–ps

统一分析保存了 64 个逐残基 CSV（4 arms × 8 systems × H4/H8）和 100 张图：所有体系的
逐残基 MD/generated RMSF、每体系 fixed-history MSD–physical-time 曲线，以及每臂
system-mean RMSF scatter。文件位于
[`per_residue_profiles/`](../../runs/gm_motion_memory_v3/A/r21/final_analysis/per_residue_profiles/)
和 [`figures/`](../../runs/gm_motion_memory_v3/A/r21/final_analysis/figures/)，不是挑选单个有利体系。
按 MD amplitude 预注册式排序的低/中/高代表体系为 `atlas_3hdj_B / atlas_2ejn_B /
atlas_3rv1_B`，但选择使用全部 8 个体系。

A2 把 ordinary RMSF MAE 降低 10.16%、fixed MSD MAE 降低 12.68%，同时 ordinary
clash/severity 反弹 22.93%/29.10%。A3 相对 A0 的 motion 改善为 9.69%/10.67%，
ordinary clash/severity 仍反弹 8.06%/12.80%；它相对 A2 修复 bond 与部分 clash。
A3 相对 A2 的 RMSF/MSD 恶化均低于 5%，平均生成 RMSF 也未下降超过 10%，所以预冻结的
Keep/Release 条件为 **false**，没有追加 fork。

### Cost 与显存

| arm | train GPU·h | ordinary eval | fixed eval | 合计 | peak allocated |
|---|---:|---:|---:|---:|---:|
| A0 | 6.473 | 1.028 | 0.492 | 7.993 | 56.94 GB (53.03 GiB) |
| A1 | 8.703 | 1.063 | 0.507 | 10.273 | 81.12 GB (75.55 GiB) |
| A2 | 9.217 | 1.080 | 0.519 | 10.816 | 81.12 GB (75.55 GiB) |
| A3 | 9.452 | 1.181 | 0.560 | 11.193 | 81.12 GB (75.55 GiB) |

正式训练合计 33.845 GPU·h，正式评估 6.430 GPU·h，正式总计 40.275 GPU·h。
把两次失败 pilot、最终 pilot、两轮 smoke/profile 也计入的可审计总成本为
41.051 GPU·h，低于本线 80 GPU·h 总预算。成本以 wall-clock×占用 GPU 数计，ledger 在
`runs/gm_motion_memory_v3/A/r16/configs/formal_cost_ledger.json`。

## 机制证据和选择

四臂共享 parent、schedule、曝光、training seed、evaluation seeds 和积分步数，且 review 后
模型源代码相对固定提交没有变化，因此差异不能用初始化、额外 epoch 或不同采样器解释。
A1 在 8/8 体系上同时降低 ordinary clash count 与 severity，且 fixed family 同向，说明
actual-source clash 确实提供了弱但一致的几何信号；训练中该项 active-update weighted mean
只有 `3.17e-5`，也解释了为什么没有接近 30% 目标。

A2/A3 的生成 mean RMSF 从 A0 的 0.854 Å 提高到 0.973/0.953 Å，更接近 MD 的
1.114 Å，spread 从 0.292 提高到 0.374/0.370，motion MAE 同步改善；这不是仅靠加噪声
提高幅度，因为 residue correlation 也略升，但 system Pearson 从 0.897 降至约 0.885，且
clash 明显增加，仍是运动—几何 trade-off。单 training seed、8-system validation 和 aux
权重很小是剩余替代解释；这里不声称统计确证或 production parity。

最终默认发布 **A1**：它是唯一同时满足 motion guardrail 且不发生 clash rebound 的增强臂，
也是较简单的 Pareto 候选。A0–A3 都在预定义五轴 Pareto 前沿并通过实用 guardrail；A2/A3
作为 motion-oriented 研究候选保留，但不覆盖默认 recipe。A1 没有解决“约 30% 降 clash”
的科学目标，这一点不是软件失败，也不被报告成成功。

## 独立 review、可复现入口与下一步

正式候选由全新无实现对话的 `/root/formal_a_review_r02` 对固定提交
`af60c05…36e` 审查；[REVIEW.md](reviews/A/formal_r02/REVIEW.md) 为 PASS，SHA256
`4b653591…c2d1`；[review_gate.json](reviews/A/formal_r02/review_gate.json) 固定 review、
candidate 与 frozen configs，SHA256 `486d045b…92f9`。reviewer 没有修改模型代码。
CUDA smoke 验证 FP32/BF16、teacher unchanged、future mutation=0、exact resume 和四项
sampled loss 到 history/M/DiT/decoder 的非零梯度；未运行不必要的全量 pytest。

[commands.sh](artifacts/A/commands.sh) 保存复现命令；
[runtime_paths.json](runtime_paths.json) 保存本线 host/container 路径；
[resolved_experiment.json](artifacts/A/resolved_experiment.json)、model/data/loss/train cards、
四臂 configs、checkpoint identity、protocol/summary、逐体系/逐残基结果和 figures 均已落盘。
旧 GM 路径只读；neibu 只做备用数据/代码 staging，没有重复启动本次正式训练。未 push、
未升级依赖、未读 sealed test。

最有信息量的下一步是用一个新的 training seed 对 A0/A1 做同配方成对复验；在确认约 4–6%
改善可复现之前，不应仅凭这 8 个体系直接提高 clash 权重或宣称已达到目标。
