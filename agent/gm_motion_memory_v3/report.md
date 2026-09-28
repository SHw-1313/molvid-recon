# A 线：GM + actual-source 几何与 motion-only ES 实现进度报告

状态：**blocked（无空闲 GPU）**。日期：2026-09-28。
Base `cc74a385577e57c64c9623111e0db86e54e48770`；共同修复
`eec76455a7d9a36723e7c1a2aad1043bba553198`，review 后共同修复
`5a745960f673d5397b883c2b129f39cf57593856`；A loss 候选
`7323adc7c1895e045812288d582c2e2bce26013a`。
环境：enter-container / torch-ito；实际 GPU 分配为空，模型执行次数为零。
这是一份按报告模板保存的中间阻塞报告，不是完成训练/评估的科学报告。

## 本实验回答什么

在相同 P2-GM 权重、基础 loss、曝光与对称重置下，比较 actual-source
clash、motion-only ES、bond band 对几何和运动幅度的影响。G/M 均开启，
它们是架构模块，无专属 loss。旧 J0/J1 从 B0 开始且 G/M 关闭，不作为
本轮初始化；它们的历史结果不替代新 A0 对照。

## 数据

| 项目 | 实际值/身份 |
|---|---|
| train | 已读 train 索引：192 systems / 576 trajectories / 69,120 clips |
| ordinary stores | 100/200/400 ps 分别 35,712 / 17,856 / 8,640 clips |
| fixed-history train | 100/200/400 ps 各 2,304 clips |
| valid source | 已读 valid 索引：8 systems / 24 trajectories / 480 clips；原始 valid store 为 300 ps |
| 正式 valid views | 仍需核对 paired ordinary/fixed-history 派生评估清单与每 replica 2 anchors；尚未评估 |
| native dt | 规格为 10 ps；本次未重新读取 raw trajectory 核验 |
| ordinary H/Q | H4/Q12、H8/Q8；H 在每 lag 内按 batch 平衡，实际 clip 数另报 |
| fixed history | [-300,-200,-100,0] ps；future horizon 1200 ps，按真实 frame mask |
| 拟定 epoch 曝光 | 每轮 13,824 clips，576 条轨迹各 24；ordinary 75%、fixed 25%，各 lag 等权 |
| 拟定 updates | epoch0=10,606、epoch1=10,598，总计21,204；不是旧21,208，待 profile 冻结 |
| 已训练曝光 | 0 clips / 0 valid atom-frames |
| sealed test | false；未打开 |

Data hash `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1`；
normalization base `54bbdc198373ae9f477bf5803643aab77f960fe4498da8db1fcf9c5bfdd770c0`。
Statistics file SHA 已实测
`117dbcb674f9932c40216c4e2c3ee69153dacc3d3efebcca6e67c876a3b6bee9`，
semantic hash 与 parent 为 `7be1542ce10e35bd7f2e170ca88eaf6db464bae4f7721402214bf2c4f0fad76a`。
详见 [data_card.json](artifacts/A/metadata/data_card.json) 及同目录两份完整拟定 schedule。
普通 H4/H8 的 clip 数因 packed batch 大小略有差异，不能把 batch 平衡写成精确 clip 50/50。

## 模型与初始化

| 完整方案 | 架构 | 初始化 | 可训练 / 冻结 |
|---|---|---|---|
| A0 GM + base | G blocks2/4、M on；DiT4 blocks、256/128、8 heads | 同一P2-GM | history/M/DiT/decoder；teacher frozen |
| A1 GM + base + sampled clash | 同上 | 同一P2-GM | 同上 |
| A2 GM + base + clash + motion-only ES | 同上 | 同一P2-GM | 同上 |
| A3 GM + base + clash + motion-only ES + band | 同上 | 同一P2-GM | 同上 |

Parent 文件与 SHA 已实际核验：
`/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P2/formal/GM/frame_joint_step_00021208.pt`，
`aa3b89925d988024bb7ab43e6c38a916e8c2908393817873ce54561be91cd540`。
旧 runs 只读；没有向旧 GM 工作区写入代码。

Observed teacher h/v → trainable history/M condition；source = 最后 observed
normalized latent 重复 + 独立单位高斯；DiT 联合处理 future → coordinate decoder。
h `[H或Q,N,128]`，v `[H或Q,N,3,128]`；clock/mask `[B,T]`，abid/atom_ptr
隔开 packed systems。Physical time 为 ps，与 flow time s 不同。
Future 仅进入 RF target、loss 和评估，actual-source 分支使用 observed/query/source。
正式权重加载、参数计数和 load report 尚未执行；四臂必须 warm start 同一 parent，
optimizer/scheduler/cursor/RNG 对称 reset，禁止从 pilot 或前一臂续接。

## Loss 账本与训练

以下 base 数值直接读取实际 checkpoint contract，不使用手抄近似值。

| loss | 输入 / target | 权重 | 生效条件 | 梯度 |
|---|---|---:|---|---|
| flow（h/v） | target-interpolated latent velocity / teacher target-source | 1（0.5/0.5） | 每步 | DiT/history/M；teacher冻结 |
| clean_coordinate | clean teacher latent解码 / true future坐标MSE | 1 | 每步 | decoder |
| clean_bond | clean解码 / true future键长 | 0.1 | 每步 | decoder |
| near_coordinate | detached endpoint解码 / true future坐标MSE | 0.1 | s≥0.9 | decoder |
| near_bond | detached endpoint解码 / true future键长 | 0.01 | s≥0.9 | decoder |
| generated_bond | **target-interpolated单步endpoint** / future键长 | 0.051707249134778976 | s≥0.75 | DiT/history/M/decoder |
| sampled_clash | K2 actual-source解码 / 冻结vdW及拓扑类别阈值 | 未校准 | A1–A3分层约1/8 | DiT/history/M/decoder；邻居索引detach、距离不detach |
| sampled_rmsf | K2 observed-reference residue RMSF / detached真值特征 | 未校准 | A2/A3；ordinary Q≥8 | 同上；FP32 Kabsch旋转detach |
| sampled_displacement | K2 MSD/相邻平方位移 / detached真值特征 | 未校准 | A2/A3；所有family有效Q≥1 | 同上；单位Å²，不称速度 |
| sampled_bond_band | K2实际键长 / observed键长中心+冻结train宽度 | 未校准 | A3 | DiT/history/M/decoder；future不定义逐样本带宽 |

新 ES 不含旧 covalent increment/product 组。固定历史 Q12 也不进入主 RMSF。
clash 每对一次、排除真实1跳/1–3（含真实二硫键），1–4独立规则；按有效帧及原子数归一化。
当前后端为分块 CUDA 距离正确性参考，没有邻居 cap，不宣称稀疏性能。
精确重叠计数单独输出；零距离有限零次梯度不能保证自行解除重叠。
阈值、band宽度、feature scales、lambda、aux Euler4/8 均尚未冻结，配置拒绝缺失必需表。

拟定设置：seed20260928、BF16、AdamW wd0.01（norm/bias 0）、clip1；
history/M 1e-4、DiT2e-4、decoder5e-5；5%warmup+cosine到0.1。
已核对 M 归入 history_encoder decay/no_decay 两组，不漏训。
一臂一进程/一卡，A最多同时两卡；当前新 CLI 明确拒绝DDP，避免独立rank aux mask的归一化偏差。
暂定40,000 atom-frames/GPU及2epochs，必须经最大样本profile后统一冻结。
详见 [loss_manifest](artifacts/A/loss_manifest.json)、[model_card](artifacts/A/model_card.json)、
[train_manifest](artifacts/A/train_manifest.json)、[resolved_experiment](artifacts/A/resolved_experiment.json)。

## 结果

已完成4项 metadata/config 定向测试（5.03s）、语法编译检查、真实索引全schedule检查，
没有全量pytest。它们不验证CUDA前向/梯度、BF16、future mutation或真实resume。

| 项目 | A0 / A1 / A2 / A3 |
|---|---|
| CUDA smoke、pilot/profile | 全部未运行；无可分配GPU |
| geometry / ordinary motion / fixed MSD | 全部N/A；无正式checkpoint |
| 逐体系pairedCI / 逐残基profile / scatter / MSD曲线 | 全部N/A；没有可合法绘制的新科学结果 |
| persistence / 每decoder oracle | 未运行 |
| 训练seed数量 / generation seed数量 | 实际0 / 0；计划1 / 3（0/1/2） |
| GPU-hours / peak VRAM / GPU throughput | 0 / N/A / N/A |

不生成虚构per_system数值、逐残基曲线或cost曲线。完整评估、指标扩充、校准/诊断和
正式四臂编排仍未完成。仅common family分离不等于完整统一评估实现。

## 机制证据和选择

尚无新模型科学证据，不选择任何臂，不宣称解决clash或改善运动。
没有发布capacity_release/recipe_release；A1完成评估且配方稳定后才能发布。
Keep/Release触发尚不可判断。[selection.json](artifacts/A/selection.json)明确为未评估。

## 独立 review、可复现入口与下一步

Common r01 独立会话 `/root/common_review` 无上下文继承，结论FIX_REQUIRED；
[原报告](reviews/A/common_r01/REVIEW.md)与[修复响应](reviews/A/common_r01/FIX_RESPONSE.md)真实保存。
r02 全新会话 `/root/common_review_r02` 对固定共同提交5a745960复核三项修复，
[报告](reviews/A/common_r02/REVIEW.md)为静态范围通过，**不是A loss/formal PASS**。
A科学candidate review必须在CUDA smoke/pilot/profile后另开全新会话。

[commands.sh](artifacts/A/commands.sh)记录实际科学/检查命令；
[runtime_paths.json](runtime_paths.json)记录主机/容器路径。
协调目录已写resources.json（A2/B1/C1逻辑槽，物理UUID均未分配）和common_release.json。
多次nvidia-smi显示所有8卡都有其他用户计算进程，没有抢占/kill/借用占用卡。

资源阻塞解除后的顺序：补齐A校准/运行工具 → CUDA数值与真实resume smoke →
train-only MD阈值和band统计、GM/J0/J1积分诊断、共享梯度校准 → 每臂pilot/最大样本profile →
新candidate/config独立review闭环 → 同parent对称正式训练 → 统一评估/报告。
常规成功节点不需再次确认；当前没有后台训练/资源轮询守护进程。
预算实际0/80 GPU-hours，未push、未升级依赖、未读sealed test。
