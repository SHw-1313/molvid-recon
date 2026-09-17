# R4 source 两臂实验合同

## 1. 冻结输入

- manifest：`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/manifest_20260904_token80000`。
- 系统数48 train/8 valid/8 test；train 8928 clips、valid 1488 clips。只打开 train/valid store，允许读取 manifest 中 test 的数量，不读取/hash test payload。
- 当前数据是有序连续16帧、100 ps；不随机抽16帧，不增加其他 dt/static 数据。
- R4 codec：`/data4/users/sihao/workspace/PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt`；同目录 `result.json`。
- R4 codec SHA256：`ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df`。
- R4 旧 pilot 根：`/data4/users/sihao/workspace/molvid-dit-state-detail-pilot-v1/outputs/dit_state_detail_pilot_v1/full_20260908_S4500/ratio4_state_detail`。
- 旧 DiT 检查权重：`seed20260907_S4500/best_validation.pt`；SHA256 `bbd9c3710addecc05c0ecf41cd779b0ceaebcd4ed5f2e1bbd80e482e0b523992`。
- R4 statistics：`shared/statistics.pt`；文件 SHA256 `4d07bf53f317a6f2937a027695d2d70993674903d0fcefc1fa49e0482a529583`。另验证内部 semantic statistics hash，不把这两个 hash 混淆。
- R2 仅作 parity：路径复用 B `config/dit_backend_profile_v2.yaml`，codec SHA256 `b15cb92c34aec0e0f0c44e796def518d7ad89cda3dbc7f2fc3de83d55b4c64e9`，旧 DiT SHA256 `2d48363d272d0768281948df13e20d068b9da3a86fd3ece1540cdd8e6d173771`。

输入路径可以按已验证容器映射覆盖，内容 hash 不变；不能根据文件名相似就替换。

## 2. 模型与 tensor

固定 R4、C=128、K=4。packed atom 数为 N：

- state_h/detail_h：`[K,N,128]`。
- state_v/detail_v：`[K,N,3,128]`。
- DiT scalar_width=256、vector_width=128、depth=4、heads=8、FFN multiplier=4、dropout=0；参数量11,173,120。
- 同时保留 topology、atom/sample identity、真实 time_ps、valid masks、origin 和 observation mask。
- H4 的 observed token 数1，H8 的 observed token 数2。未来是其余完整 blocks。
- 固定 codec/decoder 的 eval/FP32 策略；DiT BF16 autocast，loss/指标 FP32。TF32 暂关闭，与 parity 口径一致。

## 3. Source 中心

默认候选 `repeat_last_coordinate_encode`：

1. 每个 sample 只读取 x[0:H]、已知 topology、masks 和 query times。
2. 构造16帧 template：前 H 帧保留观测，未来全部为 x[H-1]。
3. 沿用该 sample 的原始 frame0 origin 和冻结 codec 编码策略，得到 mu_raw；仅取 future fields 作为中心。
4. 生产 statistics 只作用一次，得到 m=N(mu_raw)。sample 间不共享中心。

替代诊断 `block_state_zero_detail`：最后完整 observed token 的 raw state_h/state_v 复制到 future；raw detail_h/detail_v 设0，再标准化。

两者都不使用未来坐标、未来 latent 均值或未来对齐。template 是条件构造，不作为静止 MD 监督数据。H4/H8 的中心分别计算并缓存。

验证与预先选择规则：

- 固定8个 validation 系统各 R1/window30；两种 H。另用 CUDA synthetic fixture 做 future mutation。
- 首选 repeat center 的 decode 与它自己的 coordinate template 比较，而非用真实未来挑中心。每 clip/H 的 raw 坐标 RMSD <= max(0.10 Å, 5倍该 clip codec oracle raw RMSD)，bond RMSE 相对 template <=0.10 Å，未来内部位移 RMS <=1e-4 Å。阈值是实现一致性检查，不是论文成功标准。
- 同时报告 decoded first-future 与 last-observed 的实际位移、对 MD 的 boundary error、bond/contact；这些读数不能替代上述实现检查。
- repeat center 符合条件即固定采用。若失败，先检查归一化、origin、mask、partial block、解码路径，修确定性 bug 后复查。
- 只有确认为 repeat 构造在冻结 codec 下存在实质问题，且 block-state center 的 finite/zero-detail/静止性/无泄漏检查通过、相对 template bond RMSE<=0.20 Å 时，才可采用已声明的替代；记录实际值和原因。不按未来 RMSD 决胜。两者都不满足时停止正式训练，报告中心/codec 问题。
- source 冻结后不得在训练过程中换中心或缩小噪声。

## 4. RF 数学与训练—推理一致性

以下只写 future 部分；observed clean clamp，invalid fields 归零。

```text
z = N(z_data_raw)
m = N(mu_raw)
eps ~ Normal(0, I)                  # 四 fields，每个有效系数相同 sigma=1

gaussian:    source = eps
conditional: source = m + eps

tau ~ Uniform(0,1), one value per sample
z_tau = (1-tau)*source + tau*z
target_velocity = z-source
loss = 1/4 * sum(masked_mean_square(predicted_velocity-target_velocity))
```

两臂都输入相同含义的 z_tau/history/time/metadata；conditional 没有额外 prior-only 通道。不将输入也切换成 residual 参数化，不重新拟合 residual 方差。

raw detail=0 通常对应 normalized scalar detail=-mean/std，不一定是0。Gaussian 的零均值指标准化空间；它不等于 raw-zero latent。

推理：按 source_mode 初始化 future，运行16步 Euler，逐步 clamp observed，最后 inverse-normalize 一次后 decode。8步仅诊断。sample 的 prior 只算一次；未来 fields 从未被 clamp 到 prior。

向量 center 按 SO(3) 旋转，噪声分布各向同性；逐样本等变测试必须将同一 eps 的 xyz 也一起旋转，不能用“旋转输入后同随机数张量”错误要求逐样本等变。

## 5. 公平训练与随机性

- 两臂都从新随机初始化训练，不加载旧已训练 DiT 作为其中一臂的起点；旧 checkpoint 只用于诊断和可丢弃 profile。
- `init_seed=20260910`：在 adapter/model 构造前固定 RNG，生成一份 init state_dict，分别 strict load 到两臂；按 tensor 字节验证 init_hash 相同。新 optimizer 独立。
- AdamW lr=2e-4、weight_decay=0.01、grad_clip=1.0；沿用当前无额外 LR scheduler 的配方，不单臂改 warmup/decay。
- 原 sampler：每 trajectory 每 epoch 抽24个 windows，max atom-frame tokens=80000；同一确定性 batch schedule，H 按 `(4,8)` 随步数交替。
- training_seed=20260910，单独的 tau/noise generator；两臂 generator 初始状态和每步 draw 数量相同。source 构造/缓存/validation 不能消耗 training RNG。
- checkpoint 保存 model/optimizer/scaler、实际成功 optimizer updates、epoch/batch cursor、RNG/generator、source/normalization/cache/预算合同、init_hash、code commit。
- 两臂 tau/noise shape 完全相同，可直接核对；未来 source 差异只应为 m。
- resume 恢复相同批次/噪声顺序；拒绝错误 source_mode/center/hash。旧 checkpoint 仅允许显式 weight-only 验证加载，不能静默作为新实验 resume。

## 6. 缓存与预算

缓存是本轮可选加速，先测后选：

- 统计 target fields 与两个 H 的 center 大小，估算全量 disk/RAM 和首次编码成本。默认 RAM cache 上限8 GiB、disk cache 上限64 GiB；不自动占满服务器存储。
- raw fields 保存 FP32，metadata 完整；若实验性 BF16 存储需另开变量，本轮不采用。缓存只读命中与直接 encode 在相同 gauge 下须数值一致。
- key 至少包含 clip 内容标识、codec hash、atom identity/order、time/mask/gauge、center_kind/H。中心缓存不允许携带真实未来信息。
- 原始 encode 是按 packed batch 实现的，若改成逐 sample 编码后重组，必须用真实 ragged batch 比较四 fields、origin、token mask、loss_mask、atom index/abid；不能假定数值和索引天然一致。
- 有界未命中可在线编码；并行两臂不竞争写同一文件。优先单一预计算 writer 后只读共享，或独立缓存 namespace。临时写入后原子发布；不能读取未完成 shard。
- 不按 cache 命中率重排训练数据、不减少所见样本；报告 compute 与 prep/I/O 两种成本。

预算：

- 使用1–2张空闲 GPU，无 DDP。两卡可各一臂；一卡按4500/10000/20000共同检查点轮流推进。
- 从 source/prep 到最终评估累计最多32 GPU-hours，最长36小时 wall time；无空闲 GPU 时不抢占，记录等待造成的中断。
- 在训练前实测至少20个有代表性的不同 train batches 的新 runner wall time，每臂5 warmup+20 measured disposable updates。包括中心和缓存模式，不用单次 encode 加法作为预算依据。
- 设总GPU秒预算B=32*3600；预留评估额度E=max(0.25*B, 1.3*estimated_evaluation_gpu_seconds)。仅当 `1.3*(candidate_steps*(p90_gaussian+p90_conditional)+estimated_prep_gpu_seconds) <= B-E` 时选择该共同终点。估计评估成本包含途中RF/生成监控以及终点评估。GPU秒与wall秒分开记；两卡不能把GPU成本除2，也要满足36小时wall上限。
- 优先共同20000步；超预算则共同10000步；仍不满足可共同4500步并标 `PARTIAL_BUDGET`。4500也不可行则只交付实现/检查，不超额开跑。
- 写 `budget.json` 后开始科学训练，不根据谁领先变更共同终点。固定检查点4500、10000（若达到）、20000（若达到）；训练效果差不自动截断一臂。
- 临近资源上限保存last，比较两臂都达到的最高共同检查点；未收敛如实标注，不以固定步数宣布收敛。

## 7. Validation 与终点评估

- validation_master_seed=20260910；stable hash(sample_id,H,draw_id,kind)，不得依赖 training step/batch offset/Python hash。
- RF validation：固定72 clips（8 systems ×3 replicas ×windows0/30/61），H4/H8，每个sample/H固定1组tau/eps；每500步及共同检查点运行。两臂各自记录loss，不能直接跨 source target 排名。
- 生成监控：每1000步及共同检查点，固定8 clips（R1/window30），H4/H8、16步、draws0/1。用独立 eval RNG；之后恢复 train 状态。
- 共同终点主评估：72 clips、H4/H8、16步、draw0。8-clip子集补到4 draws并增加8步对照；16步已有draw0/1可复用，避免重复采样。
- 主文表比较固定共同终点；附4500/10000过程曲线，另列各臂 best-RF 的结果和 step。主结论不靠挑一个最好看的 checkpoint/draw。
- future H4=[4,16)、H8=[8,16)；另报同 L4/L8。H4/H8目标区间不同，不能直接把差异当作更多history的因果收益。
- 主指标：future aligned RMSD/dRMSD（路径参考）、bond RMSE、contact F1/真正occupancy MAE、边界位移向量误差、future RMSF预测/目标及其分布、物理dt下位移分布、velocity-lag1 ACF。
- diversity 仅在 future 帧计算，observed不混入分母；报告对齐/raw pairwise RMSD，注明4 draws不是4个独立蛋白。
- per-draw→per-sample→per-system，配对报告8体系的差异；不使用best-of-N或把draws当统计独立重复。
- 常数轨迹是低点误差/几何基线，不是动力学成功标准。短窗口RMSF/ACF不能声称自由能面或长期跃迁时间正确。

## 8. 判断与下一阶段边界

本轮输出支持/反对/证据不足，并给出以下方向：

- conditional 几何更好且保留合理运动：保留 source 方案，下一阶段考虑多dt和更大数据。
- 几何更好但运动塌缩：报告分布和diversity，不自动调 sigma；单独设计下一轮。
- 两臂都几何失真：检查训练曲线后，下一阶段考虑 endpoint 几何监督或局部原子相互作用。
- state 稳定后detail仍受损：下一阶段才补R2同预算对照。

本轮不执行这些后续架构更新，也不宣称已经得到参数/数据 scaling law。
