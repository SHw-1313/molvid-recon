# B — 数值等价的 scalable backend v2

## 目的与边界

去掉现有block-spatial Python循环、碎片化kernel、热路径device-host同步以及重复codec encode。保留同一factorized DiT数学模型、参数量、weights和RF学习问题。不能在一次性能重构中加入local atom attention、relative geometry bias、persistent geometry refiner或新的conditional prior。

历史实测只说明DiT train_step R2约11.13s、R4约5.48s；没有operator级trace，不能把全部差值事先归到某个算子。R2 K=8 / R4 K=4，FFN/空间注意力等线性项约2倍、temporal logits项约4倍，但不是整网必然4倍。

实现文件按共享CONTRACT；新helper放 `module/dit_backend_v2.py`，benchmark放 `scripts/profile_dit_backend.py`。checkpoint/模型signature尽量兼容；旧backend保留明确reference路径用于验收，不允许silent fallback。

## B1. 冻结reference与contract

以a22f60c的MolecularDiT/factorized block/ScalarVectorAttention为reference。可把原函数复制到 `tests/reference/dit_backend_v1_reference.py`，注明commit，或保留可选legacy backend；不能用新实现同时冒充reference。

记录scalar_width=256、vector_width=128、depth=4、heads=8、ffn_multiplier=4、dropout=0及可训练参数数。R2/R4各使用原codec/schema/statistics。旧state_dict strict加载保持；非参数运行后端选择单独写execution_backend与semantic_contract_hash，不把它伪装为新物理模型。

如果当前contract把backend字符串包含在hash中，定义显式、仅针对本次等价backend的迁移/加载验证：完整检查所有语义字段和tensor shape/key，不删除hash检查或任意strict=False。原checkpoint文件不改；未批准的semantic mismatch仍报错。

## B2. 拓扑布局与vectorized pooling/broadcast

当前输入h `[K,N,Dh]`、v `[K,N,3,Dv]`，atom有sample与block归属。构造连续block index时键为(sample_id, block_id)，不同sample同block编号绝不合并。

在CPU已有metadata或一次性batch准备阶段生成：atom_to_block、block_to_sample、block_counts、local block slots、padding/gather映射。数据搬到CUDA后重复使用。不得每层/每tau重跑 `.tolist()` 或创建每block的小tensor。

整个K一次性scatter/index_add求和并除以atom_counts，保持原本的mean语义；不是sum，不擅自把loss_mask改成pooling_mask。得到block h/v，再pack为 `[B,K,M_max,Dh]` 和 `[B,K,M_max,3,Dv]`，attention可折叠B*K。不平衡分子大小明显时用少量size buckets或segment策略，但batch/样本权重保持。

得到block context后一次性gather到每atom。padding key不被看见，padding query与无效time token输出为0，处理全mask不会NaN。所有映射与batch/mask绑定，cache key不能只看N；碰到重排/同N不同topology必须更新。派生缓存不进入可学习latent、不依赖future坐标，也不修改序列化latent字段定义。

## B3. Attention执行实现

原attention的q/k来自scalar；同一attention权重同时作用于scalar V与vector V。vector xyz不进入learned q/k，不跨xyz任意投影。保持原scale=`1/sqrt(scalar_head)`、投影bias、head reshape顺序和dropout语义。

尝试SDPA，scalar/vector values可在保持head/xyz语义的前提下拼接或分别执行。先核验现有PyTorch支持与实际kernel选择；q/k与value head维度不同可能退回math backend，不能仅调用SDPA就宣称用上FlashAttention。当前dropout=0，若分别SDPA两次有nonzero dropout，其权重mask可能不同，应限制本优化支持dropout=0并保留显式reference行为，或证明shared-mask等价；不把dropout非零静默改成0。

不得通过把xyz混入任意线性层、改变attention scale或截断vector通道来让fast kernel可用。若SDPA无法加速，可以vectorized matmul+softmax作为实测后端，明确标记。空间和时间attention均保持bidirectional/noncausal既有语义。

除attention以外SO3ChannelNorm、vector FFN、AdaLN不改变。scalar/vector FFN原来每atom做，本次仍然每atom做；不能降为block-only模型来制造速度收益。

## B4. Runner与吞吐plumbing

- 删除训练loop中第二次`codec.model.encode(batch)`，直接复用 `_encode_batch` 已返回的latent；原helper pack重复若有也去重，不能保留名为优化实际又encode一次。
- frozen encoder/codec eval+no_grad，编码精度保持原状；需要改AMP必须单独记录为另一变量，本轮不改。
- batch布局从初次准备传到model，多ODE步复用；不在热路径hash整个模型/statistics，不把tensor验证简单删掉以换速度。可将可验证的静态metadata检查移动到创建边界，mutation仍必须被验证。
- 日志CPU转换允许发生在日志边界，不应散落每层attention。保留必要非finite检查；若 `.item()` loss count/log同步仍明显，在profile说明，优先收敛本次主变更，不触及冻结RF objective。
- 减少validation重复encoder可以在缓存证明正确后做，但不是本次通过的前置；完整磁盘latent缓存不做强制项。仅做小批量内存复用：固定geometry+无随机增强+相同gauge/codec hash时才有效。缓存规模必须profile后估算，不复制全dataset。
- 原pilot runner禁止意外被profile脚本用于开启1000/4500步训练。实现独立短profile runner。不要扩大原phase的训练step上限或切换学习率配方。
- 报告步数区分 `training_completed_steps`、`selected_checkpoint_step`、`resume_check_step`；不要复用旧summary里completed_steps=4100含混语义。旧summary不改。

## B5. CUDA数值等价性

新旧模型加载完全相同state_dict，在同device、输入、tau、noise、mask上对比：

- forward四fields，masked RF loss，所有参数梯度，单步optimizer更新；
- R2/R4、H4/H8、多个sample、ragged blocks、非连续block ID、相同block ID不同sample、padding与全mask；
- SO(3)旋转、平移origin、observation clamp及sample isolation；
- 16次ODE forward复用layout与重建layout结果一致；改变topology后旧cache失效；
- gradient测试使用非零modulation/gates（实际checkpoint或非退化fixture），不能只测AdaLN-zero导致attention分支全被乘0的初始化。

先FP32禁用TF32、固定权重与mask。默认目标 `atol=2e-5, rtol=2e-4`，同时报告max_abs与relative_L2（near-zero使用absolute尺度）。BF16目标 `atol=2e-2, rtol=5e-2` 为起始预声明；若真实data超界，应分析非结合加法/AMP/错误，而不是看结果后无限放宽。梯度对norm极小tensor单列绝对误差，不能用max-relative制造伪失败或遮掩关键偏差。

一个真实R4 clip从编码到采样解码smoke，R2再做同类加载/forward检查。旧checkpoint测试不是以随机模型取代；若缺真实权重，synthetic测试可完成，但真实证据标BLOCKED。

## B6. 短profile设计

先预载同一批真实train-only数据、latent及权重；不要在precision comparison或compute-only计时内读取文件。端到端另计，包括一次encoder；两种路径都使用相同输入batch。

在train split元数据中预先按原子/block规模选small/median/large合法batch，不读取test。R4三档、R2 median。每档reference/optimized串行，默认各5 warmup + 20 measured optimizer steps；每次从同权重/optimizer/RNG副本起步，交替测试先后顺序或重复forward计时以检测时钟偏差。合计默认200个短训练步（两backend × 四case × 25），不是两轮pilot。必要补测先说明原因，单candidate不自动升级成200-step以上持续训练。

warmup后reset CUDA peak；先释放另一个模型的GPU拷贝与不必要缓存，防止reference+optimized同时常驻污染显存。用CUDA events分别计前向/反向/optimizer；用perf_counter+必要同步计host端总耗时。trace采集只取几步，trace开销不计入steady throughput。

报告model-only与end-to-end两套：median/p90 seconds、samples/s、atom-frame tokens/s、latent atom-time tokens/s（K*N）、scalar+vector coefficient volume、max/padding block count、GPU allocated/reserved、GPU UUID、dtype/AMP/TF32、实际attention kernel/backend。旧all-process peak=~26GB不能替代本轮reset后的train peak。

trace拆出block grouping、pool/broadcast、spatial/temporal attention、FFN、encoder、H2D/metadata、logging/synchronization。不能单凭K翻倍推断具体耗时占比。

没有强行规定“必须快5倍”；如果未提速或只小幅提速，就交付瓶颈定位和实测结果，不宣称已经scalable-ready。至少输出支持/不支持进入下一轮的判断：等价性、吞吐收益、padding长尾是否可接受。R2仍慢于R4可完全合理。

## 实现后的CLI与输出

需要实现、验证以下新接口：

```text
python scripts/profile_dit_backend.py --config config/dit_backend_profile_v2.yaml --stage preflight
python scripts/profile_dit_backend.py --config config/dit_backend_profile_v2.yaml --stage equivalence --device cuda:0
python scripts/profile_dit_backend.py --config config/dit_backend_profile_v2.yaml --stage smoke --device cuda:0
python scripts/profile_dit_backend.py --config config/dit_backend_profile_v2.yaml --stage profile --device cuda:0
```

所有命令在enter-container/torch-ito中执行，先实现再`--help`。checkpoint/data路径可指向A的只读output；不要假定Git worktree携带被ignore的二进制。

新输出：`outputs/dit_factorized_backend_v2/<run_id>/`，包含profile inputs/hash、parity.json、profile.json、report.md、短trace（本地）。小型摘要复制到本任务 `evidence/` 提交。memory/thread/CPU sync瓶颈都如实写出。

完成更新TASKS/DECISIONS/HANDOFF，本地提交B拥有文件。等价/真实profile完成后停止；不自动合并A、不建conditional-source模型。
