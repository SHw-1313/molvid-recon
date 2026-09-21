# Checkpoint 来源与生命周期

## 首选 target encoder 来源

重构仓库 `docs/migration.md` 记录的已选 codec：

```text
PVB/outputs/state_detail_codec_v2/t1/full_20260904_seed20260903/ratio4_state_detail/codec_best.pt
SHA-256: ba10c44189cca837430abbd64afce2109a0daf0bda4f05971e0441abb2a5e6df
```

远程工作空间历史位置为 `/data4/users/sihao/workspace`，容器内可能有不同挂载。先从现有配置/记录解析实际路径，再校验 hash；不要因相同 basename 就选另一份权重。此交付包没有包含 checkpoint，也没有在当前环境重新加载其张量。

引用来源：https://github.com/SHw-1313/molvid-recon/blob/93d191f/docs/migration.md

这是旧 codec 的来源，不是新 DiT 或新 decoder 的权重。提取的是 Haar 前 `frame_encoder` 和 `coordinate_stem`，因此使用 R4 checkpoint 不意味着未来还生成 R4 token。不因目标逐帧而重新寻找所谓“R1 teacher”。

## 严格提取

1. 读取实际 artifact schema、model contract、source SHA、空间构造参数和 stem 模式。
2. 复用 `molvid/checkpoints.py` 中经过验证的名字映射：旧 `frame_encoder.spatial_encoder.*` → `frame_encoder.backbone.*`，旧 `coordinate_vector_stem.projection.*` → `coordinate_stem.projection.*`。
3. 逐项核对目标模块的 key、shape、dtype；每个目标参数必须有来源，不能静默随机补齐。不要从另一个 checkpoint 拼接 stem。
4. 加载完整 codec 后提取，或增加清楚命名的 weights-only 提取函数均可。新 encoder warm start 不需要迁移旧 optimizer；不要为了只读权重强制运行完整旧 optimizer 恢复。
5. 在真实 CUDA batch 上比较旧 codec Haar 前 h/v 与新 frame encoder 的输出，使用同一 center、拓扑与 atom ordering。复用已有数值容差证据，不为通过测试擅自放宽容差。
6. 输出 `target_encoder_provenance.json`：源路径/hash、映射清单、子模块权重 hash、构造参数、坐标约定、检查结果、当前代码 commit。

迁移记录同时出现 `result.json` 的 45,844 steps 与实际 best artifact 的 39,721 step 描述。不要把 final run 的步数硬贴给 best checkpoint；以实际加载 artifact 的计数为准并说明差别。

## 新模型初始化

- geometry+stem：从上面的同一 codec 加载后冻结。
- history Haar/state/detail：只加载形状和语义都兼容的参数；time/temporal 新增层明确初始化。
- decoder：坐标头可以复制作为起点；新 temporal/local block 初始化。由于现在读的是逐帧特征，不能宣称直接继承了旧 codec 的重建精度，必须做 decoder 预热。
- Frame DiT：主干若明确兼容可部分加载；输入/输出 adapter、history cross-attention、时间模块新建。记录完整加载报告，不用无报告的 strict=False。
- 新逐帧 latent statistics：只在训练 split 拟合，不加载旧 state/detail 的 statistics.pt；vector 的 xyz 三轴共享尺度。

本版固定逐帧目标空间，不实现在线更新 teacher。报告应称“历史编码器—DiT—decoder 联合训练”，不要称 TorchMD 也端到端解冻。

## 新模型 checkpoint 必须能恢复什么

模型各组件、frozen teacher 身份、latent 统计及 provenance、optimizer param-group 名称与状态、scheduler、成功 update 计数、sampler/epoch/cursor、每 rank RNG/训练随机 generator、精度/scaler（如有）、已处理数据曝光量、代码/config/manifest hash。

普通 resume 必须保持目标函数/数据等语义。改变架构或 loss 是新 experiment ID，明确 parent。不要强行通过旧的 strict resume 契约来伪装这种变化。

## bond continuation

`bond_keep` 与 `bond_release` 从同一个新模型 joint checkpoint 分叉，复制相同模型、统计、optimizer moments、sampler cursor 与 RNG。两臂进入同样的新 continuation scheduler，详见 TRAINING_EVAL.md。记录 parent 完整 SHA 与分叉 step。

新任务的 optimizer 与统计要由新路径正常创建。旧 codec 缺失 RNG 等情况不阻塞 weights-only 初始化，但绝不把它写成 exact training resume。
