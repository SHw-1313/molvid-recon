# molvid 文件与类重构设计（v2：浅层包布局）

基准：`exp/dit-architecture-sequential-v1` / `d8f674aad692c8426cf9240d07b3384f7a043888`。这是沿用上一版源码审阅基准的目标设计。本次仅修改设计文件，没有重新核验远端 HEAD、修改仓库或运行数值等价性验证。

保持当前 TorchMD + state/detail codec + DiT/rectified-flow 路径；保留现有 codec 对照、静态数据、conditional/gaussian source、future-bond 与 history corruption 选项。

版本：`v2-flat-20260918`。正式运行代码仍为 50 个 Python 文件、74 个类（不计包标记）；类、函数和保留范围与上一版一致。下列函数名是目标接口；由旧实现迁移/提取，不表示仓库中已存在同名函数。

## 0. 本版变更与数据位置

删除 `src/` 容器层；把原 `models/` 下的 `spatial/`、`codec/`、`dit/` 提到 `molvid/` 一级；`equivariant.py` 也提到包根。保留 `losses/` 的两个文件：上一次简化示意树漏列它们，不表示删除损失实现。

| 内容 | 位置 |
|---|---|
| 正式 Python 包 | 仓库根目录的 `molvid/` |
| 数据处理代码 | `molvid/data/` |
| 原始轨迹、静态结构、clip store | 原 NAS/数据盘，位置由配置指定，不移动已有数据 |
| 可选本地数据入口 | 根目录 `datasets/`，不跟踪、不打包；无需创建，更不自动创建符号链接 |
| 小型回归样本 | `tests/fixtures/`，仅提交适合共享的小样本 |
| 大型 checkpoint、数值参考和生成输出 | 外部存储或根目录 `outputs/`，不打包、不提交大文件 |

`<repo>/` 只是下面树中的仓库根目录标签，不是再增加一层文件夹。`molvid.data` 是代码包；根目录不再新增一个同名的 `data` Python 包。

## 1. 完整目标文件树

```text
<repo>/
├── .gitignore
├── AGENTS.md
├── LICENSE
├── NOTICE.md
├── README.md
├── benchmarks/
│   ├── overfit.py
│   └── profile.py
├── configs/
│   ├── codec_train.yaml
│   ├── dit_train.yaml
│   ├── evaluate.yaml
│   ├── machine.example.yaml
│   ├── preprocess.yaml
│   └── sample.yaml
├── docs/
│   ├── architecture.md
│   └── migration.md
├── env.yaml
├── experiments/
│   └── architecture_20260914/
│       ├── baseline.yaml
│       ├── protocol.py
│       ├── results.md
│       ├── round1.yaml
│       ├── round2.yaml
│       ├── round3.yaml
│       └── round4.yaml
├── molvid/
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── evaluate.py
│   │   ├── preprocess.py
│   │   ├── sample.py
│   │   ├── train_codec.py
│   │   └── train_dit.py
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── haar.py
│   │   ├── heads.py
│   │   ├── model.py
│   │   ├── state_detail.py
│   │   └── types.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── batch.py
│   │   ├── chemistry.py
│   │   ├── io.py
│   │   ├── manifest.py
│   │   ├── preprocess.py
│   │   ├── sampling.py
│   │   └── store.py
│   ├── dit/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   ├── blocks.py
│   │   └── model.py
│   ├── equivariant.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   ├── latent.py
│   │   ├── motion.py
│   │   ├── report.py
│   │   └── runner.py
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── objective.py
│   │   ├── sampling.py
│   │   └── source.py
│   ├── generation.py
│   ├── geometry/
│   │   ├── __init__.py
│   │   ├── coordinates.py
│   │   ├── frames.py
│   │   ├── neighbors.py
│   │   ├── topology.py
│   │   └── types.py
│   ├── latent/
│   │   ├── __init__.py
│   │   ├── adapter.py
│   │   ├── conditioning.py
│   │   ├── statistics.py
│   │   └── types.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── geometry.py
│   │   └── reconstruction.py
│   ├── runtime.py
│   ├── spatial/
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── ops.py
│   │   └── torchmd.py
│   └── training/
│       ├── __init__.py
│       ├── batches.py
│       ├── codec.py
│       └── dit.py
├── pyproject.toml
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   ├── reference.py
│   ├── test_codec.py
│   ├── test_conditioning.py
│   ├── test_data.py
│   ├── test_dit.py
│   ├── test_flow.py
│   ├── test_generation.py
│   ├── test_geometry.py
│   ├── test_imports.py
│   ├── test_latent.py
│   ├── test_losses.py
│   ├── test_metrics.py
│   ├── test_migration.py
│   └── test_training.py
└── tools/
    ├── inspect_dataset.py
    ├── inspect_model.py
    └── migrate_artifacts.py
```

所有包的 `__init__.py` 仅作为包标记；根包可定义 `__version__`，不重导出旧模型。

包发现必须限定到根目录 `molvid` 和 `molvid.*`，不能把 tests、tools、benchmarks、experiments、datasets 或旧根目录包一起安装；`pyproject.toml` 不再配置 `src` 映射。使用现有环境，不顺手升级依赖。

所有 import 和配置中的目标模块路径同步更新，例如：

```python
from molvid.codec.model import TrajectoryCodec
from molvid.spatial.encoder import FrameEncoder
from molvid.data.batch import ClipBatch
```

正式入口维持：

```bash
python -m molvid.cli.train_codec --config configs/codec_train.yaml
python -m molvid.cli.train_dit --config configs/dit_train.yaml
python -m molvid.cli.sample --config configs/sample.yaml
python -m molvid.cli.evaluate --config configs/evaluate.yaml
```

上述 Python 命令均在既有 `enter-container` / `torch-ito` 环境中执行；这是重构完成后的入口，不表示当前旧仓库已存在这些模块。


## 2. 每个运行文件保留的类与函数

### `molvid/config.py`

**类：** 无。

**公共函数：** `load_config()`、`resolve_config()`、`validate_config()`

**来源：** `现有 YAML 读取、配置校验逻辑`

读配置、解析路径、校验结构；阶段预算放 experiments/architecture_20260914/protocol.py。

### `molvid/runtime.py`

**类：** 无。

**公共函数：** `configure_device()`、`seed_all()`、`autocast_context()`、`sha256_file()`、`canonical_hash()`、`atomic_write_json()`、`append_metrics()`

**来源：** `各 runner 的 CUDA/RNG/日志/哈希函数`

运行环境与结果写入；不定义模型或实验选择规则。

### `molvid/checkpoints.py`

**类：** `CheckpointLoadReport`、`CodecArtifact`

**公共函数：** `load_codec_artifact()`、`save_training_checkpoint()`、`load_training_checkpoint()`、`capture_rng_state()`、`restore_rng_state()`

**来源：** `module/multiframe_codec.py`；`trainer/codec_trainer.py`；`trainer/dit_trainer.py`；`scripts/run_state_detail_dit_pilot.py`

CodecArtifact 替代 FrozenCodec；保留权重、optimizer、scheduler、scaler、RNG、sampler 游标与来源核查。

### `molvid/data/batch.py`

**类：** `ClipBatch`、`ClipValidationError`

**公共函数：** `validate_clip_record()`、`collate_clip_records()`、`canonical_time_fields()`

**来源：** `data/clip_dataset.py`

数据字段、单位、mask、collate；不负责磁盘文件。

### `molvid/data/store.py`

**类：** `BlockStore`、`ClipMMapDataset`、`ClipMMapWriter`、`StaticClipDataset`

**公共函数：** `static_record_to_clip()`

**来源：** `data/mmap_dataset.py`；`data/clip_dataset.py`

BlockStore 由 MMAPDataset 的存储读取能力迁入；保留已有静态数据和 clip store，不保留 x0/x1 训练工作流。

### `molvid/data/sampling.py`

**类：** `ClipItemSpec`、`ClipSpecTable`、`TaskAwareClipBatchSampler`、`TrajectoryCappedBatchSampler`

**公共函数：** `make_clip_dataloader()`、`get_clip_specs()`

**来源：** `data/clip_batching.py`；`scripts/run_state_detail_codec_v2_t1.py`

统一 T*N 预算、静动态分组、每轨迹采样上限及已有分布式切分。

### `molvid/data/io.py`

**类：** 无。

**公共函数：** `read_topology()`、`read_trajectory()`、`read_static_records()`、`write_trajectory()`

**来源：** `data/trajectory_clips.py`；`现有静态数据读取与推理输出函数`

原始 PDB/轨迹/HDF5/静态记录的读写，及生成结果导出；外部格式库局部导入。

### `molvid/data/preprocess.py`

**类：** `ClipPreprocessConfig`、`SourceDataError`

**公共函数：** `preprocess_atlas()`、`preprocess_misato()`、`preprocess_static()`、`iterate_windows()`

**来源：** `data/trajectory_clips.py`；`scripts/preprocess_trajectory_clips.py`；`现有 ANI1x/PCQM4Mv2/PDB/PDBBind 静态预处理逻辑`

原始数据转 clip；保留原子顺序、物理时间及静态 T=1 语义。函数名为迁移后的目标接口。

### `molvid/data/manifest.py`

**类：** `DatasetSplits`、`ValidationPlan`

**公共函数：** `build_manifest()`、`load_datasets()`、`make_validation_plan()`、`check_split_overlap()`

**来源：** `scripts/run_state_detail_dit_pilot.py`；`scripts/build_state_detail_codec_v2_t1_manifest.py`

DatasetSplits 替代 PilotData；样本身份与划分独立于某次 64-system 实验。

### `molvid/data/chemistry.py`

**类：** 无。

**公共函数：** `get_block_from_top()`、`get_block_from_complex()`、`parse_static_molecule()`

**来源：** `utils/bio_utils.py`；`静态源预处理函数`

保留原子、残基、block 类型词表和化学解析；不改变现有类型编号。

### `molvid/geometry/types.py`

**类：** `StaticTopologyMetadata`、`FrameNodeBatch`、`FrameGraphBatch`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_codec_v2.py`；`module/multiframe_codec.py`

坐标无关的拓扑元数据及逐帧图数据结构。

### `molvid/geometry/frames.py`

**类：** 无。

**公共函数：** `pack_frame_nodes()`、`build_frame_graph()`、`unpack_frame_features()`

**来源：** `module/multiframe_codec.py`

从 FrameEncoder 抽取打包、跨帧隔离建图与特征恢复。

### `molvid/geometry/neighbors.py`

**类：** `NeighborList`、`CudaRadiusNeighborList`

**公共函数：** 使用上列类的方法。

**来源：** `module/neighbor_graph.py`

真实 CUDA 邻居搜索和可微 FP32 几何；dense 参考实现迁到测试。

### `molvid/geometry/topology.py`

**类：** `TopologyEntry`、`BoundedTopologyCache`、`DistanceOnlyBondCache`

**公共函数：** `build_canonical_reference_index()`、`stable_topology_id()`

**来源：** `module/topology_cache.py`；`module/bond_sources.py`

缓存静态拓扑和 distance-only 参考图，合并重复 topology-id 函数。

### `molvid/geometry/coordinates.py`

**类：** 无。

**公共函数：** `compute_masked_centroid_origin()`、`center_coordinates()`、`restore_origin()`、`kabsch_align()`、`align_and_center()`

**来源：** `module/state_detail_codec_v2.py`；`data/trajectory_clips.py`；`utils/geometry.py`

原点、旋转、对齐、恢复；保持现有对齐策略与公式。

### `molvid/spatial/encoder.py`

**类：** `FrameEncoder`、`FrameEncoderOutput`

**公共函数：** 使用上列类的方法。

**来源：** `module/multiframe_codec.py`

PVBFrameEncoder 改为 FrameEncoder；只组装骨干、图缓存和多帧编码，不加载 checkpoint。

### `molvid/spatial/torchmd.py`

**类：** `TorchMDEncoder`、`EquivariantMultiHeadAttention`

**公共函数：** 使用上列类的方法。

**来源：** `module/torchmd_et.py`

TorchMD_VQ_ET 的当前空间编码路径迁为 TorchMDEncoder；去掉旧桥 decoder 专用路径。

### `molvid/spatial/ops.py`

**类：** `NeighborEmbedding`、`CosineCutoff`、`GaussianSmearing`、`ExpNormalSmearing`、`ShiftedSoftplus`

**公共函数：** `scatter()`

**来源：** `utils/torchmd_utils.py`

保留骨干用到的 RBF、cutoff、邻居嵌入、激活映射和 scatter；外部图路径不用内部 OptimizedDistance。

### `molvid/equivariant.py`

**类：** `AxisPreservingLinear`、`SO3ChannelNorm`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_latent_adapter.py`；`module/molecular_dit.py`

这里只迁 DiT 的 SO3ChannelNorm，不把旧 TorchMD SO3LayerNorm 当作同一种归一化。

### `molvid/codec/model.py`

**类：** `TrajectoryCodec`

**公共函数：** `build_codec()`

**来源：** `trainer/codec_trainer.py::PVBCodecModel`

总装 frame encoder、coordinate stem、temporal codec、coordinate head；方法 encode/decode/forward/prepare_batch。

### `molvid/codec/types.py`

**类：** `StateDetailLatent`、`MatchedPoolingLatent`、`CodecOutput`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_codec_v2.py`

CodecOutput 替代 StateDetailDecoderOutput；不定义神经网络。

### `molvid/codec/haar.py`

**类：** `HaarCoefficients`

**公共函数：** `haar_lift()`、`haar_inverse()`

**来源：** `module/state_detail_codec_v2.py`

HaarCoefficients 替代数据类 HaarLift；固定系数和有效性 mask，不新增小波网络。

### `molvid/codec/state_detail.py`

**类：** `StateDetailCodec`、`MatchedPoolingCodec`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_codec_v2.py`

移除 V2 和重复别名；只编码/解码时间特征，坐标 head 迁到总装。保留现有 codec 对照。

### `molvid/codec/heads.py`

**类：** `CoordinateVectorStem`、`EquivariantCoordinateHead`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_codec_v2.py`

分别由 CenteredCoordinateVectorStem 和 _EquivariantCoordinateHead 改名；计算不变。

### `molvid/dit/model.py`

**类：** `MolecularDiT`

**公共函数：** `build_dit()`

**来源：** `module/molecular_dit.py`

DiT 总装、条件 embedding、forward；保留 adapter 为唯一共享对象。

### `molvid/dit/blocks.py`

**类：** `FactorizedDiTBlock`、`ScalarVectorAttention`、`AdaLNZero`、`ScalarVectorFFN`

**公共函数：** 使用上列类的方法。

**来源：** `module/molecular_dit.py`

DiT 的参数层和块计算；不拆成一类一文件。

### `molvid/dit/backend.py`

**类：** `FactorizedLayout`

**公共函数：** `build_factorized_layout()`、`factorized_block_forward()`、`reference_block_forward()`

**来源：** `module/dit_backend_v2.py`；`module/molecular_dit.py`

保留向量化 CUDA 与原数值参考路径；v2 不再写入文件名。

### `molvid/latent/types.py`

**类：** `LatentFields`、`LatentBatch`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_latent_adapter.py`

分别替代 LatentFieldSet 和 DiTLatentBatch；只承载字段、mask 和基本结构操作。

### `molvid/latent/adapter.py`

**类：** `StateDetailLatentAdapter`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_latent_adapter.py`

目标方法为 from_codec_latent/project_inputs/project_outputs/make_generated_latent，由既有转换逻辑提取；不负责统计、哈希、观测噪声。

### `molvid/latent/statistics.py`

**类：** `LatentStatistics`

**公共函数：** 使用上列类的方法。

**来源：** `module/state_detail_latent_adapter.py`

fit、normalize、inverse_normalize、state_dict；训练集统计和原有向量缩放语义不变。

### `molvid/latent/conditioning.py`

**类：** `ObservationCondition`、`HistoryCorruptionViews`、`HistoryConditionedLatents`

**公共函数：** `build_observation_condition()`、`frame_prefix_observation_mask()`、`sample_history_sigmas()`、`make_history_corruption_views()`、`combine_clean_target_with_condition()`

**来源：** `module/state_detail_latent_adapter.py`；`module/dit_history_corruption.py`

观测条件与在线历史噪声；保留干净 target / 加噪 condition / origin 关系。固定 Round4 概率迁到配置。

### `molvid/flow/objective.py`

**类：** `FlowSample`、`FlowLoss`、`RectifiedFlowObjective`

**公共函数：** `rectified_flow_interpolate()`、`rectified_flow_velocity()`、`four_field_loss()`、`endpoint_from_velocity()`

**来源：** `module/latent_rectified_flow.py`；`module/dit_geometry_supervision.py`

flow 数学与目标；endpoint 公式不再藏在 Round2 辅助损失文件中。

### `molvid/flow/source.py`

**类：** 无。

**公共函数：** `build_observed_center()`、`repeat_last_coordinate_template()`、`sample_source()`

**来源：** `module/latent_flow_source.py`

高斯或条件源构造；不含整套旧 bridge matcher。

### `molvid/flow/sampling.py`

**类：** 无。

**公共函数：** `euler_sample()`、`apply_observation_clamp()`、`generate_state_detail_latent()`

**来源：** `module/latent_rectified_flow.py`

latent 数值积分和观测钳制；不计算 RMSD、不写报告。

### `molvid/generation.py`

**类：** 无。

**公共函数：** `sample_clip()`、`rollout()`

**来源：** `scripts/run_dit_architecture_round4.py`；`evaluation/dit_reassessment.py`；`现有生成函数`

坐标条件→latent 采样→解码，以及滚动续写；接口不接收真实未来坐标。

### `molvid/training/codec.py`

**类：** `CodecTrainConfig`、`CodecTrainer`

**公共函数：** `train_codec()`

**来源：** `trainer/codec_trainer.py`；`train_codec.py`

codec 的优化和训练调度；模型总装、loss、保存分别迁出。

### `molvid/training/dit.py`

**类：** `DiTTrainConfig`、`DiTTrainer`

**公共函数：** `train_dit()`

**来源：** `trainer/dit_trainer.py`；`各 DiT runner`

DiT 的优化和训练调度；冻结参数与几何损失对输入的梯度路径分开处理。

### `molvid/training/batches.py`

**类：** `PreparedDiTBatch`

**公共函数：** `prepare_batch_then_to_device()`、`encode_batch()`、`prepare_dit_batch()`

**来源：** `trainer/codec_trainer.py`；`scripts/run_state_detail_dit_pilot.py`；`scripts/run_dit_architecture_round4.py`

PreparedDiTBatch 替代 PreparedRound4Batch；清楚组织真值、观测、源中心和噪声视图。

### `molvid/losses/reconstruction.py`

**类：** `CodecLossWeights`、`BucketNormalization`、`TimeBucketSpec`

**公共函数：** `compute_codec_losses()`、`masked_coordinate_loss()`、`bond_length_loss()`、`local_contact_distance_loss()`、`velocity_loss()`、`acceleration_loss()`、`fit_time_bucket_normalization()`

**来源：** `trainer/codec_losses.py`；`trainer/codec_trainer.py`

codec 重建目标和按物理时间分桶的损失归一化。

### `molvid/losses/geometry.py`

**类：** `FutureBondLoss`、`FutureBondAuxiliary`

**公共函数：** `future_bond_distance_loss()`

**来源：** `module/dit_geometry_supervision.py`

保留已有 future-bond 辅助监督；不添加新几何约束。

### `molvid/evaluation/geometry.py`

**类：** 无。

**公共函数：** `coordinate_errors()`、`bond_errors()`、`contact_scores()`、`clash_statistics()`

**来源：** `evaluation/codec_evaluation.py`；`evaluation/dit_reassessment.py`；`utils/geometry.py`

迁移现有坐标/键/接触/碰撞指标；不同对齐或归约定义保留明确命名，不强行合并。

### `molvid/evaluation/motion.py`

**类：** `MotionMetricError`、`RMSFRead`

**公共函数：** `rmsf()`、`velocity_metrics()`、`acceleration_metrics()`、`fft_retention()`、`read_rmsf_value()`

**来源：** `evaluation/motion_metrics.py`；`evaluation/codec_evaluation.py`；`evaluation/dit_reassessment.py`

计算与读取运动指标；保留零值、缺失、冲突、分母不足的区别。

### `molvid/evaluation/latent.py`

**类：** 无。

**公共函数：** `latent_field_errors()`、`oracle_vs_generated()`

**来源：** `evaluation/dit_diagnostics.py`；`evaluation/dit_evaluation.py`

四字段误差、oracle reconstruction 与生成误差分离。

### `molvid/evaluation/runner.py`

**类：** `EvalConfig`、`RolloutTrack`

**公共函数：** `evaluate_codec()`、`evaluate_generation()`、`evaluate_rollout()`

**来源：** `evaluation/codec_evaluation.py`；`evaluation/dit_evaluation.py`；`scripts/run_dit_architecture_round4.py`

EvalConfig 是从现有评估参数抽取的数据类；RolloutTrack 为评测参考轨迹，不进入 generation 条件。

### `molvid/evaluation/report.py`

**类：** 无。

**公共函数：** `aggregate_by_system()`、`aggregate_by_time_bucket()`、`compare_runs()`、`write_report()`、`plot_report()`

**来源：** `evaluation/dit_reassessment.py`；`evaluation/dit_reassessment_interpretation.py`；`scripts/plot_codec_results.py`

结果聚合与展示；各轮 KEEP/REJECT 规则留在实验 protocol。

### `molvid/cli/preprocess.py`

**类：** 无。

**公共函数：** `main()`

**来源：** `相应旧入口和实验 runner`

仅 argparse、配置解析和调用公共流程，不定义模型和训练循环。

### `molvid/cli/train_codec.py`

**类：** 无。

**公共函数：** `main()`

**来源：** `相应旧入口和实验 runner`

仅 argparse、配置解析和调用公共流程，不定义模型和训练循环。

### `molvid/cli/train_dit.py`

**类：** 无。

**公共函数：** `main()`

**来源：** `相应旧入口和实验 runner`

仅 argparse、配置解析和调用公共流程，不定义模型和训练循环。

### `molvid/cli/sample.py`

**类：** 无。

**公共函数：** `main()`

**来源：** `相应旧入口和实验 runner`

仅 argparse、配置解析和调用公共流程，不定义模型和训练循环。

### `molvid/cli/evaluate.py`

**类：** 无。

**公共函数：** `main()`

**来源：** `相应旧入口和实验 runner`

仅 argparse、配置解析和调用公共流程，不定义模型和训练循环。

## 3. 旧名到新名

| 旧名 | 新名/处理 |
|---|---|
| `PVBCodecModel` | `TrajectoryCodec` |
| `PVBFrameEncoder` | `FrameEncoder` |
| `PVBFrameGraph` | `删除别名；使用 FrameGraphBatch` |
| `TorchMD_VQ_ET` | `TorchMDEncoder` |
| `StateDetailCodecV2` | `StateDetailCodec` |
| `MatchedPoolingCodecV2` | `MatchedPoolingCodec` |
| `StateDetailLatentV2` | `删除别名；使用 StateDetailLatent` |
| `StateDetailDecoderOutput` | `CodecOutput` |
| `CenteredCoordinateVectorStem` | `CoordinateVectorStem` |
| `_EquivariantCoordinateHead` | `EquivariantCoordinateHead` |
| `HaarLift` | `HaarCoefficients` |
| `LatentFieldSet` | `LatentFields` |
| `DiTLatentBatch` | `LatentBatch` |
| `FrozenCodec` | `CodecArtifact` |
| `PilotData` | `DatasetSplits` |
| `PreparedRound4Batch` | `PreparedDiTBatch` |
| `MMAPDataset` | `BlockStore` |

## 4. 模型组合关系

```text
TrajectoryCodec
├── frame_encoder: FrameEncoder
│   ├── backbone: TorchMDEncoder
│   ├── neighbor_list: CudaRadiusNeighborList
│   └── topology/cache: BoundedTopologyCache / DistanceOnlyBondCache
├── coordinate_stem: CoordinateVectorStem
├── temporal_codec: StateDetailCodec 或 MatchedPoolingCodec
└── coordinate_head: EquivariantCoordinateHead

MolecularDiT
├── adapter: StateDetailLatentAdapter
├── 原有 flow/物理时间/类型/观测 embeddings
└── blocks: FactorizedDiTBlock × depth
    ├── spatial: ScalarVectorAttention
    ├── temporal: ScalarVectorAttention
    ├── ffn: ScalarVectorFFN
    └── 三个 AdaLNZero
```

`TrajectoryCodec.encode`：居中 → FrameEncoder → CoordinateVectorStem → temporal_codec.encode。
`TrajectoryCodec.decode`：temporal_codec.decode_features → coordinate_head → 恢复 sample_origin。
`TrajectoryCodec.forward`：仅重建调用 encode/decode，不计算 loss。
`DiTTrainer.model.adapter is DiTTrainer.adapter` 必须成立，不能复制训练两个 adapter。
未来真实坐标只进入训练标签/评估参考，不能进入 `generation.sample_clip` 和 `generation.rollout` 的条件接口。

## 5. Checkpoint 映射

映射较具体的前缀必须先于一般前缀处理：

```text
state_detail_codec.coordinate_head.*  -> coordinate_head.*
coordinate_vector_stem.projection.*   -> coordinate_stem.projection.*
frame_encoder.spatial_encoder.*       -> frame_encoder.backbone.*
state_detail_codec.*                  -> temporal_codec.*
```

只按锚定前缀匹配一次，不做全局字符串替换；实际 checkpoint 的 model/module/EMA 外层前缀先按格式确认。检查同名冲突、遗漏键、shape/dtype 和张量值。转换还要校验 optimizer 参数与名称/顺序的对应关系，不靠重建 optimizer 忽略旧状态。旧 schema/hash 先按原格式验证，再转换并记录新格式与来源。数据 reader 可在内存适配已有 pvb.clip.v1，不要求为改名重打包数据。完整 pickle 模型在旧 Git 提交环境导出 state_dict 后迁移；新主包不保留 PVB 别名。

## 6. 测试、工具、实验文件

| 文件 | 保留类 | 职责 |
|---|---|---|
| `tests/conftest.py` | 无 | 共享真实小模型、tiny clip、固定噪声 fixtures |
| `tests/reference.py` | `DenseReferenceNeighborList` | 从现有 dense_test 提取的仅测试邻居参考；不作为 CUDA fallback |
| `tests/test_data.py` | 无 | record/时间单位/静态 T=1、store、采样调度、split 隔离 |
| `tests/test_geometry.py` | 无 | 跨帧跨样本边、原子顺序、缓存、CUDA 邻居与参考结果 |
| `tests/test_codec.py` | 无 | Haar 往返、静态零运动、encode/decode、codec 对照 |
| `tests/test_latent.py` | 无 | 四字段形状、adapter 往返、train-only 统计、归一化往返 |
| `tests/test_conditioning.py` | 无 | 改变真实未来不影响 condition；加噪 condition 不污染 clean target；origin 一致 |
| `tests/test_dit.py` | 无 | DiT shape、旋转行为、reference/optimized 前向与梯度一致 |
| `tests/test_flow.py` | 无 | 插值/速度/终点、source、Euler 与观测钳制 |
| `tests/test_losses.py` | 无 | 重建/几何辅助目标、mask、冻结 decoder 仍向 DiT 回传梯度 |
| `tests/test_generation.py` | 无 | 新入口生成与多段 rollout；纯观测推理不读取未来真值 |
| `tests/test_training.py` | 无 | 单步更新、冻结边界、resume、失败加载不污染状态 |
| `tests/test_metrics.py` | 无 | RMSD 等已有定义、RMSF 零值/缺失/冲突、系统级聚合 |
| `tests/test_migration.py` | 无 | 旧→新 artifact/权重映射与数值回归，不使用源代码字符串断言 |
| `tests/test_imports.py` | 无 | 导入新包不加载旧 module/trainer/scripts；入口不反向成为依赖 |
| `tools/migrate_artifacts.py` | 无 | 旧命名空间、权重键、统计格式和数据记录 schema 的显式兼容/转换；不重写原始数据大文件 |
| `tools/inspect_model.py` | 无 | 打印真实模型树、模块参数量、冻结项和一次实际 forward 的形状 |
| `tools/inspect_dataset.py` | 无 | 打印 system/trajectory/clip/时间间隔/原子数统计 |
| `benchmarks/overfit.py` | 无 | 真实 tiny clip 和 3 系统 9 轨迹过拟合 |
| `benchmarks/profile.py` | 无 | 图/encoder/codec/DiT/decoder/反向的耗时和峰值显存 |
| `experiments/architecture_20260914/protocol.py` | 无 | validate_protocol/compare_arms/decide；保留各轮协议，不放模型和训练循环 |

## 7. 旧文件退休清单

以下为目标清单，不是已经证明全仓零引用的删除执行结果。先按本文件搬出需要的符号和有用断言，再删除旧路径。

| 旧文件/目录或待解析模式 | 处理 |
|---|---|
| `module/model.py`；`module/interpolant_matcher.py` | 删除旧 dyVAE/bridge 数学和模型；latent rectified flow 保留在 flow/。 |
| `trainer/abs_trainer.py`；`trainer/dyvae_trainer.py`；`trainer/dynamic_trainer.py`；`trainer/adj_match_trainer.py`；`trainer/dpo_trainer.py` | 退出新主线；不搬入 legacy 包。 |
| `module/temporal_codec.py`；`module/coordinate_decoder.py` | 旧 temporal/anchor decoder 退出主线；当前坐标 head 来自 state_detail_codec_v2.py。 |
| `module/equiformer_v2/`；`ept/` | 不进入当前主包；旧实验留在固定 Git 提交。 |
| `module/visnet.py`；`module/visnet_v2.py`；`module/trajectory_temporal_refiner.py` | 作为非当前主线的历史研究实现留在原提交，不等于判定算法无价值。 |
| `module/dit_latent_cache.py` | 不进入本轮新主线；在线 history corruption 路径不得复用不匹配的 latent 缓存。 |
| `train.py`；`train.sh`；`infer_prot.py`；`infer_complex.py`；`eval_dock.py` | 旧模型入口退出；仍需的格式读写函数先迁至 molvid/data/io.py。 |
| `data/atlas_dataset.py`；`data/misato_dataset.py`；`data/mdcath_dataset.py`；`data/dypdbbind_dataset.py` | 旧 pair 数据入口退出；ordered clip 和静态源读取保留。 |
| `data/data_init.py`；`data/dataset_wrapper.py`；`data/collate.py` | 旧数据调度退出；新 batch/store/sampling 接管当前能力。 |
| `scripts/run_*round*.py（待解析模式）`；`scripts/run_*pilot*.py（待解析模式）`；`scripts/run_dit_source_ab.py（待解析模式）`；`scripts/run_dit_source_reassessment.py（待解析模式）`；`scripts/*acceptance*.py（待解析模式）` | 公共能力迁出；协议进入 experiments，长测进入 benchmarks，之后旧文件退出。 |
| `tests/test_luna_review_contracts.py`；`tests/test_*round*.py（待解析模式）` | 有用断言按 tests 清单迁移；阶段专属断言入 protocol；不按旧文件名保留。 |
| `module/`；`trainer/`；`evaluation/`；`utils/`；`agents/`；`LUNA*.md（待解析模式）` | 完成公共符号迁移后退休；历史来源通过旧 Git 快照追溯。 |

## 8. 迁移验收对应的具体测试

test_migration.py：固定 tiny clip、同一权重与噪声，比较 frame features、codec latent、重建坐标、DiT velocity、loss、梯度、一次 optimizer 更新与 resume 下一步。CUDA 数值测试使用真实 CUDA；不以 CPU fallback 替代。
test_imports.py：检查新包不可 import 根目录旧 module/trainer/scripts。
test_metrics.py：保留最近 motion reader 的零值/缺失/冲突回归，避免整理时退回旧指标读取错误。

原有许可、版权和代码来源保留于 LICENSE/NOTICE 与相关文件头；主 API 不使用 PVB 类名。

## 9. 核对过的主要源码位置

- `module/multiframe_codec.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `trainer/codec_trainer.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/state_detail_codec_v2.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/state_detail_latent_adapter.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/molecular_dit.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/torchmd_et.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `utils/torchmd_utils.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/dit_geometry_supervision.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `module/dit_history_corruption.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `data/clip_dataset.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `data/clip_batching.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `data/trajectory_clips.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`
- `evaluation/motion_metrics.py` @ `d8f674aad692c8426cf9240d07b3384f7a043888`


## 10. 给执行 agent 的分批顺序

| 阶段 | 本批工作 | 验收后才进入下一批 |
|---|---|---|
| P0 | 核对实际 HEAD、保留入口、源符号与旧数值基线；不改生产代码、不删文件 | 记录可用基线及缺失资源；确认哪些旧符号仍在真实调用路径上 |
| P1 | 包配置、数据、几何、共享算子（15 个正式文件） | 数据单位、原子顺序、mask、跨帧隔离与新包基础导入 |
| P2 | FrameEncoder、codec 总装、权重映射（9 个正式文件） | 同权重、同输入下的 frame features、latent、坐标、重建损失/梯度 |
| P3 | latent、条件、DiT、flow（10 个正式文件） | 同 tau/noise 下 velocity、flow loss、参考/优化后端与条件不泄露 |
| P4 | loss、训练、生成、评估、CLI（16 个正式文件；按下列子步骤提交） | 单步更新、resume、冻结 decoder 梯度、短 rollout、指标与统一入口 |
| P5 | 全链验证、包外导入检查，最后退休旧代码；不发布或推送远端 | 保留能力已验证；旧路径无活跃依赖；未运行检查明确标注 |

P4 内部按 `P4a：loss 与 codec 训练` → `P4b：DiT batch 与训练` → `P4c：生成和评估` → `P4d：CLI/config/tools` 逐段完成；不要在一个尚不可运行的大补丁里同时改完。JSON 的 `files[].phase` 与 `execution.phases[].owns_runtime_files` 指定了每个正式文件的初次迁移批次；验收测试随对应批次迁移，不留到最后才写。

采用一个主写入者、一个独立只读审阅者。审阅应关注数值语义、遗漏依赖、权重和梯度，不以格式建议淹没问题。不同时让多个 agent 改公共类型、codec、adapter 和 checkpoint。

每批完成后形成可回滚的本地提交/检查点，记录当前阶段、实际命令及结果、未决项和下一步。复用 `docs/migration.md`，不要为每批再生成一套 PLAN/TASKS/DECISIONS/HANDOFF。常规迁移检查通过后可继续，不必每个文件都等待用户确认。

P0 必须读取用户实际 checkout 的 HEAD；基准提交是比较锚点，不是强制回滚目标。保留工作区未提交修改，不 reset、不 force-push、不改远端。实际代码比此设计更新时，先记录增量和映射差异。

需要 CUDA、真实 checkpoint 或样本的检查不可用时，标记 `not_run`，暂停对应切换/删除，而不是报告通过或使用 CPU fallback 冒充。可继续不依赖这些资源的工作。默认不跑长训练、不重跑完整四阶段研究。

### 10.1 重构期间不能顺便改变的内容

保持已有词表编号、数据划分、shape/单位/mask、时间处理、源分布、loss 归约、冻结设置、采样算法、随机数消耗顺序和初始化对应关系。允许改名、搬文件、拆函数和移动模块所有权；不混入联合训练、per-frame 生成或新压缩方案。

旧新实现的对照用独立进程/独立 worktree 运行：新生产代码不导入旧包；旧版本在切换前仍可作为数值基准。测试断言先迁移再清理；不得以降低容差、删除失败断言或新增空类占位来完成“清单”。若实际业务确实需要一个未列出的 helper，记录它的调用来源和最终归属，而非复制完整旧模型。

### 10.2 退休清单的解释

JSON 的 `retirement_targets` 已拆成明确路径、待解析模式、P5 条件和核查项；`automatic_delete` 全部为 false。它不是 `rm` 清单，更不是要求按“没有被 import”删除测试。先提取共享能力，再核对代码、配置分派、测试和 checkpoint 使用者。

某个历史实验依赖已退休模型时，在旧提交复现，不为了让所有历史配置在新包里运行而偷偷保留旧模型。`round3.yaml` 中依赖已退休 refiner 的部分只记录原协议/证据，不承诺新主包重跑。当前实际需要的 codec/encoder/DiT 权重若与保留范围冲突，先记录并解决冲突再删除依赖。

### 10.3 两个交付文件如何一起使用

`molvid_refactor_inventory.json` 是目标文件、类、函数、阶段和映射的机器可读清单；本 Markdown 解释职责与约束。JSON 中 `source` 既包含源码路径，也有需要 P0 解析的描述，不可直接当成 shell 文件清单。两者都不是已经完成的仓库审计或重构结果。

本版仅降低目录深度并补充执行边界；没有新增模型类，没有撤销静态数据、codec 对照、future-bond、history corruption 或 reference backend。完成条件不是只让 import 和 pytest collect 通过，而是保留工作流的数值与训练行为通过相应检查。
