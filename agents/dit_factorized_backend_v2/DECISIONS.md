# B decisions

## Prepared decisions — 2026-09-09

- B独立worktree；base a22f60c固定，不随A分支更新。
- 优化的是同一个factorized block-spatial/atom-temporal模型，不是新空间结构。
- R4主线、R2仅性能与等价性对照，R1/pooling不进入本轮DiT。
- 向量化与消除重复encode优先；SDPA是实现手段，不保证自动得到Flash或提速。
- 相同scalar attention权重作用于xyz vector，不改head scale，不混xyz学任意旋转矩阵。
- derived layout不进入可学习latent/数据合同；不根据future坐标建立隐式条件。
- 旧checkpoint/statistics和实验结果不可写；backend-only contract差异显式迁移。
- 全磁盘latent cache不是本轮必做项，先用预载固定输入得到可靠model-only证据。
- 没有profile trace不能断言所有耗时来自Python循环；没有重设CUDA peak不能比较纯DiT内存。

## Implementation-forced additions

worker追加实际kernel、layout结构、兼容处理、数值误差与瓶颈事实。不得删除prepared decisions。

## Implementation-forced additions — 2026-09-09

- module/dit_backend_v2.py uses one CPU metadata materialization per batch view, keyed by (sample_id, block_id), then CUDA index_add_ mean pooling, padded BxK block packing, one shared scalar q/k softmax weight matrix for scalar/vector values, and one atom-time dense call.
- The measured optimized attention backend is explicit matmul_softmax_shared_weights_v2; no SDPA/FlashAttention claim is made. factorized_v2 rejects nonzero dropout so shared-weight parity remains explicit.
- The semantic model contract and strict checkpoint state_dict keys remain unchanged; runtime selection is exposed separately as execution_backend plus semantic_contract_hash.
- Layout cache validity binds batch metadata object identity and tensor versions; topology mutation and reordered metadata rebuild the layout. Derived layout is not serialized.
- The pilot training loop now reuses _encode_batch output and performs exactly one codec encode per batch.
- Real profile evidence on the audited A100 shows factorized model-only median speedups of approximately 75x/71x/101x/142x for R4 small/median/large and R2 median; reference time remains dominated by the old Python spatial loop.
