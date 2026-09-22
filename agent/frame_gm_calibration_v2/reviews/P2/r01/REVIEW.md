# 独立审阅报告

REVIEW_STAGE: P2
REVIEWED_COMMIT: f6ec672c7d8582772c9056cac0d116c1eddf1ffc
VERDICT: FIX_REQUIRED

- reviewer/session：全新独立只读 reviewer；固定 detached worktree `/data4/users/sihao/workspace/molvid-recon-gm-calibration-v2-review-P2-r01`
- base commit：`0895fa7abe0ee475e6acf0d3fe20b54a8164f48f`
- 阅读的规格版本与pilot artifact：候选提交中的 `PLAN.md`、`ARCHITECTURE.md`、`DATA_TRAIN_EVAL.md`、`REVIEW_PROTOCOL.md`；请求绑定的 `pilot_summary.json`、`resolved_experiment.json`、`commands.sh`
- 审阅环境限制：无读取阻碍；按只读审阅要求未修改文件、未启动训练、未运行全量 pytest。三份请求 artifact 的 SHA-256 均与 `review_request.json` 一致，HEAD 正确且工作树干净。

## 结论

暂不能开始 P2 正式四臂训练。

G/M 主体实现与大部分数值证据方向正确：G 在第 2/4 block 使用稀疏 1-hop/1–3 图、从归一化前向量读取不变量并采用单侧零门；M 区分 query horizon、query interval、history span 和 flow time，静态 history 输出为零，并在每个 block 提供持续 value/门控路径。四臂 pilot 的采样清单、曝光、parent、loss、optimizer 重置和最大真实样本 profile 也保持对称。

但现有 pilot 明确记录为 base commit，而不是被审阅的 candidate；同时配置拒绝未知键、派生数据身份强校验和恢复后的曝光审计仍有可复现缺口。这些问题会破坏正式实验的代码/配方绑定或关键审计，必须先修复并复核。

## 必须修复的问题

### R1 — Pilot 证据没有绑定到候选提交

- 优先级：P1
- 定位：
  - `molvid/cli/train_frame_joint.py:362-365`，`_git_commit`
  - `molvid/cli/train_frame_joint.py:604-610`，写入 target provenance
  - `tools/summarize_frame_gm_p2_pilot.py:63-107`
  - 四臂 `pilot/*/target_encoder_provenance.json`
- 证据与触发条件：
  - B0/G/M/GM 四份 provenance 的 `code_commit` 均为 base commit `0895fa7abe0ee475e6acf0d3fe20b54a8164f48f`。
  - 请求审阅的候选为 `f6ec672c7d8582772c9056cac0d116c1eddf1ffc`。
  - pilot checkpoint 在候选提交之前生成；顶层 hash-bound `pilot_summary.json` 又没有记录代码 SHA，也没有绑定 provenance 文件的哈希。因此无法从现有证据证明运行时源码等于候选快照。
- 影响：128-update 稳定性、target mutation、梯度启动、最大样本显存和推理检查不能作为该候选提交的审阅放行证据。直接开始正式训练会违反“正式训练从获审阅代码快照运行”的协议。
- 最小修复建议：完成其余代码修复后先提交最终候选，从该干净提交运行四臂 pilot、最大样本 profile 和必要的 seed-0/leakage 检查；在顶层摘要中直接写入代码 SHA，并绑定各 provenance 文件哈希。不能用文件时间或事后声明追溯证明旧 dirty-tree 运行。
- 最小复验：新四臂 provenance、checkpoint contract、pilot summary 和 review request 全部指向同一最终候选 SHA；artifact 哈希重新生成；四臂均为 128 个成功 update、有限 loss/gradient、相同 sampler/exposure，G/M 新门和消息参数确有梯度或 optimizer state。

### R2 — Frame Joint 配置会静默忽略未知或误拼键

- 优先级：P1
- 定位：
  - `molvid/config.py:30-34`，`load_config`
  - `molvid/cli/train_frame_joint.py:379-383`
  - `molvid/cli/train_frame_joint.py:429-439`
  - `molvid/training/joint.py:41-64`，`JointLossConfig.resolve`
- 证据与触发条件：配置加载只核对顶层 schema 和六个 section 是否为 mapping；model、loss、data、training 采用 `.get()` 读取已知字段，却不拒绝剩余字段。例如把 `geometry_enabled` 拼成 `geometry_enabeld` 后配置仍会通过，随后 G 会按默认 `False` 构造；字符串 `"false"` 经过 `bool(...)` 还会变成 `True`。
- 影响：正式臂可能在名称、归档和选择规则仍标成 G/M/GM 时实际运行另一架构；错误 loss、数据权重或训练预算键也可能被无声忽略，直接破坏四臂比较。
- 最小修复建议：为根配置以及 data/derived item、codec、statistics、model、loss、training/stage 建立显式 allowlist 和类型检查；布尔开关必须要求真正的 boolean。应在打开数据、加载 checkpoint 或申请 CUDA 前失败。
- 最小复验：针对每个新 section 加最小解析测试；`geometry_enabeld`、未知 loss/training 键及字符串布尔均抛出明确错误；四份正式配置仍可解析并得到预期 B0/G/M/GM 开关及统一 loss/预算。

### R3 — 派生视图 manifest 声明的 store index 哈希未被校验

- 优先级：P1
- 定位：
  - `tools/build_frame_gm_fixed_history_train_views.py:242-245`
  - `molvid/cli/train_frame_joint.py:111-132`，`_open_data`
  - 四份正式配置 `data.derived_train`
- 证据与触发条件：builder 将 `store_index_sha256` 写入已固定哈希的 manifest；加载端验证 manifest 文件 SHA、split 和 source data hash，却只把当前 `index.txt` 哈希记录到 `identities`，没有与 manifest 的 `store_index_sha256` 比较。保持 manifest 不变而替换或改动合法的 store/index，当前代码仍会开始训练，只是在事后产生另一个 data hash。
- 影响：正式配置表面绑定已审阅 manifest，实际可消费不同的派生记录或样本清单；冻结的两 epoch schedule、曝光和预算估计随之失效。
- 最小修复建议：打开派生 store 时校验当前 `index.txt` SHA 等于 manifest 的 `store_index_sha256`，并核对 record count；建议同时让正式配置显式声明预期 derived data hash，并在启动前拒绝不一致。
- 最小复验：临时复制一个最小 store/manifest，原样加载成功；仅改变 index 后必须在构造 sampler 前失败。真实 6912-record store 应解析为已记录的 index SHA `3a1be208b90f77bf78452cb6e7402b14f9066b8b0208ce6e250852133d250045` 和 derived data hash `8b972def15babf848015699e527e28bc53fc195292ce446bda8555d86aaed0b1`。

### R4 — Strict resume 后 `exposure.json` 只统计恢复后的后缀

- 优先级：P1
- 定位：
  - `molvid/cli/train_frame_joint.py:528-563`，resume 恢复
  - `molvid/cli/train_frame_joint.py:612-615`，曝光计数器重新置零
  - `molvid/cli/train_frame_joint.py:698-706`，后缀累计
  - `molvid/cli/train_frame_joint.py:740-748`，最终覆盖写入
- 证据与触发条件：resume 会恢复 step、sampler cursor 和 RNG，但之后无条件新建空 Counter；完成时 `successful_updates` 是累计总数，bucket/trajectory/history/valid-atom-frame 则只来自当前进程后缀。正式训练只要从 checkpoint 恢复一次就会产生内部不一致的曝光报告。
- 影响：无法用交付的曝光文件核验四臂是否具有相同主数据、H4/H8 比例、bucket 配方和 valid atom-frame 数；不同臂发生不同次数中断时尤其会污染比较审计。
- 最小修复建议：把累计曝光状态写入并严格恢复 checkpoint，或从完整、去重的训练记录重算最终曝光；不要把后缀统计伪装为完整 run。
- 最小复验：同一确定性短 schedule 分别连续运行和“中途 checkpoint → resume”运行；最终 exposure、sampler cursor、selected sample IDs、bucket/history/trajectory counts 和 valid atom-frames 必须逐项一致。

## 非阻塞建议

- `resolved_experiment.json` 写的 generated-bond 系数为 `0.05170724987983704`，实际 parent/pilot checkpoint contract 为 `0.051707249134778976`。所有臂当前继承同一实际值，数值差异不会改变训练，但应直接从 parent contract 生成记录，避免手工小数不一致。
- `git diff --check` 仅报告两个新工具文件末尾多余空行，不影响科学或运行行为。
- 正式结果产生前补齐可执行的 system-level paired bootstrap/`selection.json` 工具；目前冻结了规则，但候选中还没有对应实现。

## 已核验的关键路径

- `prepare_frame_joint_batch` 将 observed context、query clock、future target 分离；sample origin 只来自 frame 0。生成路径会重建 observed-only batch，future-coordinate mutation 逻辑不会进入 context/source。
- teacher 在模型构造、训练模式切换和 trainer 中均保持 frozen/eval；四臂 pilot checkpoint 的 teacher state hash 相同。
- G 使用去重双向共价 1-hop 与 1–3 图，无跨体系边；最后 observed 距离、hop 和 atom type 构成静态 edge 特征，向量 norm/inner/difference 在 SO(3) normalization 前以 FP32 计算。全局 spatial attention 仍保留。
- G 的输出分支为普通初始化、外层 residual gate 为零；CUDA 测试覆盖等变性、平移、padding、packed 边界和第 1/后续步梯度启动。128-step checkpoint 中 G gate 已离开零点。
- M motion summary 只读取 observed h/v/time；保留 signed/absolute/vector-magnitude 和 relative-to-last 信息，静态 history 精确为零。query horizon、query delta、history span、observed delta 与 flow time 分离，时间平移不变。
- M 在每个 DiT block 注入 motion value/门控，并给 history/future temporal attention 增加非负平方根时间衰减；H4 单 key 不只依靠 softmax bias。128-step checkpoint 中各 block motion gate 已离开零点。
- B0 保持 v1 contract，warm start 加载全部 386 个旧 tensors；G/M/GM 仅初始化允许的新增 tensors。四臂均未恢复 optimizer、scheduler、cursor 或 RNG。
- 四臂 pilot 使用相同 sampler schedule/sample-ID hash、相同曝光和 parent。最大真实训练样本为 4975 atoms、16 slots；GM 峰值约 56.6 GB，冻结的两 epoch 四臂预算估算约 31.62 GPU-hours。
- train-only fixed-history builder 使用 100/200/400 ps，不含 300 ps；有效 query 分别为 12/6/3，padding mask 为 false。正式配置的 24 clips/trajectory 配方解析为 75% legacy、25% fixed-history。
- 三份请求绑定 artifact 的哈希和四份正式配置哈希均核对一致；正式训练尚未开始，sealed test 未打开。

## 主session下一步

先修复 R2、R3、R4 并提交新的候选；随后按 R1 从该干净提交重新产生候选绑定的四臂 pilot/profile/leakage 证据，写 `FIX_RESPONSE.md`，发起 P2 r02 定向独立复核。r02 通过前不要启动四臂正式训练。
