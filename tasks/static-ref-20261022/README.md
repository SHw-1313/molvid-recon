# 静态0.2Å扰动、MD快照混合与reference融合

授权归因更正：此前concat是在解释PVB，PM误读为实施要求；用户2026-10-22随后明确同意保留继续试验，并新增固定/learnable source对照。两固定source臂已完成，新增learnable在独立MOLVID-STATIC-SOURCE-20261022继续。

2026-10-22用户已授权，独立工作区由CAPACITY科学冻结44b5197建立。两臂均已完成8192更新、4096/8192两保存点及全部60个匹配恢复Q1。错误/缺失为空，8192数据/RNG/source/曝光及有限权重检查全部通过。旧INPAINT所有训练后生成/评分继续，依赖不改；中央三文件仍唯一管理状态。


## 已完成结果（2026-10-23）

12个固定DEV快照、每体系两个快照先均值再体系等权；均为真实source 35NFE，不是teacher插值单步。全部同0.2Å受损输入、同标准normal及固定source，clean/corrupted/source逐位配对。只是本6体系MD快照恢复，不是PDB外部泛化或动态运动测试。

|阶段|ATLAS主链/侧链误差Å|MISATO主链/侧链误差Å|ATLAS蛋白键/角异常%|MISATO蛋白键/角异常%|MISATO配体误差Å|
|---|---:|---:|---:|---:|---:|
|old_static4096|0.860/1.207|0.895/1.214|4.774/10.106|4.800/10.677|1.063|
|control4096|0.233/0.313|0.235/0.303|1.038/0.898|0.401/0.502|0.342|
|concat4096|0.225/0.307|0.230/0.296|0.345/0.473|0.153/0.415|0.344|
|control8192|0.218/0.299|0.222/0.288|0.107/0.124|0.048/0.057|0.356|
|concat8192|0.209/0.291|0.214/0.281|0.078/0.041|0.063/0.065|0.343|

新配方相对旧static的收益明显，但同时改变了训练扰动、静态快照数据及曝光，不能全部归因concat。同step control→concat才隔离融合：8192蛋白坐标误差继续小幅改善；ATLAS几何改善，MISATO含侧链键/角异常.082/.062%→.111/.094%，不是全项胜。protein soft clash/1000原子ATLAS44.48→40.46、MISATO51.63→46.61（clean约21.88/20.55）。配体键/角14.93/10.99%→9.14/8.49%，soft clash197.62→158.20（clean72.22），仍明显有缺口；配体raw误差约等于0.2受损输入的.342Å，不能和蛋白一概称恢复成功。

主链与含侧链键/角、protein/ligand手性/平面/clash及共同主链fit误差保留原CSV/JSON；手性/平面/clash未另外声称有侧链专属分解。原始输入/source/full/teacher输出均保留，不修坐标。

成本：20554.189992 GPU秒＝5.709497220 A100 GPUh/12h（control8809.142361、concat11653.493722、检查/旧probe91.553909），无失败。主评分及合并CPU156.417842墙秒/174.059290 CPU秒。两条训练/推理lane均结束，SOURCE与INPAINT独立继续；未使用额度不新增实验。

结果：[summary](../../runs/static_ref_20261022/static_probe/scores/summary.json)、[成本](../../runs/static_ref_20261022/resource_summary.json)、[8192配对](../../runs/static_ref_20261022/pair_check_step8192.json)。

两个静态臂共同query_post 10,009,356参数骨架，seed71随机初始化，corruption=0.2Å、source rho=1.1397942064485225Å、sigma_data保持原值，原RF目标与18区间35NFE。control沿旧条件路径；reference_concat复用受损reference的同深度逐原子h/v，与当前h/v通道concat投影，current块初始I/ref块0，向量无XYZ偏置，独立初始化RNG。它补充静态原先未消费的编码reference；旧dynamic本来已有historyattention和reference-relative displacement，不能称旧dynamic不知道初始构象。

PDB/ATLAS快照/MISATO快照=50/25/25，每16更新8/4/4；PDB按source均匀，MD按TRAIN system均匀再取帧，复用5Å名单与原DEV/封存排除，不构造零运动伪轨迹。两臂都8192固定schedule，lr1e-4/warmup128/save256，4096和8192保留；首32保存恢复及4096暂存均不改变schedule。主比较只隔离reference融合，旧静态4096与新模型还同时改变扰动、数据和曝光，不能单因子归因。

核心恢复评估60Q1：原12DEV快照，同0.2Åcorruption/同原rho source/35NFE，旧static4096一组，以及两个新臂各4096/8192。记录真实source与单步teacher proxy分别，蛋白主侧链/配体raw和共同蛋白主链fit误差、键角/手性/平面/clash；不会用旧1.14输入成绩作为同协议基线。旧static0.2可先测，不等待8192。

独立总43200 GPU秒=12 A100 GPUh，含必要验证/失败/两臂训练/60Q1，不挪INPAINT额度。实际卡、显存和吞吐在必要小跑后记录。共享卡不作为架构速度比较。

1.14来源：2026-10-08 e8e7f91 的 tasks/coordinate-state-denoising-v1/prepare_static_stage.py 将static_corruption_A直接设为base.scales.sigma_data_A，即动态TRAIN位移分量尺度；原trainer双噪声公式更早由b7810b27定义。10/20capacity沿用。git作者为共享工作区用户身份，不据此断言具体agent或将配置决定归给用户。

## 实际运行与必要验证

科学实现冻结c2f0275；4项CUDA与17002原子最大Q1 FB通过，reserved16.83GiB。两臂GPU5/7、allocator20GiB的有限训练/恢复均已结束；首32保存恢复及8192最终配对通过，新增reference列实际更新。参数control10,009,356/concat10,271,500；总8192 Q1/18,508,734 atom-query pairs，dynamic_examples=0。全部60Q1与原CPU尾部已完成，结果及结算见上表。

首32后早期吞吐约control1.18秒/更新、concat1.47秒/更新，粗估完整训练约2.7/3.4GPUh；共享吞吐会变，不作架构速度结论。12h总上限包含训练、失败与60Q1，单臂有限timeout含其24Q1，不会每段重置。数据核实在[data_scope.json](data_scope.json)，复用PDB2569、ATLAS512/165888窗口、MISATO5Å过滤1022/10152窗口；未将NAS/12800原始全集冒称训练ready。

旧static4096的0.2对照已12/12、errors=[]，原生35NFE输出raw主链RMS为ATLAS0.859960/MISATO0.894813Å，高于输入0.338889/0.342243Å；蛋白坏键4.774%/4.800%、角10.106%/10.677%。侧链raw误差1.207142/1.214047Å，MISATO配体1.063230Å。局部键角相对0.2受损输入改善，但构象偏移、soft碰撞和手性仍坏；不能只归因旧1.14输入太强。配体从未进入旧protein-only静态训练，独列。原评分[summary](../../runs/static_ref_20261022/static_probe/scores_old02/summary.json)，CPU32.793秒；新四stage现已完成，同协议结果见页首。

复现：既有enter-container/torch-ito，PYTHONPATH为本工作区；训练命令由[run_study.py](run_study.py)调用原tools/train_coordinate_state.py，后续[post_run.sh](post_run.sh)仅等待两个有限lane并调用已有Q1评分/合并。原始clean/corrupted/source/full/teacher坐标保留在run/static_probe/generated，不钳输出或事后修几何。
