# 新静态父MD续训至4096与配体几何

## 2026-10-27：4096续训与全部评估已完成

C/L均从同一个P1 new_static MD512完整状态新增3584步，到total4096。新96/96条四协议Q50、8条中间joint探针和32条Q1恢复均完成；最终CPU及1024/2048/joint先评均completed，errors=[]。只读复用旧参照82/84，缺的两条是既有P1 new_static512/5TV1/coarse-fine seed17/29历史缺项，不是本轮失败、不补跑。

**MISATO80ps jointQ50的蛋白坏键目标已达到：旧query_post4096为0.176746%，新纯RF C为0.088400%，L为0.054119%。** 原新父MD512为1.181067%。但这不是旧4096单日程的严格复现：父静态训练、512后新阶段LR等谱系差异仍保留，真正配对因果是本轮C与L。ATLAS100ps joint也从C蛋白坏键0.243613%降到L0.177093%，旧query_post4096为0.421982%。

配体辅助loss有真实生成收益：MISATO80ps joint配体坏键14.3801%→5.6840%，坏角14.3145%→9.6805%，三个配体各自均改善；四协议的配体键角也各3/3改善。joint蛋白整体CA幅度基本不变，不能将本项收益笼统说成通过压低蛋白运动得到。配体内部幅度下降且仍普遍高于MD，其中包含被去除的坏局部形变，不能直接称其真实动力学变好。joint时间ACF/G基本不变，欠幅与时间结构问题仍在。

### MISATO80ps：每项均为C → L，同case先两seed均值，再三体系等权

| 协议（输出主评50帧） | 配体坏键% | 配体坏角% | 配体soft clash/千原子 | 内部自对齐RMSF/MD | 相对口袋质心RMSF/MD |
|---|---:|---:|---:|---:|---:|
| joint | 14.3801 → 5.6840 | 14.3145 → 9.6805 | 398.1931 → 335.2407 | 1.8147 → 1.5821 | 1.4309 → 1.3525 |
| q12 | 19.9590 → 9.7065 | 25.4554 → 20.0229 | 731.0688 → 627.5423 | 2.7610 → 2.3806 | 2.3191 → 2.1369 |
| q4 | 25.1447 → 13.3199 | 35.5110 → 28.7833 | 1557.2302 → 1462.6772 | 4.4307 → 3.7614 | 4.0543 → 3.7709 |
| coarse_fine | 16.0421 → 6.8116 | 17.0924 → 12.4979 | 513.8016 → 435.7937 | 2.2407 → 1.9667 | 1.7532 → 1.6492 |

### 蛋白保护与时间指标：严格分dataset/实际dt/协议，C → L

| 条件 | 协议 | 蛋白坏键% | 蛋白坏角% | CA RMSF/MD | CA ACF误差 | PCA1 ACF误差 |
|---|---|---:|---:|---:|---:|---:|
| ATLAS100ps | joint | 0.2436 → 0.1771 | 0.2499 → 0.2253 | 0.6010 → 0.6061 | 0.4441 → 0.4438 | 0.6200 → 0.6220 |
| ATLAS100ps | q12 | 0.4666 → 0.3759 | 0.5272 → 0.4727 | 0.7025 → 0.7034 | 0.1713 → 0.1752 | 0.4074 → 0.4187 |
| ATLAS100ps | q4 | 1.9526 → 1.6511 | 2.3817 → 2.2124 | 1.1961 → 1.1837 | 0.1925 → 0.1912 | 0.2615 → 0.2701 |
| ATLAS100ps | coarse_fine | 0.4748 → 0.3664 | 0.4761 → 0.4233 | 0.6673 → 0.6720 | 0.4600 → 0.4591 | 0.6315 → 0.6347 |
| MISATO80ps | joint | 0.0884 → 0.0541 | 0.1065 → 0.0896 | 0.6517 → 0.6568 | 0.3940 → 0.3936 | 0.6662 → 0.6636 |
| MISATO80ps | q12 | 0.1605 → 0.1118 | 0.2383 → 0.2129 | 0.7489 → 0.7513 | 0.1395 → 0.1414 | 0.5456 → 0.5491 |
| MISATO80ps | q4 | 0.8813 → 0.6902 | 1.3555 → 1.2522 | 1.1952 → 1.1889 | 0.2083 → 0.2062 | 0.2477 → 0.2497 |
| MISATO80ps | coarse_fine | 0.1460 → 0.0969 | 0.1793 → 0.1590 | 0.6960 → 0.7014 | 0.4074 → 0.4071 | 0.6952 → 0.6934 |

Q4仍是明确失败风险：配体手性异常34.1389%→44.7500%、平面0.25%→0.50%（各仅在有可用特征的体系平均，不把N/A当0）；2XSB手性29.5%→51.5%。Q12的2XSB手性3.25%→6.50%，不能被整体键角收益掩盖。5TV1虽改善明显，joint配体仍有12.75%坏键/25.58%坏角，Q4仍25.14%/51.65%，配体内部运动不能在这种坏几何下解释为正确柔性。

6e5y_A大运动样本joint CA幅度C0.343495→L0.345473、固定MD柔区0.288219→0.289018，仍严重欠幅；Q12约0.37669→0.37616，Q4约0.60931→0.60010。joint MISATO G1到MD误差0.469419→0.469306，ATLAS0.543993→0.543566，没有实质修复时间结构。所有G1/2/4/8、D1..24、同位置接缝/转折和逐块坏原子统计保留原比较文件，未新增评分定义。

配体运动保护的具体限制：四协议12个case×协议的内部幅度均下降且仍全部高于同MD，不见普遍冻结；但4J8T joint口袋运动原已有欠幅，0.91120→0.90549略降，Q12末帧相对口袋位移三体系均值1.02762→1.04626Å反增，不能称所有漂移改善。

最终Q1恢复仅MISATO同6快照、0.2Å损坏观测/完整35NFE的独立口径：配体坏键12.0370%→5.38194%、坏角17.8152%→13.1020%、共同主链对齐后配体RMS误差0.695874→0.673574Å。蛋白坏键0.294475%→0.299352%微退、坏角0.858967%→0.795636%改善，同评分器蛋白坐标误差0.745250→0.758483Å略退；不是所有静态恢复项都改善。详见[evaluation/static_probe/summary.json](../../runs/flow_ligand_4096_20261026/evaluation/static_probe/summary.json)。

### 终态配对、资源与入口

[check_pair4096.json](../../runs/flow_ligand_4096_20261026/check_pair4096.json) passed=true、differences=[]：同数据窗口/clock/H/query/progress、主source及Python/NumPy/CPU/CUDA RNG、mixture/exposure均一致；模型/AdamW有限、step4096/LR1e-5。累计40960 target-frame、105490204 atom-query；新增protein/complex各1792。L实际complex H4/H8访问905/887，对应113/110次辅助，共223/223无skip，未为名义224调整数据。lambda固定0.09840946099484672。末段aux loss下降只作优化诊断，生成收益以上述真实source结果为准。

GPU进程总96719.956216秒=26.866654504 A100共享作业小时/72h：C35061.346549秒，L59501.950167秒，检查/校准/失败2156.659500秒；含首fixture失败8.232200秒及因外部占用仅中止本任务校准195.854837秒，训练/生成无失败。CPU四批评分合计墙1584.109461秒、CPU1610.899174秒；最终12阶段全部exit0。共享墙时不作模型独占吞吐。全部本任务GPU/CPU有限链已退出，未使用预算不追加任务、不升默认。

- [完整分条件/逐例比较](../../runs/flow_ligand_4096_20261026/evaluation/comparison/comparison_summary.json)、[逐draw全指标CSV](../../runs/flow_ligand_4096_20261026/evaluation/comparison/draw_metrics.csv)。
- [原始播放器](../../runs/flow_ligand_4096_20261026/evaluation/comparison_viewer/index.html)、[最终CPU回执](../../runs/flow_ligand_4096_20261026/evaluation/cpu_execution.json)。
- 资源原账：run/control_execution.json、ligand_execution.json、auxiliary_execution.json；模型checkpoint保存在run/control与run/ligand，旧parent不改。
- Q1恢复、五类几何、配体内部/口袋相对运动与接缝均在evaluation既有目录；Q1是0.2Å受损观测条件修复，和clean-history动态Q50分表。

## 2026-10-26执行设计与启动记录（历史，终态以上述为准）

2026-10-26 两臂均已实际从step513续训：C在GPU5/cap32GiB，已保存544并自动恢复，核读时727；L四TRAIN校准及最大完整complex通过后，自动在GPU2/cap64GiB启动，容器lane1265490/train1265921、宿主lane1180222/train1249701，首步loss.3770372272/grad1.2189218998/LR1e-5有限。科学冻结496a7b2，CPU薄层eec69f4；尚无训练后生成结论。唯一管理仍为molvid_recon/{AGENTS,HANDOFF,TASKS}.md，本任务不覆盖旧checkpoint/运行依赖。

两臂从P1 new_static MD512完整coordinate_final.pt继续，共同10,009,356参数、query_post/groupQK/atomV、固定rho1.1397942064485225Å、同TRAIN/原FM/双向attention；C纯RF，L原生source终点配体内部键与1–3角loss，无replay/concat/新噪声。保留AdamW moments、数据cursor、全部RF RNG、step与累计曝光，各新增3584到total4096。step513 lr1e-5，128新步线性到1e-4，之后cosine到4096的1e-5。旧512完整日程不改写，不能冒称从头4096单日程复现。

L用18interval/35NFE真实前向，前14无梯度、末4截断反传/重计算。仅同正component内部的真实键/角，逐frame/feature→类型均值→配体→体系等权，没有蛋白分母。正常带TRAIN统计detach：bond floor.1Å/relative.08、cos floor.15、MAD multiplier8.8956；它们是训练带宽，非重拟评价阈值。complex×H每第8访，3584步约224次（H4/H8各112仅为估算；原16任务随机shuffle与按step交替H不强制每域均半，最终按保存counts真实事件数报告，不为凑数改数据配方）；辅助data/default/source RNG与RF隔离并保存。4个预定complex TRAIN（H4/H8各2）按min(.5 median RF/aux gradient norm, min ratio)一次标定lambda后冻结，不按DEV选值。有限lambda不保证运动受保护；仅键角不代表手性/平面/自由二面角已约束。

独立总259200GPU秒=72 A100h：两lane各115200秒=32h，辅助/失败28800秒=8h，全过程实际减剩余。容器torch-ito，C32GiB/L64GiB受限allocator，启动前复查设备。有限run_arm.py首32新增更新保存step544，再1024/2048/4096；中间每臂6e5y/4J8T seed17 joint与各2Q1，最终四协议×6DEV×2seed=96draw及各12Q1。joint35NFE；Q12固定5×12保留60主评50/175NFE；Q4沿旧13块/455NFE；pure-clampCF70NFE，不注入GTfuture，不声称端点训练模型。失败保留partial、不自动重试。

结果严格按dataset、实际dt_ps、history间隔、Q、协议和参考窗分表：MISATO80ps与ATLAS100ps不可混平均；同case先seed均值，再该条件等权。旧query_post MD4096 MISATO80ps joint坏键0.176746%仅为该条件参照。蛋白主侧链/配体五类几何、真实幅度/RMSF/ACF/G/D及接缝都保留，另检查配体内部与口袋相对运动。恢复表是corruption.2Å、source rho1.139794、Q1同构象测试，独立于动态评分。

执行入口 tasks/flow-ligand-4096-20261026/run_arm.py --arm control 或 ligand。run为 runs/flow_ligand_4096_20261026；每臂execution.json记录真实命令/PID/累计GPU秒。评分尾部沿已有common/geometry/G/D/viewer薄适配，允许partial。

实际有限链补充：C首32保存step544并恢复，491.448085进程秒。L原GPU6检查因外部占用增加仅中止本任务，195.854837秒计失败；同4窗+最大窗GPU2成功进程1933.903645秒，全部辅助已结账2156.659500秒。固定lambda=0.09840946099484672；最大6D6U16999原子H4Q12 aux检查753.822291秒、peak18.197GiB、auxgrad5.966254，主RF RNG保留/蛋白endpoint梯度0。四窗weighted aux/RF梯度比.185452/.813007/.509142/.491181；这不是运动保护保证。证据run/calibration_gpu2/calibration.json及configs/ligand.json。after_calibration.sh已实际exec L有限训练链，首aux/global537已实用（6NPU/H4：18键/25角，raw .1078448221、weighted .01061295066、grad6.031399有限，211.348312秒）。check_pair544.json通过/differences=[]：各32新增输入/320监督帧、mixture及全部主RNG一致；实际complex:H4=9/H8=7、1事件。L首32进程664.889113秒，已自动resume真实545；C核读809。CPU唯一尾部容器1265599已实际运行run_cpu.sh --wait --allow-partial，先1024/2048小joint及4096joint，再复用已评分joint补其它协议/Q1；日志run/cpu_tail.log。有限链继续到两臂4096、96draw/评分/播放器或明确预算停止。

启动配对证据：runs/flow_ligand_4096_20261026/check_pair544.json（纯CPU0.722368秒，无新模型调用）。后续1024/2048中间与4096终态仍按同检查脚本读取已保存状态；不重复生成基线或改变数据顺序。
