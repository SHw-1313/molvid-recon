# 新静态父MD续训至4096与配体几何

2026-10-26 两臂均已实际从step513续训：C在GPU5/cap32GiB，已保存544并自动恢复，核读时727；L四TRAIN校准及最大完整complex通过后，自动在GPU2/cap64GiB启动，容器lane1265490/train1265921、宿主lane1180222/train1249701，首步loss.3770372272/grad1.2189218998/LR1e-5有限。科学冻结496a7b2，CPU薄层eec69f4；尚无训练后生成结论。唯一管理仍为molvid_recon/{AGENTS,HANDOFF,TASKS}.md，本任务不覆盖旧checkpoint/运行依赖。

两臂从P1 new_static MD512完整coordinate_final.pt继续，共同10,009,356参数、query_post/groupQK/atomV、固定rho1.1397942064485225Å、同TRAIN/原FM/双向attention；C纯RF，L原生source终点配体内部键与1–3角loss，无replay/concat/新噪声。保留AdamW moments、数据cursor、全部RF RNG、step与累计曝光，各新增3584到total4096。step513 lr1e-5，128新步线性到1e-4，之后cosine到4096的1e-5。旧512完整日程不改写，不能冒称从头4096单日程复现。

L用18interval/35NFE真实前向，前14无梯度、末4截断反传/重计算。仅同正component内部的真实键/角，逐frame/feature→类型均值→配体→体系等权，没有蛋白分母。正常带TRAIN统计detach：bond floor.1Å/relative.08、cos floor.15、MAD multiplier8.8956；它们是训练带宽，非重拟评价阈值。complex×H每第8访，3584步约224次（H4/H8各112仅为估算；原16任务随机shuffle与按step交替H不强制每域均半，最终按保存counts真实事件数报告，不为凑数改数据配方）；辅助data/default/source RNG与RF隔离并保存。4个预定complex TRAIN（H4/H8各2）按min(.5 median RF/aux gradient norm, min ratio)一次标定lambda后冻结，不按DEV选值。有限lambda不保证运动受保护；仅键角不代表手性/平面/自由二面角已约束。

独立总259200GPU秒=72 A100h：两lane各115200秒=32h，辅助/失败28800秒=8h，全过程实际减剩余。容器torch-ito，C32GiB/L64GiB受限allocator，启动前复查设备。有限run_arm.py首32新增更新保存step544，再1024/2048/4096；中间每臂6e5y/4J8T seed17 joint与各2Q1，最终四协议×6DEV×2seed=96draw及各12Q1。joint35NFE；Q12固定5×12保留60主评50/175NFE；Q4沿旧13块/455NFE；pure-clampCF70NFE，不注入GTfuture，不声称端点训练模型。失败保留partial、不自动重试。

结果严格按dataset、实际dt_ps、history间隔、Q、协议和参考窗分表：MISATO80ps与ATLAS100ps不可混平均；同case先seed均值，再该条件等权。旧query_post MD4096 MISATO80ps joint坏键0.176746%仅为该条件参照。蛋白主侧链/配体五类几何、真实幅度/RMSF/ACF/G/D及接缝都保留，另检查配体内部与口袋相对运动。恢复表是corruption.2Å、source rho1.139794、Q1同构象测试，独立于动态评分。

执行入口 tasks/flow-ligand-4096-20261026/run_arm.py --arm control 或 ligand。run为 runs/flow_ligand_4096_20261026；每臂execution.json记录真实命令/PID/累计GPU秒。评分尾部沿已有common/geometry/G/D/viewer薄适配，允许partial。

实际有限链补充：C首32保存step544并恢复，491.448085进程秒。L原GPU6检查因外部占用增加仅中止本任务，195.854837秒计失败；同4窗+最大窗GPU2成功进程1933.903645秒，全部辅助已结账2156.659500秒。固定lambda=0.09840946099484672；最大6D6U16999原子H4Q12 aux检查753.822291秒、peak18.197GiB、auxgrad5.966254，主RF RNG保留/蛋白endpoint梯度0。四窗weighted aux/RF梯度比.185452/.813007/.509142/.491181；这不是运动保护保证。证据run/calibration_gpu2/calibration.json及configs/ligand.json。after_calibration.sh已实际exec L有限训练链，首aux/global537已实用（6NPU/H4：18键/25角，raw .1078448221、weighted .01061295066、grad6.031399有限，211.348312秒）。check_pair544.json通过/differences=[]：各32新增输入/320监督帧、mixture及全部主RNG一致；实际complex:H4=9/H8=7、1事件。L首32进程664.889113秒，已自动resume真实545；C核读809。CPU唯一尾部容器1265599已实际运行run_cpu.sh --wait --allow-partial，先1024/2048小joint及4096joint，再复用已评分joint补其它协议/Q1；日志run/cpu_tail.log。有限链继续到两臂4096、96draw/评分/播放器或明确预算停止。

启动配对证据：runs/flow_ligand_4096_20261026/check_pair544.json（纯CPU0.722368秒，无新模型调用）。后续1024/2048中间与4096终态仍按同检查脚本读取已保存状态；不重复生成基线或改变数据顺序。
