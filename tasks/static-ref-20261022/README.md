# 静态0.2Å扰动、MD快照混合与reference融合

2026-10-22用户已授权，独立工作区由CAPACITY科学冻结44b5197建立。当前CPU实现准备，无新GPU启动。旧INPAINT所有训练后生成/评分继续，依赖不改；中央三文件仍唯一管理状态。

两个静态臂共同query_post 10,009,356参数骨架，seed71随机初始化，corruption=0.2Å、source rho=1.1397942064485225Å、sigma_data保持原值，原RF目标与18区间35NFE。control沿旧条件路径；reference_concat复用受损reference的同深度逐原子h/v，与当前h/v通道concat投影，current块初始I/ref块0，向量无XYZ偏置，独立初始化RNG。它补充静态原先未消费的编码reference；旧dynamic本来已有historyattention和reference-relative displacement，不能称旧dynamic不知道初始构象。

PDB/ATLAS快照/MISATO快照=50/25/25，每16更新8/4/4；PDB按source均匀，MD按TRAIN system均匀再取帧，复用5Å名单与原DEV/封存排除，不构造零运动伪轨迹。两臂都8192固定schedule，lr1e-4/warmup128/save256，4096和8192保留；首32保存恢复及4096暂存均不改变schedule。主比较只隔离reference融合，旧静态4096与新模型还同时改变扰动、数据和曝光，不能单因子归因。

核心恢复评估60Q1：原12DEV快照，同0.2Åcorruption/同原rho source/35NFE，旧static4096一组，以及两个新臂各4096/8192。记录真实source与单步teacher proxy分别，蛋白主侧链/配体raw和共同蛋白主链fit误差、键角/手性/平面/clash；不会用旧1.14输入成绩作为同协议基线。旧static0.2可先测，不等待8192。

独立总43200 GPU秒=12 A100 GPUh，含必要验证/失败/两臂训练/60Q1，不挪INPAINT额度。实际卡、显存和吞吐在必要小跑后记录。共享卡不作为架构速度比较。

1.14来源：2026-10-08 e8e7f91 的 tasks/coordinate-state-denoising-v1/prepare_static_stage.py 将static_corruption_A直接设为base.scales.sigma_data_A，即动态TRAIN位移分量尺度；原trainer双噪声公式更早由b7810b27定义。10/20capacity沿用。git作者为共享工作区用户身份，不据此断言具体agent或将配置决定归给用户。
