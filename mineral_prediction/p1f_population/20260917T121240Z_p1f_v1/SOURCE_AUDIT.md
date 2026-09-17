# P1-F source and history audit

【源码事实】模型直接复用 acawlr_ppp_bridge.ReducedRankOneSyntheticPredictor；675参数，x=xy/[60000,40000]，h=tanh(raw(xy))-tanh(raw(reference))，b=beta+u*h，g=x^T b；buffers完整加载。P1-F不重写结构、feature、anchor或penalty类。

【旧artifact事实】P1-E v1 POPULATION_RESULTS.json 的 count=0、status=BLOCKED；唯一GATES.failed是fidelity_legacy_0.001_71。其96→192 |ΔlogZ|=.0005061547841709313，阈值.0001；本轮GATES.legacy_audit保留，原文件哈希不变。它是不同的经验数据拟合结果，没有进入本轮目标计算。

【旧artifact事实】P1-C正式archive包含20组300步历史；P1-D optimization有3条2001点trajectory；P1-D LR有9条2001点trajectory。B_n256_data23/53/71_init1011共享完全相同初态。Population无event draws，所以这三个data seed不构成三次独立population重复。

【数学与实现】本轮pop_Q(g)=logsumexp(logw+g)-pi_teacher@g，pi_teacher detach；256*pop_Q+multiplier*原R。P0的effective penalty为0，但另存reference penalty便于公平比较。Pλ只增加原始显式L2，没有AdamW或optimizer weight_decay。标准/teacher-local各自P0与Pλ完整初态相同，teacher-local扰动不改buffers。

【协议范围】与v1不同的是依赖范围，不是把.0001阈值改大。Fresh gates检查teacher logZ/density/矩、复制与梯度、同目标global/fixed-h controls和实际initial states；endpoints逐个独立96/192复核。旧结果不参与fresh target，不能当阻断新目标的数值反例。检查teacher矩使用固定无量纲基X,hX,g，绝对容差5e-5在新诊断前冻结。

【可核验证据】PROTOCOL.json保存阈值/预算/顺序/定义/剂量条件；PRETRAIN_CONFIRMATION.json在更新数0时记录protocol与source hash；SOURCE_MANIFEST.json与raw/source保留实际训练代码。PROVENANCE.json记录基线bf97a33、环境、teacher/init/grid哈希和parameter顺序。父memory read_audit清单记录完整读取相关文本，不执行旧GIS或training入口。

【解释限制】使用48²训练与96/192评价可检验当前离散population程序；不能直接与旧24² empirical轨迹作单因素比较。固定两种初态不是多随机seed稳定性试验，不估计collapse概率。若本轮Pλ未collapse，不能把“不观察到”写成“对所有初态都排除”；若观察到，也不能自动归因历史seed23/71。

【读取型报告】report_acawlr_ppp_p1f_population.py在训练启动后编写，仅消费保存的JSON/数组；不在训练程序调用路径，不改变已冻结训练源文件、定义或gate。其单独source hash与输入hash由REPORT_PROVENANCE记录。
