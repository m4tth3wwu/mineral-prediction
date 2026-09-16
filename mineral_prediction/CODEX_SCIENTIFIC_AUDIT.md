# Stage 1A — Static Scientific Audit

审计日期：2026-09-16。仓库：m4tth3wwu/mineral-prediction。审计对象基线提交：`a3a3f186c0bdef76fd4753366874c1d61b2fe103`。

**首要结论：当前推荐 PPP 属于 B，是与 ACAWLR reproduction 分离的全局 log-linear presence-only Poisson point process。它没有保留 ASPNN / GASPG / CAWNN / 局部系数空间加权架构。当前核心似然、面积权重和 profiled intercept 的代数实现正确；主要科学风险在模型身份、目录选择过程、验证适用范围、积分域与岩性定义，以及报告表述。没有发现足以据静态代码判定当前 PPP 核心数学失效的证据。**

## 1. Current pipeline

### 1.1 仓库和执行边界

- 实际 Git 根目录：`D:/code/ResearchPractice`；主工程：`D:/code/ResearchPractice/mineral_prediction`。用户消息中的反斜杠下划线按 Markdown 转义理解；不存在字面路径 `mineral/_prediction`。
- 开始时 `git status --short --branch` 为 `## codex-refactor...origin/codex-refactor`，无工作区改动；`git branch --show-current` 为 `codex-refactor`。
- `main`、`codex-refactor`、`pre-codex-refactor^{commit}` 均为上述基线提交。本地 `origin/main`、`origin/codex-refactor` 也指向该提交；本阶段未 fetch，因此这是本地引用状态，不是实时远端核验。
- `pre-codex-refactor` 是 annotated tag：tag object 为 `48bf5edee80ebbf6e08dffe64b23f0f2646abe25`，其指向的 commit 与上述分支相同，不能把 tag object ID 误认成不同代码提交。
- 本阶段只读取代码、文档、原文及少量已有结果，做 AST 语法解析与轻量表格读取。未完整训练、未 spatial CV、未单折重拟合、未新 sensitivity experiment、未运行可能重建缓存或覆盖历史输出的 audit/verify/plot 入口。
- 唯一科学输出是本报告；只为它追加精确 gitignore allowlist，并按要求提交。main、标签及历史结果保持原样；不 merge、不 push、不进入 Stage 1B。

### 1.2 当前调用链

根目录 `run_current_ppp.ps1` → `PPP_binary_lithology_v3.py:main` → `PPP_three_papers_simple.py` 的拟合、评价和重复分块函数；原始地学特征提取复用 `PPP_lithology_spatial_validation.py`，**没有使用该 legacy 模块的旧网格计数拟合器**。

1. 读取已有 5 km support 和原 44/16 点，保持事件输入坐标。
2. v3 从原始地化、断层、重力和岩性重新提取特征；通过输入、代码、参数和输出 SHA-256 判断是否复用缓存。
3. 固定 fine support；10 km 分组汇总面积，并选择真实 fine 点作为积分代表。
4. 五个并列方案：M0 基础变量；M1 加侵入岩指示；M2-old 加旧图斑边界距离；M2-new 改用二分类共同边界距离；M3 在 M2-new 上加地化覆盖代理。
5. 200 km blocks、50 km guard、五折、种子 42/43/44；另有四个 compact regions。
6. 全 44 点拟合；输出 5 km 地图；M2-new 做逐事件删除、含事件块删除和区域残差；最后评价历史 16 点。

M0 的变量为 log1p(Cu)、log1p(Au)、log1p(Mo)、Fe、log1p(断层距离 km)、Bouguer、最近重力数据距离 degree、has_gravity。M1 增加 lith_intrusive；M2-new 再增加 log1p(共同接触距离 km)。坐标用于 GIS、分块和验证，**不作为当前 PPP 的普通回归项或空间权重网络输入**。

### 1.3 已阅读证据

根目录 BASELINE、当前运行与目录说明；四份指定 PPP / alignment 文档；全部十二个指定 Python 文件的定义与有关调用链、特征、训练、验证、审计、作图和测试段落。另追踪了 reproduction 真正通过 `core_path` 动态加载的 `scripts/ACAWLR_improved.py`，避免把文件名相似的 `ACAWLR_paper_aligned.py` 当成实际核心。

本地原文：`references/g53947.pdf` 第 2 页 Methods，公式 (1)–(9)；`references/G53947_SuppMat.docx` 的 Data and Method、OLS/GWR/GNNWR、Attention Module、Table S1。已直接提取并阅读文本和数学文本，没有依赖搜索摘要。原文定位：[论文 DOI](https://doi.org/10.1130/G53947.1)、[补充材料 DOI](https://doi.org/10.1130/GEOL.S.30888581)。这些链接来自本地原文，本阶段未联网验证链接。

下文源码定位以基线文件的函数名及行号为准；历史数值只用于识别现有证据与风险，不宣称重新算出或复现。

## 2. Model identity

### 2.1 代码判定：B

直接证据：

- `PPP_binary_lithology_v3.py:25–37,371–428` 导入 ppp/legacy，最终调用 `ppp.fit_points`。
- `PPP_three_papers_simple.py:166–211` 优化一个长度等于 feature 数量的 slopes 向量；`log_intensity = intercept + matrix @ slopes`。同一模型在全区使用同一组系数。
- ACAWLR 核心 `scripts/ACAWLR_improved.py:192–210` 明确是 ASPNN → GASPG → CAWNN → local_weight × beta_global → 局部线性预测值；reproduction 的 `make_model` 明确创建 `core.ACAWLR`，随后 BCEWithLogitsLoss 训练并 sigmoid 预测。
- `ACAWLR_paper_aligned.py:113–218,711–761` 有相应网络；其存在不意味着 PPP 调用它。multiscale 的 `make_model/run_strict_acawlr` 同样属于另一条训练流程。

| ACAWLR 核心或 reproduction 模块 | 当前 PPP 是否具备 |
|---|---|
| 各向异性距离编码、方向及平行/垂直尺度 | 无；只有标量断层距离等协变量 |
| ASPNN 2→16→8→4→1 非线性 proximity 学习 | 无 |
| 120×100 GASPG 空间 proximity 网格 | 无；PPP quadrature 是积分工具，不能称 GASPG |
| CAWNN 卷积、残差块、通道/空间注意力 | 无 |
| W(s) 对 global beta 的位置依赖加权、beta(s) 局部系数 | 无；PPP 是全局 beta |
| 局部断层方向/置信度及密度、multiring 等增强变量 | 当前 PPP 变量表未包含这些模块 |
| logistic / BCE、正负样本权重、2 km 正样本栅格扩增 | 已改为独立事件似然；不是原网络末层替换 |

真正继承的是研究问题、相近研究窗口、论文图件匹配后的原 44/16 点、部分 USGS 地化/断层/Bouguer 数据及部分变量概念。IDW 和基础特征提取有方法延续，但坐标投影、目标位置、背景定义与评价都并不完全相同。新二分类岩性共同边界是 PPP 特征工程，不是注意力或各向异性架构继承。

允许表述：“以同一研究区和原 44/16 点为基础，构建独立的 presence-only log-linear PPP，并审计岩性接触特征与空间验证。”

禁止表述：

- “仅将 ACAWLR 的 binary logistic formulation 换成 PPP，其余架构不变。”
- “ACAWLR-PPP 已实现”“保留原论文的空间注意力/ASPNN/GASPG/CAWNN”。
- “当前 PPP 已建模局部系数非平稳性或学习了各向异性。”
- “与 ACAWLR 的差异仅在 loss，因此 AUC 差异可归因于 PPP。”
- “精确复现原论文 44/16 随机划分”。现有点是图件/数据库重建，非作者公开的原始划分坐标。

### 2.2 真正 ACAWLR + PPP 需要哪些改动（本阶段不实现）

1. 在事件位置和积分位置都计算同一套各向异性距离、ASPNN、GASPG、CAWNN，让网络输出空间变化的线性预测量 g_theta(s)。
2. 定义 `log lambda(s)=b+g_theta(s)`，用事件项与面积积分共同反向传播；将 BCE、class weights、正样本缓冲扩增、任意采样负类接口替换为 event/quadrature 数据接口。
3. 单独保留不惩罚的全局截距，才可使用 `b=log n-log integral exp(g_theta)`。原 ACAWLR 局部加权常数项不是可直接解析消元的单一全局截距；需处理它与 b 的可辨识性。
4. 重新定义 beta_global 的来源及与 W(s) 的约束。不能不加说明保留由 presence-background logistic 得出的固定 beta，却声称完全由 PPP 估计。
5. 重做折内网络训练、预处理、方向估计/网格构造及内层模型选择的边界；若方向从矿点估计，只能用训练点。重新审查 logits clamp、积分数值稳定性和批处理积分权重。
6. 用统一域、特征、划分和指标设计 architecture/formulation 对照；这是一项新模型研究，超出小修复与本阶段范围。

原论文 Methods 使用方向加权协方差/PCA、OLS global coefficients、ASPNN/GASPG/CAWNN；当前 reproduction 用局部断层方向旋转和固定尺度，并以 LogisticRegression 得 global beta。补充材料空间 attention 写 5×5，aligned 核心实现为 7×7。这些也要求 reproduction 自身保留“近似/工程对齐”限定。补充材料明确 2 km 正样本缓冲的段落属于 Meguma 20 金矿点；不能据此声称 western-US 44/16 的所有重建细节已被原文确认。

## 3. PPP formulation

证据：`PPP_three_papers_simple.py:112–234`。下面是对代码的推导，不依赖模型运行。

设训练事件 s_i，i=1…n；训练积分位置 q_j，面积 a_j>0，单位 km²；折内预处理后的特征 x(s)，全局斜率 beta：

```
lambda(s) = exp(b + x(s)^T beta)
Z(beta) = sum_j a_j exp(x(q_j)^T beta)
S = sum_i x(s_i)

ell(b,beta) = n*b + S^T beta - exp(b)*Z(beta)
penalized ell = ell - alpha/2 * beta^T beta
```

这是 `sum_i log lambda(s_i) - integral_D lambda(s) ds` 的 quadrature 近似，不是 binary logistic likelihood。

对 b 求导为 `n-exp(b)Z=0`，故：

```
b_hat(beta) = log(n) - log Z(beta)
ell_profile(beta) = n*log(n) - n*log Z(beta) + S^T beta - n
negative objective, dropping constants:
  n*log Z(beta) - S^T beta + alpha/2 * beta^T beta
p_j = a_j exp(x(q_j)^T beta) / Z(beta)
gradient = n*sum_j p_j x(q_j) - S + alpha*beta
```

与 `profiled_objective`、`fit_points` 完全一致。Hessian 为 n 倍 p 加权特征协方差加 alpha I；alpha>0 时对 slopes 严格凸（在有限有效输入条件下）。未发现符号、面积 offset 或截距惩罚错误。

### 3.1 各函数的科学含义

| 函数 | 实际行为与判定 |
|---|---|
| preprocess | 只从传入训练 quadrature 估计面积加权 median、填补后 mean 与 population SD；全缺失填 0；常量 SD 置 1。has_gravity/lith_intrusive 保持 0/1。 |
| matrix | 使用已拟合 prep 填补非有限值、标准化；不会从测试点重新估计参数。 |
| profiled_objective | 用真实事件特征之和与面积积分优化 slopes；不读取 event_count；惩罚全部 slopes，包含两个 binary 项，不惩罚 b。 |
| fit_points | L-BFGS-B；检查 success 及最大梯度<=0.01；解析算截距。阈值是优化器保护，不是科学有效性证明。 |
| log_intensity | b+x beta；M3 可显式将覆盖代理标准化列置 0，等于固定在训练域平均 log1p 距离。v3 主流程未输出这个标准化 M3 图。 |
| evaluate | 测试事件得分、测试域面积积分、presence-background AUC、条件对数增益、Top 面积阈值；不重新标定测试域截距。 |
| coarsen_quadrature | 以投影坐标 floor 分组，面积求和；选距离 coarse cell 中心最近的 fine 代表点，cell_id 打破平局；删除 event_count。 |
| spatial_validation | 训练事件和训练 quadrature 同时剔除 test blocks 和 guard；每折调用 fit_points，因此重估 prep。 |
| summarize_cv | 折 AUC 按 test_events×test_area_km2 汇总；capture 与 conditional gain 按事件数汇总；跨 seeds 报均值和范围。 |

### 3.2 event / background / likelihood / mass

- event term 使用保留的矿点真实输入坐标处提取的特征，不取最近积分点特征，不把矿点移到格心。“真实输入坐标”不等于已证明真实矿床位置精确；论文匹配误差另论。
- quadrature/background 是对 D 的面积积分支撑；并非确认无矿、也不是 132,888 个独立负样本。事件点不必与积分点重合。
- area_km2 在 `log_area=log(a_j)` 中进入 logsumexp；lambda 的数值单位是拟合目录事件/km²。面积统一乘 c 时，slopes 不变、b 减 log c；raw likelihood 的单位依赖不能忽略。
- `PointFit.log_likelihood` = `n*b+S^T beta-n`，是**惩罚估计值处、未扣 L2 的 quadrature PPP log likelihood**，不是 optimizer 的负目标值，也不是 conditional likelihood。省略点过程参考测度约定下与参数无关的常数。
- simple 的 `fitted_models.json.training_log_likelihood` 保存上述值；v3 的 `fitted_models.json` **没有保存 training likelihood、n、梯度或收敛状态**，只存参数与 prep。v3 外部/折表的 `point_log_likelihood` 来自 evaluate。
- legacy `fit_ppp` 保存的是包含 `y log(area)` 和 `-gammaln(y+1)` 的**网格 Poisson count** likelihood，不能与上述 exact-event likelihood 的数值直接比较。
- 在用于拟合的同一 quadrature、同一特征设置上，`sum a_j lambda(q_j)=n` 是 profiled intercept 的恒等式，即使 slopes 尚不理想也成立。它不是恢复未知矿床数、成功校准或泛化良好的证据。
- 用 coarse fit 在 fine grid 上积分、在 held-out region 上积分、或修改覆盖代理后积分，不保证 mass=n；须分别标明。
- `logsumexp(log_area+xq@beta)` 及 softmax 权重正确避免了中间指数溢出；evaluate 最后 exp(log mass)、地图 exp(log intensity)、diagnostics 的 exp(scores) 仍可能在极端外推时溢出。已有数值稳定测试仅覆盖 profiled_objective，不能泛化到全部输出。

### 3.3 10 km 聚合近似

面积总和守恒不等于积分准确。`sum_{fine in cell} a_f exp(x_f beta)` 被 `A_cell exp(x_rep beta)` 替代；不是平均强度，也不是面积平均连续特征，更不是分类比例积分。完整 2×2 组的代表点可由 cell_id 决定，存在确定方向的采样偏好。岩性类别、近接触带梯度和小岩体尤其可能被代表点遗漏。

coarse 聚合也改变了用于加权 median/mean/SD 的特征分布；固定 alpha 在标准化坐标里施加，因而 5/10 km 对比并非纯粹“只换积分精度、其余完全不变”。这是需要披露的附加差异。

默认 200 km blocks 与 10 km origin 网格对齐，coarse 组不跨 block；但 50 km guard 的弧形边缘及 compact Voronoi 边缘仍按代表点留/删整个 coarse 面积，未精确裁切部分单元。5 km 原 support 本身是中心筛选的 25 km²整格，不是实际海岸/岩性面积。

## 4. Existing guarantees

“测试代码覆盖”“历史产物记录 passed”“本阶段重新核实”是不同证据级别。本阶段没有重跑现有测试，也不重复现有单折 replay。

| 现有检查 | 已覆盖什么 | 不证明什么 |
|---|---|---|
| simple PointProcessTests（8 项） | 数值梯度；mass与事件数、保存LL一致；忽略event_count且响应事件特征；面积单位变化只改截距；常量强度 AUC=.5/gain=0；ties可使Top10实际面积=100%；训练prep不被matrix修改；覆盖标准化显式性；大logit下objective有限 | 不证明模型选择无偏、真实事件定位、完整域积分或全部输出稳定 |
| simple SpatialTests（3 项） | coarse面积守恒及真实代表点；到方块而非格心的guard距离；按真实坐标分块与seed确定性 | 不证明真实GIS拓扑、所有compact边界面积、独立地域泛化 |
| simple RealDataResultTests（4 项，有条件跳过） | 针对旧simple_v2的44点每重复留出、50 km隔离、梯度、输出完整/有限；旧特征5/10 km质量指标门槛 | 非当前v3全套集成回归；目录不存在即skip；不是所有分辨率的收敛证明 |
| v3 GeometryTests（2 项） | 合成相邻方块中内部接缝/水边排除；pink优先、Unknown不成接触；距离转换 | 不覆盖真实sliver、repair产生GeometryCollection、错位边界、source CRS变化 |
| v3 RegionTests（1 项） | 小合成数据的Voronoi序与最近中心标签一致、polygon有效 | 不证明划分独立于事件、测试域精确积分 |
| v3 CacheTests（2 项） | mock signature下复用；缓存文件损坏/规则版本变化触发重建 | 不证明signature穷尽依赖、原始GIS科学质量；未覆盖所有缓存schema错误 |
| verify_ppp_v3_results.py | 数量、binary、事件有效域、接触距离、留出覆盖、隔离、influence/stability输出、residual总量、地图关联等一致性 | 调用load_features可能重建缓存并写verification.json；不是纯只读，也不是外部独立验证 |
| audit_completed_run.py | 原始点坐标/投影；独立几何重建15个block折和4个compact折；核对缓存哈希；一次fold replay、Cu训练均值、外部得分和4个未覆盖点 | replay复用同一数学实现，只独立验证部分几何和一致性；compact audit未独立核对全部训练quadrature；仅一个模型一个折replay |

读取到历史 `ppp_run_20260905_195450_443/verification/audit_summary.json` 记录19项fold checks、26项hash checks、一次replay误差约1.07e-14。旧 `ppp_binary_lithology_v3/verification.json` 记录 checks=passed、25次稳定性refits、1个无效岩性support格。这些是**已有记录**，本阶段未重新核验其全部原始数据哈希，不能写“今天全部测试通过”。

本阶段新检查：上述12个指定Python文件全部由 ast.parse 成功解析；读取两份 .prj；只读扫描grid.csv定位无效格；核对小型配置/汇总。没有导入这些模型脚本，没有产生pyc，没有重拟合。

## 5. Critical findings

这里 Critical 指会使研究主张失真，而不是已经发现程序崩溃。

### C1 模型身份必须立即统一

若研究目标是“A，仅替换ACAWLR最终二分类形式”，当前代码没有完成目标。B可以是独立且合理的研究方案，但不能被包装成A。证据见第2节。当前推荐入口文档并未全部犯此错误；风险在后续题目、汇报与比较归因。必要动作是先确定研究主张，不应以新训练掩盖身份不一致。

### C2 目录事件、绝对概率和盲测不可混用

44点不是经完整调查得到的全部事件，也没有发现/收录概率；lambda混合地质关联与被发现、被选入论文匹配子集的过程。16点历史已反复评价，当前代码把它们放在最后只能避免本次自动调参，不能恢复盲测资格。raw mass及训练mass恒等式不能支持未知矿床数量；intensity、percentile、target stability均不是 calibrated mineral probability。

若报告只主张“既有调查协变量条件下的目录事件相对排序及探索性验证”，上述限制可诚实容纳；若声称无偏矿床概率、未知矿床总数或真正独立盲测，当前证据不支持。

## 6. Major findings

### M1 积分域与新的有效岩性定义不完全一致

`binary_geometry` 排除 Water/Unknown/Ice/Dam；legacy `build_domain_grid:361` 只排除 Water/Unknown。v3 `load_features:149–151` 只对事件要求 lithology_valid，对fine积分支撑不删除无效格，也不把该标志放进MODELS。

本阶段只读定位到 cell_id=84489、(-109.6091247,43.1439585)、原 lith_symbol=Ice、area=25 km²、lithology_valid=False，仍编码 lith_intrusive=0并参与积分/输出。单格占固定fine总面积约0.0007525%，不能仅凭这点断定性能失效，但“所有other均有效岩石”是不准确的。与之分开，Kelsey 位于mapped rock内、旧support格心掩膜漏格，已有audit记最近积分点3.2226 km。两者都是域/积分近似问题，不应静默删矿点或重写历史缓存。

### M2 原始协变量可能携带调查过程；CV只条件于现有资料

地化IDW对全区原始地化样本先计算，重力/断层/岩性同理。没有用测试矿点标签拟合预处理的直接代码路径；但测试区已有调查样本参与了特征，不能声称预测“完全没有调查的新地区”。地化勘查若因已知矿点而密集，即使折内scaling正确，仍可能携带发现过程。需要调查时间/采样设计证据才能作更强结论。

M3是最近地化样点距离的普通协变量，不是已知effort offset、检测概率模型或抽样逆权重；固定它不能识别真实地质强度。M0本来就有 gravity_distance_deg/has_gravity 两个覆盖相关变量，也不能把“采样覆盖影响”完全归于M3。

### M3 对当前模型的积分精度与域边界证据不完整

旧simple_v2 RealDataResultTests和quadrature_convergence比较的是旧接触特征。v3 main没有重做新共同边界M2的5/10 km convergence。已有结果提示旧设置不敏感，但不能据此声称当前M2-new已数值收敛，更不能把25 km²格心面积当精确陆地面积。未来检查应先固定fitted参数在更细/分组求和积分上审计质量，而非为AUC搜索分辨率。

### M4 原论文/ACAWLR/PPP现有数值不是受控算法对比

ACAWLR reproduction使用1 km、2 km矿点缓冲正样本、采样背景、class weights、不同CRS/协变量、分组验证及BCE；旧PPP用cell counts；当前PPP用exact input event features和面积背景。AUC的样本单位、背景域、折与目标均不同。不能将原论文或旧reproduction的AUC与0.9488直接相减解释“PPP优于ACAWLR”。

reproduction的普通 `train_models` 用同一个fold validation选择best epoch并报告该折分数，会带来模型选择偏差；multiscale的 `run_strict_acawlr` 另设内层epoch选择，不能把前者与后者混称同等严格。传统logistic在随机StratifiedKFold之前用全部训练表median填补，存在内层OOF预处理泄漏；其背景池又排除了全部60点周围区域，历史16点坐标影响训练背景构造。这些不是当前PPP的新泄漏，但限制旧结果的公平性和“从未使用外部点”表述。

### M5 事件分布参与分折；固定alpha不等于从未选择过

重复blocks按全部44点的块事件数平衡，compact直接对全部44点坐标KMeans。并未使用预测得分或16点构造分折，不是把测试响应直接放进训练loss，但不是外生、事先固定的地理测试区。结果应称“事件平衡分块验证”和“基于现有事件坐标的连续区域留出诊断”。

v3固定alpha=.1，无外部评价反馈到参数的自动路径。文档也明确历史16点比较；现有单一baseline Git历史不能证明alpha或方法选择从未受以往外部结果影响。结论是**当前代码无直接external-set调参，历史selection风险无法排除**，不是指控已证实alpha按外部AUC优化。

### M6 岩性共同边界有更合理的语义，但不等于真实地质接触真值

make_valid → 同类union（等价于按binary dissolve）→ other减pink → 两者boundary相交，排除同类内部接缝、单侧水边/Unknown边和通常的外框边。pink优先是明确的冲突解决规则，不是地质学证据；若源面重叠，也可能在差集边缘产生人为优先级界线。

无snap避免任意移动边界，同时exact intersection会漏掉不吻合sliver。已存几何audit中overlap=0、shared长度155,525.72 km、merged pink边界158,940.47 km；这些是已有窗口范围记录，不能把差长全部解释为“已识别的错误海岸/接缝”，也不能把窗口polygon面积视为最终研究域面积。geometry repair产生的低维残片、空结果、近邻错位和分区接缝缺少真实数据定量测试。

## 7. Moderate/minor findings

### 7.1 GIS / feature 逐项审计

| 环节 | 已核实实现 | 限制/未覆盖风险 |
|---|---|---|
| 地化IDW | legacy:403–442；12最近样本、power=2、距离floor=1000 m，在ESRI:102039投影距离上加权；按元素分别忽略非finite值重归一化 | 先找12邻点再排缺失，不补搜第13个有效元素样本；可能全空后交给折内median。没有最大插值半径、检测限/负哨兵值解析、样本重复/质量标记审计。clip负Cu/Au/Mo到0会掩盖原始编码问题，尚未确认数据实际含此问题。 |
| 覆盖代理 | nearest_geochemistry_km=所有地化样本最近距离，不管该点元素是否有测值 | 不等于每种元素的有效覆盖，更不等于勘查努力；IDW源先裁study bounds，边缘可能遗漏区外近邻。 |
| 断层 | legacy:445–463；丢弃null/empty，to_crs等面积投影，sjoin_nearest，m/1000→km | 未检查/修复nonempty但invalid geometry；CRS缺失时直接假设4326不稳健。本地.prj实际为WGS84 EPSG:4326，所以不能声称当前数据已错投影。 |
| 重力 | legacy:467–492；XYZ当lon/lat/value，float32，最近邻使用二维degree距离；<=0.25 degree才has_gravity=1，否则gravity_bouguer=NaN | 0.25不是固定km圆；经向距离随纬度变化，degree欧氏最近邻不严格等于测地最近邻。gravity_distance_deg在missing时仍保留，连同indicator可学到覆盖边界。没有显式no-data sentinel、0–360经度或header/单位验证。 |
| 重力missing | 折内训练quadrature median填补异常值，has_gravity保留 | 不是自动校正missing-not-at-random；数据包命名指Bouguer，不足以证明mGal单位/异常基准与作者完全一致。 |
| 岩性分类 | Symbol strip/fill Unknown，严格匹配Igneous, intrusive；其余非排除类为other | “粉色”是属性映射的简称，不是PDF取色、岩石年龄/成因或成矿岩体鉴别。未验证source taxonomy变更/大小写别名。 |
| bbox与crop | read_lithology:267–288先用投影bbox+100 km读取，再to_crs；读取相交面而非逐面clip | 本地MapUnitPolys.prj确为USGS Albers、米单位，与目标定义相符；换成别的source CRS时bbox元组可能误筛，缺少入口断言。无显式crop不会制造裁切面边，但读取窗口仍可能遗漏较远的接触对象；100 km不是全域最近接触充分性的证明。 |

坐标/单位链：输入 longitude/latitude 为经度/纬度；Transformer均采用always_xy=True，从EPSG:4326到ESRI:102039，metric_x/metric_y为米；断层/接触/地化邻距除1000得到km，block/guard乘1000；面积从m²除1e6或用grid_km²。gravity_distance_deg特意保留degree，不能与km列直接相加或使用共同距离阈值。ESRI:102039等面积投影适合面积积分，但不是处处等距；50 km是投影平面隔离距离，非精确测地线50 km。

reproduction默认ESRI:102008；当前PPP用102039，历史表格不能仅凭metric_x/y同名混用。legacy缓存loader只校验config中的grid/CRS和坐标有限性，未对所有fine点重算lon/lat↔metric一致性；既有audit只对60点做了投影核对。未发现当前实际经纬度互换证据，但完整缓存CRS一致性未被上述测试覆盖。

### 7.2 工程和解释细节

- binary 0/1与连续变量SD尺度上的斜率都用同一L2；这是合法但不“尺度中性”的先验/正则化。折事件数变动而alpha固定于未除n的总likelihood，较小训练集相对惩罚更强。
- 所有quadrature某feature缺失或常量时，prep用0/SD=1；若事件该feature仍有变化，模型可能由事件项和L2在缺乏积分域变异信息时支撑系数。现有测试没有这一科学边界情形。
- cache signature覆盖主要原始数据和三个代码文件，值得保留；但环境/GEOS/GDAL版本、所有可选sidecar和返回schema未完全封装。cache hit路径不再调用validate_inputs，哈希一致证明字节一致，不证明含义正确。
- v3的config保存alpha、seeds、模型、feature manifest；block/guard/folds/固定support等部分设置只在代码内，完整run provenance还依赖源哈希与文件路径。GitHub仅SOURCE baseline，不含数据/结果，不能称可在干净环境立即复现。
- `PPP_LITHOLOGY_METHOD_README.md` “16点始终封存”与后续文档及历史比较不一致；其推荐10 km旧入口也已被v3入口取代。保留历史可以，但正式报告需标记版本与时间语境。
- slope_l2_change只在逐事件删除、quadrature相同因而prep相同时可直接解释为同一标准化参数空间的变化；跨区域/尺度重估prep后的斜率差不能直接照搬此含义。

## 8. Validation assessment

| 设计 | 代码行为 | 可以回答 / 不能回答 |
|---|---|---|
| repeated block CV | 200 km正方形按44点数/面积分配五折；三seed只改变部分平局次序；同一fold可由不连续块组成 | 区域内空间隔离排序的划分敏感性；不是3个独立数据集，范围不是CI，200 km/50 km未由相关长度估计证明普适合理 |
| 50 km guard | 到test square的精确平面距离；同时删除训练event和训练quadrature；不只删标签 | 没有把测试域quadrature留在训练积分的代码漏洞；但保留单元代表面积近似，非精确buffer裁切 |
| compact validation | 44点KMeans固定seed42、4中心、Voronoi，分别删test及50 km邻近event/quadrature | 比交错块更接近连续地域迁移；是依事件位置构造的4区诊断，不是预先指定未知区域盲测；只有1次划分 |
| event influence | 每次删1事件、积分域不变，43点重拟合，比较Top10 Jaccard/斜率 | 特定目录点的拟合影响；不是LOO空间预测性能，也未覆盖坐标误差的方向/幅度 |
| target stability | 依次删含事件的200 km块及50 km guard，在共同fine域预测；归一化图形，统计Top10选择频率 | 删除设计下靶区敏感性；不是bootstrap概率或后验置信度，且地图混合各次训练/留出区域 |
| regional residual | 全44点全域拟合，200 km块observed-predicted | in-sample模型错配诊断；不是OOF独立计数检验，残差总和约0由截距恒等式约束 |
| historical 16 | 不进入当前loss、prep、fold map；所有内部过程后评价 | 历史探索性检验；不能重新命名独立blind external set |

fold map用测试事件位置/块计数是否“不该使用”：用于事件分层平衡在本研究可公开描述，不等同训练泄漏；但若主张事前未知区域预测，则这类划分不符合主张，需要将研究问题限于现有目录条件下的验证。compact利用坐标定义区域也不因“没有用模型分数”就自动成为独立验证。

外部16点与44训练点之间没有额外50 km隔离；不能把内部guard属性推广到历史16点评价。报告应同时展示逐折事件数、训练数、测试面积、novelty，避免一个加权平均掩盖困难区域。compact novelty只检查测试事件逐变量min/max，不检查联合特征支持和所有测试背景。

## 9. Metric assessment

| 指标 | 准确定义及解释 | 主要边界 |
|---|---|---|
| presence-background AUC | 从目录事件等概率取一点，与按面积取背景位置相比的排序概率，ties计0.5 | 背景不是确认无矿；高AUC不等于absolute probability或资源发现率 |
| Top5/10 capture | 背景强度面积加权分位阈值；事件得分>=阈值即命中 | 需一起报actual_area_fraction；离散格/ties可超名义面积。内部每折用该测试区阈值，外部用全域阈值，两者靶区预算含义不同 |
| conditional log gain/event | mean(log lambda(event))-log(M_test)+log(A_test) | 等于对测试域归一化位置密度相对uniform的平均log gain，截距抵消；不是计数校准，不能只称“排序”，因为还反映密度集中程度 |
| point_log_likelihood | sum(log lambda(event))-M_test | 同域、同单位、同事件过程才可比较；空间测试区可评价目录计数与位置，但全域44拟合对历史16子集的raw LL会混入子集抽样率差异 |
| predicted event mass | 面积积分lambda | 拟合目录尺度的期望量；同训练quadrature=n属构造恒等式。若16是保留子样本，未建thinning/exposure便不能把全域预测mass与16对比当校准 |
| regional residual | observed目录数-predicted目录mass | 未标准化、未考虑过离散/残差相关；大域或高期望块可有较大绝对残差，不等同“漏矿数” |
| Spearman | 网格分数秩相关，代码为逐格未额外面积加权 | 固定5 km等面积格时等价于面积均匀抽样的格点秩比较；变化面积网格不成立。全区高相关可掩盖Top区域变化 |
| Top10 Jaccard | sum(area in intersection)/sum(area in union) | 靶区重叠、非性能或概率；必须同域、相同阈值规则 |
| event influence | 删点后的Top10 Jaccard和标准化slopes L2变化 | 稳健性诊断，无因果解释、无自动“坏点”删除理由 |

### AUC 的 test_events × test_area_km2 权重

对折f，AUC_f的分母为 `n_f A_f`，分子为该折所有事件与面积背景的加权正确排序总量。因此

```
sum_f (n_f*A_f*AUC_f) / sum_f(n_f*A_f)
```

恰好是**只比较同一折模型下的事件—背景配对，再将这些配对汇总**。它避免跨折不同强度标定的直接比较，有可解释的estimand，不是数学错误。

但大面积且事件多的折权重更大，不是每个事件在整个研究区随机背景下的AUC，也不是按事件数简单平均。尤其compact面积差异大，图注必须写“事件数×测试面积”；同时给逐折值即可，无需为了指标更好再换汇总规则。summarize_cv中capture与conditional gain才是事件加权，不能把三者混成一个说明。

所有intensity / percentile / target area / stability频率都不能解释为 absolute mineral probability、calibrated probability或unknown deposit count。即便数学Poisson过程可以从已知强度导出区域至少一事件概率，本目录抽样机制未知，当前lambda也不支持这种地质概率解释。

## 10. Visualization assessment

审计对象 `build_paper_figures.py`；这里只静态审查生成逻辑，未重新绘图，未对现有PNG/SVG像素或最终字体做视觉验收。

| 项目 | 发现和判定 |
|---|---|
| hardcoded run path | 第8行固定S=ppp_run_20260905_195450_443，O=reports/paper_figures；脚本顶层立即执行/写文件。只适合作为该历史run的专用图，不会自动跟随新run。 |
| provenance | g/metrics来自S，ev/ex却来自可变的共享ppp_binary_features_v3，misses来自S/verification；未比较manifest、点表或cell_id签名。未来缓存重建后可把不同版本静默混画，这是必须改的来源绑定风险。 |
| intensity / percentile | Fig3先对M2 intensity算面积累计rank×100，色条准确写面积百分位、不是有矿概率；没有把原始lambda误标成百分位。图展示排序，不展示强度倍数或校准。 |
| Top10 | 用累计area取0.9阈值，v>=threshold；确实面积加权。需要用actual_area_fraction标明“约10%”，不能只报固定整数10%。 |
| scale公平性 | Fig2所有模型AUC共同0–1.1轴、capture同轴范围，没有截断0轴夸大差异；1.1为文字留白，AUC有效范围仍应明确0–1。Fig3百分位固定0–100；右图是二值分类，二者不是强度尺度对照；没有跨模型地图尺度比较。 |
| smoothing/interpolation | pcolormesh(shading='nearest')按格点着色，无Gaussian smoothing、KDE、空间插值；IDW已经发生在原始地化特征阶段。rasterized是输出编码，不是平滑。 |
| aspect ratio / coords | Fig3用metric_x/y除1000并set_aspect('equal')，正确。二维数组由unique xs/ys构造；没有验证所有轴间隔都为5 km，若未来整行缺失pcolormesh可跨缺口扩宽格子。 |
| blank area | NaN支撑外；图注已说明不是无矿。白色没有进一步区分域外、水体、缺图或格心漏格；当前无效Ice格反而仍着色，所以不能称“仅有效岩石预测图”。 |
| validation legend | 黑点44、白三角历史16、蓝圈4未覆盖点，性质基本准确；未覆盖人数及4个ID/offset硬编码，只适用于当前run。若换数据可能错误或KeyError。 |
| y-axis / error bars | 柱从0起，无截轴误导；散点(a)为3次repeat，(b)为4个fold，含义不同，图注已区分且非CI。禁止把散点散布当不确定性区间。 |
| 图2文字错误 | 生成图注写compact汇总“按测试事件数加权”，实际summarize_cv按事件数×面积；现有图注文件同样须纠正，柱数值本身不因此错误。 |
| 图1流程 | 可以表达数据流，但预处理折内估计只在图注说明；不能让人以为一次全量fit后再做内部CV。可在下一阶段加“折内拟合”短语。 |

可用于报告的部分：Fig1作为独立PPP概念流程；Fig2现有数值作为同一run五个特征方案探索性比较；Fig3作为该历史run的面积百分位/Top区域图。使用前须核对该run与共享cache一致，修正图2权重图注，并在正文保留历史16点、目录强度、面积近似限制。**不能在尚未视觉验收时称“全部出版就绪”。**

必须改的是图注权重与可变cache来源绑定；未来复用图脚本时还需参数化run路径、动态生成miss labels/counts、记录实际Top面积。单纯把图变平滑或美化色带不解决科学问题，不建议为此做新实验。

补充：v3 `plots` 的经纬度scatter概览没有统一投影aspect；其compact_fold_maps在投影坐标下equal aspect较稳健。正式空间图优先使用build_paper_figures的投影地图，避免把经纬度方格外观解释为等距几何。

## 11. Remaining limitations

尚无可靠依据确认：

1. 44/16图件匹配坐标完全复原作者原始点位及随机划分；现有8点复核标记只定位审查对象，不量化位置误差。
2. 真实勘查努力、发现概率、收录/匹配概率和调查时间顺序；不能分离地质强度与观测过程。
3. 当前新接触模型对精确陆地域、sliver处理和5/10 km积分已经收敛。
4. 50 km足以消除所有相关性，200 km代表目标部署地理尺度；CV并未估计空间相关长度。
5. 冻结alpha=.1及M2推荐完全独立于历史16点评价；只有当前代码路径可确认无自动选择。
6. 全部原始地化/重力no-data编码、单位、重复采样、所有GIS geometry validity均已逐条审计；本次没有重建原始特征。
7. PPP独立事件假设充分，或Poisson count方差适合该选中目录；in-sample raw residual不等于过程拟合优度检验。
8. 全球系数的符号可作为地质因果效应，或binary intrusive指示能替代岩体年龄、成因、埋深和矿化相关性。
9. 当前图件在字体、遮挡、分辨率和期刊要求上均已通过人工视觉验收。

这些是范围边界，不是已证实的所有数据错误。现有静态证据足以判断模型身份和数学结构，不足以替代新数据的独立验证。

## 12. MUST / SHOULD / NICE experiments

以下是**后续建议，Stage 1A均未执行**。先区分无需实验的必要修正与真正需要计算/新资料的项目。所有新结果应新建目录、保留历史44/16原方案，不按AUC更换点、alpha或模型。

### MUST HAVE

| 工作 | 具体回答导师/审稿人的问题 | 最小可审查产物/完成条件 |
|---|---|---|
| 非实验：冻结模型身份、研究主张与版本说明；纠正旧“封存”和图2权重文字 | “这到底是ACAWLR+PPP，还是独立PPP？所谓外部验证是否独立？” | 一段明确B的Methods、模型模块对照、16点历史使用声明、精确AUC定义；不需要训练 |
| 非训练的积分域/岩性与地学schema审计 | “模型究竟在哪个D上积分？水/冰/Unknown是否被当岩石？坐标和缺失值含义可靠吗？” | 定位已知Ice格和Kelsey缺口；所有支持格有效性/面积规则、CRS与重力/地化单位/no-data编码的可追踪清单；先量化，不静默改域 |
| 当前M2-new的固定参数积分检查；仅必要时预先定义成对5/10 km重拟合 | “新共同接触特征很陡，结果是否只是代表点积分误差？” | 先重用已有参数与fine特征比较coarse/fine mass与积分项、定位贡献最大的单元；不扫分辨率找AUC。若再比较拟合，报告prep/L2随尺度变化，区别数值积分和预处理影响 |
| 非实验：绑定图/报告到同一run与feature manifest，修正图注并视觉验收 | “这张图和这张表是否真的来自同一批点、同一个模型？” | 每个图源校验、动态命中标记、实际Top面积、明确blank定义；不用新训练 |

“必须有全新盲测”只在论文保留“独立外部泛化已验证”的主张时成立；当前若限定探索性结论，不应为凑MUST虚构可取得的新样本。

### SHOULD HAVE

| 工作 | 具体回答导师/审稿人的问题 | 最小范围 |
|---|---|---|
| 冻结方案后，用与事件无关的明确地理区留出；有条件时用真正新目录/时间留出 | “你的高分是不是事件平衡或KMeans分区设计带来的？能否迁移到预先指定区域？” | 先预注册一个有地学/部署含义的划分；固定features/alpha/metric；不试多划分挑最好；新数据须明确发现时间与重叠排除 |
| 基于已有标记的点位来源复核，再决定有限坐标不确定性评估 | “接触距离效果是否由图件匹配位置误差驱动？” | 先查8点来源和合理误差范围；只有依据充分才做有限扰动/成对方案。保留原44/16，不能因低分删点 |
| 调查覆盖来源与时间核对；仅在问题确实包含无调查区时做相应留出设计 | “模型学到的是矿化规律还是采样位置？” | 报元素有效样本覆盖与区域差异；有真实effort信息再建观测过程。M3存在本身不能结题 |
| 用既有逐折输出补充分区指标和适用域说明；必要时才做OOF区域计数诊断 | “平均AUC是否掩盖某地区失败或计数失配？” | 先展示每折n、area、gain、capture、novelty；若跨折比较mass，必须解释各fold目录暴露与训练n不同，不能拼成一个校准强度图 |

### NICE TO HAVE

| 工作 | 具体回答导师/审稿人的问题 | 为什么不是现在必须做 |
|---|---|---|
| 外部证据支持下的更细地质类别、年龄或多尺度特征 | “binary intrusive把不相关岩体混在一起，会不会限制地质解释？” | 新数据质量/样本量可能先限制可辨识性，不能用特征堆叠替代验证 |
| 同域同点同折的ACAWLR与PPP受控对照，或正式ACAWLR+PPP研究 | “收益来自presence-only formulation还是空间权重架构？” | 必须重构训练接口/设计公平对照；属于独立sprint，不是当前小修复 |
| 在明确科学问题下做残差空间过程/过离散扩展 | “给定协变量后仍有无法解释的聚集吗？” | 先证明残差问题与观测偏差无法由现有诊断解释，再考虑Cox/随机场；不应先上复杂模型 |

不建议重复已经覆盖的梯度、mass恒等式、合成guard和cache损坏测试来充当新科学证据。不建议为了提高AUC调alpha、扩大量级seed搜索、盲目增加sensitivity矩阵或重新实现PPP。

## 13. Exact next recommended action

**下一步应先做一个不训练、不改模型的“科学表述与数据契约修正”任务，而不是开始完整运行。** 本报告提交后停止，后续任务须另行启动。

具体验收范围：

1. 将主方法身份固定为B，并让当前入口文档、方法段和图注一致；如研究目标必须是A，先另立设计任务，不称已实现。
2. 纠正图2 AUC权重说明、历史16点“封存/盲测”措辞；保留旧结果原始记录。
3. 写清固定legacy support与binary valid-rock domain的差异，记录已知25 km² Ice格和Kelsey漏格；先审查域定义，不自动删除或重新生成支撑。
4. 为未来制图要求run/manifest/点表绑定及实际Top面积说明，明确shared cache不等于不可变run输入。
5. 上述契约明确后，才批准一项范围有限的当前M2-new固定参数积分审计；是否需要重拟合由积分证据决定，而不是AUC是否提高。

本Stage 1A的交付仅为本报告及其精确allowlist。没有实施上述后续修正，没有自动开始Stage 1B。


## Stage 1B verification results

### B1. Scope, environment and execution record

日期：2026-09-16。起点为 Stage 1A 提交 `7b22319d2e37a6050067cbc83461f58bb7bc24c9`；开始时工作区干净，分支为 `codex-refactor`。上文第1–13节保留为 Stage 1A 的历史审计记录，其中“本阶段未运行测试”等表述指 Stage 1A；本节记录新执行的 Stage 1B 证据。

**结果：现有20项测试全部通过，新增2项无拟合的微型指标回归测试通过；已有审计中的19项几何/记录检查和26项文件哈希检查通过；实际仅重放1个既有空间折，保存预测的最大绝对误差为1.0658141036401503e-14。没有发现新的测试失败或已证实的科学实现错误。**

运行时：`D:/anaconda/python.exe`，Python 3.12.7；NumPy 1.26.4、pandas 2.2.2、SciPy 1.13.1、scikit-learn 1.5.1、Shapely 2.1.2、pyproj 3.7.2、GeoPandas 1.1.3。设置 OMP/OPENBLAS/MKL 线程数为1；使用 `-B` / PYTHONDONTWRITEBYTECODE 避免写入pyc。未安装或升级依赖。

先执行用户指定套件：

```powershell
& 'D:/anaconda/python.exe' -B -m unittest -v test_ppp_binary_lithology_v3 test_ppp_three_papers_simple
```

结果：`Ran 20 tests in 0.526s — OK`，0 failures、0 errors、0 skips。其中v3合成几何/分区/缓存5项，simple数学8项、空间3项、旧simple_v2已存结果4项。后四项只读取历史产物，**没有重新运行旧simple_v2模型或CV**。

原始日志及deterministic audit新输出保存在独立目录：
`mineral_prediction/output/stage1b_verification_20260916/`。
目录含 `unit_tests.txt`、`metric_regression_tests.txt`、`environment.json`、`audit_console.txt`、`replay_call_budget.json` 与 `audit/` 下的摘要/哈希/坐标/折检查表。该目录按既有规则local-only、未加入Git；本节和新增测试源码是随提交保存的可审查证据。未修改.gitignore。

### B2. Existing deterministic audit and the single replay

使用已有 `audit_completed_run.py` 的 `main()`，原始输入仍是 `ppp_run_20260905_195450_443` 和其feature cache；仅在调用时把模块的 `OUT` 指向新的 `stage1b_verification_20260916/audit`。没有执行原脚本的默认历史输出写入路径，没有改动审计程序源码。

另以临时 `unittest.mock.patch` 包装 `ppp.fit_points`：首次调用原函数、记录参数和收敛状态；若第二次调用立即报错。审计结束断言调用数恰为1。因此19项fold checks是读取已存划分并重建几何/核对成员关系，**不代表19次训练，不是完整 spatial CV**。合成unit tests里的微型拟合也不是实际数据fold replay。

| 单折核验项 | 本次实测 |
|---|---|
| 模型 / seed / fold | M2_binary_contact / 42 / 1 |
| alpha | 0.1，未修改或搜索 |
| 训练事件 / 留出事件 / guard排除事件 | 28 / 9 / 7，共44 |
| 训练quadrature | 21,194 |
| 训练与留出事件最小平面距离 | 75.53254375314215 km，>50 km |
| 优化器success | True |
| 最大绝对梯度 | 4.1592140340246386e-05，小于原代码保护阈值0.01 |
| 9个留出事件的保存log-intensity最大复现误差 | 1.0658141036401503e-14，小于原审计容差1e-7 |
| 训练quadrature面积加权Cu均值（独立重算） | 3.089988564850937 |
| replay模型prep中的Cu均值 | 3.089988564850937 |

held-out名单与保存预测表按deposit核对一致；训练/测试名单无交集。审计以Shapely test-square并集距离重建训练事件和训练quadrature掩膜，同时排除测试块与50 km guard，并核对保存的train/test/quadrature数量。重放使用这个训练子集，成功复现保存分数，为“该折没有把测试quadrature纳入训练积分、prep使用训练域”提供运行证据。**这仍不是独立重写PPP求解器的数值交叉验证**，也未对所有特征的median/SD各做一份独立实现。

其他既有轻量检查结果：

- 15个block折（42/43/44各5折）以及4个compact折的成员、保存记录和事件间隔全部匹配；所有检查的最小事件间隔为54.90892622792758 km。compact检查核对事件及区域，未独立核对全部compact训练quadrature积分权重。
- 60个事件的坐标与来源表最大差为0 degree，split标签一致。EPSG:4326 → ESRI:102039重投影最大误差为6.984919309616089e-10 m，小于0.01 m容差；60点都不恰好位于积分点中心。这支持“保留输入事件坐标”，不证明论文图件匹配坐标准确。
- 19个signature输入文件（含原始数据、主要shapefile sidecars、三个生产模块、旧support和点表）与7个缓存输出文件共26项SHA-256全部匹配；另行通过run config中feature_manifest_sha256与当前manifest的绑定断言。没有调用会自动重建特征的 `v3.load_features`。
- 原审计最后用已保存全量参数直接重算历史16点分数，满足其误差<1e-10的断言，Top10未覆盖仍是4点：Fish Creek、Gabbs Group、San Xavier North、Two Peaks。这里只复核保存预测，没有用44点重新拟合或启动新44/16实验。
- 缓存哈希一致现在是本次核验结果，不再只是读取历史“passed”文字。但它只保证被记录文件的字节一致；不证明原始资料正确、所有依赖均已记录或发现偏差已消除。

### B3. Only two small new regression tests

没有重复新增梯度、mass、合成guard或cache corruption测试。仅在已有 `test_ppp_three_papers_simple.py` 增加27行 `MetricWeightingTests`，补足现有套件未直接覆盖的指标面积权重。两项测试均不调用fit_points，不读取真实数据，不改变生产算法：

1. `test_auc_uses_background_area_and_half_credit_for_ties`：背景分数[0,1,1,2]，面积[1,2,3,4]，事件分数1。按面积的手算AUC是 `(1 + 0.5*(2+3))/10 = 0.35`；若误用等权格点则为0.5。测试得到0.35，同时核对该离散例的Top10实际面积为0.4、事件命中0。
2. `test_cv_auc_uses_event_area_pairs_not_event_only_weights`：两折事件数[2,1]、面积[10,100]、AUC[0.2,0.8]，正确汇总为 `(20*0.2+100*0.8)/120=0.7`，仅事件加权会是0.4。测试得到0.7，并核对capture和conditional gain各自仍按事件口径汇总。

只运行新增类：

```powershell
& 'D:/anaconda/python.exe' -B -m unittest -v test_ppp_three_papers_simple.MetricWeightingTests
```

结果：`Ran 2 tests in 0.008s — OK`。因此本次共执行22个不同测试，全部通过；并非把整个22项套件又重复运行一次。

这使Stage 1A关于 `test_events * test_area_km2` 的判断从源码推导获得了独立手算例的运行支持，也确认现有图2“按测试事件数加权”的图注需要修正。图注修正不属于本次verification的代码变更。

### B4. Which Stage 1A judgments are now verified?

| Stage 1A判断 | Stage 1B证据级别与范围 |
|---|---|
| PPP gradient、profiled intercept/mass、saved LL关系正确 | 已通过现有合成数值梯度、mass/LL一致性和logsumexp大值测试；单折拟合成功。对完整似然等价性、惩罚解释和凸性的普遍结论仍来自第3节数学推导，有限测试不构成所有输入的证明。 |
| area weighting和unit scaling正确 | 现有非等面积数学fixture、面积统一放大只改截距、coarsen面积守恒测试通过；新增指标测试直接确认背景AUC及跨折AUC权重。未据此宣称实际海岸面积或当前M2-new积分已收敛。 |
| event/quadrature separation | 现有测试证明改event_count不影响拟合、改真实事件特征会影响拟合；实测60点保留来源坐标，单折名单/积分域/保存预测一致。 |
| preprocessing没有直接测试集fit泄漏 | 现有prep不可被测试matrix修改的测试通过；replay中训练域Cu均值独立核对一致。支持代码所声明的折内处理，不证明全区预先调查协变量没有观测过程信息。 |
| 50 km spatial guard | 合成方块距离测试通过；实际19组事件几何检查通过；单折训练quadrature掩膜与保存记录/预测一致。不是地质相关性已在50 km消失的证明。 |
| GIS单位/坐标链 | 合成接触距离m→km、面积单位变换、60点投影检查通过；没有完整重算原始地化/重力/断层，不扩张为全部GIS单位已核查。 |
| cache provenance | mock缓存复用/损坏失效测试通过；实际26项哈希及run-manifest绑定通过；本次没有缓存重建。 |
| 新模型是独立log-linear PPP，即B | 与运行所用fit_points一致，但身份判断主要仍是Stage 1A调用链证据；没有通过测试变成ACAWLR+PPP。 |
| 旧simple_v2的已保存5/10 km质量门槛满足 | RealDataResultTests读取已有结果后通过；不是本次重跑，也不能推广到v3新接触特征。 |

### B5. Findings that remain static or scientifically unresolved

以下事项本阶段没有获得新的实验验证，继续保留Stage 1A结论的边界：

- 目录选择/发现偏差、M3是否能代表effort、历史16点的重复使用和alpha历史选择风险，不能由当前实现测试消除。
- compact区域由全部44点坐标构造、同折AUC的estimand、全域44拟合对16子集raw likelihood的解释，仍是代码/设计与数学层面的判断，未做新验证设计。
- 1个Ice积分格与Kelsey漏格是Stage 1A已读取的证据；本阶段没有再扫描重做该检查、改support或运行域敏感性实验。其对预测的影响仍未量化。
- 新M2共同接触距离的积分收敛、sliver/拓扑真值、全部原始missing编码、全fine网格坐标一致性、输出极端外推稳定性，均未新增验证。
- 图2权重实现已有运行支持；hardcoded路径、可变cache来源绑定和图件视觉质量仍是此前静态审计/待验收项。生产图脚本未修改。
- 高AUC、预测重现、mass恒等式与稳定缓存，仍不能被解释成绝对有矿概率、未知矿床数量或独立盲测成功。

### B6. New issues and final scope

**未发现新的失败测试、缓存不一致、坐标差异、guard违反或单折预测复现失败；无需纠正Stage 1A核心数学或模型身份结论。** 新增测试补上了非等面积AUC及跨折权重的覆盖缺口，并未发现需要修复的生产计算错误。已有科学限制与图注问题仍未解决；“验证通过”不等于Stage 1A所有风险关闭。

最终改动仅为本报告新增本节和现有测试文件新增两个微型回归测试。未改生产模型/参数、未重跑完整CV、未新44/16实验、未sensitivity analysis、未新模型、未重构；实际数据只拟合一个既有fold。历史结果和默认审计输出未覆盖，main与pre-codex-refactor仍指向基线a3a3f186；不merge、不push。按要求提交 `stage1b: verify scientific audit` 后停止，不自动进入Stage 2。
