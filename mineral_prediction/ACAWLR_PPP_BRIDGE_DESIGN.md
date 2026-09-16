# ACAWLR–PPP Bridge Design

日期：2026-09-16。Stage 1C，仅架构、数学推导与静态接口核查。基于 codex-refactor 的 a5e76e6（Stage 1B），先读取 [科学审计及 Stage 1B 结果](CODEX_SCIENTIFIC_AUDIT.md)，再核对下述实际源码。本阶段未训练、未运行 spatial CV、未做 sensitivity experiment、未实现新模型。

**结论：可以设计真正的 ACAWLR–PPP，但不能只把 sigmoid 换成 exp。** 应保留 location → anisotropic GASPG → convolution/attention → spatial coefficients 的路径，同时重建 event/quadrature 的共同评价接口、确定性归一化、PPP 积分及全局截距。当前推荐 PPP 仍是独立的全局 log-linear PPP，不包含这些神经空间模块。完整原网络约 22,966 个可训练参数，44 个事件不足以为其稳定泛化提供可靠证据；本设计推荐先做纯 synthetic P0，之后才考虑缩减的 P1。P1 是“reduced ACAWLR–PPP”，不是论文架构的逐层精确复现。

证据层级：代码行为为静态阅读/AST 核对；公式为解析推导；P0/P1 为尚未实现、未经实验验证的设计。Stage 1B 的测试和单折 replay 只支持已有 PPP，不能转移为新桥接模型的保证。

## 1. Original ACAWLR computational graph

### 1.1 实际入口与一个必须先解决的接口阻断

主要证据：

- [44/16 reproduction](ACAWLR_paper_44_16_reproduction.py)：load_core、make_core_config、fit_preprocessor、global_beta、train_epoch_loop、train_models。
- [multiscale reproduction](ACAWLR_paper_44_16_multiscale.py)：同类入口、feature_variants、run_strict_acawlr。
- [默认 core](../scripts/ACAWLR_improved.py)：compute_fault_features、generate_anisotropic_grid、ASPNN、ResBlock、CBAM、CAWNN、ACAWLR、predict_prob。
- [aligned](ACAWLR_paper_aligned.py)、[另一个 core](../scripts/ACAWLR.py)、[fallback core](ACAWLR_gravity_fault_maps.py)。

两个 reproduction 默认指向 scripts/ACAWLR_improved.py；只有路径不存在时才 fallback。**当前默认 core 的 Config 不声明 wrappers 传入的 deposit_path、study_bounds、deposit_buffer_km、make_maps 四个参数。** 静态 AST 比较已经确认，因此照当前签名构造会在训练之前产生 unexpected-keyword TypeError。本阶段没有执行训练入口来触发它。aligned.Config 也不接受 make_maps，不能仅改路径就宣称解决。fallback core 接受这四项，但仍需未来显式绑定完整接口和来源，不能静默替换以掩盖身份差异。

这补充了 Stage 1A 的边界：下面计算图来自真实已实现的类及 wrappers 的预期调用；**不能据此声称当前默认 reproduction 入口可直接运行**。Stage 1B 没有测试这些入口。此发现不证明历史结果无效；历史运行实际使用的 core 路径、版本、配置须另行核实。此次所读 improved.py 的 SHA-256 为 80AAF0431B209498303116BF85A76BFBA9DD882827549CB26BB783F8A262BA56。

ASPNN、ResBlock、CBAM、CAWNN、ACAWLR 五个类在 scripts/ACAWLR.py、scripts/ACAWLR_improved.py、mineral_prediction/ACAWLR_paper_aligned.py 中 AST 相同；这不是说整个文件、数据接口或实验流程相同。multiscale 默认 skip_acawlr=True，文件存在不等于该网络已经执行。

### 1.2 输入与空间路径

默认 reproduction 地学向量有十项：Cu、Au、Mo、Fe、dist_fault、fault_density、local_fault_confidence、gravity_bouguer、gravity_distance_deg、has_gravity。它不是现有含岩性接触项的 PPP M2 特征集。训练行计算 median/mean/std 后标准化，再附加常数 1；二元项也进入原标准化流程。coords 和 local_fault_angle 另行进入空间支路。

compute_fault_features 根据目标点附近断层线段计算局部轴向方向：长度加权的 sin(2φ)、cos(2φ) 合成角度，再除以 2；confidence 为合成向量强度。搜索半径通常 50 km，无候选时取最近断层，至多保留 80 个近邻。方向计算使用候选线的整条分段，而非严格裁到搜索圆内的线段。它是**轴向平均方向近似，不是论文的加权 PCA 各向异性估计**。

原函数自行 choose_metric_crs，可能选 UTM，也可能 ESRI:102008；不可默认与 PPP 的 ESRI:102039 一致。distance_normalize_m=50,000，parallel/perpendicular ratios=3/1；confidence 是普通特征，未自动改变这两个比例。

make_global_grid 用训练样本 coords 的极值加 5% padding，构造等间距 anchor lattice。reproduction 使用 120×100；core standalone 默认 48×48。这些 anchors 是空间编码参照点，**不是 PPP 有面积权重的积分格**。

对目标 s、anchor z，原实现计算：

    d_parallel = [(z_x-s_x) cosφ(s) + (z_y-s_y) sinφ(s)] / (3 × 50,000 m)
    d_perp     = [-(z_x-s_x) sinφ(s) + (z_y-s_y) cosφ(s)] / (1 × 50,000 m)

两个 signed offsets 截断到 [-20,20] 后输入 ASPNN。ASPNN 为 2→16→8→4→1 MLP，隐藏 ReLU，末层 sigmoid。逐 anchor 的输出重排成 GASPG，并截断到 [1e-6,1−1e-6]。这是真实实现的 learned proximity，但没有保证随距离单调下降，也没有保证 φ 与 φ+π 给出相同结果：signed offsets 同时翻转，而普通 MLP 并不必然是偶函数。

### 1.3 CAWNN 与最后的 latent predictor

CAWNN 的真实路径：

    Conv2d(1→32, 3×3)
      → ResBlock(32→8) → ResBlock(8→16) → ResBlock(16→32)
      → CBAM channel attention → CBAM spatial attention
      → global average pooling → Linear(32→p+1)

每个 ResBlock 是两次 3×3 convolution、BatchNorm、ReLU 和 shortcut；没有空间下采样。channel attention 用空间 avg/max pooling 和共享 32→8→32 的 1×1 MLP；spatial attention 用通道 avg/max pooling 和 7×7 convolution。两个 attention 内部均用 sigmoid 门控。代码空间卷积为 7×7，与已读补充材料中的 5×5 描述不一致；应称本地 reproduction 实现。ASPNN 的内部 sigmoid 与这些 attention sigmoid **应保留**，不能因为删除 binary 输出而一并删掉。

global_beta 来自 class-weighted L2 LogisticRegression（C=1），含截距；注册为不可训练 buffer。它不是论文描述的 OLS global coefficients。CAWNN 输出的是可正可负的 **local coefficient multipliers**，不是类别概率、不是归一化权重，也不是直接的标量 predictor。

令 x̃(s)=[x(s),1]，γ 为冻结的 global_beta，wθ(s) 为 CAWNN 输出，则：

    w_code(s) = clamp(wθ(s), -50, 50)
    β_local,j(s) = γ_j w_code,j(s)
    tθ(s) = Σ_j x̃_j(s) β_local,j(s)
    η_code(s) = clamp(tθ(s), -30, 30)

**桥接位置就在 ACAWLR.forward 中的 (beta_local * x_features).sum(dim=1)，紧接着的 clamp 是原 forward 的返回值。** 在所读 core 的约 199–210 行。predict_prob 才调用 torch.sigmoid(model(...))。训练使用 BCEWithLogitsLoss，内部合并 sigmoid 与 BCE，并不先调用 predict_prob。参见 [PyTorch BCEWithLogitsLoss 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)。

最后一个常数特征对应 γ_0 w_0(s)，所以它是**空间变化的截距项**，不是能够直接 profile 的独立全局截距。原 γ 某项若恰为 0，乘法网络无论怎样调整对应 w 都不能恢复该项的效应；沿用 binary γ 也把 pseudo-negative 选择带入新 PPP。

原训练用 BCE pos_weight、Adam weight_decay、梯度裁剪、AP 选 epoch；普通 reproduction 的同一验证折可能既选 epoch 又报告指标。multiscale 的 strict 路径区分 inner epoch selection 与 outer evaluation，不能把两者混称为相同严格验证。core standalone 另有 Cu 阈值样本标签逻辑；wrapper 的矿床 buffer 样本逻辑不同，未来不可直接调用 core.main 代替桥接入口。

~~~mermaid
flowchart LR
  S["目标坐标 s"] --> D["相对 anchors 的有符号各向异性偏移"]
  F["附近断层线段"] --> O["轴向平均角度 φ(s)"]
  O --> D
  A["训练坐标范围构造的 anchor grid"] --> D
  D --> M["ASPNN：MLP + 内部 sigmoid"]
  M --> G["目标 s 的 GASPG"]
  G --> C["卷积 + 残差块 + BatchNorm"]
  C --> AT["channel attention → spatial attention"]
  AT --> W["pool + FC → local multipliers w(s)"]
  X["标准化地学变量 + 常数 1"] --> T["Σ x_j(s) γ_j w_j(s)"]
  B["binary logistic 的冻结 γ"] --> T
  W --> T
  T --> E["clamp → η_code(s)"]
  E --> L["BCEWithLogitsLoss：训练"]
  E --> P["sigmoid：predict_prob"]
~~~

与原论文的方法对照，以本地 references 中已读 Methods 为依据；[原文 DOI](https://doi.org/10.1130/G53947.1) 与 [补充材料 DOI](https://doi.org/10.1130/GEOL.S.30888581) 仅用于定位来源。应区分“保留原方法空间思路”“复用当前 reproduction 类”“逐层精确论文复现”，三者不是同一承诺。

## 2. Current PPP computational graph

证据：[PPP_three_papers_simple.py](PPP_three_papers_simple.py) 的 preprocess、matrix、profiled_objective、fit_points、log_intensity、evaluate、coarsen_quadrature，以及 [PPP_binary_lithology_v3.py](PPP_binary_lithology_v3.py) 的支持域、特征和验证编排。

当前模型：

    η(s) = b + x(s)^T β
    λ(s) = exp(η(s))，单位为 observed catalogue events / km²

事件在真实 event coordinates 评价 feature，不以最近 quadrature 点替代。quadrature 提供域 D 上地学变量和正面积 a_q（km²），不是确认无矿负样本。训练 quadrature 上的面积加权中位数、均值、标准差用于两类位置；binary 列保持其约定编码。事件不能单独拟合另一套 scaler。

    Z(β) = Σ_q a_q exp(x_q^T β)
    b*(β) = log n − log Z(β)
    J(β) = n log Z(β) − Σ_i x_i^T β + α/2 ||β||²

这是忽略与 β 无关常数的 profiled negative PPP log likelihood，加 slope L2；原配置 α=.1。saved log_likelihood 是未加惩罚的点过程 likelihood 值。Σ a_q λ_q=n 是 profile 截距导致的训练质量恒等式，不是独立正确性或矿床总数校准证据。

~~~mermaid
flowchart LR
  Q["训练 quadrature 原始特征 + area"] --> PRE["训练域面积加权 preprocessing"]
  E["真实事件位置原始特征"] --> XE["共享 matrix"]
  PRE --> XE
  PRE --> XQ["共享 matrix"]
  Q --> XQ
  XE --> SE["x_event · β"]
  XQ --> SQ["x_q · β"]
  SQ --> Z["logsumexp(log area + x_q · β)"]
  Z --> I["profile b = log n − log Z"]
  SE --> OBJ["n log Z − Σ event score + L2"]
  Z --> OBJ
  I --> OUT["log intensity = b + x · β"]
~~~

可复用与需改造的边界：

| 组件 | 可复用内容 | 不可直接沿用的部分 |
|---|---|---|
| raw GIS/covariate evaluation | 真实位置取值、明确 units、缺失指标、合格 support | 原缓存不含完整方向支路；不能冒充新模型完整输入 |
| preprocess / matrix | 训练 q 拟合、event/q 同参数、area weighting | binary ACAWLR 的按样本标准化与其不同；需对共同特征统一 schema |
| area/support/guard masks | 经审计的定义与掩码机制 | 已知域/拓扑误差尚未解决；不能靠桥接自动消除 |
| profiled likelihood | logsumexp、全局 additive intercept 的解析消元 | convex global β optimizer、固定 design-matrix 梯度不适用于 nonlinear θ |
| regularization | 不惩罚可 profile 的全局 b | .1 不能照搬为全部 network weights 的合理强度 |
| evaluate / reporting | area AUC、capture、conditional gain 等数学定义 | PointFit、β 字段和 score 调用需 adapter；不能伪装成线性模型 |
| coarsen_quadrature | 求和面积、固定代表点的构造原则 | nonlinear network 可能格内变化更快，旧模型积分精度不能直接继承 |
| spatial validation | split/guard 的既定设计 | fit_points 硬编码调用及模型注册需显式注入新 fit/predict 接口 |

## 3. Exact architectural gap

当前 PPP 的 β 在整个域内固定；没有 ASPNN、逐位置 GASPG、卷积、双重注意力、CAWNN local multipliers、局部方向控制的 learned spatial mapping。共享研究区、部分地学变量、矿床清单、GIS 数据或验证方案，不构成“保留 ACAWLR architecture”。

真正桥接须新增/接通：

1. 从同一地学坐标函数获得 x(s)、φ(s) 与 metric coordinates。
2. 同一个 spatial encoder 对任意 s（event 或 q）返回 local coefficients。
3. 从 local coefficients 形成 raw η，替换最后 binary observation model。
4. 从全部训练积分域计算面积加权 PPP normalizer。
5. 用共享、确定性的网络状态计算 event 和 q 两项。
6. 重做 regularization、intercept gauge、保存格式和 fold 内训练流程。

不能把已有 global log-linear PPP 重命名为 ACAWLR–PPP；不能只在图上添加 ACAWLR 名称，也不能把 geochemical IDW 当作 CAWNN 的替代。新模型与论文的差异必须写入 model identity。

## 4. Proposed ACAWLR–PPP equations

### 4.1 同一个空间函数

固定域几何 D 和 anchor lattice A；用原始断层、地化、重力、岩性信息定义确定性 raw evaluator。所有坐标转到 ESRI:102039，metric_x/y 为 m，方向为 radians，面积为 km²。longitude/latitude 仅用于来源、必要地理查询和制图；重力来源中的 degree 距离如保留，必须是明确独立的特征，不能参与 m 制的 anisotropic distance。

定义：

    Hθ(s) = ASPNNθ(anisotropic_offsets(s, A, φ(s)))
    Wθ(s) = CAWNNθ(Hθ(s))
    gθ(s) = spatial_geological_predictor(x(s), Wθ(s))
    ηθ,b(s) = b + gθ(s)
    λθ,b(s) = exp(b + gθ(s))

event coordinates 和 quadrature coordinates 都进入同一个 evaluator/network；接口不含“是否事件”的标签，不允许按角色换 scaler、anchor、clamp、BN 状态或权重。area 只进入积分和明确声明的训练正则，不作为 event/q 身份提示输入网络。

ASPNN/GASPG 不必对整个研究域一次存成巨大张量。**每个目标位置都有自己的 GASPG**；可以按位置分块计算。可一次固定的是 anchor grid、坐标变换、raw covariates 和几何偏移；ASPNN 参数变化后 learned GASPG 必须重新计算。只有相应 trunk frozen 时才可复用 learned outputs。

新 anchors 用预先声明的研究域几何构造，避免训练 event 极值驱动范围。已知 held-out 地区的无标签坐标作为 anchor 参照不是把 held-out q 放回训练积分；但不能用其事件标签、全域 fitted scaler 或有监督 learned state。正则若对 held-out q 施加额外训练约束，也须视为另一种 transductive 设计，不能悄悄加入。

### 4.2 PPP 与 global intercept 的重新推导

对 n>0 的训练事件和训练域 quadrature：

    ℓ(b,θ) = n b + Σ_i gθ(s_i) − exp(b) Zθ
    Zθ = Σ_q a_q exp(gθ(q))
    J(b,θ) = −ℓ(b,θ) + R(θ)

若 b 为不受约束、无惩罚、独立于 θ 的全局加性参数：

    ∂ℓ/∂b = n − exp(b) Zθ
    ∂²ℓ/∂b² = −exp(b) Zθ < 0
    b*(θ) = log n − log Zθ

代入后：

    ℓ_profile(θ) = n log n − n log Zθ + Σ_i gθ(s_i) − n
    J_profile(θ) = n log Zθ − Σ_i gθ(s_i) + R(θ)   [省略常数]
    ∇J = n Σ_q p_q ∇gθ(q) − Σ_i ∇gθ(s_i) + ∇R
    p_q = exp(log a_q + gθ(q) − log Zθ)

因此 nonlinear spatial network **并不自动破坏 profiling**；决定因素是是否仍有独立 global additive b。θ 的优化不再是当前线性 PPP 的凸问题。n=0 时有限 b* 不存在；不得用 epsilon 假装正常事件折。

若只有原结构 η(s)=c w_0(s)+Σ_j β_j(s)x_j(s)，试图把 c 当 global intercept，则：

    ∂ℓ/∂c = Σ_i w_0(s_i) − Σ_q a_q w_0(q) exp(η(q))

它一般不能化为 n−exp(c)Z，**不能使用 log n−log Z 的旧公式**。若另加 b，并把 c w_0(s) 纳入 g，b 又能 profile，但出现 b 与 g 常数分量的 gauge，并增加不受地学变量约束的 spatial intercept。推荐 P1 不设自由 local intercept。

保存真实 unpenalized ℓ、penalty、profile objective、b、logZ、n、training area，避免混用。测试时沿用训练所得 b；不能拿 test n 重新 profile 来“验证”预测 mass。conditional density/gain 使用测试域积分作条件归一化是另一项明确命名的指标，不等同于重新拟合 raw intensity。

### 4.3 BatchNorm 与一致性

原 CAWNN 在 train mode 下按当前 batch 统计 BN；分开 event/q、不同 chunk size，甚至把两者拼接后重分块，都会改变输出函数。因此“共享同一个 model 对象”仍不够。PyTorch 明确区分训练 batch statistics 与评估 running statistics，关闭 track_running_stats 也不会解决问题：[BatchNorm2d 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)。

P0 使用固定网络与固定 eval 状态。P1 默认删除 BN；如未来改用每个样本内部的 GroupNorm，应作为显式架构版本并用于匹配 binary control，不能混用。其统计在训练和评估均按输入的样本内分组计算：[GroupNorm 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GroupNorm.html)。禁止从全44点拟合的网络获取 BN running stats，再声称 fold 内无泄漏。

### 4.4 稳定性与 identifiability

- 计算 logZ=logsumexp(log a+g)，以 log intensity 为主要输出；不直接累计巨大 exp。
- profile 模式下 quadrature mass 可用 n·softmax(log a+g) 得到；求 log likelihood 无需对 event η 取 exp。
- 不沿用最终 [-30,30] predictor clamp：它改变似然函数、制造零梯度平台，还使 exp(b+g) 的 profile 推导与实际 clamp 不符。应正则化、限制空间偏差幅度，并对非有限值显式失败，不能静默跳过 loss。
- 数值累加优先 float64，分块 logsumexp 合并；面积为正且单位固定。log intensity 的数值随强度单位变换，必须记录 reference unit。
- 若 g 可整体加常数、b 可减同常数，模型不唯一。通过独立 b、禁用自由 local intercept、P1 的参考点约束减少这一自由度；神经参数的排列/缩放对称性仍存在，不能声称所有参数可识别。
- 原 γ×w 分解若两者都训练会产生额外尺度混淆，若冻结 binary γ 又继承负样本信息。P1 改为 global coefficient + shrinkable spatial deviation，不复用 binary γ。

## 5. P0 minimal prototype

**目的只有数学/接口验收，不作真实数据模型或性能结论。** 下一阶段先新增极小 synthetic prototype，不能借此启动 44/16 训练。

保留原 signed anisotropy、ASPNN、GASPG、CAWNN、local weighting 路径；使用很小的 synthetic anchor lattice（例如 4×3）、少量地学维度、非等面积 quadrature、真实独立的 synthetic event coordinates。网络权重预先固定，BN eval，global coefficient vector 也预设；不是用44点训练完再 frozen。

输出原未截断 latent t_fixed(s)，定义：

    gτ(s) = τ [t_fixed(s) − t_fixed(s_ref)]
    η(s) = b + gτ(s)

仅 τ 是可选的低维验证参数，b 可解析 profile。fixture 必须产生可观察的空间变化；若空间支路输出常数，验收失败，不能退回 log-linear 测试假装通过。固定/随机权重不是有科学依据的地质模型，P0 没有研究模型身份。

未来 P0 tests 只新增桥接缺口，不重跑 Stage 1B 已证明的基础指标：

1. 同一坐标分别作为 event/q、单点/分块/重排序输入，得到相同 raw predictor。
2. raw predictor 与原 forward 的未截断乘积和一致，内部 sigmoid 保留，最终 binary sigmoid/BCE 不在 PPP 路径。
3. network 参数/τ 的有限差分梯度与 profiled 公式一致；验证的是新增 nonlinear 路径。
4. profile b 与固定 θ 下直接一维解一致；单独设置 spatial intercept multiplier 的反例，拒绝错误 profiling。
5. 全量与两遍 streaming gradient 一致；非等面积及稳定大 logit fixture。
6. 固定 x 改变位置/方向时空间支路确实有作用；不是检查 AUC。

P0 不需要修复所有旧入口；应导入明确绑定的纯模型组件，避免执行 core.main。先用静态签名核对和小型构造测试确认所选 core 兼容。

## 6. P1 research model

### 6.1 推荐 reduced architecture

P1 保留主要机制，但限制不同地学系数场的自由度为 rank 1：

    anisotropic offsets
      → ASPNN(2→16→8→4→1, internal sigmoid)
      → GASPG
      → Conv(1→4, 3×3)
      → one residual block(4→4, two 3×3 conv, no BN)
      → channel attention(4→1→4)
      → spatial attention(7×7)
      → global average pooling
      → bias-free Linear(4→1) = rθ(s)

    hθ(s) = tanh(rθ(s)) − tanh(rθ(s_ref))
    β_j(s) = β_j + u_j hθ(s)
    gθ(s) = Σ_j β_j(s) x_j(s)
    η(s) = b + gθ(s)

s_ref 从预先固定的域几何选择，不由事件分布或16点表现选择。h(s_ref)=0，使 β 是该参考位置的 coefficient，而非含任意常数偏移的 latent factor。h∈[-2,2]，但 u 仍须收缩。必要的 reference branch 梯度不能 detach 掉。

这里 CAWNN 输出一个共享的 spatial weight latent，u 把它映射成多项 local coefficient deviations；CAWNN **不是直接输出 intensity**。相比原每项独立 multiplier，rank1 强制系数场共享一种空间模式。保留的地学作用由 local coefficients 进入最后 linear predictor，避免用独立 spatial intercept 吞掉全部地学解释。

原各向异性为 axial orientation，却用 signed MLP；研究 P1 应显式采用 [f(d)+f(−d)]/2 作为 proximity 以保证 φ+π 不变。它保留 ASPNN 的学习能力，但属于与当前 reproduction 的明确差异；匹配的 binary P1 control 也用同一修改。原实现结果单列历史参考。

| 机制 | P0 | P1 |
|---|---|---|
| anisotropy | 原方向/比例路径，synthetic 固定 | 地质方向固定、比例预声明，轴向对称 |
| GASPG | 每位置生成，极小 anchor grid | 每位置生成，固定域 anchors，分块评价 |
| CAWNN | 原组件固定，验证接口 | 缩窄卷积/残差，保留双 attention |
| spatial weights | fixed 非常数 latent，τ 验证 | rank1 local coefficient field，u 可收缩 |
| binary γ / pseudo-negative | 不使用拟合的 binary γ 或 labels | 不使用；β 与 u 在 PPP 内学习 |
| local intercept | P0 latent 内可存在，另设 b 明确验证 | 不设自由 local intercept，单独 global b |

P1 的空间分支有能力在相同 x 下产生不同 predictor，故架构不是当前 log-linear PPP。u=0 是其嵌套的无空间效应情形；若数据支持收缩到零，应如实接受，不能为名称强迫空间效应非零。

### 6.2 参数规模、regularization 与解释

原 p=10、K=p+1=11 时的参数静态计数：

    ASPNN 225 + head 320 + residual blocks (3192+3696+14560)
    + CBAM 610 + FC 33K = 22,966

不含冻结 γ 和 BN running buffers。这是依据层形状的算术计数，未实例化训练或测量运行资源。

上述 P1 的 trunk 约 671 参数：ASPNN225、head40、residual296、attention106、readout4；另有 2p 个 β/u，p=10 时约691，b profile 消元。方向和比例不新增可训练参数。**691 仍明显多于44个事件，不能把缩小网络等同于已有统计可行性。**

建议明确的 regularization：

    R = λβ/2 ||β||² + λu/2 ||u||² + λθ/2 ||θ||²
        + λs/(2 A_train) Σ_{q∈D_train} a_q Σ_j ||L ∇_s [u_j hθ(q)]||²

λ 非负，L 是声明的长度尺度，使 smoothness units 可解释；b 不惩罚。最后一项可采用明确的邻接 finite-difference 版本，但不能跨越留出域/guard偷偷加训练约束。它是可选的额外实现复杂度：首个 P1 先使用前3项，只有最小验证显示必要且预算允许才加入 smoothness。不得在本阶段指定“最优 λ”，不得沿用 .1 作为所有参数的默认科学依据。记录 loss 是总和还是按 n 平均，相应正则尺度必须一致。

冻结地质方向、anisotropy ratios、anchors、CRS、feature schema。优先强收缩 u；不学习额外深层、多尺度搜索或自由 spatial intercept。不把完整44点训练得到的 neural trunk 用于留出验证。若存在真正独立、无矿床标签的预训练来源，才可另行讨论冻结 trunk；目前未确认这样的来源，冻结随机网络只能用于 P0，不能自动成为可靠研究模型。

β/u/h 的缩放与神经对称性仍使内部参数不唯一；报告 β_j(s)、β_j(s)x_j(s)、g(s) 与其跨fold稳定性，而非把 attention 大小当作因果重要性。相关地化项使系数符号也不能直接作因果结论。保存参考点、标准化单位、原始量纲转换及方向定义，支持复算。

## 7. Training strategy

### 7.1 首选：完整训练域积分，分块累积

44 events 可每步全部使用；quadrature 只分块计算，不删去任何训练积分格。所有 preprocessing 只在训练 q 拟合，支持域/guard 与事件掩码各自明确。训练 fold 不能加载 full44 的 fitted scaler、β、BN stats 或 selected epochs。

精确 profiled gradient 的内存受控方案：

1. θ 不变、网络确定，第一遍 no_grad 遍历所有训练 q，以分块 logsumexp 得到精确离散 logZ。
2. 第二遍重新计算每块 g，用 detached p_q=exp(log a_q+g_q−logZ) 累积 n Σ p_q ∇g_q。
3. 累积 −Σ_i ∇g_i 和一次 ∇R，全部结束后才 optimizer.step。
4. reference-point 支路、可训练空间参数等参与链式求导；状态不能在两遍之间更新。
5. 若以 n Σ stop_gradient(p_q)g_q 作梯度 surrogate，日志仍用真实 J_profile，不能把 surrogate 数值当 likelihood。仅将 logZ detach 后对 n logZ 求导会漏掉全部积分梯度。

若增加依赖全域 moments 的 normalization/regularizer，必须再推导其梯度；本 P1 选固定 reference point，避免默认忽略全域 centering 导数。无 dropout，BN 不更新。分块时加 penalty 一次，不能每块重复正则。

### 7.2 quadrature minibatching：允许，但必须区分两个目标

对于 **未 profile** 的 J(b,θ)，从所有训练 q 按 π_q>0 有放回抽 m 次：

    I_hat = (1/m) Σ_{k=1}^m [a_{q_k}/π_{q_k}] exp(b+gθ(q_k))

这是离散积分的无偏估计；使用固定或停止梯度的 proposal 并正确处理 sampling，梯度也可无偏。无放回抽样须用 inclusion probability 的 Horvitz–Thompson 权重，不能照搬 draw probability。event 若均匀抽 k 个，事件总和估计为 n/k 倍样本和；44点无需这样做。所有抽样只在训练域，不能遗漏低分地区。

**把 Z_hat 填进 profiled log Z 不是同一回事。** E[log Z_hat]≠log Z，ratio gradient 与 log n−log Z_hat 通常有偏；不能每个 minibatch 重新归一化后称为精确 PPP。大样本/合适抽样条件下可一致近似，但必须另做误差研究。P0/P1 首选完整离散积分的两遍 streaming，不把这一近似加入首个模型。

### 7.3 计算量与保存

120×100 anchors 对约33,000 q 意味着约4亿个 location-anchor pairs。原 first conv 的32通道、float32 activation 约1.46 MiB/目标，batch32仅该层约46.9 MiB；全 q 约47 GiB，尚未计反传和其他层。这是量纲估计，不是硬件 profiling。P1 缩窄通道仍须 streaming；完整网络一次放进显存不可作为默认策略。

保存 model state、architecture version、source/core hash、feature顺序、preprocessor、CRS/units、raw GIS signatures、方向定义、anchor geometry、reference point、domain/fold masks、q coordinates/areas/hash、regularization、训练停止规则、b/logZ。新 cache 命名空间不可覆写旧 PPP 或历史 reproduction 缓存。原 audit 已记录的 cache provenance 保证不自动覆盖新增方向场和 neural state。

## 8. Validation strategy

本节是未来实验设计，不在 Stage 1C 执行。

### 8.1 A/B/C 与归因边界

| 比较对象 | 明确定义 | 科学问题 |
|---|---|---|
| A_historical | 原 binary ACAWLR reproduction，原 feature/label/architecture 差异完整记录 | 与历史方案的描述性关系，不能作纯 loss 因果归因 |
| A_matched | 与 C 相同 P1 spatial architecture、raw geology、preprocessing、anchor、初始化规则；最终 binary likelihood，训练域内 pseudo-background | 为 A vs C 提供架构匹配对照 |
| B_current | 当前 log-linear PPP，已审计 formulation | 实际当前基线 |
| B_matched（若与 B_current 信息不同） | 与 C 相同共同地学信息的 global log-linear PPP | 排除 feature 增补造成的改善 |
| C | 本文 reduced ACAWLR–PPP | 检验 PPP 下空间结构的额外价值 |

A_matched vs C 回答：在同一空间架构下，**PPP observation model 与 pseudo-negative removal 的联合影响**。不是只隔离一个 loss 名称。为避免再混入 positive-support 改变，匹配 A 的 positives 应为同一44个真实坐标；若保留历史 deposit buffers，则报告还包含标签支持范围改变，不称纯 observation-model 比较。binary pseudo-background 只从训练域生成，其抽样密度、权重、seed 预声明；只用于 A。

B_matched vs C 回答：在 PPP observation model 下，anisotropic spatially varying architecture 是否增加泛化价值。若 C 多拿到了 fault angle/coords，而 B 没有对应信息，这同时是信息与架构的改变；不能单独归因 CAWNN。严格比较应给 B 同一已声明的方向信息（例如 sin2φ/cos2φ/confidence 的固定编码），或明确报告这个无法完全隔离的差异。B_current 同时保留为真实现有基线，不能悄悄修改后仍叫原模型。

共同地学 schema 应先声明：是否使用当前 M2 的岩性接触项、是否包含额外密度/重力距离，各模型尽量一致。原 default reproduction 不含 lithology，不能用其旧 AUC 与新 C 直接证明 likelihood 改善。

### 8.2 空间验证与16点

沿用预声明的 repeated 200 km blocks、50 km guard、seeds42/43/44 和相同 fold masks；本阶段不执行。训练事件、训练积分域、guard exclusions 分开保存。被留出的 q 不进入训练 normalizer、scaler、网络正则或监督选择。若使用 inner selection，必须在 outer train 内重新建立上述边界；普通 reproduction 的同折 early stopping 不够。

44个事件在空间训练折中更少，已有 folds 约18–31事件；复杂 inner tuning 非常不稳定。优先预声明单一缩减架构及有限训练规则，不能大规模搜索后称“无调参”。未形成充分证据前，不升级完整网络。

compact validation 若依据全部44点坐标构区，应继续标为 event-informed geography 的压力测试，不作为完全独立的泛化证据。16个历史验证点已用于方案比较，只作 exploratory 描述，不能用于选 α、网络宽度、epoch、support或图件版本，也不能恢复为 untouched external test。

共同主指标用 area-weighted presence-background AUC、top5/10% area capture；背景不是真负样本。B/C 可比较统一单位的 conditional log gain 与 PPP likelihood；不要直接比较 BCE 数值和 PPP likelihood 数值。对 A 若另作 score-to-intensity 转换，必须明确额外映射及归一化，不把 sigmoid 当 point-process intensity。模型 intensity、percentile、target area 均不等于 absolute mineral probability 或未知矿床数量。

未来 stability 检查应关注 local coefficient fields、target overlap、event influence；其目的是回答“少数点是否支配空间场”，不是为更高 AUC 调参。本阶段不新增这些实验。

## 9. Risks

| 等级/类型 | 风险与后果 | 设计响应及仍未解决部分 |
|---|---|---|
| CRITICAL technical | 当前默认 wrapper/core Config 不兼容 | 实现前锁定入口/API/hash；本阶段只记录，不修代码 |
| CRITICAL technical | event/q 因 train BN、chunking、不同 scaler 成为不同函数 | P0 role/chunk invariance；P1 无BN，共享唯一 evaluator |
| CRITICAL technical | local intercept 被错误 profile | 独立 additive b；明确上述反例及解析条件 |
| CRITICAL technical | guard/test q 在积分、preprocessing或正则中回流 | 同一 fold domain contract，保存并检查所有 masks |
| CRITICAL technical | 使用旧 binary γ 或 full44 neural cache | 从新 PPP 参数化开始；fold 内来源绑定，不隐含有监督 warm start |
| STATISTICAL | 观测矿床位置同时反映真实地质与发现/采样过程 | λ 是观察目录强度；网络不能消除 preferential sampling，effort模型仍未识别 |
| STATISTICAL | 灵活 intensity 在少量事件附近尖峰，积分格漏检 | shrinkage、有限空间尺度、后续积分精度检查；高训练ℓ不是成功 |
| STATISTICAL | φ轴向不变性、原 clamp、signed proximity 的物理含义不足 | P1对称化、删除最终clamp，保持与binary control一致 |
| STATISTICAL | 参数/系数场不唯一、相关covariates | reference gauge、rank1与收缩；只解释可复算输出场，不保证因果识别 |
| SMALL-SAMPLE | 原约23k参数，44个正事件且空间有效样本更少 | 不建议直接全网络训练作为主结果；P1约691仍需谨慎 |
| SMALL-SAMPLE | 多个fold/seed不是独立矿床样本扩增 | 不以 q数量、repeated folds或训练 epochs论证样本充足 |
| SMALL-SAMPLE | 局部方向/attention可能只记住事件 | 最小架构、预声明规则、未来event influence；目前无稳定性结论 |
| COMPUTATIONAL | 逐q生成120×100 GASPG及反传过重 | 小P0、窄P1、分块两遍；尚未实测耗时 |
| COMPUTATIONAL | profile小批量归一化偏差、exp overflow | 完整离散logsumexp；若抽样只用明确无偏unprofiled estimator |
| DATA/GIS | 旧support边界、重力缺失、contact topology风险延续 | 沿用审计限制；新模型复杂度不能作为修复证据 |

**44点是否足够？** 当前证据不支持“足以稳定训练完整 ACAWLR neural weighting”这一断言。参数数目本身不是不可训练的数学证明，但结合空间相关、发现偏差、折内更少事件与 flexible intensity spike 风险，完整网络作为主要科学结论载体明显过重。P1 仅是保持核心思想的更小研究候选；若其空间权重在留出条件下不稳定，应接受当前 log-linear PPP 更有依据，而不是进一步增宽网络。没有新增实验前，不宣称 P1 解决了小样本问题。

## 10. Files/functions that would need modification

以下全是未来变更清单，本阶段不实施。

| 位置 | 最小必要变化 | 边界 |
|---|---|---|
| 新 acawlr_ppp_bridge.py（建议名） | 独立 deterministic spatial predictor、P0/P1模块、raw log-shape接口、global b/profile loss、streaming gradient | 不在旧 PPP 内冒名替换 PointFit |
| 新 test_acawlr_ppp_bridge.py | §5 中少量synthetic tests，含role/chunk一致性和profile反例 | 不运行44/16或CV |
| scripts/ACAWLR_improved.py 的模型类 | 如抽取复用，显式暴露 unclipped latent 和 normalization选择；绑定源码来源 | 不调用 standalone main，不覆盖历史实现；优先新adapter而非全局改行为 |
| wrappers 的 load_core / make_core_config | 显式匹配被选 core Config 和完整接口，取消不透明身份切换 | 入口缺陷应单独小修复，保留历史core说明 |
| compute_fault_features / generate_anisotropic_grid | 可复用公式，新增固定CRS/轴向定义的纯geometry接口 | 不直接使用旧labels-dependent cache或自动CRS覆盖 |
| PPP_three_papers_simple.py 的 preprocess / matrix | 复用训练q统计与schema，adapter接神经score | 保持当前线性模型结果可复算 |
| fit_points / log_intensity / evaluate 周边 | 新模型实现自己的fit/score；metrics接受统一scores | 不把nonlinear θ伪装成β向量；不先改全套生产流水线 |
| PPP_binary_lithology_v3.py spatial_validation 编排 | 未来显式注入fit/predict adapter，复用已审计masks | 不改全局MODELS状态凑接口，本阶段不接CV |
| cache/artifact schema | 增加方向、anchors、state hash、fold domain和source provenance | 独立namespace，不删改历史结果 |
| build_paper_figures.py（后续才需要） | 显式run路径、模型身份、intensity/percentile名称及area weighting | 设计阶段不生成新性能图 |

旧入口修复、P0、P1接生产数据是不同小步骤，不能把这张清单理解为立即批准一次性重构。

## 11. Recommended next implementation step

**下一步仅建议一个 bounded synthetic P0 sprint，不直接实现研究模型。**

交付：显式绑定一个兼容的纯模型接口；新增最小 prototype 与 §5 的新测试；在人工坐标、人工covariates、非等面积小quadrature上，验证共同空间函数、nonlinear梯度、全局截距profile与streaming一致性。必要的旧Config接口修复应最小化并单独说明，不运行旧训练入口。

验收门槛：保留空间路径且输出确实随位置变化；event/q互换角色及chunk大小不改变函数；profile解析式仅在正确条件下使用；小规模直接计算与streaming的objective/gradient一致；没有历史数据或16点参与选择。若未通过，先解决接口/数学问题，不进入P1。

P0通过后，另行决定是否授权P1及其数据/验证预算。本文没有证明完整网络或缩减网络的实证优越性，没有授权完整训练或spatial CV。本Stage 1C只提交此设计文档和必要的精确gitignore allowlist；提交后停止，不自动开始实现。
