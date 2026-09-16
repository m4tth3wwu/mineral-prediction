# ACAWLR–PPP P0 notes

- 范围：仅 synthetic P0。新增两个 Python 文件及本 notes；未修改旧 pipeline，未读取真实数据、运行44/16、spatial CV或P1。
- 实现：4×3 anchors、3个人工事件、6个非等面积 quadrature；固定各向异性偏移 → 原 ASPNN → GASPG → 原 CAWNN（含双 attention）→ local spatial weighting。网络权重为预设常数，BN锁定eval。统一 g(s) 同时用于两项，独立 b=log(n)−logZ，objective=n·logZ−Σevent g；无负样本、BCE、最终sigmoid或旧binary global_beta。
- 适配：只从指定 core 的 AST 加载 ASPNN、ResBlock、CBAM、CAWNN 四个纯类，不执行其顶层依赖、Config或训练入口。未修复旧Config不兼容问题。
- 验证：在 mineral_prediction 下运行 `python -B -m unittest -v test_acawlr_ppp_bridge`，7/7通过（1.297 s）。覆盖角色一致性、chunk/reorder一致性、位置变化、独立一维截距求解、有限差分、全量/两遍streaming一致性及极大正负gain下finite loss/gradient。
- 梯度边界：检查标量gain及人工event/q坐标；坐标梯度经过固定的非线性ASPNN/CAWNN。未训练权重，未验证所有神经参数的优化。面积固定；streaming返回真实objective及一阶诊断梯度，不使用小批量profile近似。
- P1条件：P0数学与接口门槛通过，可以另行评审/授权P1；不代表已证明P1统计可行性、真实GIS输入正确性或44事件下的稳定性，本次不进入P1。
- 阻断：本P0未发现失败项；旧Config入口兼容性仍未解决，由纯类adapter绕开。固定人工权重不是已学习的地质模型，不能报告真实预测性能。
