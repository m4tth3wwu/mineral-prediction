# ACAWLR–PPP P1-A synthetic notes

- 基线：codex-refactor / f4fbb2c6571c941065b5345e8ec1d29a98fcdb85；开始时工作区干净。仅扩展 synthetic bridge，未读取真实数据，未运行44/16、spatial CV或旧训练入口。
- 实现：4×3 anchors，3个人工事件、6个非等面积积分点、2个人工坐标协变量；轴向对称 ASPNN → GASPG → 4通道无BN残差卷积 → 双attention → 无bias标量readout。h=tanh(r(s))−tanh(r(ref))，β(s)=β+u·h；主干671参数，总计675。参考支路参与梯度；无自由local intercept、binary global_beta、负标签、BCE或最终sigmoid。
- 训练：共享g(s)，独立b解析profile；sum尺度PPP目标加显式β/u/θ三组L2，完整域全量或两遍streaming梯度，正则每步仅加一次。40步Adam为固定synthetic测试预算；日志含真实profile objective、penalty、unpenalized log likelihood、最终权重对应的b/logZ/n/area。λ=(0.02,0.2,0.001)仅为测试设置，未作科学选优。
- 验证：`python -B -m unittest discover -s mineral_prediction -p 'test_acawlr_ppp*.py' -v`，14/14通过（P0 7项、P1-A 7项，2.622s）。检查角色/chunk/reorder/train-eval一致性、轴向对称、reference/rank-1/u=0约束、固定covariates下空间变化；每个参数张量选一个非零似然梯度与有限差分比较；全量/streaming含正则梯度和40步更新一致、目标降低、各参数组更新、最终截距独立一维求解、finite及非法输入拒绝。
- 发现并修复：随机初始化可使单单元channel-attention ReLU关闭；其第一层改为固定正初始化0.25，不依据事件选择。修复后两个attention权重张量均有非零似然梯度。旧Config继续由P0纯类adapter绕开，未修改旧pipeline。
- 结论：P1-A synthetic可训练/数学接口验收通过，无当前测试阻断。尚未验证真值恢复、多seed稳定性、积分收敛、真实GIS或44事件统计可行性；本阶段停止，不接入真实数据或继续后续阶段。
