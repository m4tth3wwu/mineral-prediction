# 当前交接：P1-G 完成；未复现 collapse，数值保真度未解决

本轮基线 codex-refactor / 28b0bd3c14e9b3ff3b92fccbb0ba1e92de4a0bf7。P1-C/D/D-LR/E/F全冻结，305文件哈希未变。
本轮完整代码、结果与交接记录随本提交发布至既有origin/codex-refactor；实际远端状态以Git核对为准。用户已在P1-G任务中明确授权普通push，不改main。

五层对象分别为p、g、b、beta/u/h、theta。不得将参数收缩、未collapse或更低经验目标等同于恢复成功。

## 实际执行

run: mineral_prediction/p1g_empirical/20260922T054836Z_p1g_v1
session: docs/project_memory/sessions/20260922T054836Z_p1g.md
仅新增empirical_P0/empirical_Plambda，严格恢复P1-D/LR data23/init1011；相同teacher、Adam .003、float64/CPU1线程、48²训练/96²监测/192²终点。各2000更新，共4000，2001行轨迹/81次监测；模型与Adam checkpoints为595/1238份。未重跑历史、未加seed/dose、未延长预算、未进入真实数据。
20项训练前gate通过；训练exit0；32项独立验证通过、4项无训练解释测试通过。初次代码写入审核超时未执行，按允许重试一次成功；无训练失败/重试。原始日志与失败的科学门槛完整保存。

## 2×2固定终点（192²）

| objective / λ | KL | q/teacher | field/teacher | collapse | stationarity |
|---|---:|---:|---:|---|---|
| population / 0 | 1.33455722e-5 | 1.147552 | .579335 | 未观察到 | 未通过 |
| population / 原λ | 5.35918125e-5 | 1.061500 | .538393 | 未观察到 | 未通过 |
| empirical / 0 | .169594595 | 7.156446 | 3.983157 | 未观察到 | 未通过 |
| empirical / 原λ | .088306149 | 3.539910 | 2.329680 | 未观察到 | 未通过 |

## 已确认与解释限制

empirical原λ相对λ0使192² KL下降47.93%、q下降50.54%、field下降41.51%，方向是压低过大支路幅度，未达到P1-F相对teacher的shrink/collapse标准。KL的empirical效应=-.081288447，population效应=+4.024624e-5，interaction=-.081328693；只有一个固定dataset/init，不是统计显著性或最优解结论。
原λ并未全面改善优化：gradient峰值2320.57→2978.12，单步total绝对跳变>1次数273→353。两条empirical均未恢复teacher、未stationary。没有预声明objective conflict（步级48²、监测96²和持续条件均未触发）；相反经验目标下降而teacher KL/g误差上升的监测区间为46/36个。
两条empirical的96→192 logZ误差分别.000602173/.000112211，都大于1e-4。KL/TV一致性条件通过但不豁免logZ失败。必须保留numerically_unresolved；192²数值和interaction只能作当前离散评估。没有放宽阈值/重训。48→192经验目标修正约+.844975/+.164026。
当前更支持经验目标拟合与teacher恢复脱钩，伴随优化振荡和积分误差；没有证明其各自因果贡献。原λ在这个终点减轻错误幅度扩张，不支持正则稳定诱导collapse或结构性不可辨识性强机制。只符合CaseC“未复现collapse”部分，不能说两条恢复正常。历史24²/.01 collapse仍未解释。

## 唯一下一步建议（没有执行）

固定两份step2000模型，不训练、不改参数，只做192²/384²独立积分一致性检查，沿用容差。先解决终点评价可信度，不自动测试lr=.01、24²、额外seed或新模型。

## 阅读顺序

本轮session → PROTOCOL/PROVENANCE/GATES → NOTES/SUMMARY/OBJECTIVE_REGULARIZATION_INTERACTION/NUMERICAL_FIDELITY → TRAJECTORY_DYNAMICS/raw/states → FINAL_VERIFY。训练source manifest在SOURCE_MANIFEST，报告/验证/测试的最终manifest在DELIVERY_SOURCE_MANIFEST；P1-F与旧文件不改。
