# P1-F 中文报告

1. P0 达到预定密度恢复门槛的初态：standard、local；standard 与 local 分开报告，local 的保持成功不等于从远处恢复。
2. Pλ 在末段满足预声明稳定收缩定义的初态：无。
3. 观察到持续功能层面 collapse 的轨迹：无；至少一次监测到 collapse 的轨迹：无。
4. 当前证据判别：regularization_changes_recovery_without_stable_field_shrinkage。这是固定teacher/optimizer/初态下的匹配结果，不自动解释历史经验样本轨迹。

## 实验与数值边界

本轮run：20260917T121240Z_p1f_v1。标准初态为正式归档init1011；local为seed20260918、相对范数0.001的teacher附近扰动。每条固定2000步、Adam lr=.003、float64/CPU1线程、48²训练/96²监测/192²终点评价；唯一配对差异为原显式L2的整体倍数。实际4条、8000次更新，无有限样本训练、真实数据或新方法。
旧23/53/71是数据seed，在population目标下没有抽样作用，未当作三个独立初始化。P1-E旧gate与BLOCKED原样保留；本轮按真正依赖的teacher/controls/新endpoint检查fresh gate，并未放宽1e-4的logZ误差门槛。

| 初态/条件 | KL192 | TV192 | centered g RMSE(面积/teacher) | b RMSE | beta误差 | q/teacher | 密度/预测通过 | 持续collapse |
|---|---:|---:|---:|---:|---:|---:|---|---|
| standard_P0 | 1.3345572e-05 | 0.001921125 | 0.00527087/0.00516556 | 0.312424 | 0.0760511 | 1.14755 | True/True | False |
| standard_Plambda | 5.3591812e-05 | 0.003262326 | 0.0103932/0.0103591 | 0.302934 | 0.0655522 | 1.0615 | True/True | False |
| local_P0 | 2.6627589e-11 | 2.518798e-06 | 8.5883e-06/7.29783e-06 | 1.51119e-05 | 7.8787e-06 | 0.999966 | True/True | False |
| local_Plambda | 4.3569661e-05 | 0.003220352 | 0.00934653/0.00933267 | 0.0455853 | 0.0680126 | 1.06201 | True/True | False |

## 终点数值保真与stationarity

| 条件 | ΔlogZ96→192 | ΔKL96→192 | fidelity | grad/256末段最大 | total范围/256 | stationarity |
|---|---:|---:|---|---:|---:|---|
| standard_P0 | 9.076112e-06 | 3.39866e-08 | True | 0.0497115 | 8.21379e-05 | False |
| standard_Plambda | 7.055283e-06 | 1.782965e-07 | True | 0.00472623 | 6.0116e-06 | False |
| local_P0 | 9.221153e-06 | 1.598721e-14 | True | 4.08185e-07 | 2.14602e-11 | True |
| local_Plambda | 9.067264e-06 | 2.203856e-08 | True | 0.00178062 | 0.000438809 | False |

固定2000步不早停。恢复门槛与stationarity分别报告；后者未通过时不宣称已经求得人口目标最优解。

## 目标与参数惩罚

| 条件 | beta penalty（原recipe） | u penalty | theta penalty | Fλ−Fλ(teacher),192² |
|---|---:|---:|---:|---:|
| standard_P0 | 0.0008541302 | 0.16915251 | 0.03554635 | 0.0034164665 |
| standard_Plambda | 0.00098161374 | 0.060248786 | 0.024116873 | -4.4127451 |
| local_P0 | 0.0013000438 | 4.4991535 | 0.010558223 | 6.8166628e-09 |
| local_Plambda | 0.0010156516 | 0.22813361 | 0.010757333 | -4.2607514 |

上表惩罚分量使用原recipe作为共同评价尺度；P0训练实际惩罚为0，effective分量另存在逐步日志。正惩罚排除exact teacher为stationary点，并不排除通过改变参数表示仍达到小KL的近似密度恢复。不能把u的收缩直接解释成q或field塌缩。
同一P0目标下，local终点比standard终点的dense目标高-0.0034164597（负数表示local更低）。若一条轨迹高于另一已知可行点，说明该轨迹未达到已知更好目标，不能把其偏差全解释为目标最优点的性质。
同一Plambda目标下，local终点比standard终点的dense目标高0.15199365（负数表示local更低）。若一条轨迹高于另一已知可行点，说明该轨迹未达到已知更好目标，不能把其偏差全解释为目标最优点的性质。

## 轨迹是否把收缩与目标下降联系起来

以下仅描述固定轨迹的首次密度门槛交叉及预定checkpoint区间，未据此选择checkpoint或启动额外搜索。
- standard_P0：首次达到48²密度门槛在step771；此后到终点Δtotal=-0.02216514，ΔKL=-8.658257e-05，ΔqRMS=-0.02137021，Δpenalty=0。 相邻步同时total下降/KL上升/q下降的次数=0。
- standard_Plambda：首次达到48²密度门槛在step862；此后到终点Δtotal=-0.0176431，ΔKL=-4.709824e-05，ΔqRMS=-0.009986602，Δpenalty=-0.005585946。 相邻步同时total下降/KL上升/q下降的次数=37。
- local_P0：首次达到48²密度门槛在step0；此后到终点Δtotal=-5.495383e-06，ΔKL=-2.146634e-08，ΔqRMS=5.523052e-05，Δpenalty=0。 相邻步同时total下降/KL上升/q下降的次数=0。
- local_Plambda：首次达到48²密度门槛在step0；此后到终点Δtotal=-4.260609，ΔKL=4.343228e-05，ΔqRMS=0.009362539，Δpenalty=-4.271728。 相邻步同时total下降/KL上升/q下降的次数=524。

## 第一阶段判别与剂量实验

- standard：Case C；Pλ−P0的KL差=4.024624e-05；q相对teacher由1.14755变为1.0615。
- local：Case C；Pλ−P0的KL差=4.3569634e-05；q相对teacher由0.999966变为1.06201。
预声明dose条件：{'eligible': False, 'qualifying_pairs': [], 'all_main_complete_and_fidelity': True, 'reason': 'predeclared paired criterion not met; no dose search'}。没有执行任何剂量训练。

## 五层恢复与 local 初态的解释

p/g的门槛分别报告；b-field误差和beta误差不作为密度通过的替代。u/h的变化需结合q与field norm，不能因u减小就称collapse；theta距离只作内部表示描述，不能据此判科学失败。低KL也不能建立连续域field唯一性。
local_P0：初始KL192=2.15046558e-08；最终KL192=2.6627589e-11；局部probe={'initial_gap_resolved': True, 'interpretation': 'local_recovery_probe', 'excess_gap_reduction90': True}。固定小扰动可能已满足绝对恢复门槛，必须检查excess是否降低，不把保持成功包装为冷启动恢复。
local_Plambda：初始KL192=2.15046558e-08；最终KL192=4.35696606e-05；局部probe={'initial_gap_resolved': True, 'interpretation': 'local_recovery_probe', 'excess_gap_reduction90': False}。固定小扰动可能已满足绝对恢复门槛，必须检查excess是否降低，不把保持成功包装为冷启动恢复。

## 排除与未排除

本轮训练目标没有finite-sample噪声，因此实际观察到的population现象无需有限样本噪声即可发生。teacher完整state属于模型类，copy/梯度检查通过；使用固定同初态的匹配比较可以隔离“加入原正则”在这套离散优化程序中的因果影响。通过96/192只说明预定节点与容差内的数值可信度，不能证明连续积分完全精确。
本轮没有复现历史data23/71的empirical目标，训练网格也由旧24²变为48²。因此即使观察到population收缩/塌缩，也只能证明候选机制可以发生，不能据此断言历史collapse主要由它导致。没有出现collapse同样不能排除别的初态、样本或优化设置下的collapse。
本轮只有一个teacher、两个固定初态；没有collapse概率估计，没有参数/field不可辨识性的全局证明，没有跨模型类推广。stationarity只是末200步描述，低梯度不是全局最优证明。

## SECOND_PAPER_TRIAGE

Level 1 上限或机制未决。有理论正则偏差，但本轮不足以建立跨初始化稳定病理及历史collapse主因；优先作为第一篇的诊断证据。
Level3需要推广到一类模型、原则性修复及理论/实验检验；本轮明确未实现这些。Level2候选也不能等同于可发表结论。

## 下一项最小实验

下一轮只做归档data23/init1011的empirical matched pair（λ=0与原λ），固定本轮相同48²/Adam/.003/2000步，和现有standard population pair比较；不扩seed。这个2×2目标/正则对照才开始检验收缩与历史经验collapse的联系。本轮未执行。

代码与证据入口：PROTOCOL.json、PROVENANCE.json、GATES.json、CONTROLS.json、POPULATION_COMPARISON.json、TRAJECTORIES.json、raw/及figures/。失败门槛和所有实际trajectory均保留。

## 审校补充：q与系数场范数不能混为一谈

本轮预声明shrinkage分级以q RMS相对teacher为主；该分级未达到，不等于所有field幅度不变。标准初态的(b-beta)场范数相对teacher分别为P0=0.579335、Pλ=0.538393；local分别为0.999970、0.892874。Pλ比匹配P0减少约7.07%和10.71%，这是真实的温和系数场幅度变化，应保留。与此同时q RMS终点都没有低于teacher，且标准P0本身已有明显field幅度差异，因此不能把所有teacher-field差异归因于正则。这里补充的是既存原始指标的解释，不重定义门槛、CaseC或剂量触发。没有观察到collapse、剂量条件未满足仍成立。“第二篇论文假设降温”针对稳定严重收缩/塌缩强机制，不是否认任何正则相关变化。数值见FIELD_NORM_ADDENDUM.json。
