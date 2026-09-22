# P1-G 中文判读

1. empirical λ=0：截至2000步，逐步48²及每25步96²监测均未观察到P1-F定义的collapse；密度/预测恢复不通过。
2. empirical 原λ：同样未观察到collapse；密度/预测恢复不通过。
3. 正则效应与population不同：192²终点empirical KL下降约47.93%，q幅度下降50.54%，field contribution下降41.51%；这些是相对无正则膨胀支路的压低，原λ的q/teacher仍约3.54、field/teacher仍约2.33，并非相对teacher塌缩。population中KL效应为+4.0246e-5，empirical为-0.0812884，interaction=-0.0813287。无统计显著性含义。
4. 本轮没有预声明的objective conflict：未出现total下降、经验数据项上升、KL和centered-g误差共同上升的相邻步或独立监测区间，持续条件也未触发。反而，经验数据项与total共同改善而teacher恢复变差的96²区间，在P0/原λ分别有46/36个；这与Case F不同。
5. 当前最支持有限样本经验目标拟合与teacher恢复脱钩，伴随优化振荡和积分分辨率问题。λ不是坏恢复的必要条件，原λ在固定终点减轻了函数幅度扩张，但未恢复teacher，也未消除优化不稳定。不能把它写成正则诱导collapse、已证明结构性不可辨识性或模型bug。

优化动态并未随终点KL改善而全面改善。最大gradient为P0=2320.57、原λ=2978.12；预声明abs(单步total变化)>1的次数为273/353。原λ的gradient/n>10或相对尖峰条件触发2步；P0虽无该阈值标志，绝对梯度仍很大，不能说稳定。实际更新量另列，不能用梯度大小替代Adam的真实步长。候选transition不是已证明的basin改变。

## 必须保留的数值限制

两个empirical终点都没有通过96²→192² logZ绝对误差≤1e-4的门槛：P0为0.000602173，原λ为0.000112211。KL/TV的对应一致性条件通过，但不能以此豁免logZ失败。48²→192²目标修正分别约+0.844975/+0.164026。这是数值未解决，不是执行失败；没有放宽门槛或重训。报告的192²值及interaction是当前离散评价结果，不能当作已验证的连续域精确量。

全部四个2×2终点都未通过沿用的末200步stationarity，不宣称任何一个是目标最优解。λ=0/原λ的empirical末段max gradient/256约9.065/11.633，total范围/256约0.03049/0.04224。

Case A/B（collapse）不获支持；Case D（原λ恢复更差）与本轮固定终点方向相反；Case E不能描述明显的功能输出差异；Case F未触发。只符合Case C中“本设置未复现collapse”这部分，不符合“两条恢复正常”，因此不强行贴完整Case C。历史24²/.01的collapse仍未解释。

## 唯一下一步建议（未执行）

固定本轮两份step2000模型，不训练、不改参数，只做192²与384²的独立积分一致性检查，沿用原数值容差；先辨别终点评价误差，随后再决定是否能解释优化/正则机制。本轮没有执行384²或任何新训练。

# P1-G fixed 2x2 evidence

All endpoints are fixed step2000, not selected checkpoints. KL/g metrics use grid192. Functional collapse and density recovery are separate.

| objective / penalty | KL | q ratio | field ratio | collapse ever | stationarity | fidelity |
|---|---:|---:|---:|---|---|---|
| standard_P0 | 1.33455722e-05 | 1.14755176 | 0.579334824 | False | False | True |
| standard_Plambda | 5.35918125e-05 | 1.0614995 | 0.538393354 | False | False | True |
| empirical_P0 | 0.169594595 | 7.15644558 | 3.98315731 | False | False | False |
| empirical_Plambda | 0.0883061486 | 3.53991004 | 2.32968002 | False | False | False |

## Interaction
| metric | empirical effect | population effect | interaction |
|---|---:|---:|---:|
| KL | -0.0812884469 | 4.02462403e-05 | -0.0813286932 |
| TV | -0.0118514651 | 0.00134120112 | -0.0131926663 |
| centered_g_area_RMSE | -0.473592321 | 0.00512236937 | -0.47871469 |
| centered_g_teacher_RMSE | -0.33781686 | 0.0051935893 | -0.34301045 |
| q_ratio | -3.61653554 | -0.0860522646 | -3.53048327 |
| field_ratio | -1.65347729 | -0.0409414697 | -1.61253582 |
| b_RMSE | -0.413209691 | -0.00949031287 | -0.403719378 |
| beta_error | -0.0597506522 | -0.0104989281 | -0.0492517241 |
| u_norm | -2.99014012 | -0.524385549 | -2.46575458 |
| theta_norm | -7.4126838 | -1.48659676 | -5.92608704 |
| gradient_max | 657.544675 | -72.480473 | 730.025148 |
| gradient_max_after300 | 657.544675 | -72.480473 | 730.025148 |
| gradient_tail_median | -104.504069 | -0.660811224 | -103.843257 |
| update_max | -0.00678370293 | 0.0178834378 | -0.0246671407 |
| update_max_after300 | 0.0059057672 | 0.00516093721 | 0.00074482999 |
| relative_update_max | 0.00432057204 | 0.00712260777 | -0.00280203573 |
| relative_update_max_after300 | 0.000888978649 | 0.00297119715 | -0.0020822185 |
| total_tail_range | 3.00840881 | -0.0194883211 | 3.02789713 |
| objective_spike_count | 80 | 0 | 80 |
| gradient_spike_count | 2 | -5 | 7 |
| transition_candidate_count | 394 | 1 | 393 |
| collapse_indicator | 0 | 0 | 0 |

## Objective conflict
{
  "empirical_P0": {
    "sustained_conflict": false,
    "coarse_sustained_conflict": false,
    "longest_monitor_conflict_streak": 0,
    "step48_count": 0,
    "monitor96_count": 0,
    "overfit_monitor_count": 46
  },
  "empirical_Plambda": {
    "sustained_conflict": false,
    "coarse_sustained_conflict": false,
    "longest_monitor_conflict_streak": 0,
    "step48_count": 0,
    "monitor96_count": 0,
    "overfit_monitor_count": 36
  }
}

Full intervals, deltas and transition steps: TRAJECTORY_DYNAMICS.json. No significance estimate; one dataset/init. Fidelity failure limits continuous-domain interpretation. All saved states are descriptive; no restart/extension.
