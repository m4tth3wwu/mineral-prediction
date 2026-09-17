# 当前交接：P1-F 已完成，未观察到 population collapse

## 1. 快照
更新时间 UTC：2026-09-17T12:40:41.676720+00:00。基线 HEAD `bf97a33056911b84240df612d78795006f77446d`，codex-refactor；用户已明确要求直接push；本轮完整交付随本提交发布至origin/codex-refactor，远端完成状态以Git提交核对为准。源码清单 `docs/project_memory/snapshots/20260917T120519Z_p1f_final_sources.json`，SHA256 `7f3d13369d16bea4e928b3333bb3245ca6e96292bc5f0415d8cdf9aed0fd4c5a`。训练源码及protocol在运行前冻结；report独立记录其生成源码hash。

## 2. 五层对象
p（密度）、g（预测函数）、b（系数场）、beta/u/h（anchored分解）、theta（神经参数）分开。所有四条p/g通过不意味着b/分解/权重全恢复；小KL不是连续域field唯一性证明。

## 3. 阶段状态
P1-C/P1-D/P1-D-LR/P1-E：frozen；P1-E v1依旧BLOCKED，训练0。P1-F审计/协议/fresh gate/四条主轨迹/报告：COMPLETED。第二阶段剂量：SKIPPED（预声明条件不满足，不是失败）。所有fresh endpoint fidelity通过；候选“稳定收缩/塌缩机制”在本设置下未获支持。四条共8000更新，无其他训练。

## 4. 已确认结果（本轮数值，固定step2000/192²）
| 初态 | 正则倍数 | KL | TV | q RMS/teacher | b RMSE |
|---|---:|---:|---:|---:|---:|
| standard_P0 | 0.0 | 1.33455722e-05 | 0.001921125 | 1.147552 | 0.3124241 |
| standard_Plambda | 1.0 | 5.35918125e-05 | 0.003262326 | 1.0615 | 0.3029337 |
| local_P0 | 0.0 | 2.6627589e-11 | 2.518798e-06 | 0.9999657 | 1.511192e-05 |
| local_Plambda | 1.0 | 4.35696606e-05 | 0.003220352 | 1.062014 | 0.04558534 |

1. 四条density与predictor gates通过，两个matched pair均CaseC；没有任何monitor点的collapse，未出现预声明稳定收缩。证据 `mineral_prediction/p1f_population/20260917T121240Z_p1f_v1/POPULATION_COMPARISON.json` / SUMMARY.json。
2. local Plambda的u norm从6.70807降至1.51041，h RMS从0.0578917升至0.229656，q没有塌缩。参数收缩不能代替功能收缩。证据同目录raw/local_Plambda_trajectory.jsonl。
3. 原正则终点KL比对应P0大约4e-5，支持该固定优化程序下的精度取舍；local从step0起total下降约4.2606、penalty下降约4.2717、qRMS增加约.00936。未把正则bias连接成历史collapse主因。
4. 96→192 logZ误差均约7–9e-6；独立标准库复算KL/TV/bRMSE一致到1e-12内。协议、训练源码及143项旧文件哈希未变。证据GATES.json/FINAL_VERIFY.json。
5. 只有localP0通过末200步stationarity；另三条未通过，不能宣称全局优化完成。source/阈值/步数未事后改动。

## 5. 未解决与反证
“原正则在population中稳定诱导场收缩/塌缩”的强预期在这一个teacher、两个初态、固定2000步下未获支持；不是对所有初态的普遍排除。理论exact-density penalty bias仍成立。标准P0密度好而field误差明显，不能因此证明field结构不可辨识。历史empirical24²/.01等交互未复现，原因仍未决。第二篇论文假设降温，至多Level1/机制未决，不支持Level2/3。

## 6. 最近执行与异常
run_id=20260917T121240Z_p1f_v1；prepare/train/report均exit0。命令与完整记录见session；protocol/provenance在 `mineral_prediction/p1f_population/20260917T121240Z_p1f_v1/`。10项新无训练测试通过。训练后独立验证首次因OpenMP双运行库冲突退出，已留日志；改用标准库复算通过，未重训、未使用冲突绕过开关。一次验证审批超时后重试成功，不影响训练结果。

## 7. 下一项最小实验（仅建议）
固定归档data23/init1011，在同48²、Adam .003、2000步做empirical λ0/原λ一对，与本轮standard population pair匹配。先检验目标×正则交互，不扩seed、不改架构/优化器；本轮未执行。若未来endpoint fidelity失败先停解释，不调阈值或用最好checkpoint替代。

## 8. 阅读顺序
`docs/project_memory/sessions/20260917T120519Z_p1f.md` → `mineral_prediction/p1f_population/20260917T121240Z_p1f_v1/PROTOCOL.json` / PROVENANCE.json → NOTES.md / SUMMARY.json → raw/trajectory/states → 新P1-F源码。根AGENTS未改，历史session/decision/index保留；本轮仅更新当前摘要与追加记录。

审校补充：q分级未出现收缩，不代表field完全不变。Pλ的(b-beta)范数比匹配P0下降约7.07%/10.71%；标准P0本身相对teacher只有0.5793。原始结果FIELD_NORM_ADDENDUM.json记录区别；未改门槛或CaseC/剂量决定。
