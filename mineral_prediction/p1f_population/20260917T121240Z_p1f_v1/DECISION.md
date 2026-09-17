# P1-F decision

当前证据判别：regularization_changes_recovery_without_stable_field_shrinkage。这是固定teacher/optimizer/初态下的匹配结果，不自动解释历史经验样本轨迹。

{'eligible': False, 'qualifying_pairs': [], 'all_main_complete_and_fidelity': True, 'reason': 'predeclared paired criterion not met; no dose search'}

SECOND_PAPER_TRIAGE: Level 1 上限或机制未决。有理论正则偏差，但本轮不足以建立跨初始化稳定病理及历史collapse主因；优先作为第一篇的诊断证据。

Next experiment is only proposed in NOTES; no authorization inferred from this decision.

## 审校补充：q与系数场范数不能混为一谈

本轮预声明shrinkage分级以q RMS相对teacher为主；该分级未达到，不等于所有field幅度不变。标准初态的(b-beta)场范数相对teacher分别为P0=0.579335、Pλ=0.538393；local分别为0.999970、0.892874。Pλ比匹配P0减少约7.07%和10.71%，这是真实的温和系数场幅度变化，应保留。与此同时q RMS终点都没有低于teacher，且标准P0本身已有明显field幅度差异，因此不能把所有teacher-field差异归因于正则。这里补充的是既存原始指标的解释，不重定义门槛、CaseC或剂量触发。没有观察到collapse、剂量条件未满足仍成立。“第二篇论文假设降温”针对稳定严重收缩/塌缩强机制，不是否认任何正则相关变化。数值见FIELD_NORM_ADDENDUM.json。
