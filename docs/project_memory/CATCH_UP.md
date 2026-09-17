# 当前交接：P1-E 静态诊断已完成，population 被门槛阻断

## 1. 时间与代码快照
更新时间：2026-09-17T11:31:09.821571+00:00（UTC）；北京时间见会话记录。分支 codex-refactor，HEAD `af633edd655a6e14398f59a222b69d07a347620b`。已跟踪文件 diff 为空；用户已要求将本轮新代码、记录、结果提交并推送；本条记录随该提交发布。原始忽略规则保持不变，明确核对的交付文件会强制加入跟踪。实际提交 SHA 与远端推送结果以 Git 历史和当前发布会话为准。实际 Git 根为 D:/code/ResearchPractice；代码目录为 mineral_prediction。
源码清单：`docs/project_memory/snapshots/20260917T111349Z_p1e_final_sources.json`；SHA256 `3aa2d0c364884203d99b49134d6a572895db82c2f13f6fffb8bcc92fdd947105`。运行时源码完整副本：`mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/raw/source_at_execution/`。报告程序运行后修订仅涉及路径纠正与图坐标，详见 REPORT_SOURCE_CHANGE.json。

## 2. 研究目标与五层对象
分别考察 p（归一化密度）、g（标量预测函数）、b（系数场）、beta/u/h（anchored 分解）、theta（网络权重）。以 p/g 为主要恢复对象；小 KL、高相关、branch 非零和 theta 距离不能互相替代。有限节点检查不证明连续域 identification。

## 3. 阶段状态
P1-C、P1-D、P1-D LR：frozen。P1-E 数学/结构、凸对照、Jacobian：COMPLETED（已执行，不等于所有科学门槛通过）。Population：BLOCKED；实际轨迹数 0。整体 scientific_status=numerical_unresolved。唯一失败 gate 为历史 .001/71 终点积分，其他记录的 gate 通过。原14项 no-training tests 通过；最终复核日志见本次 session。

## 4. 重要已确认结论
1. 【本轮数值诊断】正式 teacher/init 完整状态哈希匹配；teacher-copy gap 与 per-event gradient max 均为0；旧55文件 SHA256 不变。证据：mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/PROVENANCE.json、STRUCTURE.json#/teacher_identity、FROZEN_VERIFY.json。
2. 【数学推导与数值诊断】anchor 固定 g(0)=0；连续 h 给出 ∇g(0)=beta。sign/ReLU scaling/permutation/axial sign 是表示对称性；任意 readout scaling 不是 exact h scaling。证据：MATH_REVIEW.md、STRUCTURE.json#/symmetries（均在上述运行目录）。
3. 【本轮数值诊断】192² teacher 矩下 global-only 最优 KL≈0.00633107473，TV≈0.04397763；96→192通过。本值是受控积分近似，不是全解析 truth 积分。证据：CONVEX_CONTROLS.json#/192/global。
4. 【数学推导与数值诊断】teacher 的 R≈4.511811843；缩小 u 的 penalized population 方向导数为−9，FD通过。全局对照相对 teacher 的 `256 KL+R-R_teacher≈−2.888223`；fixed-h ridge 对照约−3.402710，KL≈0.003038141。因此 exact teacher 不是旧正则目标的最优点；尚不能据此解释每次 collapse。证据：STRUCTURE.json#/shrink、CONVEX_CONTROLS.json#/192。
5. 【本轮数值诊断】旧 .001/71 step2000 的 |ΔlogZ96→192|≈0.000506155，超过0.0001；24→48 SUM 修正≈11.60365435。证据：STRUCTURE.json#/historical/0.001_71。未放宽门槛或重训。

## 5. 未解决解释
已有条件证明支持正惩罚下全零 branch 的 strict local minimum；不是所有历史 collapsed state 的机制证明。正式 teacher 的 recipe 零线上 h 最大约5.50e−6，说明其在已测点不再严格为零；不能证明所有方向上无零线。Jacobian 谱和有限位移已保存，但未建立固定网络中的 field-changing exact counterexample，也未证明 global identification。

## 6. 最近实际执行
run_id：20260917T112555Z_p1e_v1。
命令（cwd=矿产代码目录）：`python -B acawlr_ppp_p1e_identifiability.py --output-dir p1e_identifiability/20260917T112555Z_p1e_v1 --run-population`；runner exit 0，门槛阻断训练。报告命令 `python -B report_acawlr_ppp_p1e_identifiability.py --input-dir p1e_identifiability/20260917T112555Z_p1e_v1`；exit 0。protocol/provenance 与 logs 均在 `mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/`。

## 7. 下一项最小允许行动
先审查下一版本是否应让仅涉及旧经验终点的积分失败阻断 fresh population 对照。保留本版 BLOCKED 与原阈值，不自动启动新轨迹或扩大网格。Population 基线可信以后，才考虑同归档data23/init1011、同grid/optimizer/penalty的 population/empirical matched comparison；尚未执行。

## 8. 阅读顺序
`docs/project_memory/sessions/20260917T111349Z_p1e.md` → `mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/PROTOCOL.json` / PROVENANCE.json → SUMMARY.json / NOTES.md → raw/ → `mineral_prediction/acawlr_ppp_p1e_identifiability.py`。根 AGENTS 未修改；建议补入口见 session。本轮用户已授权发布；此前仅本地的历史描述保留在原会话中。

最终验收：14项新测试再次通过；canonical索引路径、protocol哈希和55项旧文件哈希通过；参见运行目录 FINAL_VERIFY.json。

发布会话：`docs/project_memory/sessions/20260917T113838Z_github_push.md`。实验基线 HEAD 保持记录为 af633edd，不代表本次发布提交。
