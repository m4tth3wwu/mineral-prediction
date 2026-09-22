# 当前交接：P1-J已完成并核验

工作目录 D:\code\ResearchPractice，分支 codex-refactor。源基准 bdfee6c5d3570a3b8146fad133f9a3f00160f465；本轮发布提交以 Git HEAD 和远端分支为准。

run：mineral_prediction/p1j_budget/20260922T091632Z_p1j_step6000。P1J_LAST_RUN.json 为最新结果指针；P1-I及更早指针、结果和阈值保持冻结。完整历史记忆快照见 snapshots/20260922T091632Z_p1j_before_CATCH_UP.md 及同名前缀 DECISIONS/RUN_INDEX。

P1-J从P1-I第2000步完整模型及Adam状态续训到6000步，lr=.001、96²、data23/init1011及lambda配对不变，共新增8000更新，已正常结束，禁止重复训练。评价及核验均完成：21/71/65项全部通过，1834个检查点，4631个历史文件未变。

统一768² KL：λ=0在2000/4000/6000步为0.07335/0.16127/0.30395；原始λ为0.03534/0.05166/0.06380。经验目标下降，教师密度、标量g和向量场b恢复变差；所有固定状态均值一致性通过、恢复与平稳性未通过。没有触发既定塌缩判据，不等于恢复。原始λ减缓本次路径的场扩张及恢复退化。尚不能证明不可约误差、惩罚偏差或论文机制，不能把不同lr×预算当等价进度。

[完整结论](../../mineral_prediction/p1j_budget/20260922T091632Z_p1j_step6000/FINDINGS.md) · [数值报告](../../mineral_prediction/p1j_budget/20260922T091632Z_p1j_step6000/REPORT.md) · [独立核验](../../mineral_prediction/p1j_budget/20260922T091632Z_p1j_step6000/FINAL_VERIFY.json)。

下一步建议：先用存档轨迹整合P1-F–J证据，分析相同经验目标改善幅度下的恢复变化，再冻结最小区分实验；不盲目续训或扫描lr。本轮未启动下一阶段。P1-E原BLOCKED记录保持原样；其他population后续证据不能回写成P1-E成功。区分p、g、b以及beta/u/h/theta。

沿用用户commit及直接push授权。新文件统一D盘项目目录；实际矿产任务仍默认原44训练/16验证及粉色/其他有效岩性，不自动切换选点实验。
