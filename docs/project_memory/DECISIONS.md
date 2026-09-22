# Decisions (append-only)

## D-20260917-001
2026-09-17T11:30:07.968028+00:00. Continue P1-E with frozen prior stages, no empirical/real-data training. Preserve the original decision record at mineral_prediction/docs/project_memory/DECISIONS.md. Evidence: user request and archived attached task; numerical protocol: mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/PROTOCOL.json. Does not authorize future searches.

## D-20260917-002
2026-09-17T11:30:07.968028+00:00. Correct repository-root identification. Earlier baseline described mineral_prediction as repository root. git rev-parse --show-toplevel instead returns D:/code/ResearchPractice, with prefix mineral_prediction/. This does not affect loaded states or diagnostics. Canonical memory now at docs/project_memory; original history retained. All paths here are Git-root-relative. Source/protocol files from the numerical execution are preserved, not silently rewritten.

## D-20260917-003
2026-09-17T11:30:07.968028+00:00. Population BLOCKED under the predeclared conservative gate: old .001/71 endpoint has |ΔlogZ96→192|=.0005061547841709313 > .0001. Source: mineral_prediction/p1e_identifiability/20260917T112555Z_p1e_v1/GATES.json and STRUCTURE.json#/historical/0.001_71/fidelity. All other recorded gates pass. This failure does not prove that fresh population integration would fail; the conservative gate is a protocol choice. Do not change it after seeing results. A subsequent explicit protocol review can distinguish legacy-only failures from teacher/control failures; no new experiment authorized by this note.

## D-20260917-004
2026-09-17T11:30:07.968028+00:00. Keep existing Git allowlist and AGENTS unchanged. .gitignore lines 2/6 ignore new root docs and mineral_prediction additions. New files are local, ignored, not staged/committed/pushed; an empty tracked diff does not mean new work is committed. Add no exceptions automatically. Root AGENTS already exists; suggested handoff text is in current session.

## D-20260917-005
2026-09-17T11:38:38.577506+00:00. User explicitly requested GitHub push. Commit and push this reviewed P1-E delivery to existing origin/codex-refactor; include complete raw evidence and both canonical and historical memory. No force push, visibility change, new training or edits to frozen prior stages. Earlier local-only/no-push statements describe the preceding session, not the current authorization.

## D-20260917-006
2026-09-17T12:20:18.405132+00:00. User explicitly authorized P1-F causal discrimination. Freeze a new protocol with separate legacy audit and fresh gates; retain P1-E v1 BLOCKED/0 trajectories. Four main runs standard/local × zero/original penalty, no seed expansion. 0.1 dose only on the predeclared matched criterion. Evidence: mineral_prediction/p1f_population/20260917T121240Z_p1f_v1/PROTOCOL.json, GATES.json, PRETRAIN_CONFIRMATION.json. This is not permission to alter P1-E or launch further searches. Current source snapshot and threshold hashes fixed before any updates.

## D-20260917-007
2026-09-17T12:40:41.676720+00:00. P1-F four matched population trajectories completed2000 each, all endpoint fidelity and density/predictor recovery gates pass. Both starts give CaseC. Neither stable function-level shrinkage nor any collapse observed. Conditional dose trigger not met, so no0.1 runs. Evidence: mineral_prediction/p1f_population/20260917T121240Z_p1f_v1/POPULATION_COMPARISON.json, DOSE_DECISION.json, FINAL_VERIFY.json. Parameter u reduction is not field collapse: local Plambda u=6.70807→1.5104092, q/teacher=1.0620138. This weighs against the proposed stable population shrink/collapse mechanism in this tested setup; not a universal exclusion. Second-paper candidate cools; Level1 upper bound/uncertain, not Level2/3. Next only propose archived data23/init1011 empirical matched lambda0/1 at identical grid/optimizer/budget, no run authorization inferred.


## P1-F publication authorization 2026-09-17T12:47:00.171604+00:00
用户要求“直接push”：提交完整P1-F代码、结果及交接记录至现有origin/codex-refactor；使用普通push，保持冻结历史文件不变。


## D-20260922-001 — P1-G matched empirical pair completed
Only archived data23/init1011 lambda0/1, 4000 total updates. No observed collapse; original lambda reduces endpoint KL/branch expansion but increases some instability measures. No predeclared objective conflict. Both empirical endpoints fail unchanged logZ96/192 gate; scientific status numerically_unresolved, not structural discovery. All32 integrity checks pass and305 historical hashes unchanged. Recommend only fixed-endpoint192/384 numerical check next, not executed. User explicitly authorizes complete results commit and ordinary push codex-refactor.


## D-20260922-002 — Fixed P1-G endpoint192/384 check
Original lambda passes the unchanged logZ/KL/TV pairwise gates; lambda0 logZ delta=.0001541823 still fails1e-4. KL effect direction stable; no training, no claim of exact integration or convergence. Historical gates remain frozen. Only proposed next check: fixed lambda0 endpoint384/768, not executed.


## D-20260922-003 — lambda0 endpoint384/768 passes
Unchanged endpoint model, zero updates. logZ delta3.8467674e-5, KL delta3.5270307e-5, TV delta1.5755961e-5 pass original tolerances. Poor recovery persists on refined evaluation. Original lambda passed192/384 earlier; do not imply both models evaluated768 or optimizer convergence. No historical gates rewritten. Next proposed only: fixed two endpoints empirical gradient/data-term comparison48/384, no training; not executed.


## D-20260922-004 — fixed endpoints48/384 gradient audit
Data gradient relative differences23.9377%/9.60879%, whole-vector angles4.3902/2.8441deg, both negative48 directions remain local descent for384. Lambda0 u-block angle56.0895deg; full-vector angle hides block sensitivity. No updates, no Adam-direction or path-causality conclusion;384 derivative convergence not established. Next proposed only: read-only candidate next Adam direction from archived moments and existing48 gradient, dot against48/384 gradients; not executed.


## D-20260922-005 — candidate Adam direction remains local descent
Frozen step2000 moments plus saved48 total gradients yield candidate t2001 displacement only, no update/forward/backward. Total g dot d: P0 -0.565320/-0.738614, original lambda -3.406370/-3.739032 on48/384. All total/data blocks negative. No candidate local-ascent or grid sign-reversal evidence here; finite-step loss/path cause unresolved. Next proposed finite-displacement diagnostic on copies only, not executed or authorized as a training extension.


## D-20260922-006 — one finite candidate perturbation per frozen endpoint
P0 total changes48/384=-.371054/-.525037; original lambda=+2.072713/+1.665135 despite linear predictions-3.406370/-3.739032. Confirms finite-displacement nonlinearity at the regularized endpoint, not cause of historical collapse or isolated causal regularization effect. Two temporary copies, no optimizer calls/backward/training continuation, no alpha scan. All49 checks pass;2315 frozen hashes unchanged. Item1 completed. Next: freeze a single-variable training quadrature protocol before any new runs; not executed.

## 20260922T072637Z_p1h
用户要求一次多推进；预声明P1-H整批固定train96配对对照及统一评价。无自适应扩展，完整证据核验后沿用commit/push授权。正在执行，结果以对应run为准。

## 20260922T072637Z_p1h 完成
单变量train96配对实验完成4000更新。统一768评价：lambda0 KL改善24.68%，原lambda恶化31.04%；四终点fidelity通过、恢复及stationarity均失败，无collapse。实际更新反转702/679；两个新终点分别出现候选方向上升和有限位移反转。保留混合结果，不宣称网格加密解决问题或正则机制成立。下一阶段建议仅改预定学习率，尚未启动。20/61/35组检查通过，2345旧文件未变，1935新检查点完整。沿用直接push授权。

## 20260922T082159Z_p1i
继续预声明学习率对照，固定96²，仅lr.003改.001，两条2000步。23项前置门槛通过；完整评价/报告/独立核验后提交push。所有旧证据冻结。

## 20260922T082159Z_p1i 完成
4000更新完成，lr.001较.003在统一768评价的KL改善42.64%/69.47%，目标突变420/353→0/1，反转702/679→460/498。仍未通过恢复及stationarity，无collapse。原lambda经验拟合较少，不能将恢复改善当更接近经验最优解。下一步建议固定.001进行预声明预算延长，未执行，不再扫lr。23/63/47项检查通过，4372旧文件冻结，167新检查点完整。沿用提交push授权。

## 20260922T091632Z_p1j
预声明预算延长至累计6000步，保留P1-I两条模型和完整Adam状态，各新增4000更新，不改其他设置。21项续训前检查通过。固定2000/4000/6000评价；完整交付后沿用push授权。


## P1-J完成：固定预算续训



统一768²评价的密度 KL（越小越好）：

| 固定累计步数 | λ=0 | 原始λ |
|---|---:|---:|
| 2000 | 0.07334896 | 0.03534388 |
| 4000 | 0.16126925 | 0.05166135 |
| 6000 | 0.30395350 | 0.06380285 |

2000→6000步，经验 PPP 目标分别下降32.0965和6.3151，而 KL 分别上升314.39%和80.52%。向量场 b 的 RMSE 分别从0.58324升至1.77971、从0.36749升至0.52660；标量预测器 g 的面积 RMSE 也分别从0.43773升至1.20533、从0.25858升至0.39233。训练目标改善没有带来教师恢复改善。

原始λ组的场扩张和恢复退化较小；这是一组固定数据、初始化和优化路径下的观察，不能据此证明正则化的普遍因果效果。6000步 λ=0 的 KL 已高于历史 lr=.003、2000步的0.12787；原始λ组仍低于该历史对照的0.11576。预算不同，6000×.001不能视作2000×.003的等价优化进度。

## 动态与边界

三个等长区间1–2000、2001–4000、4001–6000中，线性预测下降但实际上升的次数：λ=0 为460/760/760，原始λ为498/703/725；目标大幅变化次数为0/100/456和1/42/53。定义及标记历史沿用冻结协议，续训边界未重置。区间事件增加不等于已定位原因。

六个固定状态均通过384²/768²值一致性门槛，但均未达到密度恢复、预测器恢复或平稳性门槛。48²全程监测未触发既定塌缩判据；没有塌缩并不意味着恢复成功。末200步最大梯度范数/256在6000步为19.3474和7.3907，远高于0.001门槛。

终点96²与384²总梯度相对差为5.46%和17.03%。临时副本上的候选6001步，λ=0在两网格均实际下降；原始λ在两网格均出现预测下降、实际上升。只有两次临时位移，未执行训练第6001步。值一致性不证明梯度已收敛，单点诊断不证明整个轨迹的原因。

观察支持本次路径上“经验拟合改善而教师恢复变差”。由于尚未满足平稳性，不能将残差唯一归因于不可约有限样本误差、惩罚偏差或论文机制，也不能作统计显著性结论。独立192²监测的严格 Case F 次数为0，不强行套用该归因。


核验21/71/65项通过；4631个历史文件未变。下一步优先存档证据整合与等经验目标改善幅度分析，不自动扩展训练或扫描超参数。详见 mineral_prediction/p1j_budget/20260922T091632Z_p1j_step6000/FINDINGS.md。
