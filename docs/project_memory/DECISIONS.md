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
