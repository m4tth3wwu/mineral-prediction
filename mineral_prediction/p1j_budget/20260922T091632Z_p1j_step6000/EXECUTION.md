# P1-J execution and evidence map

Working directory D:\code\ResearchPractice, branch codex-refactor.

Completed: `python -B mineral_prediction/acawlr_ppp_p1j_budget.py --phase prepare`.21 zero-update resume gates passed. Model and complete optimizer state restored exactly; next candidate matches archived P1-I candidate within stated tolerance.

Completed (exit 0): `python -B mineral_prediction/acawlr_ppp_p1j_budget.py --phase train`.Two continuations2000..6000,4000 new updates each. Full stdout/stderr in logs/training.log. Never restart or reset moments.

Completed in order (all exit 0; logs/evaluation.log, logs/report.log, logs/verification.log):
1. `python -B mineral_prediction/evaluate_acawlr_ppp_p1j_budget.py`: fixed2000/4000/6000 six states on384/768; final6000 gradient96/384 and one candidate full displacement on each temporary copy; zero optimizer steps.
2. `python -B mineral_prediction/report_acawlr_ppp_p1j_budget.py`: six-state budget effects and three equal2000-update blocks; old0..2000 records read-only, new2000 boundary deduplicated.
3. `python -B mineral_prediction/verify_acawlr_ppp_p1j_budget.py`: full checkpoint coverage, restored boundary, actual first2001 state vs archived probe, fixed2000/4000/5975 alignment checks, report rates and contrasts; zero optimizer steps.

SOURCE_MANIFEST freezes training before updates; DELIVERY_SOURCE_MANIFEST freezes evaluation/report/verification during training. Snapshots under raw/source. FROZEN_BEFORE preserves all previous tracked files except3 evolving root memory files. Separate P1J pointers preserve old pointers.

The segment includes2000 as an archived boundary with its original actual-update/linear fields; no update2000 is executed again. Step2001 is the first new update. Flags retain old history, and transition checkpoint saving applies within2000..6000. Mandatory states2000/2001 plus every25 and triggered neighbors are saved. Equal-block counts exclude the repeated boundary.

Budgets are not automatically equivalent to other learning rates. No adaptive extension, grid, lr, seed or checkpoint selection. Fidelity failure remains unresolved without extra refinement; NaN/Inf aborts. Value fidelity and stationarity are separate.

Final status: COMPLETED_VERIFIED. Exactly 8000 new training updates; evaluation used two temporary displacements and zero optimizer steps; verification used zero optimizer steps. All 21/71/65 gates passed. Four figures visually checked. See FINDINGS.md.
