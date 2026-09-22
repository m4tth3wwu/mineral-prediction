# P1-H execution and evidence map

Working directory: `D:\code\ResearchPractice`; branch: `codex-refactor`.

- Completed, zero updates: `python -B mineral_prediction/acawlr_ppp_p1h_quadrature.py --phase prepare`. Output is preserved structurally in `GATES.json`, `INITIAL_DIAGNOSTICS.json`, and `PRETRAIN_CONFIRMATION.json`; console checks were also returned in the task transcript.
- Completed: `python -B mineral_prediction/acawlr_ppp_p1h_quadrature.py --phase train`. Stdout/stderr: `logs/training.log`; exact boundary markers: `TRAINING_STARTED.json` / `TRAINING_END.json`.
- Completed after both trajectories finished: `python -B mineral_prediction/evaluate_acawlr_ppp_p1h_quadrature.py`. No optimizer updates; four endpoint evaluations and two isolated candidate probes.
- Completed: `python -B mineral_prediction/verify_acawlr_ppp_p1h_quadrature.py`. No optimizer updates; all saved checkpoints and fixed alignment checks at0/1000/1975.
- Completed: `python -B mineral_prediction/report_acawlr_ppp_p1h_quadrature.py`. Pure analysis and figures; no model execution.

`SOURCE_MANIFEST.json` freezes training and inherited helpers before training. `DELIVERY_SOURCE_MANIFEST.json` freezes evaluation/verification/report scripts during training. `EVALUATION_SOURCE_MANIFEST.json` froze read-only evaluation helpers before evaluation. Snapshot bytes are in `raw/source/`.

`FROZEN_BEFORE.json` freezes all previously tracked files except the three evolving root memory files. Old session pointers, old results, old protocols, old scripts and all old evidence remain read-only. The new P1H pointer is separate.

A read-only initial streaming-gradient audit additionally checked the inherited stricter1e-10 tolerance after launch; error3.93e-15 passed. Original pretraining checks and source remain unchanged. The initial streaming comparison had used1e-8 absolute/1e-9 relative numerical tolerance, while the regularizer gradient-difference gate used the inherited1e-10 tolerances; this extra record makes compliance with the stricter interpretation explicit and does not change a gate.

Primary scalar, density and field evidence: `ENDPOINTS.json` (four models on both384/768), `COMPARISON.json` (paired differences), `SUMMARY.json` (threshold status). Per-step96 objective/gradient diagnostics and48 recovery measurements are distinct from monitor96/192. Final384/768 fidelity is a value check, not a gradient convergence certificate.

Saved candidate/probe states are diagnostic only. Exactly4000 optimizer steps total, with no step2001 appended to training. Temporary full candidate displacements are separate from optimizer updates.

All listed commands exited0. Evaluation, verification and report stdout/stderr are archived in logs/evaluation.log, logs/verification.log and logs/report.log. Training ended after exactly4000 optimizer steps.
