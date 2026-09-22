# P1-I command and evidence map

Working directory: D:\code\ResearchPractice; branch codex-refactor.

Completed: `python -B mineral_prediction/acawlr_ppp_p1i_learning_rate.py --phase prepare`; zero updates;23 pretraining gates passed. Training trajectory AST is identical to P1-H after substituting the one Adam lr literal.

Completed: `python -B mineral_prediction/acawlr_ppp_p1i_learning_rate.py --phase train`; two2000-update trajectories, no restarts. Full stdout/stderr: logs/training.log.

Completed after training, in order:
1. `python -B mineral_prediction/evaluate_acawlr_ppp_p1i_learning_rate.py` (four fixed endpoints384/768; new endpoint96/384 gradients and two isolated temporary-copy probes; zero optimizer steps).
2. `python -B mineral_prediction/report_acawlr_ppp_p1i_learning_rate.py` (paired learning-rate effects, all4 actual-update event rates, figures).
3. `python -B mineral_prediction/verify_acawlr_ppp_p1i_learning_rate.py` (complete saved checkpoints, actual updates, fixed0/1000/1975 alignment checks, reported event rates and paired differences; zero optimizer steps).

SOURCE_MANIFEST.json freezes training implementation before training; DELIVERY_SOURCE_MANIFEST.json freezes evaluation/report/verification during training. Raw snapshots live under raw/source. FROZEN_BEFORE.json preserves every previous tracked file except3 evolving root memory files. P1I pointers are separate from old pointers.

Both learning rates optimize the same96 quadrature objective, with common independent192 monitors. Recovery endpoints384/768 and stationarity thresholds remain separate. Smallerlr may slow fitting under equal2000-step budgets; neither better teacher metrics nor lower oscillation proves convergence or a better empirical optimum. No adaptive extra steps, grids, lr sweep or checkpoint selection.

All commands exited0. Full evaluation/report/verification stdout and stderr are saved in logs/evaluation.log, logs/report.log and logs/verification.log. Exactly4000 training updates and2 isolated probe displacements.
