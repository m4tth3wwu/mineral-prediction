# P1-E results

Run: 20260917T112555Z_p1e_v1; HEAD af633edd655a6e14398f59a222b69d07a347620b. Static diagnostics completed; this does not imply all scientific gates passed.
Population trajectories executed: 0; status BLOCKED. Failed gates: ['fidelity_legacy_0.001_71'].

| Control, 192² teacher moments | KL | TV | 256 KL + R - R_teacher | Solver |
|---|---:|---:|---:|---|
| global | 0.006331074726 | 0.04397763 | -2.88822301 | True |
| global_ridge | 0.006331077427 | 0.043970805 | -2.8882237 | True |
| fixed_h | 0 | 1.2619669e-16 | -5.32907052e-15 | converged |
| fixed_h_ridge | 0.003038140792 | 0.029911452 | -3.40270978 | converged |

Teacher moments and global gap are quadrature approximations; the global normalizer alone is analytic. The unpenalized controls still display their cost under the old penalty for fair feasible-point comparison.

| Archived step2000 | KL48 | KL96 | KL192 | ΔlogZ96→192 | fidelity |
|---|---:|---:|---:|---:|---|
| 0.01_23 | 0.00749231977 | 0.00750239174 | 0.00750491561 | 3.52319803e-06 | True |
| 0.01_53 | 0.0109297656 | 0.0108923719 | 0.0109074843 | 2.75581271e-05 | True |
| 0.01_71 | 0.0072664709 | 0.00727353752 | 0.00727531129 | 5.50503055e-06 | True |
| 0.003_23 | 0.0930846819 | 0.0916830109 | 0.0917080964 | 5.28120703e-07 | True |
| 0.003_53 | 0.0231018316 | 0.0229708356 | 0.022969611 | 9.27160947e-06 | True |
| 0.003_71 | 0.0995045957 | 0.099437595 | 0.0994220262 | 8.34362165e-05 | True |
| 0.001_23 | 0.0396737928 | 0.0399480138 | 0.0399987114 | 8.47832699e-05 | True |
| 0.001_53 | 0.0145927945 | 0.0146116044 | 0.0146201812 | 3.15186872e-05 | True |
| 0.001_71 | 0.130379274 | 0.130241853 | 0.129640868 | 0.000506154784 | False |

0.001/71 24→48 check: {'logZ24': 9.006398244177472, 'delta_logZ': 0.04532677482398739, 'SUM_correction': 11.603654354940772}.

Five-level interpretation: p/g equivalence under tested symmetries is observed; small KL is not a proof of global continuous equality. b identifiability remains conditional; no finite-network field-changing exact counterexample has been established. beta is fixed by the continuous origin derivative. u/h has sign/representation ambiguity; evaluation canonicalization does not modify training. theta has exact function-preserving transformations and need not match teacher weights.

The penalty directional derivative is negative at exact teacher density; positive ridge creates bias. A strict zero-branch local minimum exists under the conditions in MATH_REVIEW, but its relationship to the historical checkpoints remains unresolved. High correlation at a collapsed endpoint does not establish exact teacher density.

No new finite-sample training, real-data runs, optimizer search, teacher replacement, checkpoint selection, commit or push. Failure of archived endpoint integration is not evidence of structural nonidentifiability. The conservative predeclared gate blocks population when any required numerical fidelity check fails.

Consequently cases involving P0 success/P_lambda bias or oracle-local rescue cannot be judged if their trajectories were blocked. Existing finite-sample failures cannot be attributed uniquely to regularization, sampling, integration or basin effects.
