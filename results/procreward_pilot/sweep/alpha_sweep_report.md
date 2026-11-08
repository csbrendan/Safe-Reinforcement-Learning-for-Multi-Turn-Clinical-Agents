# Validation alpha sweep — Claude process-reward GRPO (deephedge, seed 202)

3 GRPO runs, identical config (300 cases, 30 steps, group 8, temp 1.0, seed 1), alpha in {0.05,0.10,0.20}. Validation seed 202 only; test seed 100 untouched. support@k is a diagnostic.

> ⚠️ **α=0.20 is NOT a matched final.** INCOMPLETE — crashed@25/30 (API blip); latest saved = checkpoint-20, i.e. a 20-step result. It is shown as *screening evidence only*. Per the prespecified rule: if α=0.20(ckpt-20) is **competitive** with the completed α=0.05/α=0.10 finals, a **clean 30-step α=0.20 re-run is required before selecting α**; if **clearly dominated**, it is used as screening evidence with the limitation stated.

## Validation metrics — high subgroup  (α=0.05/0.10 = 30-step final; α=0.20 = 20-step ckpt)
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| value | 0.132 | 0.101 | 0.111 |
| unsafe_high_rate | 0.889 | 0.931 | 0.917 |
| premature_rate | 0.861 | 0.889 | 0.917 |
| mean_regret | 0.698 | 0.729 | 0.719 |
| oracle_grade_rate | 0.111 | 0.056 | 0.069 |
| benign_overtriage_rate | 0.0 | 0.0 | 0.0 |
| mean_asks | 1.71 | 1.65 | 1.5 |
| redundant_rate | 0.028 | 0.014 | 0.014 |
| one_ask_rate | 0.736 | 0.792 | 0.778 |
| over_gather_rate | 0.028 | 0.042 | 0.028 |

## Validation metrics — moderate subgroup  (α=0.05/0.10 = 30-step final; α=0.20 = 20-step ckpt)
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| value | 0.178 | 0.157 | 0.145 |
| mean_asks | 2.76 | 2.86 | 3.4 |
| benign_overtriage_rate | 0.0 | 0.0 | 0.0 |
| premature_rate | 0.68 | 0.718 | 0.649 |
| over_gather_rate | 0.174 | 0.191 | 0.243 |
| one_ask_rate | 0.532 | 0.57 | 0.5 |

## Validation metrics — benign subgroup  (α=0.05/0.10 = 30-step final; α=0.20 = 20-step ckpt)
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| value | 0.329 | 0.377 | 0.298 |
| mean_asks | 2.67 | 2.75 | 3.36 |
| benign_overtriage_rate | 0.798 | 0.741 | 0.817 |
| premature_rate | 0.612 | 0.561 | 0.565 |
| over_gather_rate | 0.143 | 0.161 | 0.22 |
| one_ask_rate | 0.555 | 0.592 | 0.531 |

## Validation metrics — overall subgroup  (α=0.05/0.10 = 30-step final; α=0.20 = 20-step ckpt)
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| value | 0.246 | 0.257 | 0.215 |
| unsafe_high_rate | 0.044 | 0.047 | 0.046 |
| premature_rate | 0.657 | 0.653 | 0.624 |
| mean_regret | 0.588 | 0.577 | 0.62 |
| oracle_grade_rate | 0.135 | 0.142 | 0.103 |
| benign_overtriage_rate | 0.372 | 0.346 | 0.381 |
| mean_asks | 2.67 | 2.75 | 3.28 |
| redundant_rate | 0.063 | 0.067 | 0.065 |
| one_ask_rate | 0.553 | 0.592 | 0.528 |
| over_gather_rate | 0.152 | 0.169 | 0.222 |

## Greedy pass@1 (k=1) high-risk
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| high value | 0.165 | -0.015 | - |
| high oracle_grade_rate | 0.0 | 0.167 | - |
| high unsafe_high_rate | 1.0 | 0.833 | - |
| high mean_asks | 1.17 | 1.67 | - |

## Selection-rule inputs (objective; deephedge oracle ≈ 3.9 asks; 1-ask-collapse guard ≥ 0.85)
| metric | α=0.05 | α=0.10 | α=0.20 *(20-step, INCOMPLETE)* |
|---|---|---|---|
| SAFETY high value | 0.132 | 0.101 | 0.111 |
| SAFETY high unsafe-commit rate | 0.889 | 0.931 | 0.917 |
| SAFETY high premature rate | 0.861 | 0.889 | 0.917 |
| BURDEN benign over-triage rate | 0.798 | 0.741 | 0.817 |
| BURDEN over-gather rate | 0.152 | 0.169 | 0.222 |
| BURDEN 1-ask rate | 0.553 | 0.592 | 0.528 |
| BURDEN mean asks | 2.67 | 2.75 | 3.28 |
| AGG overall value | 0.246 | 0.257 | 0.215 |
| AGG overall regret | 0.588 | 0.577 | 0.62 |

_Apply the prespecified rule objectively to these values: (1) reject clear benign over-triage / over-gathering / instability / aggregate regression; (2) among survivors prioritize lower unsafe-high & premature; (3) overall value only as tiebreaker; (4) prefer smaller α if practically tied. α=0.20 is a 20-step screening point — if competitive, require a clean 30-step re-run before selecting._

## Checkpoint trajectories (val seed 202)
- **high value** by checkpoint (10/20/final):
  - α=0.05: 0.15 -> 0.162 -> 0.132
  - α=0.10: 0.065 -> 0.076 -> 0.101
  - α=0.20: - -> - -> 0.111
- **high unsafe rate** by checkpoint (10/20/final):
  - α=0.05: 0.889 -> 0.861 -> 0.889
  - α=0.10: 0.972 -> 0.931 -> 0.931
  - α=0.20: - -> - -> 0.917
- **overall mean asks** by checkpoint (10/20/final):
  - α=0.05: 2.87 -> 2.88 -> 2.67
  - α=0.10: 2.6 -> 2.66 -> 2.75
  - α=0.20: - -> - -> 3.28

## Reward-component diagnostics (training rollouts)
| quantity | α=0.05 | α=0.10 | α=0.20 |
|---|---|---|---|
| analytic mean±sd | 0.2±0.29 | 0.201±0.309 | 0.193±0.304 |
| claude_raw mean±sd | -0.388±2.263 | -0.41±2.228 | -0.414±2.175 |
| claude_norm mean±sd | -0.0±0.949 | -0.0±0.97 | -0.0±0.97 |
| alpha_x_norm mean±sd | 0.0±0.047 | -0.0±0.097 | -0.0±0.194 |
| total mean±sd | 0.2±0.298 | 0.201±0.333 | 0.193±0.373 |
| % groups reordered | 63.3 | 75.8 | 80.0 |
| % groups nonzero-adv | 90.0 | 94.2 | 94.0 |
| corr(score,len) | 0.244 | 0.301 | 0.27 |

## Claude API + training stability
| quantity | α=0.05 | α=0.10 | α=0.20 |
|---|---|---|---|
| API calls | 0 | 0 | 0 |
| cache hits | 0 | 0 | 0 |
| retries | 0 | 0 | 0 |
| parse fails | 0 | 0 | 0 |
| Claude wall (s) | 0 | 0 | 0 |
| train wall (s) | 18464 | 17729 | None |

- α=0.05: asks [2.03, 3.28, 2.22, 3.34, 2.47, 2.84, 1.44, 1.72, 2.38, 3.06, 2.31, 1.75, 2.22, 2.0, 2.66, 1.62, 1.56, 3.47, 2.5, 3.72, 1.78, 2.38, 3.75, 1.53, 2.94, 3.28, 1.94, 4.34, 1.72, 3.22], reward [0.15, 0.139, 0.138, 0.311, 0.301, 0.29, 0.25, 0.163, 0.228, 0.259, 0.068, 0.137, 0.251, 0.134, 0.118, 0.245, 0.215, 0.204, 0.274, 0.032, 0.22, 0.24, 0.152, 0.365, 0.208, 0.206, 0.068, 0.142, 0.29, 0.211], parse [0.948, 0.948, 0.931, 0.948, 0.991, 0.95, 0.962, 0.93, 0.935, 0.969, 0.952, 0.92, 0.99, 0.958, 0.939, 0.881, 0.963, 0.915, 0.973, 0.932, 0.933, 0.991, 0.953, 0.938, 0.919, 0.963, 0.903, 0.976, 0.919, 0.939], committed [0.938, 0.875, 0.875, 0.75, 0.906, 0.906, 0.938, 0.812, 0.781, 0.812, 0.906, 0.906, 0.938, 0.875, 0.844, 0.781, 0.969, 0.719, 0.906, 0.812, 0.812, 0.938, 0.719, 0.875, 0.812, 0.812, 0.906, 0.75, 0.844, 0.781] (KL not logged by TRL to stdout)

- α=0.10: asks [2.09, 2.16, 2.25, 3.59, 2.25, 3.03, 2.56, 1.5, 2.28, 2.38, 2.06, 1.97, 1.78, 2.16, 1.34, 2.44, 2.0, 3.28, 2.44, 3.56, 1.88, 1.97, 2.69, 1.94, 2.31, 3.56, 2.66, 3.59, 1.59, 3.66], reward [0.205, 0.168, 0.111, 0.201, 0.43, 0.193, 0.189, 0.129, 0.281, 0.286, 0.106, 0.127, 0.245, 0.175, 0.091, 0.255, 0.253, 0.171, 0.208, 0.051, 0.283, 0.25, 0.202, 0.353, 0.287, 0.17, 0.045, 0.061, 0.281, 0.222], parse [0.959, 0.911, 0.942, 0.923, 0.981, 0.906, 0.946, 0.938, 0.922, 0.962, 0.939, 0.968, 0.977, 0.93, 0.933, 0.944, 0.958, 0.956, 0.954, 0.944, 0.934, 0.947, 0.957, 0.989, 0.971, 0.938, 0.94, 0.938, 0.928, 0.925], committed [0.875, 0.875, 0.875, 0.812, 0.938, 0.812, 0.781, 0.938, 0.812, 0.812, 0.875, 0.938, 0.906, 0.844, 0.969, 0.812, 0.938, 0.844, 0.875, 0.75, 0.844, 0.812, 0.906, 0.969, 0.938, 0.719, 0.844, 0.719, 0.875, 0.719] (KL not logged by TRL to stdout)

- α=0.20: asks [2.19, 2.78, 1.78, 4.34, 1.62, 5.25, 1.44, 2.81, 3.84, 1.56, 2.25, 2.81, 2.03, 1.47, 2.25, 2.88, 1.59, 3.19, 2.41, 4.19, 1.72, 1.69, 2.44, 2.16, 3.44], reward [0.184, -0.044, 0.023, 0.367, 0.384, 0.221, 0.231, 0.105, 0.287, 0.336, 0.093, 0.037, 0.242, 0.162, 0.195, 0.226, 0.147, 0.203, 0.201, 0.06, 0.231, 0.251, 0.159, 0.366, 0.165], parse [0.941, 0.966, 0.888, 0.976, 0.988, 0.974, 0.974, 0.95, 0.973, 0.975, 0.981, 0.951, 0.979, 0.962, 0.951, 0.943, 0.976, 0.916, 0.927, 0.92, 0.931, 0.953, 0.909, 0.96, 0.942], committed [0.844, 0.812, 0.781, 0.75, 0.969, 0.625, 0.969, 0.75, 0.75, 0.906, 0.906, 0.875, 0.906, 0.938, 0.875, 0.844, 0.938, 0.75, 0.844, 0.594, 0.906, 0.844, 0.719, 0.844, 0.781] (KL not logged by TRL to stdout)
