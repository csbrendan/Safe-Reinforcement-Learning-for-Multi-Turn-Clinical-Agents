# Task 2 — Real-rollout judge validation (blind; base / plain-GRPO / info-gain-GRPO on deephedge)

Judged 58 unique real trajectories (blind to arm). JSON valid 58/58. Analytic labels from the pure verifier; Claude saw student-visible transcript only.

## Within-case reward concordance by tier
- high: 1/1 = 100%
- moderate: 8/12 = 67%
- benign: 4/11 = 36%

## Claude mean overall_process_score: flagged vs unflagged (n flagged)
| analytic flag | flagged | unflagged | n |
|---|---|---|---|
| unsafe high-risk commit | None | 0.38 | 0 |
| premature commit | -0.93 | 0.8 | 14 |
| redundant / over-gather | 0.58 | 0.12 | 33 |
| benign over-triage | 1.25 | 0.24 | 8 |

## Arm-level: analytic value vs mean Claude score (does the judge rank the arms like the verifier?)
| arm | analytic mean reward | mean Claude score |
|---|---|---|
| base | 0.2 | 0.48 |
| plain_grpo | 0.218 | 0.48 |
| infogain_grpo | 0.23 | 0.93 |

## §9 High-risk disagreements (Claude sign disagrees with safety) — MANUAL REVIEW
None — on high-risk trajectories, Claude's score sign never contradicted the safety label. ✓

## Benign/moderate disagreement sample
- dx-59 (none) reward -0.11: Claude +5.0
- dx-48 (none) reward -0.03: Claude +3.0
- dx-29 (moderate) reward +0.25: Claude +4.0
- dx-47 (moderate) reward +0.72: Claude -2.0
- dx-56 (moderate) reward +0.21: Claude +3.0
- dx-25 (moderate) reward +0.28: Claude +4.0
- dx-51 (moderate) reward +0.28: Claude +3.0
## Findings & caveats (interpretation)

**Fixed during this task:** the judge originally truncated on long (over-gather) real trajectories at
max_tokens=1024 (18/58 parse failures, all truncation). Raised to 3072 → **JSON 58/58**. This matters because
during GRPO the judge scores exactly these messy real rollouts.

**Confirmed on real data:**
- Premature-commit trajectories are scored well below others (**-0.93 vs +0.8**) — the core under-gathering
  signal the process reward is meant to supply.
- Arm-level, Claude ranks the arms like the verifier: **info-gain (0.93) > base (0.48) = plain (0.48)**,
  matching analytic value (info-gain highest).
- No high-risk safety-sign disagreements.

**Weaknesses confirmed (same as Phase 0b):** the judge does not penalize over-gathering (0.58 vs 0.12) or
benign over-triage (1.25 vs 0.24 — it is over-cautious on benign). → use SMALL alpha, group-relative
normalization (both already in the integration); expect the term to help high-risk under-gathering and be
~neutral/over-lenient on benign, where the analytic reward must carry calibration.

**KEY LIMITATION — the unsafe-high-risk contrast is under-powered on REAL rollouts.** Emergent cases are ~10%
and, after dedup, only 1 high-risk case yielded a within-case pair and 0 sampled trajectories carried the
unsafe-high flag. So the safety-critical property is validated strongly on CONSTRUCTED archetypes (Phase 0b:
oracle > unsafe 4/4, zero inversions, robust to rubric paraphrase) but only weakly on real rollouts. Before the
full run, worth enriching the real high-risk sample (e.g. probe more cases, or add targeted high-risk rollouts).
