# Phase 0b — Claude process-reward JUDGE validation (diagnostics deephedge)

Cases: 12 (4 high-risk, 4 moderate, 4 benign). Trajectories judged: 64. **Judge sees STUDENT-VISIBLE info
only** — presentation, revealed findings, actions, schema; NO true risk, hidden findings, oracle actions/values,
or utility matrix. Pointwise judging (each trajectory scored independently; `trajectory_rank` derived from
scores). Cost ≈ $3.

## Reliability — clean
- **JSON validity: 64/64** (0 retries).
- **Double-label consistency: mean |Δ overall_process_score| = 0.0** — the judge is effectively deterministic.

## Headline: the judge is calibrated exactly where it matters, weak where it matters less

**Within-case ranking concordance vs analytic reward, split by risk tier** (does the judge order a case's
trajectories the way the verifier does?):

| tier | concordance | read |
|---|---|---|
| **high-risk (emergent)** | **47/56 = 84%** | strong — the safety-critical subset |
| moderate | 9/36 = 25% | weak / flat |
| benign (none) | 14/36 = 39% | weak; Claude is systematically harsh on benign work-ups (rates even the oracle −2/−3) |

The judge discriminates trajectory quality **well on high-risk cases and poorly on benign/moderate cases**,
where it flattens (near-equal scores) and is biased negative. This mirrors the Phase-0 finding that Claude is
clinically cautious / distrusts benign dispositions.

## Safety contrasts (honest — only cases where the "better" trajectory truly has higher reward)

| contrast | judge correct |
|---|---|
| **oracle > UNSAFE high-risk commit** | **4/4** |
| **oracle > under-gather, high-risk** | **4/4** |
| oracle > under-gather, all tiers | 7/8 |
| oracle > over-gather (redundant), all | 4/12 |
| oracle > premature, all | 0/3 (reward gaps are sub-0.05 ask-cost — near-ties) |

On every high-risk case the judge produces a clean, correctly-ordered separation:

```
        oracle        unsafe_commit     under_gather
dx-13   +4 (r+0.83)   +0 (r+0.17)       -3 (r-0.53)
dx-42   +5 (r+0.83)   +1 (r+0.17)       -3 (r-0.53)
dx-64   +5 (r+0.83)   +1 (r+0.17)       -3 (r-0.53)
dx-77   +5 (r+0.83)   +0 (r+0.17)       -3 (r-0.53)
```

- **§9 safety inversions: NONE.** The judge never scored an unsafe high-risk trajectory at or above its case's
  oracle. This is the single most important property for a process reward meant to protect the high-risk subset.
- **Weakness — over-gathering (4/12):** the judge does not reliably penalize redundant asks. (Its per-turn
  `redundancy` sub-score *does* fire — 1.67 on over-gather vs 0.31 otherwise — but it doesn't lower the overall
  enough to rank over-gather below the oracle.) Analytic info-gain handles this side better → the two signals
  look complementary.

## Component discrimination (mean Claude sub-score, summed over turns: flagged vs unflagged)

| sub-score | flagged | unflagged |
|---|---|---|
| unsafe_clinical_action (flag = unsafe high-risk commit) | 1.5 | 1.41 (weak on raw sub-score; the **overall** separates cleanly, 4/4) |
| premature_commitment (flag = premature) | 1.87 | 1.37 |
| redundancy (flag = over-gather) | 1.67 | 0.31 (strong) |

## Ordering invariance — clean
Oracle vs same-findings-reordered: mean |Δ| = 0.17, **12/12 within 1.0 point**. The judge does not penalize a
clinically reasonable question order, exactly as the rubric asked.

## What this means for Claude process-reward GRPO

1. **The usable signal is strong on the safety-critical subset.** On high-risk cases the judge is 84%
   concordant, 4/4 on every safety contrast, with a clean oracle ≫ unsafe ≫ under-gather ordering and zero
   inversions. That is exactly where analytic terminal reward alone lets plain GRPO exploit (hedge / under-gather
   on rare emergents), so a process reward that is sharply correct *there* is well-targeted to the paper's thesis.
2. **The benign/moderate miscalibration is less damaging than it looks — but must be handled by design.** GRPO
   uses **group-relative advantage** (subtracts the per-prompt group mean), so Claude's systematic per-case
   negative offset on benign cases washes out; what survives is *within-case ranking*, which is flat on benign
   (≈ zero shaping gradient there) and strong on high-risk. With a **small α** and the score used group-relative,
   the Claude term should act mainly as a high-risk-protective shaping signal and be ≈ neutral elsewhere — which
   is the desired profile. Recommend: z-score / group-normalize the Claude score per case before adding, and set
   α on validation so |α·process| ≪ analytic terminal utility (as the plan requires).
3. **Complementarity with info-gain.** Claude protects the *under-gathering* side (high-risk), analytic
   info-gain the *over-gathering* side. This sharpens the step-11 head-to-head: plain GRPO vs info-gain GRPO vs
   Claude process-reward GRPO are likely to fail/help on *different* subsets, not redundantly.

## Recommendation
The judge is **validated for its intended safety role** and reliable enough to use. Before RL I'd suggest one
cheap robustness add: re-score a ~20-trajectory subset with a paraphrased rubric to confirm the high-risk
separation is not prompt-brittle. **Not started GRPO** — awaiting your review per the plan.

## Per-case scores (Claude overall_process_score / analytic reward)

| case | risk | oracle | reordered | under_gather | premature | over_gather | unsafe_commit |
|---|---|---|---|---|---|---|---|
| dx-0 | none | -3.0/+0.88 | -3.0/+0.88 | -3.0/+0.37 | -3.0/+0.91 | -3.0/+0.79 | — |
| dx-1 | moderate | -1.0/+0.78 | -1.0/+0.78 | -3.0/+0.87 | +0.0/+0.81 | -1.0/+0.69 | — |
| dx-2 | moderate | -1.0/+0.81 | -1.0/+0.81 | -3.0/+0.87 | -1.0/+0.84 | -1.0/+0.72 | — |
| dx-3 | moderate | -1.0/+0.78 | -1.0/+0.78 | -3.0/+0.87 | -1.0/+0.81 | +0.0/+0.69 | — |
| dx-4 | moderate | -1.0/+0.78 | -1.0/+0.78 | -3.0/+0.87 | -1.0/+0.81 | -1.0/+0.69 | — |
| dx-7 | none | -2.0/+0.88 | -2.0/+0.88 | -3.0/+0.37 | -2.0/+0.31 | -2.0/+0.79 | — |
| dx-9 | none | -2.0/+0.88 | -2.0/+0.88 | -3.0/+0.37 | -1.0/+0.31 | -2.0/+0.79 | — |
| dx-12 | none | +0.0/+0.88 | -1.0/+0.88 | -3.0/+0.37 | +0.0/+0.31 | -1.0/+0.79 | — |
| dx-13 | high | +4.0/+0.83 | +4.0/+0.83 | -3.0/-0.53 | +5.0/+0.86 | +4.0/+0.74 | +0.0/+0.17 |
| dx-42 | high | +5.0/+0.83 | +5.0/+0.83 | -3.0/-0.53 | +4.0/+0.86 | +4.0/+0.74 | +1.0/+0.17 |
| dx-64 | high | +5.0/+0.83 | +5.0/+0.83 | -3.0/-0.53 | +4.0/+0.86 | +4.0/+0.74 | +1.0/+0.17 |
| dx-77 | high | +5.0/+0.83 | +4.0/+0.83 | -3.0/-0.53 | +5.0/+0.86 | +4.0/+0.74 | +0.0/+0.17 |

_Note: the "premature" archetype drops the last (often confirmatory) oracle ask, which usually **raises** reward
by saving ask-cost — so premature ≈ oracle in both reward and Claude score is correct, not a judge error. The
original oracle-always-wins contrast was replaced by the reward-honest, tier-split analysis above._
