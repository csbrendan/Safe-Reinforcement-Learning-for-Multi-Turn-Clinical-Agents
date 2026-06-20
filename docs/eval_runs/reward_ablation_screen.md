# Reward-redesign ablation — Gate-1 screen (diagnostics) + the hedge↔gate coupling finding

**Date:** 2026-06-20. Goal: test whether changing reward geometry can convert "Gate 1 + Gate 2 but RL exploits"
into "RL realizes adaptive gathering." Variants implemented in `tasks/diagnostics/env_rv.py` (select via
`DX_VARIANT`), gate framework unchanged. **Gate-1 screen (CPU) — the first, decisive step:**

| variant (DX_VARIANT) | change | ASK_COST | emergent % | BC-gap | Gate 1 |
|---|---|---|---|---|---|
| base | original | 0.03 | 12.1% | +0.117 | **PASS** |
| lowcost | per-ask penalty 0.03→0.015 | 0.015 | 12.1% | **+0.059** | FAIL (BC-gap < 0.10) |
| antihedge | admit-on-emergent 0.40→−0.30 | 0.03 | 12.1% | +0.118 | FAIL (high under-gather not penalized) |
| ah_lc | antihedge + lowcost | 0.015 | 12.1% | +0.059 | FAIL (both) |
| highprev | denser emergent-gating priors | 0.03 | 27.6% | +0.114 | FAIL (high under-gather not penalized) |

**Every naive reward variant breaks Gate 1.** Two distinct mechanisms:
- **lowcost** collapses the BC-gap to +0.059. Best-fixed already gathers the full rel-union and commits
  *correctly*; its only loss vs the oracle is ask-cost, so halving the cost halves the gap. (At low cost you'd
  need a much wider rel-union to keep BC-gap ≥0.10.)
- **antihedge / highprev** make a **1-ask "escalate on a positive screen" strategy ace the high subset**: the
  `1ask` policy scores **0.920 on high** (hi_supp 1.00) — *above the oracle's 0.849*. The screening finding
  (troponin / abd_rigidity) is informative enough that, once the cheap hedge is gone or emergents are dense,
  committing `emergent_tx` after one ask is the optimal blind move on true emergents. So under-gathering is no
  longer penalized on high → Gate 1 fails.

## The core finding: the hedge action is load-bearing for BOTH the gate and the exploit
Gate 1's "high under-gather penalized" holds in the base env **only because** the hedge (`admit`, U=0.40 on
emergent) is the optimal under-gatherer move — and scoring 0.40 (not 0.95) on true emergents is what makes
under-gathering look penalized. But that same hedge is **exactly what bounded GRPO exploited** (diagnostics GRPO
blind-hedged `admit`, collapsing high-risk support 0.40→0.00 while raising reward). So:

> **You cannot remove the hedge (Gate 1 breaks) and you cannot keep it (RL hedges).** The demonstrability gate
> conditions and RL-exploitability are *coupled through the hedge action*. Simple reward-geometry knobs
> (ask-cost, hedge value, prevalence) cannot independently make the gathering benefit less diffuse without
> violating a gate. **Reward geometry is not a free third knob — it is entangled with the gate structure.**

This sharpens the two-task result: BC-gap + latent support are necessary but not sufficient, and the obvious
"just fix the reward" move is *structurally* obstructed in this env class.

## What a VALID ablation would require (co-design, not a knob)
To make deep gathering genuinely required on the safety subset while keeping Gate 1, the *environment structure*
must change, not just the reward scalars: emergents must need a **multi-finding confirmation** where (i) one ask
leaves the hedge optimal (so Gate 1's under-gather penalty still holds), but (ii) each subsequent ask has **large
VOI** and the hedge **caps out far below** the gathered-then-correct value (so RL is strongly rewarded for
gathering past the hedge, and a 1-ask escalate is wrong on the false-positive screens). That is a deeper-high-risk
+ capped-hedge env, tuned to pass Gate 1 — a real env-design iteration, the proper next experiment.

## Status
`tasks/diagnostics/env_rv.py` (variant selector) committed for reproducibility. No GRPO run — no variant cleared
Gate 1, so none qualifies yet. Recommend co-designing the deep-high-risk + capped-hedge variant next (env work
+ Gate-1 tuning), then Gate 2 + one bounded GRPO. Awaiting go on that scope.

## Co-designed variant `deephedge` — Gate 1 PASSES (BC-gap +0.161)
After the screen, co-designed a structural variant (`DX_VARIANT=deephedge` in `tasks/diagnostics/env_rv.py`):
- **Emergent = a 4-finding conjunction**, and each confirming finding is **reused across non-emergent branches**
  so no single finding is diagnostic → a 1-ask "escalate on a positive finding" strategy is wrong (1-ask scores
  **−0.53** on true emergents ≪ oracle 0.83) → under-gathering is genuinely penalized on the high subset.
- **Capped hedge:** `admit` stays positive/safe-ish but is capped well below the correct disposition on BOTH
  moderate (0.86→0.45) and emergent (0.40→0.20) — so blind-hedging is clearly suboptimal and gathering to the
  right action is rewarded on both subsets, *yet* 1-ask still hedges (Gate 1's under-gather penalty holds).
- **Disjoint moderate findings** (4 extra items: renal_impair/anticoag_use/copd/…) make a fixed checklist
  over-order → BC-gap restored.

Gate 1: **PASS** — oracle 0.823 (≈3.9 asks) vs best-fixed 0.645 → **BC-gap +0.161** (chest +0.167, abd +0.155);
emergent ≈6.9%; all 10 checks pass (incl. high under-gather penalized, no_ask < best_fixed, oracle selective).
rx_triage + base-diagnostics Gate 1 unaffected; deterministic across PYTHONHASHSEED. Next: Gate 2 base probe.
