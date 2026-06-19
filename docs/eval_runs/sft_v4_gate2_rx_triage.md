# Curriculum v4 SFT Gate 2 — rx_triage

**Date:** 2026-06-19. **What:** v4 curriculum (generator `2090ab0`: rho mix 0.68/0.22/0.10 — v3's gathering mix
with a touch more 2–4-ask termination; 100%-valid vocabulary) + **more training** (n=800, **epochs=3 → 600
steps**, 2× v3) — the directive-9 lever to raise per-turn validity and cut v3's invalid-turn no-commits. Same
widened env, reward, deterministic ordering, MAX_TURNS=10, probe k=12/n=60. Outputs `results/probe_sft_v4_t10.*`;
raw 6-way table `sft_v4_gate2_rx_triage.txt`.

## Comparison (hidden) — value / support@12 / mean asks, + behavior
| policy | reward | support@12 | mean asks | 1-ask | no-commit | parse | high reward | high supp | high asks | benign over-esc |
|---|---|---|---|---|---|---|---|---|---|---|
| base | +0.386 | 0.58 | 2.19 | 0.73 | 0.085 | 0.994 | +0.151 | 0.00 | 3.0 | 0.181 |
| smoke-SFT | +0.168 | 0.60 | 4.48 | 0.44 | 0.304 | 0.985 | −0.428 | 0.14 | 4.7 | 0.022 |
| GRPO #1 | +0.261 | 0.85 | 1.10 | 0.98 | 0.004 | 1.000 | −0.770 | 0.43 | 1.1 | 0.029 |
| v2 improved | +0.409 | 0.93 | 1.33 | 0.90 | 0.010 | 0.998 | −0.080 | 1.00 | 1.4 | 0.13 |
| v3 | +0.318 | 0.97 | 2.78 | 0.539 | 0.196 | 0.9499 | +0.195 | 1.00 | 3.4 | 0.101 |
| **v4** | **+0.372** | **0.88** | **1.53** | **0.796** | **0.029** | **0.9896** | **+0.197** | **1.00** | **2.11** | **0.094** |
| best-fixed | +0.749 | — | ~4–6 | — | — | — | — | — | — | — |
| oracle | +0.854 | — | 2.94 | — | — | — | — | — | — | — |

## What v4 fixed (the v3 problem) — and what it kept
- **Validity + termination fixed:** parse 0.9499 → **0.9896** (≥0.98 target essentially met), no-commit 0.196 →
  **0.029** (<0.05 target). The extra training did raise per-turn validity as intended.
- **High-risk safety kept:** high-risk reward **+0.197** (positive, best-tier), high-risk support 1.00, benign
  over-escalation 0.094. **Risk-grading preserved** (high 2.11 > moderate 1.53 > benign 1.36; high oracle-grade
  ask-hist still 34% at 3+).

## What v4 broke — gathering reverted toward 1-ask (FAILS the guard)
- **mean asks 2.78 → 1.53**, **1-ask 0.539 → 0.796** → **Gate 2 FAILS** "no collapse to 1-ask shortcut" (the only
  failing check; 9/10 pass). Overall reward 0.372 sits below v2 (0.409) and base (0.386), far below best-fixed
  0.749 — it does **not** close the headroom.
- The model under-gathers on the benign/moderate mass (1.3–1.5 asks) while still escalating gathering on high
  (2.11). So it's an *adaptive* under-gatherer, not a flat collapse — but the overall budget is well below oracle.

## The real finding: an SFT oscillation driven by training amount
| variant | epochs | mean asks | 1-ask | parse | no-commit |
|---|---|---|---|---|---|
| v3 | 2 (300 steps) | **2.78** ✓ | 0.539 ✓ | 0.9499 ✗ | 0.196 ✗ |
| v4 | 3 (600 steps) | 1.53 ✗ | 0.796 ✗ | **0.9896** ✓ | **0.029** ✓ |

Holding the data mix roughly constant, **more training fixes validity/termination but collapses gathering toward
the 1-ask basin**; less training preserves gathering but leaves validity/termination poor. Under the ask-cost,
the *commit-after-~1-ask* policy is a strong attractor that better-converged models fall into — exactly the
**demonstrability** difficulty: the optimal adaptive-gathering policy is hard to *instill by demonstration*; the
model reverts to the cheaper shortcut. No single (mix, epochs) point we've tried hits gather≈2.8 AND parse≈0.99
AND no-commit≈0 simultaneously.

## Verdict: NOT a clean pass → do NOT propose GRPO.
Per the pre-registered gate, v4 fails "1-ask shortcut controlled" and "mean asks risk-graded near oracle range",
so **no GRPO is proposed or launched.** v4 is arguably the best-*balanced* SFT so far (validity ✓, termination ✓,
high-risk safety ✓, risk-grading ✓) — only the overall ask budget is too low.

## Recommendation (no GRPO; await direction)
We've now mapped the SFT tradeoff surface (v2 under-gather / v3 gather-bad-validity / v4 good-validity-under-gather).
The cleanest single-variable next try is **v5 = v4 data @ 2 epochs (400 steps)** — between v3 (2 ep, gathered) and
v4 (3 ep, collapsed) — to find the validity↔gathering sweet spot (target gather ≈2.5–3 risk-graded WITH parse
≈0.98, no-commit <0.05). If that also fails to hold both, the evidence says **adaptive oracle-depth gathering is
not reliably instillable by SFT here** — which would make a bounded GRPO (to push the policy out of the 1-ask
basin toward gathering) the *intended* experiment, evaluated against the cleanest available baseline with the
ask-collapse / high-risk guards already in `framework.report`. Recommend deciding between (a) run v5, or
(b) treat v3/v4 as the baseline frontier and proceed to a guard-instrumented GRPO. **No GRPO until that call.**
