# Diagnostics — Gate 2 + SFT frontier (replication test, pre-GRPO)

**Date:** 2026-06-19. Same gate framework/probe settings as rx_triage (k=12, n=60, MAX_TURNS=10, temp 1.0,
deterministic ordering). Base = text-only MedGemma-4B. SFT = v3-style curriculum (70% ρ=0 full-checklist, 600
traj, 2 ep / 300 steps). Outputs `results/probe_dx_{base,sft_v3}_t10.*`. **No GRPO launched** — pausing for review.

## Numbers (hidden) vs analytic baselines (oracle 0.842, best-fixed 0.724)
| metric | dx base | dx SFT-v3 |
|---|---|---|
| support@12 | **0.667** | 0.417 |
| mean reward | +0.261 | +0.123 |
| mean asks | 2.58 | **8.03** |
| 1-ask rate | 0.64 | 0.108 |
| no-commit | 0.185 | **0.707** |
| parse | 0.954 | 0.999 |
| high-risk reward | +0.193 | −0.062 |
| high-risk support | 0.40 | 0.40 |
| benign over-escalation | 0.354 | 0.025 |

## The five pre-GRPO questions
1. **Base latent support?** Yes, strongly — support@12 **0.667** (> rx_triage base 0.58); high-risk support 0.40
   (rx_triage base was 0.00). Gate 2 base passes; the capability to reach oracle-grade is present.
2. **Did SFT learn the gatherer behavior?** It gathers far **too much** (mean asks 8.0, 85% ≥3 asks) and **fails
   to terminate** — 70.7% no-commit (hits the 10-turn cap mid-work). It learned "keep ordering tests," not
   "stop and commit."
3. **1-ask collapse like rx_triage?** **No — the opposite.** rx_triage SFT under-gathered (→1-ask); diagnostics
   SFT over-gathers into the turn cap (1-ask only 0.11).
4. **A clean v3-like gatherer init for GRPO?** Not from this SFT — dx_sft_v3 over-gathers + 71% no-commit, and
   its support (0.417) is **worse than the base (0.667)**. Here the **base** is the better-balanced policy
   (support 0.667, asks 2.58, no-commit 0.185, high-risk support 0.40).
5. **Parse / no-commit / 1-ask / high-risk reward vs rx_triage?** parse excellent (0.999); no-commit catastrophic
   (0.71 vs rx_triage v3's 0.196); 1-ask opposite (0.11); high-risk reward regressed (+0.193→−0.062). Same
   *gathering↔termination* tension as rx_triage v3, **amplified** by longer checklists (6–7) vs the 10-turn budget.

## Replication verdict (SFT half)
- **Gates replicate:** Gate 1 PASS (BC-gap +0.117) and Gate 2 base latent support PASS (0.667) — necessary
  conditions hold on a structurally different task with the same geometry.
- **SFT degeneracy replicates in kind, not direction:** the curriculum recipe that under-gathered on rx_triage
  **over-gathers + fails to terminate** here, *dropping support below the base*. The robust meta-finding from
  rx_triage holds: **demonstration-curriculum SFT does not reliably instill a clean adaptive gatherer** — it
  lands in a degenerate mode whose direction is task-dependent (checklist length × turn budget).
- **New wrinkle:** on diagnostics the **base out-performs the SFT** (support 0.667 vs 0.417) — SFT actively hurt.

## Options for the (paused) GRPO step — for review
- **(a) GRPO from base** — the best-balanced diagnostics policy (support 0.667, asks 2.58). Tests whether RL
  refines a reasonable starting policy (no SFT degeneracy to fight).
- **(b) Fix the SFT first** — the over-gather+no-commit is partly a turn-budget mismatch (checklists 6–7 vs
  MAX_TURNS 10). Either raise the diagnostics horizon (~14) or use shorter/adaptive demos, re-probe, then GRPO
  from a clean gatherer. Most apples-to-apples with rx_triage.
- **(c) GRPO from dx_sft_v3** — mirror-image stress test: from an *over*-gatherer (8 asks), does RL trim toward
  oracle (2.9) *with* the ask-cost gradient (the opposite of rx_triage, where RL had to add asks against it)?
  Could be more favorable to RL, or could overshoot into the same 1-ask basin. Most informative for the thesis.

**Recommendation:** (c) is the most thesis-informative single run (does the ask-cost gradient help when the SFT
errs on the over-gathering side?), with (a) as a clean control. **No GRPO until your call.**
