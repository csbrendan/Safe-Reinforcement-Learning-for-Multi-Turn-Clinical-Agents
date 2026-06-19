# v5 Gate 2 + temperature diagnostic + SFT frontier — rx_triage

**Date:** 2026-06-19. Caps the SFT-cleanliness search (v2–v5) and runs the decoding-temperature diagnostic to
decide whether "under-gathering" is a learned policy or an eval-temperature artifact. No env/reward/Gate changes.

## v5 Gate 2 (v4 data @ 2 epochs / 400 steps) — FAIL
Hidden: reward **0.411** (best of all), parse **0.996**, no-commit **0.011**, support 0.95 — but mean asks **1.20**,
**1-ask 0.918** → **Gate 2 FAILS** the anti-1-ask check; high-risk asks only 1.24, high-risk reward **+0.069**
(lowest of the validity-clean variants). The interpolation hypothesis (2 epochs would preserve v3's gathering)
is **refuted**: v5 gathered *least* of all. Outputs `results/probe_sft_v5_t10.*`.

## Temperature diagnostic — under-gathering is a LEARNED policy, not a sampling artifact
Hidden, n=40; temp 0 = greedy (k=1), temp 0.5/1.0 = k=8. (`results/temp_sweep_{v3,v4,v5}.json`.)

| ckpt | temp | mean asks | 1-ask | no-commit | reward | high reward | parse |
|---|---|---|---|---|---|---|---|
| v3 | **0.0** | **2.23** | 0.675 | 0.025 | +0.243 | −0.182 | 0.992 |
| v3 | 0.5 | 3.21 | 0.506 | 0.219 | +0.267 | +0.170 | 0.950 |
| v3 | 1.0 | 2.87 | 0.541 | 0.194 | +0.285 | +0.053 | 0.951 |
| v4 | **0.0** | **1.00** | 1.000 | 0.000 | +0.311 | −0.240 | 1.000 |
| v4 | 0.5 | 1.44 | 0.844 | 0.013 | +0.354 | +0.315 | 0.995 |
| v4 | 1.0 | 1.49 | 0.809 | 0.041 | +0.351 | +0.383 | 0.984 |
| v5 | **0.0** | **1.00** | 1.000 | 0.000 | +0.355 | +0.050 | 1.000 |
| v5 | 0.5 | 1.03 | 0.988 | 0.000 | +0.401 | +0.146 | 1.000 |
| v5 | 1.0 | 1.21 | 0.922 | 0.009 | +0.390 | +0.251 | 0.997 |

**The temp-confound hypothesis is REFUTED.** v4/v5 commit after **exactly 1 ask at greedy** (mean 1.00, 1-ask
1.000) — under-gathering is the *learned greedy policy*, not stochastic eval. Higher temperature *adds* a little
gathering (1.00→1.5), the opposite of the guess; it also flips greedy's negative high-risk reward (−0.24) to
positive by occasionally sampling a deeper gather. v3 gathers at *every* temperature (greedy 2.23) — it genuinely
learned to gather. **Eval temperature is not the cause and not the fix.**

## SFT frontier (the real finding): adaptive gathering is unstable/rare under SFT
| variant | greedy asks | learned mode | Gate-2 fail |
|---|---|---|---|
| v2 | ~1 | 1-ask | 1-ask |
| **v3** | **2.23** | **gatherer** (risk-graded) | parse + no-commit |
| v4 | 1.00 | 1-ask | 1-ask |
| v5 | 1.00 | 1-ask | 1-ask |

Across four curriculum SFTs with 68–70% full-checklist demos, **three collapsed to a greedy 1-ask policy and only
v3 learned to gather** — and v3 could not be reproduced (v4/v5, near-identical data, collapsed). The gathering
mode is **bistable** and tips on small, uncontrolled differences (seed/n/epochs). This is direct, empirical
evidence of the project's premise: **demonstrations under-specify — and do not reliably instill — the optimal
adaptive-gathering policy.** No SFT variant passes Gate 2 cleanly (v3 fails validity/termination; v2/v4/v5 fail
the 1-ask guard).

## Decision
- **(b) SFT search capped at v5** — no further curriculum iteration; the frontier is mapped and the failure mode
  (bistable gather↔1-ask, not fixable by mix/epochs/temperature) is understood.
- **(c) Proceed to a guarded GRPO from the best gatherer (v3)** — see proposal below. This is now the *intended*
  experiment: the thesis predicts RL can find the adaptive policy that SFT could not reliably instill, given the
  BC-gap (Gate 1) and latent support (Gate 2) both hold.

## Proposed bounded GRPO #2 (NOT launched — awaiting review)
- **Init = v3** (`checkpoints/sft_rx_v3`, merged) — the only checkpoint that *greedily gathers* risk-gradedly
  (2.23 asks, high>benign). Starting RL from a gatherer is the best shot at staying in the gathering basin.
- **Key change vs GRPO #1 (which collapsed v2 to 1-ask): add a KL anchor, `beta = 0.02`** (was 0.0), to hold the
  policy near v3's gathering and resist the ask-cost gradient that drove the GRPO #1 collapse. Lower **rollout
  temp 1.0** (was 1.25) to reduce drift. Otherwise the validated config: max_steps 60, lr 1e-5, save_steps 20,
  n_cases 300, bs8/G8/ga4, max_completion 512, eos [1,106], IS off, MAX_TURNS=10, same widened env/reward.
- **Live guards / abort rules:** watch mean-asks and 1-ask per step — **abort if mean-asks falls toward 1 / 1-ask
  rises past ~0.7** (collapse), or if high-risk reward goes negative. Held-out judged by `framework.report`
  (now with ask-collapse + high-risk-reward guards): require reward↑ AND support↑ AND **no ask-collapse** AND
  **high-risk reward not regressed** for THESIS-POSITIVE; EXPLOIT otherwise.
- **What each outcome means:** if GRPO from v3 *holds/improves* gathering and lifts reward toward best-fixed
  0.749 → thesis-positive (RL found what SFT couldn't instill). If it collapses to 1-ask despite the anchor →
  evidence the ask-cost credit-assignment makes the adaptive policy hard to reach even for RL here (a reward/
  exploration finding, reported as such). **No launch without your go.**
