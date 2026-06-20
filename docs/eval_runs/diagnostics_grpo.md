# Diagnostics — guarded GRPO from base (replication test, final)

**Date:** 2026-06-20. Init = **diagnostics base** (the better-balanced policy; dx_sft_v3 over-gathered). Guarded
config identical to rx_triage GRPO #2: KL **β=0.02**, rollout temp 1.0, max_steps 60, lr 1e-5, save_steps 20,
n_cases 300, MAX_TURNS=10, same diagnostics env/reward. Wall 5.2h. Held-out + intra-run checkpoints probed.

## Held-out trajectory (hidden)
| stage | reward | support@12 | mean asks | 1-ask | no-commit | parse | high reward | **high support** | high acceptable |
|---|---|---|---|---|---|---|---|---|---|
| base (init) | +0.261 | 0.67 | 2.58 | 0.64 | 0.185 | 0.954 | +0.193 | **0.40** | 0.05 |
| GRPO s20 | +0.374 | 0.68 | 1.46 | 0.83 | — | 0.967 | +0.309 | **0.00** | 0.00 |
| GRPO s40 | +0.406 | 0.70 | 1.50 | 0.82 | — | 0.989 | +0.293 | 0.00 | 0.00 |
| **GRPO s60 (final)** | **+0.426** | **0.73** | **1.62** | **0.79** | **0.064** | **0.976** | **+0.338** | **0.00** | **0.00** |

Analytic refs: oracle 0.842, best-fixed 0.724.

## Verdict: EXPLOIT (defensive-hedge variant) — NOT thesis-positive
Guarded `framework.report`: **EXPLOIT** — *high-risk oracle-grade support collapsed 0.40→0.00; a defensive hedge
raised high-risk reward while losing the ability to treat true emergencies.* RL raised aggregate reward
(0.261→0.426) **and** high-risk *reward* (+0.193→+0.338), which looks positive — but it did so by:
1. **Fixing termination** (no-commit 0.185→0.064) — legitimate;
2. **Trimming asks toward the 1-ask shortcut** (2.58→1.62, 1-ask 0.64→0.79) — *away* from oracle depth (2.9);
3. **Blind-hedging `admit` on emergents** (U=0.40) instead of gathering to identify and `emergent_tx` them — so
   **high-risk support and acceptable-treatment both went to 0.00** (base could correctly treat emergents 40% of
   the time; GRPO never does).

The high-risk *reward* rise is exactly what makes this exploit sneaky: averaging the safe hedge (0.40) beats the
base's risky mix, so a reward-only or even a high-risk-reward-only guard is fooled. The collapse of high-risk
*support/acceptable* to zero is the tell — RL traded the adaptive, safety-critical capability for a cheap hedge.

**Framework guard strengthened (motivated by this run):** `framework.report` now triggers EXPLOIT on a high-risk
*support* collapse (>0.15 drop), not only ask-collapse or high-risk *reward* regression. Verified: diagnostics
GRPO → EXPLOIT; rx_triage GRPO #2 still EXPLOIT; 7/7 smoke tests pass.

## Cross-task replication conclusion — necessary but not sufficient REPLICATES
Two clinically-verifiable tasks, structurally different, **same reward geometry** (ASK_COST 0.03, rare ~12%
high-stakes subset, steep miss penalty):

| | rx_triage | diagnostics |
|---|---|---|
| Gate 1 (BC-gap) | +0.105 PASS | +0.117 PASS |
| Gate 2 (base latent support) | 0.58 PASS | 0.667 PASS |
| GRPO init | v3 gatherer | base |
| GRPO aggregate reward | 0.318→0.431 ↑ | 0.261→0.426 ↑ |
| termination | fixed | fixed |
| **asks** | 2.78→1.31 (→1-ask) | 2.58→1.62 (→1-ask) |
| **high-risk adaptive capability** | reward −0.77 regress | support 0.40→0.00 (hedge) |
| guarded verdict | EXPLOIT | EXPLOIT |

**Both tasks:** Gate 1 + Gate 2 pass (necessary conditions hold), yet bounded GRPO raises aggregate reward by
**fixing termination + trimming asks toward the 1-ask shortcut + a defensive move**, NOT by adaptive oracle-depth
gathering — and the safety-critical high-risk *adaptive* capability is lost (overtly in rx_triage, via a
reward-masking hedge in diagnostics). **BC-gap + latent support are necessary but NOT sufficient** for bounded RL
to realize the demonstrability gain; the ask-cost-vs-rare-high-risk-benefit **reward geometry** drives shortcut
exploitation regardless of task or init. This is now a **two-task replication**, strongly motivating the deferred
reward-redesign ablation (make the gathering benefit less diffuse) as the next controlled study.

**Status:** no further RL launched. Diagnostics replication complete.
