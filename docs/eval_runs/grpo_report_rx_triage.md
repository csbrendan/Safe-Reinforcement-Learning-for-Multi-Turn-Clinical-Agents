# Bounded GRPO #1 — rx_triage (held-out report)

**Date:** 2026-06-18. **Run:** disaggregated GRPO (trainer A / vLLM rollouts B:8001), init = merged smoke-SFT,
max_steps 60, lr 1e-5, beta 0.0, rollout_temp 1.25, save_steps 20, n_cases 300, bs8/G8/ga4, eos[1,106],
MAX_TURNS=10, deterministic oracle. Wall 5.5h, never frozen (`frac_reward_zero_std≈0` all 60 steps). Held-out
probe: k=12, n=60, hidden prioritized (`results/probe_grpo_t10.*`). Raw tables: `grpo_report_rx_triage.txt`.

## Held-out comparison (hidden), by risk subset — value / support@12 / mean asks
| policy | overall | high | moderate | benign | relevant | no-commit | high-risk supp | benign over-esc | parse | 1-ask rate |
|---|---|---|---|---|---|---|---|---|---|---|
| base | +0.386 / 0.58 / 2.2 | +0.151 / 0.00 / 3.0 | +0.160 / 0.50 / 2.3 | +0.752 / 0.87 / 1.9 | +0.158 / 0.41 / 2.4 | 0.085 | 0.00 | 0.181 | 0.994 | 0.73 |
| smoke-SFT | +0.168 / 0.60 / 4.5 | −0.428 / 0.14 / 4.7 | −0.060 / 0.43 / 5.1 | +0.647 / 0.96 / 3.6 | −0.130 / 0.38 / 5.0 | 0.304 | 0.143 | 0.022 | 0.985 | 0.44 |
| **GRPO** | **+0.261 / 0.85 / 1.1** | **−0.770 / 0.43 / 1.1** | **+0.130 / 0.87 / 1.1** | **+0.747 / 0.96 / 1.1** | **−0.041 / 0.78 / 1.1** | **0.004** | **0.429** | **0.029** | **1.000** | **0.982** |

Analytic baselines (hidden, cost 0.03): **oracle 0.854, best-fixed 0.749**, ask-all 0.747, no-ask 0.601,
1-ask 0.608, blind-escalate 0.376. `oracle_grade_blind_rate = 0.0` for all three (support is never blind-commit).

## The four required separations
**1. Termination gains — LARGE and clean.** no-commit 0.304 → **0.004**; parse 0.985 → **1.000**; committed-rate
≈100%. GRPO completely fixed the smoke-SFT non-termination pathology. This is real and is the bulk of the
honest gain over SFT.

**2. Ask-efficiency — OVERSHOT into a 1-ask shortcut, NOT oracle-like.** mean asks 4.48 → **1.10**, with **98.2%
of rollouts asking exactly one item**. Oracle is ≈2.9 adaptive; GRPO did not move *toward* selective gathering —
it collapsed *past* it to a fixed 1-ask-then-commit policy. (It's not even an efficient 1-ask: overall 0.261 is
below the analytic best-1-ask baseline 0.608 — the single ask and commit are not well chosen per case.)

**3. Clinical-action — benign/moderate OK, high-risk REGRESSED on reward.** benign +0.747 (≈oracle benign),
benign over-escalation 0.029 (controlled). moderate reward −0.060 → +0.130. BUT **high-risk reward −0.428 →
−0.770** (worse than SFT *and* base +0.151): a 1-ask policy cannot gather what high-risk cases need (≈3–4 asks),
so it commits wrong (`continue`/`dose_adjust` on a high case, U=−1.0). High-risk *support@12* rose 0.143 → 0.429,
but that is a k=12 sampling artifact — with temperature 1.0, ~1 of 12 one-ask rollouts occasionally lands the
right `escalate`; the *average* high-risk outcome is catastrophic. **Support up while reward craters is the tell.**

**4. Exploit checks — 1-ask under-gathering CONFIRMED.** 1-ask rate 0.982; mean asks 1.10 ≪ oracle 2.9;
high-risk gets 1.11 asks when it needs 3–4. No-ask-commit 0, blind-escalate 0, blind-hold 0, benign
over-escalation 0.029 — so the exploit is specifically **1-ask shortcut under-gathering**, not blind/defensive.

## Verdict: **Exploit (1-ask under-gathering) layered on a genuine termination fix — NOT thesis-positive.**
Overall reward rose vs SFT (+0.168 → +0.261) but (i) remains **below base (+0.386)** and far below **best-fixed
0.749 / oracle 0.854**; (ii) the gain came from eliminating no-commit **plus** cutting ask-cost via a 1-ask
collapse — asks did **not** become oracle-like selective; (iii) high-risk action quality **regressed** (−0.77).
By the pre-registered rules this is **not** "thesis-positive" (fails "moves toward oracle-like selective
gathering" and "high-risk preserved"); it is a blend of **termination-only improvement** and **1-ask shortcut
exploit**.

**Note on the automated diagnostic:** `framework.report` labels this THESIS-POSITIVE because its four-way
heuristic keys on reward↑ + support↑ + high-risk-*support*-not-regressed, and is blind to (a) the ask-collapse
(d_asks −3.38) and (b) high-risk *reward* regression. This run is a concrete counterexample → the diagnostic
needs an ask-collapse / action-quality guard (follow-up, not changed here).

## Why this is the expected outcome, and what it justifies
The smoke-SFT init was confounded by ~30% non-termination. RL's lowest-resistance path was therefore to **fix
termination by committing immediately after a single ask** — which simultaneously harvests ask-cost and dodges
the hard adaptive-gathering problem, at the cost of high-risk safety. This is exactly why crediting RL for
"improvement" here would be wrong, and it directly validates the planned next step: **rebuild the SFT baseline to
teach ask→stop→commit (curriculum generator `cb3bce0`), re-run Gate 2, and only then consider further RL** — so
that RL is evaluated against a baseline that already terminates, and must earn its keep via *adaptive gathering*
rather than a 1-ask shortcut. No second GRPO run until that review.
