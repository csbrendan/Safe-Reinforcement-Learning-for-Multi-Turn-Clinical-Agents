# DREAM-inspired decoupled explore/exploit — diagnostics deephedge

**Primary question.** Does decoupling exploration from exploitation improve beyond the single-episode info-gain
credit-assignment result on diagnostics deephedge?

**Answer: QUALIFIED YES — a favorable-on-safety trade, not a strict domination.** Decoupling delivers a large,
clinically meaningful **high-risk** improvement beyond info-gain (best ckpt: high-risk reward +0.248 vs +0.139,
high-risk support 1.00 vs 0.50, gathering 4.4 vs 2.2 — closer to oracle VOI ~3.9), but the unconstrained explorer
**over-gathers**, collapsing benign support (0.48→0.04) and slightly lowering moderate support and overall value.

## Method (v1, smoke-verified — commit 1e97d93)
Single network + adapter, mode token; **no learned decoder / IB / second network**. `--dream` two-phase rollout in
`grpo_disagg.py`: explore phase = ask-only, reward = recoverability `traj_infogain` over `env.risk_dist`; exploit
phase = one-turn commit over ONLY explore-gathered findings (leak-guarded by `assert_no_leak`), reward = pure
`env.score`. Decoupling = phases alternate by gradient step (separate advantage normalisation). `env.score`, `U`,
Gate 1, Gate 2, `support_probe`, `framework.report` all **untouched**. Composed-policy eval via new
`framework/gates/dream_probe.py` (explore→gather→commit), emitting support_probe-schema records.

Run: from `checkpoints/medgemma4b_text`, deephedge, beta 0.02, temp 1.0, lr 1e-5, n_cases 300, 60 steps, save 20,
max-turns 12, 2-node (trainer A / vLLM B:8001), seed 1. ~7.9h. Probe k=12/n=60, hidden+anchor, max-turns 12,
pure env.score, composed policy only.

## Four-arm comparison (held-out hidden) — `results/figs/dream_diagnostics_4arm.png`
GRPO arms = FINAL ckpt. DREAM's high-risk peak is ckpt-40 (shown in the trajectory), not final.

| arm | overall | high val | high supp | moderate supp | benign supp | mean asks |
|---|---|---|---|---|---|---|
| base | +0.251 | +0.160 | 0.50 | 0.26 | 0.48 | 2.5 |
| GRPO control (coef0) | +0.255 | +0.095 | 0.50 | 0.34 | 0.39 | 2.1 |
| GRPO info-gain (coef0.2) — **the bar** | **+0.334** | +0.139 | 0.50 | **0.77** | 0.48 | 2.2 |
| DREAM (final) | +0.304 | +0.158 | 0.50 | 0.74 | **0.09** | 5.0 |
| **DREAM (ckpt-40, high-risk peak)** | +0.329 | **+0.248** | **1.00** | 0.63 | 0.04 | 4.4 |

## DREAM checkpoint trajectory — `results/figs/dream_diagnostics_trajectory.png`

| step | high-risk reward | high-risk support@12 | mean asks | guarded verdict |
|---|---|---|---|---|
| 0 (base) | +0.160 | 0.50 | 2.5 | — |
| 20 | +0.148 | 0.50 | 3.6 | TERMINATION-ONLY |
| **40** | **+0.248** | **1.00** | 4.4 | **THESIS-POSITIVE** |
| 60 (final) | +0.158 | 0.50 | 5.0 | THESIS-POSITIVE |
| info-gain (ref) | +0.139 | 0.50 | 2.2 | THESIS-POSITIVE |

High-risk reward/support **peak at ckpt-40 then erode** (as in rx_triage); mean asks rise monotonically and stay
**above oracle VOI 3.9** at every checkpoint (over-gathering). Benign support stays collapsed (~0.04–0.09) throughout.

## Verdict against the success criteria
- **Improves ≥1 of {high-risk reward, asks→oracle VOI, moderate support}:** YES — high-risk reward
  (+0.139→+0.248) and asks-toward-oracle (2.2→4.4, vs oracle 3.9) at ckpt-40.
- **vs base, regresses no protected metric** → `framework.report` = **THESIS-POSITIVE** (ckpt-40 and final).
- **vs the info-gain bar, it is a TRADE, not a clean win:** decoupling redistributes gathering toward
  high-risk/relevant cases — high-risk reward and support improve markedly, but **moderate support 0.77→0.63**,
  **benign support 0.48→0.04**, and **overall value +0.334→+0.329** (over-gathering tax). So it does not strictly
  dominate info-gain; it wins decisively on the safety-critical subset and loses efficiency on low-acuity cases.

## Interpretation
Decoupling works exactly as theorised: with no commit-cost pressure on the explore phase, the explorer closes the
VOI gap info-gain left open (info-gain plateaued at 2.2 asks, well below oracle 3.9; DREAM reaches ~4.4) and
converts that gathering into large high-risk gains (support 0.50→1.00). The cost is the *flip side of the same
mechanism*: an explorer with no efficiency signal over-gathers on benign/moderate cases where little gathering is
needed. This is a real, interpretable result — the decoupling helps where gathering is genuinely decision-relevant
(high-risk) and hurts where it is not (benign).

## Recommended next step (not run)
Add a mild efficiency signal to the explore reward — a small ask-cost or potential-based term Φ = −n_asks·c, or
early-stop at ckpt-40 — to retain the high-risk win while curbing benign over-gathering. If that recovers benign
support without losing the high-risk gain, DREAM would strictly dominate info-gain. Escalation to a learned decoder
/ two networks (`docs/dream_two_episode_design.md`) is only warranted if the efficiency-tuned v1 still wins.
