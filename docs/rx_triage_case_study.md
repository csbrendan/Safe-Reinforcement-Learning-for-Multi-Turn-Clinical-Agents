# Case study 1 — RX-refill triage: a gated-RL result (FROZEN)

**Status: complete and frozen as the first case study.** No further RL or reward changes on rx_triage; the
reward-redesign ablation is deferred until after a second task replicates (or breaks) the pattern below.

## Claim under test
Among clinically *verifiable* tasks, RL should help most when (1) demonstrations under-specify the optimal policy
— a measurable **BC-gap** (Gate 1) — and (2) the model has **latent support** for the optimal behavior — pass@k
(Gate 2). rx_triage is the first task instrumented against this claim.

## Task & reward (fixed throughout)
Multi-turn medication-refill triage (adaptive-workup POMDP). A visible drug family + hidden labs/features; the
model ASKs items (cost 0.03 each) then COMMITs one of {continue, dose_adjust, bridge, hold_for_labs, escalate}.
Verifiable reward `R = U[true_risk][action] − 0.03·#asks`; high/moderate mishandling steeply penalized;
`hold_for_labs` bad on high (forces disambiguation). Optimal gathering is **sequentially adaptive** (value of
information): which item matters next depends on prior answers, so no *fixed* checklist matches the oracle.
Base model: text-only MedGemma-4B (vision tower dropped). Eval: hidden condition (rho=0), MAX_TURNS=10,
k=12 rollouts/case, temp 1.0, deterministic oracle ordering.

## Gate 1 — demonstrability: PASS (BC-gap +0.105)
Analytic policy values (widened env, cost 0.03): **oracle 0.854 > best-fixed checklist 0.749** (the BC-gap,
+0.105; ACEi_ARB +0.114, NSAID +0.096) **> every shortcut** — ask-all 0.747, 1-ask 0.608, no-ask 0.601,
blind-escalate 0.376. All 10 Gate-1 checks pass (oracle highest; under-gathering penalized on high; benign
over-escalation penalized; oracle selective ≈2.9 asks). → The env genuinely rewards adaptive questioning; the
gap RL could capture is real and structural (figs 1–2).

## Gate 2 — latent support: PASS
Base MedGemma-4B already samples oracle-grade gather→commit behavior: support@12 = **0.58** overall (and every
trained stage ≥ 0.58), with `oracle_grade_blind_rate = 0` (support is real gathering, not blind-commit luck).
The capability to reach the optimal policy is present to elicit.

## SFT frontier — adaptive gathering is unstable/rare under demonstration
Five SFT variants (hidden; reward / mean-asks / 1-ask / parse / no-commit / high-risk reward):

| variant | data / training | reward | mean asks | 1-ask | parse | no-commit | high-risk reward | learned mode |
|---|---|---|---|---|---|---|---|---|
| smoke-SFT | 160 best-fixed traj, 2 ep | 0.168 | 4.48 | 0.44 | 0.985 | 0.304 | −0.428 | over-asks, won't commit |
| v2 (curriculum) | 480 mixed-ρ, 2 ep | 0.409 | 1.33 | 0.90 | 0.998 | 0.010 | −0.080 | commits, under-gathers |
| **v3** | 70% ρ=0, 600, 2 ep | 0.318 | **2.78** | 0.539 | 0.950 | 0.196 | **+0.195** | **gathers** (risk-graded), validity weak |
| v4 | v3-mix, 800, 3 ep | 0.372 | 1.53 | 0.796 | 0.990 | 0.029 | +0.197 | valid+commits, under-gathers |
| v5 | v4-data, 800, 2 ep | 0.411 | 1.20 | 0.918 | 0.996 | 0.011 | +0.069 | valid+commits, under-gathers most |

Only **v3** learned to gather (greedy 2.23 asks, risk-graded high>moderate>benign); v2/v4/v5 — different
mix/epochs/n — all collapsed to ~1.2–1.5 asks despite 68–70% full-checklist demos. The gather mode is **bistable
and not reliably controllable by curriculum**: no SFT point holds gathering AND validity AND termination together
(figs 5–6). This is itself evidence that demonstrations under-specify / cannot reliably instill the adaptive
policy.

## Two confounds ruled out
- **Determinism:** `rel_items()` set-ordering made `v_oracle` (the oracle-grade threshold) wobble ±0.03 across
  `PYTHONHASHSEED`; fixed by sorting (model reward provably unchanged, re-score 0/720). Gate 1 re-confirmed PASS.
- **Decoding temperature:** at **greedy (temp 0)** v4/v5 commit after *exactly 1 ask* (mean 1.00) — under-gathering
  is a **learned policy, not a temp-1.0 sampling artifact**; higher temp *adds* a little gathering. v3 gathers at
  all temperatures. Eval temperature is neither cause nor fix (fig 8).

## RL — two bounded GRPO runs, both collapse to the 1-ask shortcut
- **GRPO #1** (init = smoke-SFT-merged, β=0, temp 1.25, 60 steps): fixed termination (no-commit 0.30→0.004) and
  raised support, but **collapsed to 1-ask** (mean 1.10, 1-ask 0.98) and **regressed high-risk reward to −0.77**.
  Guarded verdict: EXPLOIT. (Also exposed a diagnostic gap — `framework.report` initially mislabeled it
  THESIS-POSITIVE; we added ask-collapse + high-risk-reward guards.)
- **GRPO #2** (init = **v3**, the gatherer; **β=0.02 KL anchor**, temp 1.0, 60 steps): the controlled test. Held-out
  trajectory (incl. intra-run checkpoints): asks declined **monotonically 2.78→1.81→1.38→1.31**, 1-ask 0.54→0.87;
  reward 0.318→0.431, parse 0.95→0.998, no-commit 0.196→0.004 — **but high-risk acceptable-decisions fell
  0.50→0.43 and high-risk reward 0.195→0.032.** No mid-run peak. The KL anchor + gatherer init **mitigated** the
  collapse (high-risk reward +0.032 vs GRPO #1's −0.77; asks 1.31 vs 1.10) but did **not** prevent it. Guarded
  verdict: EXPLOIT / under-gathering — **not thesis-positive** (figs 3–4, 7).

## Result & lesson
**BC-gap (Gate 1) and latent support (Gate 2) are necessary but NOT sufficient** for a bounded GRPO to realize
the gain on this task. Even with a gatherer init, a KL anchor, temperature ruled out, and both gates passing, RL
trimmed asks toward the 1-ask shortcut and eroded the very high-risk safety that gathering buys. Mechanism: the
reward geometry — a **certain −0.03 per ask** versus a **diffuse benefit concentrated in the rare (≈12%)
high-risk subset** — makes ~1-ask a strong attractor for both imitation and RL; capturing the BC-gap requires
*precise* value-of-information targeting that neither SFT nor bounded RL reliably acquired. RL's apparent reward
gain came from fixing termination/validity and cutting ask-cost, not from adaptive gathering.

**Implication for the thesis:** the two a-priori gates predict *whether the headroom and capability exist*, but a
third factor — **reward geometry / credit assignment** (how concentrated the gathering benefit is relative to its
cost) — governs whether bounded RL actually converts that headroom into adaptive behavior. This motivates, as a
later ablation, a reward-redesign study (less-diffuse gathering benefit) — deferred until a second task shows
whether the "necessary-but-not-sufficient / shortcut-collapse" pattern replicates.

## Reproduce
Gates: `python -m framework.gates.bc_gap --env tasks.rx_triage.env`; `framework.gates.support_probe`. SFT/probe:
`scripts/run_sft_gate2.sh`, `scripts/run_gate2.sh`. Temp diag: `framework.gates.temp_sweep`. GRPO:
`framework.rl.grpo_disagg` (disaggregated, vLLM on node B). Dashboard: `python scripts/make_learning_plots.py`
→ `results/figs/learning_dashboard/`. Per-run reports under `docs/eval_runs/`.
