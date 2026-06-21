# Diagnostics deephedge — DREAM-inspired info-gain credit assignment (single-episode)

**Question.** On the deephedge variant (which clears Gate 1 BC-gap +0.161 and Gate 2 latent support),
bounded GRPO with the plain verifiable reward still *exploits* — it trims toward 1-ask shortcuts and hedges
high-risk. Does explicit per-ask credit assignment prevent that while preserving final clinical performance?

**Method (single-episode DREAM proxy, RL-only).** Treat the hidden case / true severity as task identity μ, ask
turns as the exploration trajectory, the commit as exploitation. Shaped reward `R = env.score + λ·infogain`, where
`infogain = Σ max(prev−cur, 0)` over the entropy `H(risk_dist(fam, revealed))` as each *new, relevant* ask reveals
its value (`framework/rl/grpo_disagg.py: traj_infogain`). λ=0.2. Shaping lives ONLY in the rollout reward — `env.score`,
`U`, and the analytic gates are untouched (so Gate 1/Gate 2 need no re-run; confirmed deephedge BC-gap still +0.161).
Irrelevant/repeat asks → 0 gain (also discourages over-asking).

**Runs (identical config; 2-node disagg GRPO, trainer A / vLLM B:8001).**
`--base medgemma4b_text --max-steps 60 --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --save-steps 20 --n-cases 300
--seed 1 --max-turns 12`, DX_VARIANT=deephedge. Control = `--infogain-coef 0.0` (`checkpoints/dh_grpo_base`);
treatment = `--infogain-coef 0.2` (`checkpoints/dh_grpo_infogain`). Each ~7h wall.

**Probe.** `support_probe` k=12, n=60, conditions hidden,anchor, max-turns 12, **pure env.score** (shaping excluded
from evaluation). Verdict via `framework.report` (guards: ask-collapse, high-risk reward regression, high-risk
support collapse).

**Figure:** `results/figs/infogain_diagnostics_base_vs_ctrl_vs_ig.png` (gen: `scripts/plot_infogain_diagnostics.py`)
— base vs GRPO-control(coef0) vs info-gain(coef0.2), held-out hidden, by risk subset (value / support@12 / asks).

## Held-out `hidden` (gathering test)

| ckpt | overall val | high-risk val | high-risk supp | moderate supp | mean_asks | 3+ asks | verdict |
|---|---|---|---|---|---|---|---|
| base (untrained) | +0.251 | +0.160 | 0.50 | 0.26 | 2.54 | 24% | — |
| control s40 | +0.301 | +0.055 | **0.00** | 0.57 | 1.97 | 16% | EXPLOIT (HR support collapse / hedge) |
| control final | +0.255 | **+0.095** | 0.50 | 0.34 | 1.92 | 15% | EXPLOIT (HR reward regressed) |
| info-gain s40 | +0.308 | +0.171 | 0.50 | 0.57 | 2.17 | 21% | **THESIS-POSITIVE** |
| info-gain final | **+0.334** | +0.139 | 0.50 | **0.77** | 2.23 | 20% | **THESIS-POSITIVE** |

`framework.report` (base→IG-final): d_value=+0.083, d_support=+0.300, HR support 0.5→0.5 (supp_reg=False),
HR reward 0.16→0.139 (reward_reg=False), one_ask=0.674 mean_asks=2.23 (ask_collapse=False) ⇒ THESIS-POSITIVE.
IG-s40 likewise THESIS-POSITIVE (HR reward 0.16→0.171). support@12 rises monotonically across info-gain
checkpoints (0.35→0.60→0.62→0.65), vs control's peak-then-regress (0.35→0.50→0.367).

## Findings

1. **No 1-ask trim.** Info-gain holds mean_asks ~2.2 and keeps the 3+ gathering bucket ~20% (control halved it to
   ~15%, drifting to 1-ask). ask_collapse=False on every info-gain checkpoint.
2. **No high-risk hedge.** HR support stays 0.50 and HR reward ≈base (0.139–0.171), vs control which either
   regressed HR reward (final) or collapsed HR support to 0.00 (s40 — the defensive-hedge mirage where overall
   support rises while the ability to treat true emergencies is lost).
3. **Real gain, not a sampling artifact.** The lift is concentrated in the **moderate** subset (support 0.26→0.77,
   value 0.177→0.368) — exactly where gathering changes the disposition — and is monotonic across checkpoints.

## Caveat (honest)

The `anchor` condition is essentially unchanged (blind-commit ~0.94, mean_asks ~0.97 for both control and
info-gain). The shaping rewards reducing uncertainty about *hidden* severity; it does **not** confer resistance to a
*misleading anchor*, which is a distinct failure mode. Anchor robustness would need a different intervention.

## Takeaway

The single-episode DREAM-inspired info-gain shaping does what it was designed to: explicit per-ask credit assignment
prevents bounded GRPO from trimming to 1-ask shortcuts and from hedging high-risk, while *improving* held-out
clinical performance. This is the third necessary factor on top of BC-gap (Gate 1) and latent support (Gate 2):
**reward geometry / credit assignment**. The result clears the bar for trying full two-episode (decoupled
explore/exploit) DREAM next.
