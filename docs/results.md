# Results (fresh — populated only by new runs in this repo)

No experimental results yet. Each task records its results here as it clears the gated workflow. Prior-project
numbers are NOT results of this repo; see [lessons_learned.md](lessons_learned.md) for conceptual carry-over.

## Headline (to be filled): RL gain vs demonstrability
Plot: `results/figs/rl_gain_vs_bcgap.png` — one point per (task, reward/scale setting): x = BC-gap,
y = RL − SFT held-out value, only where latent support was confirmed.

| task | model | reward | BC-gap (deep) | latent support@k | RL − SFT (held-out) | verdict |
|---|---|---|---|---|---|---|
| rx_triage | — | — | — | — | — | (pending) |
| diagnostics | — | — | — | — | — | (pending) |
| longterm_care | — | — | — | — | — | (pending) |

## Per-task detail (template)
For each task, record: Gate-1 (BC-gap + policy/subset table + checks), Gate-2 (base & SFT pass@k, blind-rate,
ask-hist), the bounded GRPO config + training trajectory (KL, frac_reward_zero_std), the held-out diagnostic
(SFT vs RL by subset, blind-rate, high-risk support, benign over-escalation), and the four-way verdict.

### rx_triage
- Gate 1: PASS at env build (BC-gap +0.105; oracle 0.854 vs best-fixed 0.749; anchor ≈0). Re-run here to
  regenerate the canonical table: `python -m framework.gates.bc_gap --env tasks.rx_triage.env`.
- Gate 2: pending (re-probe base + smoke-SFT under this repo).
- GRPO: pending.

### diagnostics / longterm_care
- pending env build.
