# Results (fresh — populated only by new runs in this repo)

No experimental results yet. Each task records its results here as it clears the gated workflow. Prior-project
numbers are NOT results of this repo; see [lessons_learned.md](lessons_learned.md) for conceptual carry-over.

## Headline (to be filled): RL gain vs demonstrability
Plot: `results/figs/rl_gain_vs_bcgap.png` — one point per (task, reward/scale setting): x = BC-gap,
y = RL − SFT held-out value, only where latent support was confirmed.

| task | model | reward | BC-gap (deep) | latent support@k | RL − SFT (held-out) | verdict |
|---|---|---|---|---|---|---|
| rx_triage | medgemma-4b (text) | smoke-SFT 0.180 | +0.105 | base 0.55 / SFT 0.62 | (pending GRPO) | Gates 1+2 PASS |
| diagnostics | — | — | — | — | — | (pending) |
| longterm_care | — | — | — | — | — | (pending) |

## Per-task detail (template)
For each task, record: Gate-1 (BC-gap + policy/subset table + checks), Gate-2 (base & SFT pass@k, blind-rate,
ask-hist), the bounded GRPO config + training trajectory (KL, frac_reward_zero_std), the held-out diagnostic
(SFT vs RL by subset, blind-rate, high-risk support, benign over-escalation), and the four-way verdict.

### rx_triage
- **Gate 1: PASS** at env build (BC-gap +0.105; oracle 0.854 vs best-fixed 0.749; anchor ≈0). Regenerate:
  `python -m framework.gates.bc_gap --env tasks.rx_triage.env`.
- **Gate 2: PASS** (2026-06-18). Base = text-only medgemma-4b-it (vision tower dropped via
  `framework.infra.prep_text_base`; clean foundation, no domain SFT). Smoke SFT = 160 best-fixed-checklist
  trajectories (`tasks.rx_triage.gen_data`), 2 ep / 80 steps. Probe k=12, n=60 cases/condition, hidden
  prioritized. Driver: `scripts/run_gate2.sh`; report: `framework.gates.gate2_report`.

  Hidden-condition, by risk subset (supp@12 / mean reward / mean asks):

  | run | overall | high | moderate | benign | relevant | parse | reward std | high-risk supp | benign over-esc |
  |---|---|---|---|---|---|---|---|---|---|
  | base | 0.55 / +0.361 / 2.14 | 0.00 / +0.10 / 2.7 | 0.43 / +0.13 / 2.3 | 0.87 / +0.74 / 1.8 | 0.35 / +0.12 / 2.4 | 0.994 | 0.523 | **0.00** | 0.167 |
  | smoke-SFT | 0.62 / +0.180 / 4.05 | 0.43 / −0.38 / 4.1 | 0.40 / −0.05 / 4.7 | 0.96 / +0.65 / 3.2 | 0.41 / −0.11 / 4.6 | 0.987 | 0.551 | **0.43** | 0.025 |

  Analytic baselines (hidden, cost 0.03): oracle 0.854, best-fixed 0.749, ask-all 0.747, no-ask 0.601,
  1-ask 0.608, blind-escalate 0.376. `oracle_grade_blind_rate = 0.0` for BOTH runs → every oracle-grade
  success used real gathering (not blind-commit coincidence). All 10 Gate-2 checks PASS.

  **Verdict — latent support confirmed, capability latent not expressed.** Support@12 is non-trivial (0.55 →
  0.62) and clean (blind-rate 0). Smoke-SFT's mean reward (0.180) sits far below best-fixed (0.749) and oracle
  (0.854): SFT learned the gather→commit protocol (mean asks 2.14→4.05) but over-asks (pays ask-cost) and
  mis-commits high/moderate, so reward *fell* vs base — because the base exploits the demonstrability trap
  (fast "continue" on a benign-heavy mix scores high without solving the hard subsets). The discriminator that
  matters moved the right way: **high-risk support 0.00 → 0.43** (base never reached oracle-grade on a single
  high-risk case; SFT does, always via 3+ asks — high-risk og_ask_hist = 100% at 3+), and benign over-escalation
  0.167 → 0.025. This is the safety subset where the BC-gap lives becoming reachable. Reward std 0.55 gives
  ample rollout diversity for an RL signal, and there is no degenerate collapse (no-ask-commit 0, 1-ask 0.44,
  mean asks 4.05 < rel-union 7, blind-escalate 0). → An ideal RL starting point: the policy *can* reach
  oracle-grade (0.62) but doesn't *reliably* (reward 0.18); RL has a clear, safe gradient.
- **GRPO: proposed, NOT launched** — see [tasks/rx_triage/design.md](../tasks/rx_triage/design.md).

### diagnostics / longterm_care
- pending env build.
