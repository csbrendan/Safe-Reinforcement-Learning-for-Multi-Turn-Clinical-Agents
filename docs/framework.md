# The framework — gates, calibration, and the verdict

A task is a verifiable POMDP: hidden state (some items observed up front w.p. `rho`, the rest gathered by
ASKing), a verifiable reward computable from the TRUE state, and an analytic oracle. Each task module
(`tasks/*/env.py`) implements the interface in [`framework/envs/base.py`](../framework/envs/base.py).

## Gate 1 — demonstrability (CPU): `framework/gates/bc_gap.py`
Computes, on the task's deep families under its reward: oracle-adaptive, best fixed checklist, ask-all,
no-ask, 1-ask shortcut, blind-defensive; a value/support/mean-asks table by risk subset; and the **BC-gap**
(oracle − best fixed), per family. **Pass criteria:**
- oracle adaptive clearly highest;
- best fixed below oracle by **>= ~0.10** (the genuine non-demonstrability gap);
- ask-all below oracle; no-ask / 1-ask / blind-defensive clearly below best fixed;
- **high-risk under-gathering penalized**; **benign over-escalation penalized**;
- observed anchor gap ≈ 0 (all observed → 0 asks, by construction);
- oracle trajectories selective/adaptive (mean asks < rel-union), not fixed-checklist behavior.

## Gate 2 — latent support (GPU inference): `framework/gates/support_probe.py`
`k` rollouts/prompt (k≈12). Reports **support@k** (fraction of cases with ≥1 oracle-grade rollout), mean
reward, parse validity, ask distribution, and the **adaptive-vs-blind discriminators**:
`oracle_grade_blind_rate` (oracle-grade reached with 0 asks → blind-commit coincidence) and
`oracle_grade_ask_hist`. **Pass:** support@k non-trivial (≳15–20%) with reward variance > 0 **and** the
support comes from gathering (blind-rate ~0), not blind commits. Probe the base and the SFT init.

## Reward calibration (CPU): `framework/gates/reward_calibration.py`
Sweep candidate `(U, ask_cost)` so adaptive gathering is the true optimum and cheap shortcuts are not.
Keep the shape `R = U'[risk][action] − ask_cost·#asks` so the exact solvers apply. **Tension to respect:**
lowering ask-cost fixes under-gathering exploitation but can shrink the BC-gap (cheap asks → a thorough fixed
checklist ≈ oracle). **Resolve it by widening the env, not by softening safety penalties.**

## RL — `framework/rl/` (only after both gates pass)
`merge_adapter` (build the SFT-init merged model) → `infra/serve_b.sh` on the rollout node → `grpo_disagg`
(disaggregated on-policy GRPO; env_mask trains model tokens only). One **bounded** run first. If flat, inspect
KL (must move) and `frac_reward_zero_std` (rollout diversity) before scaling steps/LR.

## Verdict — `framework/report.py`
SFT-vs-RL on held-out probe records, four-way: **THESIS-POSITIVE** (reward AND support up, no high-risk
regression) | **ASK-COST EXPLOITATION** (reward up via fewer asks; support flat/down or high-risk regresses)
| **OPTIMIZATION-ONLY** (policy moved, wrong direction) | **INCONCLUSIVE** (policy didn't move). The verdict
hinges on *adaptive gathering vs exploitation*, never on mean reward alone.

## Plots — `framework/plotting.py`
RL-gain vs BC-gap (headline), per-subset value/support, safety (high-support vs benign over-escalation),
ask-count distribution. New runs only.
