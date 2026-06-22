# Scope — DREAM-inspired decoupled explore/exploit on diagnostics deephedge

**Status: SCOPE ONLY. Not implemented, not run.** Target = **diagnostics deephedge** (clean positive setting where
single-episode info-gain is THESIS-POSITIVE, `rl-thesis-positive-v1`). rx_triage out of scope. Raw-base RX deferred.

## Primary question
**Does decoupling exploration from exploitation improve beyond the single-episode info-gain result on diagnostics
deephedge?** Bar to beat = `dh_grpo_infogain` (held-out hidden, pure env.score): overall +0.334, high-risk support
0.50 / reward +0.139, moderate support 0.26→0.77, asks ~2.2, THESIS-POSITIVE.

## Baseline being challenged
One policy does ask→commit in a single episode; `R = env.score + 0.2·traj_infogain` (entropy reduction of the
deterministic `env.risk_dist`). Ask and commit gradients share parameters; the policy plateaus ~2.2 asks (oracle VOI
~3.9). DREAM claim: separating the two removes the interference.

## Design (single new mechanism, no broad refactor)
Per case μ, two phases of the **same** policy network distinguished by a mode token (NOT two networks):
1. **Explore phase** — ask dialogue only, no commit. Reward = recoverability `log q(μ | revealed)`, with `q` =
   the existing deterministic `env.risk_dist` posterior (same signal `traj_infogain` uses → no new learned module
   in v1).
2. **Exploit phase** — same case, explore-gathered findings pre-supplied, commit only. Reward = **`env.score`
   (unchanged, pure)**.
Decoupling = explore reward does not flow through commit and vice versa (explore tokens trained on recoverability,
commit tokens on env.score). Minimal realization of decouple-explore/exploit without a learned encoder.

### Bounded code changes
- `framework/rl/grpo_disagg.py`: a `--dream` flag → two-phase `make_rollout`; per-phase reward routing; reuses
  `traj_infogain`/`risk_dist`. No new file.
- `tasks/diagnostics/env_rv.py`: small **commit-given-fixed-revealed-set** entry point (prompt helper + 1-turn cap;
  `risk_dist` already memoized). **`env.score`, `U`, gates untouched.**
- No changes to `bc_gap` / `support_probe` / `framework.report`.

### NOT in scope (v1)
Learned `q`, IB regularizer, two policy networks, full multi-episode meta-RL, any env.score/gate change, any RX work.

## Evaluation (identical to single-episode, composed policy only)
- Deploy = **composed** policy: explore-mode gather → commit-mode on gathered findings = one ask→commit trace.
  **Probe ONLY this trace. Never probe the commit phase with privileged μ** (guard in code + assert in probe path).
- Same `support_probe`: k=12, n=60, hidden+anchor, max-turns 12, **pure env.score**.
- Same `framework.report` guards + THESIS-POSITIVE bar. Recoverability reward is training-only → gates stay pure,
  no Gate 1/Gate 2 re-run.

## Comparison arms (records on disk; reuse)
base (`probe_dh_base_t12`) · GRPO-control coef0 (`probe_dh_grpo_final_t12`) · GRPO-info-gain coef0.2
(`probe_dh_ig_final_t12`, the bar) · DREAM-decoupled (new: final + ckpt 20/40). Extend
`scripts/plot_infogain_diagnostics.py` to a 4th arm.

## Success criteria
- **Win:** clears the same guards AND beats info-gain on ≥1 of {high-risk reward, asks→oracle VOI ~3.9, moderate
  support} without regressing the others.
- **Null:** matches info-gain → decoupling adds cost without benefit here; report and stop.
- **Loss:** trips a guard info-gain passed (e.g. privileged-commit leak → over-commit) → diagnose leak, no credit.

## Cost / runtime
Two-phase rollout ≈ **1.7–2×** → ~12–14h / 60 steps on 2-node Spark (single-episode ~7h), same config (beta 0.02,
temp 1.0, lr 1e-5, n_cases 300, save 20). One trainer, one B server. Dev ≈ 0.5–1 day. Risks: (i) reward-scale
mismatch `log q(μ|·)` vs `env.score` (tune like λ); (ii) commit phase must see only explore-gathered findings,
never privileged μ.

## Recommendation
Build v1 as scoped (one network + mode token + deterministic decoder, two-phase rollout): minimal test of the
decoupling hypothesis, reuses all eval. Escalate to a learned `q` / two networks (generic plan in
`docs/dream_two_episode_design.md`) only if v1 beats info-gain.
