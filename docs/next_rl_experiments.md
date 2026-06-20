# Next RL experiments — DREAM decoupling + explicit credit assignment (planned)

Motivation: rx_triage + diagnostics both show **gates pass but bounded GRPO exploits** (1-ask shortcut /
defensive hedge); the `deephedge` ablation tests whether *structural* reward redesign fixes it. These two
techniques attack the same failure from the RL-algorithm side — the exploration↔exploitation chicken-and-egg and
the sparse end-of-episode credit that make adaptive gathering hard to learn. **Both are for the NEXT trial(s),
after the current deephedge GRPO completes.** Source: CS224R (Assignment 4, DREAM).

## A. Explicit credit assignment (do first — smaller, composable)
**Idea (user directive):** instead of rewarding only the final commit, give **intermediate reward during the
dialogue**: + for asking a clinically-relevant question (esp. one critical to differentiating a high-risk case),
− for *skipping* a key question needed to identify a high-risk condition. Keep the final commit reward.

**Clean implementation that preserves the gate framework (important):**
- Do **NOT** change `env.score` / `U` / the analytic gates. The Gate-1 BC-gap and the oracle/best-fixed analytics
  assume reward = `U[risk][action] − cost·n_asks`; adding per-ask bonuses there would change the gates (this is
  exactly why the `infogain` ablation idea couldn't be screened by `bc_gap`). Keep the *env/gates pure*.
- Add a separate **process-reward shaping term used ONLY by the GRPO `rollout_func`** (in `grpo_disagg.py`), added
  to `traj_reward` per rollout. Gates and the held-out probe keep using the pure env reward, so the guarded
  verdict (does true task reward / high-risk support improve?) stays honest.
- **Prefer potential-based shaping** Φ to avoid changing the optimal policy / reward-hacking: e.g.
  Φ(belief) = −H(risk_dist | revealed) (negative entropy of the true-severity posterior given what's been
  asked). Per-step shaping = γΦ(s') − Φ(s); summed it telescopes to a constant, so the optimal policy is
  unchanged but credit for *information-gaining* asks arrives immediately. The env already exposes
  `risk_dist(fam, revealed)` → cheap to compute H and the per-ask information gain.
- **"Missed key question" penalty:** at commit time, if the rollout committed without having asked an item that
  the oracle path required for this case (`case["oracle_asks"]`), subtract a penalty ∝ how much that omission
  risks a high-risk miss. (Phrase as shaping so it doesn't double-count the env's existing wrong-commit penalty.)
- **Guard against reward-hacking:** cap total shaping ≪ task reward; verify with the existing guards that asks
  don't balloon (over-gather) and that high-risk *support/acceptable* actually rise — shaping that only raises
  asks without raising high-risk support is hacking, not learning.

## B. DREAM — decoupled exploration/exploitation (do after, or in conjunction)
**Idea:** the ask→commit structure *is* explore→exploit. The chicken-and-egg: good commits need good gathering;
knowing what to gather needs the commit value. DREAM decouples them:
1. **Exploration policy** trained on an **intrinsic info-gain reward** (e.g. reduction in H(risk posterior) per
   ask, or novelty over relevant un-asked items) — independent of commit quality.
2. **Freeze exploration; train an exploitation policy** to commit optimally given the gathered info on the **task
   reward**.
3. **Alternate** until both are strong; deploy the exploitation policy (with the trained explorer gathering).

**Mapping to this repo:** the env already gives the pieces for the intrinsic reward — `risk_dist`/`oracle_value`
for belief entropy and VOI. Likely realization: two LoRA adapters (explore / exploit) over the same base, or a
role token; `grpo_disagg` rollout drives explorer for the ask turns and exploiter for the commit; each phase
optimizes its own reward. Compare against the end-to-end GRPO baselines (which collapsed) under the SAME gates +
guards. Hypothesis: decoupling lets the explorer learn to gather (info-gain reward has dense, immediate signal),
breaking the 1-ask collapse that end-to-end GRPO fell into.

## Self-reminder (external)
**Retrieve CS224R notes on exploration↔exploitation techniques** (Assignment 4 / DREAM, intrinsic-reward
exploration, decoupled meta-RL) for methods to improve *balanced action exploration in multi-turn RL* — use to
refine both A and B before implementation. (Not in this repo; pull from course materials.)

## Sequencing
1. Finish current deephedge GRPO (structural-redesign ablation) → report.
2. Implement **A (credit-assignment shaping in rollout_func, env/gates unchanged)** on diagnostics (deephedge or
   base) → one bounded GRPO → guarded report. Cheapest, composable.
3. Implement **B (DREAM decoupling)** → bounded run → guarded report; optionally combine with A.
4. All evaluated against the unchanged gates + the ask-collapse / high-risk-support guards.
