# Scope — Task 3: risk–benefit titration (ordered-intensity adaptive gathering)

**Status: SCOPE ONLY. Not implemented, not run.** Third task for the project. Clinically distinct from RX (refill
yes/no) and diagnostics (which disposition bucket): this is a **titration** decision — *where on an ordered
intensity scale* to treat — under a stylized clot-vs-bleed trade. Keeps the project spine (ask→single commit
adaptive-VOI POMDP, verifiable analytic reward) so every method result (Gate 1/2, info-gain, DREAM) transfers and
is directly comparable across all three tasks.

**Clinical framing.** Stylized, guideline-*inspired* (antithrombotic-intensity-like), NOT clinically authoritative —
same toy-POMDP status as `tasks/rx_triage` and `tasks/diagnostics`. Names are scaffolding for a verifiable
decision structure, not medical advice.

## Why this shape (and what it deliberately avoids)
The restrictive constraints of the earlier chronic-care idea came from a **structured/longitudinal multi-component
action** (32-plan enumeration, interacting `u_k` tables). This task removes that by using a **single ordered
commit**, so the oracle is a trivial argmax exactly like diagnostics. Out of scope by design: multi-component plans,
multi-timestep dynamics, continuous goal weights, learned reward. In scope: one ordered action, analytic utility.

## Episode (identical to env_rv shape)
`SYS` + `user_msg(case)` (presentation + chart-given factors) → ask/answer dialogue to gather hidden factors →
**one ordered commit**. Reward = analytic `score(case, asks, decision)`. Reuses the env interface verbatim:
`gen_cases / user_msg / answer / score / resolve_ask / risk_dist / oracle_rollout / best_commit / ACTIONS /
DEEP_FAMILIES / IDX / EPS`.

## State (latent μ)
Two stylized latent risk axes, each set by a small binary-factor conjunction (mirrors `true_risk(fam, assign)`):
- **clot axis** (benefit-of-treatment driver): factors e.g. `age_high, prior_clot, chf, htn, diabetes, vascular` →
  `true_clot ∈ {low, mod, high}`.
- **bleed axis** (harm-of-treatment driver): factors e.g. `bleed_history, renal_impair, labile_control,
  fall_risk, antiplatelet_use, alcohol` → `true_bleed ∈ {low, mod, high}`.
- **Shared/overlap factors** (`htn, age_high, renal_impair` feed both axes) — this overlap is the source of
  interesting VOI: one ask can inform both axes, and which ask is pivotal depends on the case.
- `family` = a few presentation archetypes (sets base rates + which factors are chart-given/`observed` vs hidden),
  analogous to `DEEP_FAMILIES`. `rho` controls chart-given fraction (hidden vs anchor conditions, as today).

## Action (ordered intensity)
`ACTIONS = (none, low, standard, high)` — an **ordered** scale (new vs the unordered categorical commits of the
other tasks; adds calibration structure with no verifier-complexity cost).

## Utility (analytic, single-objective first)
Decompose into two monotone components on the joint state `s=(true_clot, true_bleed)` and intensity `A`:
- `benefit(s, A) = clot_severity(s) · efficacy(A)`  (efficacy non-decreasing in A, saturating)
- `harm(s, A)    = bleed_severity(s) · exposure(A)`  (exposure non-decreasing in A)
- **Net utility** `U[s][A] = benefit(s,A) − harm(s,A)`. `score = U[true_s][A] − ASK_COST·n_asks`;
  `oracle_grade = reward ≥ v_oracle − EPS` (unchanged convention).
- `risk_dist(fam, revealed)` returns the posterior over the **joint state `s`** given revealed factors (entropy
  over `s` drives `traj_infogain` for info-gain shaping — works unchanged).
- Optimal intensity `argmax_A U[s][A]` genuinely **moves with `s`**: high-clot/low-bleed → high; low-clot/high-bleed
  → none/low; mixed → standard. → guarantees a positive BC-gap and real adaptive VOI when factors shift the argmax.

## Optional goal-conditioned extension (cheap here; decide AFTER base task clears gates)
Because the commit is a single action, multi-objective is nearly free — no `u_k` authoring blow-up:
- `R_w = w_b·benefit(s,A) − w_h·harm(s,A) − ASK_COST·n_asks`, goal profiles `w ∈ {clot-first, bleed-first,
  balanced}` (3 discrete, not continuous).
- `oracle(s, w) = argmax_A R_w` — goal-flips guaranteed by construction (same `s`, different `w` → different
  optimal intensity). Goal token in the prompt; regenerate per `w` (regeneration cost accepted), so no off-policy
  relabeling wrinkle. HER-style use is **analysis-only**: rescore each trajectory under all `w` analytically for the
  goal-sensitivity diagnostics / Pareto views.
- **Lead with single-objective** (drop `w` entirely): titration then slots directly alongside RX/diagnostics as the
  third clean replication of the core thesis. Goal-conditioning is a follow-on, not load-bearing.

## Dataset / case provenance
**Synthetic, env-generated — no external dataset, no human labeling** (identical to rx_triage / diagnostics).
`gen_cases(n, rho, seed, families)` uses a seeded RNG: per case it samples a `family` archetype, draws each factor
as Bernoulli from a per-factor prevalence vector `P1`, marks a `rho`-fraction as chart-given (`observed` / anchor),
and computes the latent `(clot, bleed)` state and the oracle value analytically via `oracle_rollout`. The env's
`U` / `risk_dist` **is** the ground truth — this is what makes the reward verifiable and the gates computable
without records. Prompt text is templated from hand-authored stylized string tables (`_FAM_DESC` / `_REVEAL`), not
patient data. SFT warmup traces (if used) are generated programmatically (oracle/curriculum rollouts), not collected.
(The project's only real-data assets — the PMC-derived SFT lineage and the OOD eval sets — are separate from the RL
tasks, which run entirely on env-generated cases.) Determinism: fixed `seed`; verify reproducibility across
`PYTHONHASHSEED` as done for deephedge.

## Gates
- **Gate 1 — BC-gap (analytic, `bc_gap.py`):** `oracle_value − best_fixed_intensity` (avg), must be > 0 (adaptive
  gathering beats every single fixed intensity). Note: `bc_gap.py` currently reads `ACT_HOLD`/`ACT_ESCALATE` hooks;
  for an ordered set, generalize to a "best fixed action" baseline (label-agnostic, small change).
- **Gate 2 — latent support (`support_probe`):** pass@k that the gatherer can realize the oracle intensity.
- **NEW Gate 0 — CPU design gate (before any GPU):**
  (a) BC-gap > threshold (single-objective);
  (b) no single fixed intensity dominates (else task is degenerate);
  (c) VOI concentrated near the indifference boundary (a pivotal finding flips the optimal intensity);
  (d) [if goal-conditioned] **goal-sensitivity**: optimal intensity flips across `w` on ≥ X% of cases.
  This is the hard stop — if the `U` tables don't yield (a)–(c), fix the tables, not the model.

## Guards (`framework.report`)
Reuse existing (under-gather / 1-ask collapse, support-collapse) + titration-specific:
- **Directional miscalibration** (the titration analogue of blind-hedge): collapse to a fixed intensity
  (always-`high` = over-treat, always-`none` = under-treat). Measure **mean signed intensity error vs oracle** and
  guard against fixed-intensity collapse. (Small additive guard; report otherwise unchanged.)
- **Safety-critical subset** = high-clot ∧ high-bleed (the hardest trade) — support must not collapse.
- [if GC] **goal-agnostic collapse**: intensity invariant to `w`.

## Bounded code changes
- **New:** `tasks/titration/env.py` (mirrors `env_rv.py`: factors/`ITEMS`, `FAMILIES`, `U`, `risk_dist` over joint
  state, `oracle_rollout`, `score`, `SYS`, `gen_cases`, `user_msg`, `answer`, `resolve_ask`, ordered `ACTIONS`,
  `EPS`). Optional `GOAL` env-var + `U_benefit`/`U_harm` + per-`w` oracle (mirrors the `DX_VARIANT` / DREAM-helper
  pattern).
- **Small:** `bc_gap.py` best-fixed-action baseline generalization; optional signed-error guard in
  `framework.report`.
- **Unchanged (env-agnostic):** `grpo_disagg.py` (`--infogain-coef`, `--dream` work as-is — `traj_infogain` uses
  `risk_dist`), `support_probe.py`, `dream_probe.py`, `framework.report` core.

## De-risk order (gate-first)
1. Author `env.py` (factors, `U`, families) — CPU.
2. **CPU design gate (Gate 0)** — BC-gap > 0, no dominant fixed intensity, VOI concentration [+ goal-sensitivity if
   GC]. **HARD STOP / iterate `U` here.**
3. SFT gatherer + Gate 2 latent support.
4. Plain (single-objective) GRPO control → guarded report. (Expect under-gather, as in tasks 1–2.)
5. + info-gain shaping (the proven third factor) → guarded report; compare.
6. [optional] goal-conditioned variant + per-`w` eval grid; DREAM optional.

## Success criteria
Clears Gate 1/2; GRPO control likely under-gathers / miscalibrates (replicates the thesis a third time); info-gain
prevents the 1-ask trim and holds the high-clot∧high-bleed subset; no fixed-intensity collapse. [If GC] optimal
intensity tracks `w` (goal-sensitivity) without goal-agnostic collapse. The narrative payoff: the core
"gates-necessary-but-not-sufficient + credit-assignment" result holds on a third, structurally different decision
*type* (titration), and titration generalizes deephedge's implicit safety-vs-burden hedge into an explicit ordered axis.
