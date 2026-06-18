# Gate-2 anomaly audit — rx_triage (base meanR 0.361 > smoke-SFT 0.180, yet SFT lifts high-risk support 0→0.43)

**Date:** 2026-06-18. **Inputs:** `results/probe_{base,sft}.records.jsonl` (hidden), live `tasks/rx_triage/env.py`,
`data/rx_triage_sft_smoke.jsonl`. **Tool:** `scripts/gate2_audit.py` (CPU, deterministic; raw: `gate2_audit_rx_triage.txt`).
**No changes were made to the reward, env, or data.**

## Verdict: **A (legitimate)** — the reward gap is real and correctly scored, not a scoring bug.
Independent re-score of all 1440 rollouts against the *current* env: **base 0/720, SFT 0/720 reward mismatches**;
n_asks consistent (0 mismatches); **0 hidden-state leaks** in SFT prompts. The base>SFT mean-reward gap is fully
explained by understood behavior (below), with a **C-contributor** (undertrained smoke SFT) and a **harness
confound** (turn-budget truncation) layered on top, plus one **minor unrelated B-nit** (oracle support-threshold
non-determinism). None of these is a reward/env/parser defect that would invalidate Gate 2.

## Why base scores higher (it is NOT solving the task)
Risk mix of the 60 hidden eval cases: none=23 (38%), moderate=30 (50%), high=7 (12%).
- **79% of base's mean reward comes from benign cheap-continue.** base benign: continue-rate 0.72, mean_asks 1.75,
  meanR **+0.743**, contributing **+0.285** of the +0.361 overall. High+moderate together contribute only +0.077.
- **On high-risk, base blind-hedges with `hold_for_labs` (0.68 of the time) → U[high][hold]=0.40**: a *safe-ish but
  non-oracle* payoff (oracle wants `escalate`=0.95). So base banks ~+0.10 on high while reaching oracle-grade on
  **zero** high cases (support 0.00). Its reward is high *because* it never pays for gathering and the eval mass is
  benign/moderate-continue — the demonstrability trap, exactly as hypothesized.

## Why smoke-SFT scores lower (three compounding mechanisms)
1. **Ask-cost from a non-adaptive checklist [C, by design].** SFT mean_asks 4.05 vs base 2.14 (≈ −0.06 extra cost).
   The expert *is* the best **fixed** checklist (ACEi 4 items, NSAID 6) and — confirmed — demonstrates the **full**
   checklist every case, **never stopping early** (`demos_that_stopped_early=0`; asks/traj = exactly 4.0 / 6.0 vs
   oracle adaptive ≈2.94). This over-asking is **intentional** — it defines the BC-gap RL must beat — not a bug.
2. **No-commit truncation [harness + C — the single biggest drain].** **31.8% of SFT rollouts never commit**
   (decision=None → reward 0.0), vs 10.7% for base; 38% on high, 37% on moderate. Driver: the probe/rollout turn
   budget is `MAX_TURNS=8`, barely above the 6-ask NSAID checklist, and SFT also **repeats asks (28%)** and asks
   **irrelevant items (26%)** — so it runs out of turns before committing. Examples (paired cases): benign NSAID
   wk-35 base 0.97 vs SFT 0.12 (SFT 4.6 asks, no commit); wk-39 SFT 6.4 asks, no commit; wk-16 SFT 5.9 asks, no commit.
3. **Mis-commit on high when undertrained [A — capability latent, not yet shaped].** When SFT *does* commit on high
   it plays `continue` 0.35 of the time → **U[high][continue] = −1.0** (e.g. wk-24: base +0.33 via hold, SFT −0.83
   via continue). The 80-step smoke SFT learned the *protocol* (ask→commit) and reaches oracle-grade `escalate`
   often enough to lift high-risk support 0→0.43, but has **not** reliably learned the action mapping.

## Checklist results (full table in the .txt)
| check | base | sft | reading |
|---|---|---|---|
| re-score reward mismatch | 0/720 | 0/720 | reward fn identical to now — **not B** |
| oracle_grade mismatch | 0/720 | 1/720 | one boundary flip from the v_oracle nit below |
| hidden-leak in prompt | — | 0/160 | **no prompt leak** |
| mean asks (overall) | 2.14 | 4.05 | SFT over-asks (non-adaptive checklist) |
| **no-commit rate** | 0.107 | **0.318** | turn-budget truncation — biggest reward drain |
| repeated-ask rate | 0.133 | 0.281 | undertrained mimicry |
| irrelevant-ask rate | 0.101 | 0.256 | undertrained mimicry |
| high-risk support | 0.00 | 0.43 | SFT genuinely unlocks adaptive escalate |
| SFT demos malformed / not-checklist / stop-early | — | 0 / 0 / 0 | expert demos are well-formed & faithfully non-adaptive |

## Minor, unrelated finding (B-nit, sub-threshold — flagged, not fixed)
`rel_items()` returns a **set**; `oracle_rollout()` iterates it, so on value-of-information **ties** the next-ask
pick depends on `PYTHONHASHSEED` (per-process). This makes the oracle path length differ by one ask on ~6/60 cases,
wobbling **`v_oracle` by ±0.03** (vsum 51.49/51.52/51.55 across hash seeds; intra-process deterministic). It moves
only the oracle-grade **threshold** (hence support classification on ~1 rollout), **never model reward**. Recommend
later: sort the ask candidates in `oracle_rollout` (and `rel_items` consumers) for cross-process reproducibility.
This is below Gate-1/Gate-2 thresholds and is **not** the anomaly under audit.

## Bottom line & recommendation
**Category A.** The Gate-2 reward readout is correct; base's higher mean reward is the demonstrability trap, and
SFT's lower mean reward is real over-asking + truncation + undertrained mis-commits — *with* a genuine safety/support
gain (high-risk 0→0.43, benign over-escalation 0.167→0.025). GRPO from this init is well-posed: its gradient
directly targets the three drains (commit reliably, stop asking earlier, fix high-risk action), and the positive
examples already exist (support 0.43).

**Caveat that should shape the GRPO read:** the 31.8% no-commit rate is partly a `MAX_TURNS=8` truncation artifact.
Two clean options before/around GRPO (neither touches reward/env/data):
- **(rec) De-confound the turn budget:** raise rollout `MAX_TURNS` to ~12 (probe + grpo_disagg) so a 6-ask checklist
  can commit, then re-probe SFT for a fair baseline — *or* simply ensure the final GRPO report tracks **committed-rate**
  alongside reward so a gain isn't misread as mere "commits more."
- **Proceed as-is:** GRPO will be rewarded for committing within budget and trimming asks anyway.

---

## Remediation addendum (2026-06-18) — what the fix actually showed

Applied (no reward/env/data change): (1) deterministic oracle ordering — `rel_items()` sorted, so `v_oracle`
is now hash-seed-stable (vsum 51.55 across seeds; model reward re-score still 0/720); (2) `MAX_TURNS` 8→10;
(3) re-ran Gate 1 (all 10 checks PASS, BC-gap +0.105) and Gate 2 (reusing the SFT adapter — data unchanged).
Outputs: `results/probe_{base,sft}_t10.json`; report `gate2_t10_rx_triage.txt`.

**Key finding — the truncation-confound hypothesis is partly REFUTED.** Raising the horizon did NOT reduce the
SFT no-commit rate; the model just asked *more*:

| SFT (hidden) | t8 (MAX_TURNS 8) | t10 (MAX_TURNS 10) |
|---|---|---|
| no-commit (decision=None) | 0.318 | **0.304** (flat) |
| mean asks | 4.05 | **4.48** (↑) |
| support@12 | 0.62 | 0.60 |
| mean reward | 0.180 | 0.168 |
| high-risk support | 0.43 | 0.143 (noisy, 1/7 cases) |
| benign over-escalation | 0.025 | 0.022 |

Given more rope, the smoke-SFT policy gathers more (high 4.12→4.68, moderate 4.66→5.12) and still fails to
commit — so **no-commit is a genuine non-termination / over-asking pathology of the undertrained smoke SFT, not
a measurement artifact of the turn budget.** This *sharpens* verdict A→C: the SFT reward shortfall is dominated
by behavior RL must fix (commit reliably; stop asking), not by scoring or harness. The high-risk-support drop
(0.43→0.14) is within sampling noise on 7 high cases, compounded by *more* no-commits on high (0.38→0.43) — the
extra turns hurt exactly the subset that needs disciplined deep gathering.

**Consequence for GRPO:** the #1 target is commit-termination (≈30% of rollouts score 0 purely by never
committing). The bounded GRPO config is unchanged, but the final report MUST track **committed-rate** alongside
reward/support so a reward gain that comes from "learned to commit" (legitimate) is reported distinctly from
adaptive-gathering gains. Determinism + horizon are now fixed; re-proposing the same bounded GRPO for review.
