# Task 1 — RX refill triage (adaptive-workup POMDP)

**Status: Gate-1-passing** (carried over as validated code; no data/weights).

## What it is
Multi-turn medication-refill triage. A visible family/context (ACE-inhibitor/ARB, NSAID, diuretic, metformin)
plus hidden labs/features (eGFR trajectory, potassium, volume, BP, GI-bleed risk, adherence, …). The model
ASKs items (small cost each) then COMMITs one of {continue, dose_adjust, bridge, hold_for_labs, escalate}.

## Why it's non-demonstrable (the BC-gap)
Optimal gathering is **sequentially adaptive**: which item matters next depends on the previous answer's
revealed value (disjoint per-branch item-modules). A *fixed* checklist must either over-ask the whole
rel-union (cost-crushed) or stay short and miss a branch (steep penalty), while the adaptive oracle walks a
short path. Deep families (ACEi_ARB, NSAID) carry the largest gap.

## Reward (recalibrated, safety-steep)
`R = U[true_risk][action] − ask_cost·#asks`, ask_cost 0.03. Mishandling a high OR moderate risk is strongly
penalized; `hold_for_labs` is bad on high (no moderate/high hedge → must disambiguate). Prices out **both**
under-gathering (penalized on high) and over-escalation (penalized on benign).

## Gate-1 numbers (at env build; regenerate with `python -m framework.gates.bc_gap --env tasks.rx_triage.env`)
oracle 0.854 vs best-fixed 0.749 → **BC-gap +0.105** (ACEi +0.114, NSAID +0.096); ask-all 0.747 (forced to
~6.5 asks); no-ask/1-ask/blind-defensive ≪ best-fixed; high under-gathering penalized (no-ask high 0.40 ≪
oracle 0.849); benign over-escalation penalized (blind-escalate benign 0.10 ≪ oracle 0.916); anchor ≈ 0;
oracle selective (≈2.9 asks vs rel-union 7).

## Gate 2 — PASS (2026-06-18)
See [docs/results.md](../../docs/results.md#rx_triage) for the full table/verdict. Headline: base support@12
0.55 → smoke-SFT 0.62 (hidden), `oracle_grade_blind_rate=0` both; high-risk support unlocked 0.00 → 0.43;
smoke-SFT reward 0.180 ≪ best-fixed 0.749 ≪ oracle 0.854 (capability latent, not yet expressed). Reproduce:
`bash scripts/run_gate2.sh` (base = `framework.infra.prep_text_base --src google/medgemma-4b-it`).

## Proposed bounded GRPO (NOT launched — awaiting review)
**Updated after the Gate-2 audit + remediation** (`4879cf2`; horizon now MAX_TURNS=10, deterministic oracle).
The audit refuted the turn-budget hypothesis: SFT's ~30% no-commit is genuine non-termination, not truncation.

**Goal (priority order):** (1) **commit reliably** — kill the ~30% no-commit (decision=None → reward 0); (2) trim
wasted asks (mean 4.48 → oracle ≈2.9), cutting ask-cost; (3) fix high/moderate commit accuracy — **without**
regressing high-risk support or inflating benign over-escalation.
**Report MUST track committed-rate** alongside reward/support so a gain from "learned to commit" (legitimate) is
reported distinctly from adaptive-gathering gains, and is not mistaken for either ask-cost exploitation or it.

- **Init policy:** merge `checkpoints/sft_rx_smoke/final` into `checkpoints/medgemma4b_text`
  (`framework.rl.merge_adapter`) → serves as both the GRPO init and the vLLM rollout model.
- **Topology:** disaggregated 2-node. Rollout (Spark-B): `framework/infra/serve_b.sh` (trl vllm-serve, detached
  tmux). Trainer (Spark-A): `framework.rl.grpo_disagg --env tasks.rx_triage.env --base <merged> --server-host <B>`.
- **Config:** per_device_bs 8, num_generations 8, grad_accum 4, max_completion_length 512, eos `[1,106]`,
  IS off; **lr 1e-5, beta (KL) 0.0** (validated setup) — if high-risk support regresses, fall back to beta 0.02
  to anchor the fragile smoke-SFT; rollout-temp 1.25; **max-steps 60** (bounded smoke); **save-steps 20**
  (intra-epoch saves — pick best pre-regression checkpoint); n-cases 300, seed 1.
- **Held-out gate (after):** re-probe the RL'd adapter (`support_probe --tag rl`) then
  `framework.report --sft results/probe_sft.records.jsonl --rl results/probe_rl.records.jsonl`. **Require
  THESIS-POSITIVE** (reward↑ AND oracle-grade support↑, no high-risk regression). **Abort/recalibrate reward**
  if ASK-COST EXPLOITATION (reward↑ via fewer asks while high-risk support drops).
- **Watch during:** `frac_reward_zero_std` (want low → diverse rewards per group), KL vs beta, mean-asks trend.
