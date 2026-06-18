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

## Next
Gate 2 (re-probe base + new smoke-SFT via `gen_data.py` → `framework.rl.sft_warmup` → `support_probe`) →
one bounded GRPO → `framework.report`.
