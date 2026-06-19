# Task 2 — Diagnostics (acute-presentation work-up)  [Gate-1-passing]

**Status: Gate 1 PASS** (BC-gap +0.117). Built to test whether rx_triage's "gates necessary-but-not-sufficient /
1-ask shortcut collapse" pattern **replicates** on a structurally different task with the *same reward geometry*.

## What it is
Acute-care triage by presentation. A visible presentation (chest pain, abdominal pain, syncope) + hidden
test/finding results (troponin, ECG, lactate, rebound, free air, …). The model ORDERs findings (small cost each)
then COMMITs a disposition: {discharge, symptomatic_tx, observe, admit, emergent_tx}.

## Structurally different from rx_triage, same geometry (deliberate)
- **Different content/structure:** different presentations, 14 different findings, freshly-authored branching
  trees (chest_pain rel-union 7, abd_pain rel-union 7 with disjoint per-branch modules), diagnostic dispositions.
- **Same reward geometry (mirrored from rx_triage):** identical utility matrix under canonical severity keys
  (`none`=benign, `moderate`=urgent, `high`=emergent) and the same ASK_COST 0.03 / EPS 0.03; the rare high-stakes
  subset (**emergent ≈ 12%**) carries a steep miss penalty; `admit` is the only hedge and is not good on emergent.
- **Framework reused unchanged in logic:** the gate framework had rx_triage's defensive-action names baked in;
  it now reads `ACT_HOLD`/`ACT_ESCALATE` from the env (defaults = rx_triage, so rx_triage Gate 1 is byte-identical).
  Severity tiers use the canonical `none/moderate/high` keys, so no other gate change was needed.

## Gate-1 numbers (`python -m framework.gates.bc_gap --env tasks.diagnostics.env`)
oracle 0.853 vs best-fixed 0.736 → **BC-gap +0.117** (chest_pain +0.119, abd_pain +0.116); emergent prevalence
12.1%; all 10 Gate-1 checks PASS (oracle highest; under-gathering penalized on emergent; benign over-escalation
penalized; shortcuts priced out; oracle selective ≈2.9 asks). Comparable geometry to rx_triage (BC-gap +0.105).

## Next (replication test)
Gate 2 (base support probe) → one SFT (curriculum, v3-style gatherer mix) → one guarded bounded GRPO, judged by
`framework.report`'s ask-collapse + high-risk-reward guards. Hypothesis under test: does diagnostics also show
gates-pass-but-RL-collapses-to-shortcut? Reward-redesign ablation deferred per the rx_triage decision.
