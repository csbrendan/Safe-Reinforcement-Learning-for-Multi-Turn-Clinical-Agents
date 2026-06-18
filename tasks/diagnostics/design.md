# Task 2 — Diagnostics (sequential work-up under uncertainty)  [scaffold]

**Goal:** sequential diagnostic work-up — choose which test to order, when to stop, when to commit a
diagnosis/management action. Many candidate tests; only a few decision-relevant per case; *which* depends on
revealed results (adaptive value-of-information). Verifiable reward = outcome utility with a steep penalty for
missing a dangerous diagnosis; ask-cost calibrated so adaptive selection beats any fixed test panel.

**Expected demonstrability:** LOW (information value is the whole task) → a candidate where RL should help if
the thesis holds.

**Build order:** implement `env.py` (ITEMS/priors/branching trees/U + analytic helpers, mirroring
`tasks/rx_triage/env.py`) → Gate 1 (`framework.gates.bc_gap`, require BC-gap ≥ ~0.10 with safety priced in) →
reward calibration if needed → Gate 2 (pass@k support, adaptive not blind) → one bounded GRPO → report/plots.

**Possible grounding:** mine realistic test/finding co-occurrence + dangerous-miss rates to set priors and U
(distributions only; no patient data in repo).
