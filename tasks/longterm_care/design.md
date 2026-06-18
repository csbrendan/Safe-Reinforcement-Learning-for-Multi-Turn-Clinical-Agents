# Task 3 — Long-term care (longitudinal management)  [scaffold]

**Goal:** chronic-disease management across multiple visits with partially-hidden, evolving state. The model
gathers status (ASK) then commits a management action each visit. Multiple actions are often defensible
(set-valued acceptable set). Verifiable reward = multi-objective outcome (efficacy vs side-effects vs adherence
vs monitoring burden).

**Expected demonstrability:** LOW, but with genuine trade-offs — the reward is the hardest part. **Sequence
this task LAST** (after rx_triage and diagnostics validate the framework end-to-end).

**Build order:** define state/actions/U + analytic helpers (mirror `tasks/rx_triage/env.py`) →
`framework.gates.reward_calibration` to find a U where the acceptable-set is non-trivial AND BC-gap ≥ ~0.10 →
Gate 1 → Gate 2 → bounded GRPO → report/plots.

**Open design question:** how to keep the reward verifiable while multiple actions are acceptable — likely
`score` returns max-U over the acceptable set with a steep floor for dangerous actions (already supported by
the `acceptable_hit` return slot in the env interface).
