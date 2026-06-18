# Experiment plan — three clinical tasks across the demonstrability axis

Each task is placed at a different point on the demonstrability axis and run through the same gated workflow.
The goal is a quantitative map: **RL gain vs demonstrability (BC-gap), with latent support confirmed**, across
tasks (and later model/data scales), turning "RL sometimes helps" into "RL helps under these conditions."

## Per-task workflow (gated; identical for all three)
1. **Gate 1 (CPU)** — build the env + reward; require BC-gap ≥ ~0.10 on the deep families, reward ordering
   correct (oracle highest; shortcuts << best fixed), under-gathering penalized on high-risk, over-escalation
   penalized on benign, anchor ≈ 0.
2. **Gate 2 (GPU inference)** — pass@k support probe on base + SFT-init; require non-trivial support that is
   **adaptive** (blind-rate ~0), reward variance > 0, clean parse.
3. **Bounded GRPO** — only if both gates pass. One run; inspect KL + rollout diversity before scaling.
4. **Verdict + plots** — four-way report; record (BC-gap, RL−SFT) as a point on the headline curve.

## Task 1 — RX refill triage  (`tasks/rx_triage/`)  [validated]
Multi-turn refill triage with hidden labs; sequentially-adaptive gathering (which item matters next depends on
the previous answer). **Status: Gate 1 PASSES** (BC-gap +0.105 under the recalibrated safety-steep reward;
both failure modes priced out). Next: Gate 2 (re-probe base + new smoke-SFT) → bounded GRPO.

## Task 2 — Diagnostics  (`tasks/diagnostics/`)  [scaffold]
Sequential diagnostic work-up under uncertainty: choose which test to order, when to stop, when to commit a
diagnosis/management action; many candidate tests, few decision-relevant per case, depending on revealed
findings. Outcome/verifiable reward with a steep penalty for missing a dangerous diagnosis. Expected position:
**low demonstrability** (information value is the whole task). Build env → Gate 1 → Gate 2 → GRPO.

## Task 3 — Long-term care  (`tasks/longterm_care/`)  [scaffold]
Longitudinal chronic-disease management with **competing objectives** (efficacy vs side effects vs adherence
vs monitoring burden) over multiple visits; the right plan depends on evolving, partially-hidden state.
Expected position: **low demonstrability + genuine trade-offs** (multiple defensible actions). Build env →
gates → GRPO. Hardest reward design (outcome-based + multi-objective) — sequence last.

## Scaling axes (run across the tasks once the per-task curve exists)
- **Model scale** — 4B → 27B → frontier; test whether the RL-gain↔demonstrability relationship is stable
  (scale also moves *latent support*, so it is doubly relevant).
- **Data realism** — synthetic-grounded → real-data-grounded distributions; test whether the pattern survives.

## Anti-goals
No relaxing the BC-gap bar to manufacture a positive; no softening safety penalties to widen the gap (widen
the env instead); no multi-seed/long campaigns before a single bounded point is positive.
