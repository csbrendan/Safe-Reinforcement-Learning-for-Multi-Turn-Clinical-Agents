# rl-gains-in-clinical-tasks

**Mapping the conditions for RL success in verifiable clinical tasks.**

The question is not *whether* reinforcement learning (RL) can improve a clinical model, but **when** it is
justified over supervised imitation. We operationalize a falsifiable answer: among **verifiable** clinical
tasks, RL's marginal value over a strong SFT baseline tracks two *a-priori-measurable* properties of the
task–model pair:

1. **Demonstration under-specification** — how far the available demonstrations fall short of the
   reward-maximizing policy. Measured by the **BC-gap** = value(oracle) − value(best demonstrable/fixed policy).
2. **Latent support** — whether the model can already *sample* reward-bearing behavior. Measured by **pass@k**.

These become **pre-RL gates**: RL is expected to help only when the BC-gap is meaningfully positive *and*
latent support is non-trivial — and the framing is falsifiable (RL flat when both hold, and the policy
genuinely moves, refutes the thesis for that task). See [docs/research_thesis.md](docs/research_thesis.md).

## Layout
```
framework/        # general-purpose, env-agnostic
  envs/base.py        # the Env interface tasks implement
  parsing.py          # hardened ask/commit JSON parser
  gates/bc_gap.py             # Gate 1 (demonstrability): BC-gap + policy/subset table + checks
  gates/support_probe.py      # Gate 2 (latent support): pass@k probe (adaptive-vs-blind discriminators)
  gates/reward_calibration.py # reward sweep so adaptive gathering is the true optimum
  rl/sft_warmup.py            # generic multi-turn SFT (LoRA)
  rl/grpo_disagg.py           # disaggregated 2-node on-policy GRPO
  rl/merge_adapter.py         # LoRA merge
  infra/serve_b.sh            # rollout-node vLLM launcher
  report.py           # SFT-vs-RL diagnostic + four-way verdict
  plotting.py         # fresh plots (RL-gain vs BC-gap, subsets, safety, ask-dist)
tasks/
  rx_triage/          # Task 1 — adaptive-workup POMDP (Gate-1-passing); env.py + gen_data.py
  diagnostics/        # Task 2 — scaffold
  longterm_care/      # Task 3 — scaffold
docs/                 # thesis, framework, experiment plan, lessons learned, results
scripts/              # thin runners wiring framework + a task
results/figs/         # populated by new runs only (gitignored)
```

## Workflow per task (gated)
`Gate 1 (CPU, BC-gap >= ~0.10 + reward ordering)` -> `Gate 2 (pass@k latent support; adaptive, not blind)` ->
**only then** one bounded GRPO -> diagnostic report + verdict. No SFT/GRPO until Gate 1 and Gate 2 pass.

## The three tasks (placed on the demonstrability axis)
- **RX triage** (Task 1) — multi-turn refill triage with hidden labs; the validated env (Gate-1-passing).
- **Diagnostics** (Task 2) — sequential diagnostic work-up under uncertainty (scaffold).
- **Long-term care** (Task 3) — longitudinal management with competing objectives (scaffold).

## Status
Task-1 env passes Gate 1 (BC-gap +0.105 under the recalibrated safety-steep reward). Tasks 2–3 scaffolded.
No data, weights, logs, or prior results are carried over — see [docs/lessons_learned.md](docs/lessons_learned.md)
for the conceptual carry-over (why we restructured).

## Setup
Reuses the existing `~/spark1/.venv` (torch / vllm / trl / peft / transformers). The base model is supplied at
runtime (`--base <path-or-hf-id>`); no weights live in this repo.
