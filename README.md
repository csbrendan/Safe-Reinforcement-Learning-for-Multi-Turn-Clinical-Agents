# Safe Reinforcement Learning for Multi-Turn Clinical Agents: When Dense Rewards Beat Outcome Rewards

Brendan P. Murphy · Agustina Saenz, MD, MPH

📄 Paper: arXiv preprint (link added on announcement) · source and PDF in [`paper/`](paper/)

---

Clinical decision-making is a multi-turn information-gathering problem. An agent must decide **what to ask**,
**when to stop asking**, and **which action to commit to**. Reinforcement learning with verifiable rewards
(RLVR) is an appealing way to train such agents. It is also a safety risk: a policy can raise its average
reward by learning shortcut procedures that fail on rare high-risk cases.

This repository contains the code for our study of **when RL improves the information-gathering procedure and
when it exploits the reward**.

We train MedGemma-4B with GRPO in three controlled ask-to-commit environments, each with exact verifiers and
exact adaptive oracles. We compare three ways of rewarding the policy:
- **outcome-only** rewards;
- **credit-assignment** rewards that give dense, per-question information-gain credit;
- **decoupled** explore/exploit training (inspired by DREAM).

## Key findings

- **Outcome-only GRPO can exploit the task.** In acute diagnostics it raises aggregate value while asking fewer
  questions. It also degrades high-risk performance, and one of three seeds loses high-risk support entirely.
  This happens even though the task passes both of our pre-RL screening gates.
- **Dense credit assignment helps in diagnostics.** Per-question information-gain credit improves held-out value
  and moderate-case support across three seeds, without the high-risk seed collapse.
- **Exploration trades off against burden.** Decoupled explore/exploit training gives the highest high-risk
  support, at its best checkpoint, but over-gathers on low-acuity cases.
- **The credit coefficient must be calibrated per task.** In titration it controls asking behaviour, but a
  calibrated value still cannot create value the task does not offer.
- **SFT limits what RL can add.** After a supervised warmup that already gathers carefully, RL trims questions
  and degrades high-risk performance. RL also does not reliably repair a coverage gap inherited from biased
  supervised data.

The takeaway: **evaluate RL for clinical agents as procedure learning, not final-answer optimization.** Always
audit the rare high-risk subset separately from aggregate reward.

> **Scope.** The three environments are synthetic, stylized, rule-based simulators. Their parameters were
> hand-designed for controlled experiments; they are not clinical validation benchmarks. Nothing here is
> evidence that a trained policy is suitable for patient care.

## The three environments

All three share one interface: a hidden state, binary findings the agent can query at a cost of 0.03 per
question, one terminal commit, and a verifiable reward `R = U[state][action] − 0.03 · n_asks`.

| Task | Code | Findings | Actions | Hidden state |
|---|---|---|---|---|
| Medication refill triage | `tasks/rx_triage/` | 12 | 5 | severity (none / moderate / high) |
| Acute diagnostics (main task) | `tasks/diagnostics/env_rv.py`, `DX_VARIANT=deephedge` | 18 | 5 | severity (none / moderate / high) |
| Dosing titration | `tasks/titration/` | 6 | 3 (ordered) | joint clot × bleed risk (9 states) |

Cases are generated on demand by the simulators. No external clinical data is used.

## Repository layout

```
framework/              environment-agnostic code
  envs/base.py            Env interface implemented by every task
  parsing.py              ask/commit JSON action parser
  gates/                  pre-RL screens: bc_gap.py (Gate 1), support_probe.py (Gate 2)
  rl/                     sft_warmup.py, grpo_disagg.py (plain / credit assignment / decoupled), merge_adapter.py
  infra/                  serve_b.sh (vLLM rollout server), prep_text_base.py (text-only MedGemma)
  report.py               guarded report: audits the procedure, not just the score
tasks/                  rx_triage/, diagnostics/, titration/  (longterm_care/ is an unused scaffold)
scripts/                experiment runners and figure scripts
figures/                make_fig_dialogues.py (paper Figure 2)
paper/                  LaTeX source, bibliography and compiled PDF of the preprint
docs/                   design notes and per-run reports (docs/eval_runs/), the project's lab notebook
V2/                     extension: plan, experiment designs, benchmark smoke tests
```

Model weights, checkpoints, generated data and run logs are not tracked.

## Setup

- **Python:** 3.12 with PyTorch, Transformers, TRL, PEFT and vLLM. The current environment has torch 2.11,
  transformers 5.8, trl 1.5.1 and vLLM 0.20.1.
- **Model:** text-only `medgemma-4b-it`, adapted with LoRA. Prepare it with `framework/infra/prep_text_base.py`
  and pass its path with `--base`.
- **Hardware:** a disaggregated two-node setup, with GRPO training on one node and vLLM rollout generation on the
  other.

## Running the pipeline

The workflow is gated. Both gates are *negative screens*: they flag when RL is unlikely to help, but passing them
does not guarantee that outcome-only RL will be safe.

```bash
# Gate 1 (CPU): adaptive headroom (BC-gap) of a task
DX_VARIANT=deephedge python -m framework.gates.bc_gap --env tasks.diagnostics.env_rv

# Gate 2 (GPU): latent support of the initial policy
bash scripts/run_dx_gate2.sh

# GRPO: start the rollout server on node B, then train on node A
bash framework/infra/serve_b.sh <model_dir> 8001
DX_VARIANT=deephedge python -m framework.rl.grpo_disagg --base <model> --max-steps 60 --lr 1e-5 --beta 0.02 \
    --rollout-temp 1.0 --n-cases 300 --seed 1 --max-turns 12 \
    --infogain-coef 0.2      # 0.0 = outcome-only; 0.1 for titration; add --dream for decoupled

# Figures
python scripts/plot_dx_4arm_seeds.py
python scripts/plot_titration_dial.py
python figures/make_fig_dialogues.py
```

Exact settings for every reported run, and any caveats about them, are recorded in
[`docs/eval_runs/`](docs/eval_runs/).

## Ongoing work

An extension of this study is in progress. It tests whether the findings generalize to MedGemma-27B, to an
external interactive clinical benchmark (AgentClinic is the leading candidate), and to clinician-reviewed
trajectories. It also adds larger risk-stratified evaluation sets and checkpoint selection on a separate
validation split.

See [`V2/PLAN.md`](V2/PLAN.md) and [`V2/agentclinic_experiment_plan.md`](V2/agentclinic_experiment_plan.md).

## Citation

```bibtex
@misc{murphy2026saferl,
  title  = {Safe Reinforcement Learning for Multi-Turn Clinical Agents: When Dense Rewards Beat Outcome Rewards},
  author = {Murphy, Brendan P. and Saenz, Agustina},
  year   = {2026},
  note   = {arXiv preprint}
}
```

## Contact

Brendan P. Murphy, cs.brendan@gmail.com
