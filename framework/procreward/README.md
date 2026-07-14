# Claude Process-Reward GRPO — experiment status & plan

**Primary new experiment for the paper.** Use an external LLM judge (**Claude Opus**, `claude-opus-4-8`) as an
*off-policy process-reward signal* for GRPO on the diagnostics **deephedge** task, and test whether it protects
the safety-critical high-risk subset better than plain GRPO — comparably to, or complementary with, the analytic
info-gain ("credit-assignment") reward already in the paper.

> Status as of 2026-07-14: **validation alpha sweep RUNNING** (α∈{0.05,0.10,0.20}, seed 202). Everything before
> it (judge validation, robustness, real-rollout validation, high-risk enrichment gate, 2-node live integration
> smoke) is **complete and PASSED**. No final multi-seed comparison has been launched — that awaits review of the
> sweep + an approved α. Per-phase reports live in `results/procreward_pilot/*.md`.

## Method

```
training_reward = analytic_terminal_reward  +  alpha * normalized_claude_process_reward
```

- **Judge is off-policy and student-visible-only.** After each rollout, Claude scores the trajectory from the
  *student-visible* transcript (presentation + revealed findings + action schema). It is **never** given the true
  risk, hidden findings, oracle action/value, or utility matrix. Rubric → strict JSON with per-step scores +
  `overall_process_score ∈ [-5,5]` + confidence.
- **Within-group normalization** (`framework/procreward/reward.py`): the raw Claude score is normalized *within
  each prompt/case rollout group* (z-score+clip, or centered-rank→[-1,1]) before being added. This removes
  Claude's systematic per-case offset (it is harsh on benign work-ups) so only *within-case ranking* flows into
  the gradient. The term is **added to, never replaces**, the analytic reward, and α is kept small.
- **Purity preserved.** The env / analytic verifier / gates / oracle are untouched; all shaping lives only in the
  GRPO `rollout_func`, exactly like the info-gain path. Evaluation is always on the pure analytic verifier.

## How it evolved today (the pivots)

1. **Claude process-supervision (per-turn scorer)** — original plan (`docs/claude_process_supervision_experiment.md`).
   Feasibility smoke passed (2026-07-12).
2. **→ Distilling Step-by-Step distillation** (`framework/distill/`) — brief pivot to a Claude-teacher SFT arm.
   Phase-0 pilot done: the Claude teacher is *reliable but env-suboptimal* (under-gathers / over-triages vs the
   analytic oracle). **Kept as a secondary SFT baseline, not the primary method.**
3. **→ Claude process-reward GRPO** (`framework/procreward/`, **this module**) — the current primary experiment:
   Claude as an off-policy *trajectory judge* folded into the GRPO reward. Not distillation, not a scorer of the
   final answer only — a trajectory-level process reward.

## Phases & status

| phase | what | status | headline result |
|---|---|---|---|
| **0 — distill pilot** | Claude teacher vs oracle, student-visible only | ✅ done | teacher reliable, but under-gathers/over-triages → SFT baseline only. `results/distill_pilot/pilot_comparison.md` |
| **0b — judge validation** | 64 constructed archetypes, 12 cases | ✅ done | **high-risk concordance 84%, oracle>unsafe 4/4, oracle>under-gather 4/4, 0 safety inversions, JSON 64/64, ordering-invariant**. `judge_report.md` |
| **Task 1 — rubric robustness** | re-score with a fully paraphrased rubric | ✅ done | **Pearson 0.983 / Spearman 0.956, 0/64 moved >1pt, 0 inversions**. `robustness_report.md` |
| **Task 2 — real-rollout judge** | blind-judge 58 real base/plain/infogain trajectories | ✅ done | premature −0.93 vs +0.8; arm ranking matches verifier (infogain>base=plain); 0 high-risk sign disagreements. Under-powered on real high-risk → led to enrichment. `real_rollout_report.md` |
| **Enrichment — high-risk gate** | 936 real high-risk rollouts (val seed 202), 81 unique judged | ✅ **GATE PASSES** | **safe-vs-unsafe pairwise 99% (1077/1088), 0 inversions, per-arm consistent, paraphrase Pearson 0.988**. `highrisk_report.md` |
| **Live integration smoke** | 3 steps, 2-node, α=0.1 | ✅ done, all checks PASS | 3/3 steps, no NaN, no sync errors; mean-preserving; |α·norm| = 18.6% of analytic; 92% nonzero-adv groups; corr(score,len)=+0.19; ~7.1h + ~$30–60 per 60-step run. `live_smoke_report.md` |
| **Validation alpha sweep** | α∈{0.05,0.10,0.20}, seed 202, 30 steps, ckpt/10 | ⏳ **RUNNING** | `scripts/run_alpha_sweep.sh` → aggregate with `scripts/analyze_alpha_sweep.py` |
| **Final multi-seed comparison** | plain / infogain / Claude (3 seeds), TEST seed 100 | ⛔ NOT started | awaits approved α. `scripts/run_procreward_arms.sh` (prepared, `--confirm`-gated) |

### The judge's validated profile (why it's usable)
- **Sharp exactly where it matters:** on high-risk (emergent) cases it orders oracle ≫ unsafe ≫ under-gather
  with **zero inversions** — the subset where plain GRPO exploits (hedge/under-gather on rare emergents).
- **Weak / over-cautious on benign & moderate**, and does **not** reliably penalize over-gathering — but GRPO's
  group-relative advantage + small α + within-case normalization make that ≈neutral, not harmful.
- **Complementary with analytic info-gain:** Claude protects the under-gathering (high-risk) side; info-gain
  handles the over-gathering side. → the 3-arm comparison should fail/help on *different* subsets, not redundantly.

## Infra & conventions (learned today)

- **2-node Spark, disaggregated GRPO is mandatory.** node A = `192.168.100.1` (trainer, this host `spark-b4e7`),
  node B = `192.168.100.2` (`trl vllm-serve` :8001, serving `~/dx_grpo_init`). The single-GPU colocation of
  trainer+server fails with "same CUDA device for multiple roles" — the live smoke **must** run 2-node. Node B
  needs `loginctl enable-linger` so the tmux server survives SSH logout.
- **`claude-opus-4-8` deprecates `temperature`** — never pass it. Determinism rests on fixed prompt + versioned
  rubric + the deterministic reduction; double-labeling showed mean |Δ| = 0.0 (effectively deterministic).
- **API key** at `~/.secrets/anthropic.env` (`source` it; do not export globally — see repo memory).
- **Judge `max_tokens` = 3072** (1024 truncated long over-gather trajectories → 18/58 parse failures; fixed).

## File map

```
framework/procreward/
  judge_pilot.py        # Phase 0b: build archetype trajectories, JUDGE_SYS rubric, judge, concordance/contrasts
  robustness.py         # Task 1: paraphrased-rubric re-score + stability comparison
  real_rollout_judge.py # Task 2: blind-judge real probe-record trajectories vs analytic labels
  collect_highrisk.py   # Enrichment: GPU in-process vLLM, roll out 3 arms on high-risk val cases
  enrich_judge.py       # Enrichment: blind-judge unique high-risk trajectories, safety gate
  reward.py             # normalize_group() + assemble(): within-group norm, non-replacement, logging  [CPU-tested 15/15]
  rollout_judge.py      # RolloutJudge used inside GRPO (cached, instrumented: latency/cache/retries)
  eval_metrics.py       # analytic subgroup metrics from probe records (value, unsafe-high, premature,
                        #   benign-overtriage, one-ask, over-gather, redundant, regret, asks, support)
  smoke_reward.py       # CPU unit smoke of the reward math (no GPU/API)
framework/distill/      # secondary Claude-teacher SFT baseline (teacher_client, label_states, build_sft)
framework/rl/grpo_disagg.py   # + --claude-alpha/--claude-norm/--claude-cache/--claude-log (disabled by default)
framework/gates/support_probe.py  # + --seed (validation vs test separation)
scripts/run_procreward_smoke.sh   # 3-step live smoke (--confirm)
scripts/run_alpha_sweep.sh        # validation α sweep (seed 202)
scripts/analyze_alpha_sweep.py    # aggregate sweep → table + plots + reward-hacking scan + α recommendation
scripts/run_procreward_arms.sh    # PREPARED final 3-arm × 3-seed comparison (--confirm-gated, NOT launched)
results/procreward_pilot/*.md     # per-phase reports (committed); *.jsonl caches/rows are gitignored
```

## How to run

```bash
source ~/.secrets/anthropic.env
export DX_VARIANT=deephedge
PY=~/spark1/.venv/bin/python

# node B (once): serve the rollout model, keep it up
ssh 192.168.100.2 'tmux new-session -d -s prsrv "bash ~/serve_pr.sh > ~/vllm_serve.log 2>&1"'   # trl vllm-serve :8001

# validation alpha sweep (node A) — seed 202 only
bash scripts/run_alpha_sweep.sh
PYTHONPATH=$PWD $PY scripts/analyze_alpha_sweep.py     # → results/procreward_pilot/sweep/alpha_sweep_report.md + plots

# CPU reward-math smoke (no GPU/API)
PYTHONPATH=$PWD $PY -m framework.procreward.smoke_reward
```

## α-selection rule (applied by the analyzer, then human review)
1. Reject any α with clear benign over-triage / over-gathering / instability / aggregate-value regression.
2. Among survivors, prioritize lower **unsafe high-risk** and **premature** commits.
3. Overall value only as a secondary tiebreaker.
4. Prefer the **smaller** α when two settings are practically tied.

## Guardrails / next steps
- **STOP after the validation report.** Do not launch the final multi-seed comparison until the α is approved.
- Final comparison: **plain GRPO · info-gain GRPO · Claude process-reward GRPO** (3 seeds for the Claude arm),
  identical init/data/rollout/seeds, evaluated on the **untouched TEST seed 100** with the guarded report;
  support@12 kept as a capability diagnostic only.
- **Do not build the combined info-gain+Claude arm** until Claude-only shows a measurable, stable effect.
- Selection (α + checkpoint) uses **validation seed 202 only**; the test seed is never inspected during tuning.
```
