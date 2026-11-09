# CLAUDE.md

Guidance for working in this repo. It is a **research codebase**, not a product: the deliverable is a single
paper (written up in Overleaf, **outside this repo** — the `.tex` is not here) supported by the gated
experiments that live in this repo. Treat the `docs/` files as the run-by-run lab notebook, but treat **this
file as the source of truth for the paper's final thesis and framing** — the write-up evolved past some of the
older `docs/` phrasing, and this file reflects where it landed.

> **V1 is frozen (tag `arxiv-v1`).** Do not change `paper/main.tex`, `paper/figs/` or existing results. V2 work goes in
> `V2/`, and the V2 manuscript in `paper/v2-extension/`. See "Repository separation" in `V2/PLAN.md`.

## The paper

**Title:** *RLVR for Clinical Multi-Turn Agents: When Verifiable Rewards Improve or Degrade Information
Gathering.* — Author: Brendan P. Murphy (cs.brendan@gmail.com). Single-author workshop-style paper.

**Core question.** Clinical decision-making is a multi-turn *information-gathering* process: the agent must
decide **what to ask, when to stop asking, and which action to commit**. That makes RL with verifiable rewards
(RLVR) an appealing post-training method for clinical agents — and also a **safety risk**. The paper asks: when
does RL improve the agent's information-gathering *procedure*, and when does it learn to **exploit the reward**?

**Central thesis.** Safe RL in this setting requires *more than a verifiable final reward*. Plain GRPO trained
only on the terminal outcome reward often learns **unsafe shortcut procedures** — it raises average reward while
asking too little, collapsing to one-question policies, or hedging with an intermediate action that scores
acceptably on common cases while **failing rare high-risk cases**. In verifiable clinical settings, safe
post-training needs four things together:

1. **Adaptive headroom** — the task must genuinely reward asking (a fixed action must not already be near-optimal).
2. **Useful initialization** — the init policy must be able to *sample* the desired gather-then-commit behavior.
3. **Reward geometry** — the reward must give **dense credit to information gathering**, not only the final action.
4. **Procedure-level evaluation** — audit the **rare high-risk subset separately**, never trust aggregate reward.

**The one-line takeaway:** *RL for clinical multi-turn agents should be evaluated as **procedure learning**, not
final-answer optimization.* A policy can look better on aggregate while becoming worse on the safety-critical subset.

**Three contributions.** (1) A controlled framework: clinical agents as **ask-to-commit decision processes** with
exact verifiers, exact adaptive oracles, and high-risk-subset evaluation. (2) Plain GRPO can **exploit** these
tasks even when RL has a plausible opportunity to help; **credit-assignment GRPO** improves and *stabilizes* the
main diagnostics result across three seeds. (3) Stronger exploration and SFT initialization create tradeoffs:
decoupled GRPO improves high-risk support but **over-gathers**; the credit coefficient must be **calibrated per
task**; and RL does **not reliably repair a biased SFT coverage gap** even when the missing high-risk action can
sometimes be sampled.

**Scope caveat (state it in any write-up):** the three environments are **synthetic, stylized, rule-labeled**
POMDPs — *not* clinical validation benchmarks and not deployment evidence. Their purpose is to *isolate* how RL
changes information-gathering behavior under verifiable reward.

## Key results (memorize these numbers)

All held-out, hidden condition, **pure verifier reward** (no shaping at eval), k=12. Primary safety metrics are
**high-risk value** and **high-risk oracle-grade support@12**.

**Diagnostics (the main result; plain vs credit-assignment are 3-seed mean±std):**

| policy | overall val | high val | high supp | moderate supp | asks |
|---|---|---|---|---|---|
| base | +0.251 | +0.160 | 0.50 | 0.26 | 2.5 |
| plain GRPO | +0.285±0.027 | +0.098±0.037 | **0.33±0.29** (one seed →0.0) | 0.38±0.18 | 1.82±0.11 |
| **credit-assignment GRPO** (λ=0.2) | **+0.313±0.024** | +0.157±0.057 | **0.50±0.00** (stable) | **0.67±0.10** | 2.08±0.30 |
| decoupled GRPO (best ckpt) | +0.329 | **+0.248** | **1.00** | 0.63 | 4.4 (over-gathers; benign 0.48→0.04) |

- **Plain GRPO is the central negative result:** diagnostics passes both gates (BC-gap +0.161, base support
  nonzero) yet plain GRPO *regresses* high-risk value and *destabilizes* high-risk support (one seed collapses to 0.0).
- **Credit-assignment GRPO is the main positive result:** dense per-ask credit removes the observed catastrophic
  seed collapse and lifts the moderate subset (where gathering changes the disposition).
- **Decoupled GRPO** (DREAM-inspired) buys max high-risk safety at an over-gathering cost — a valid but *costly*
  operating point, not strictly dominating.

**Titration — credit coefficient λ is a task-specific safety dial.** BC-gap only +0.075 (weak headroom). Plain
GRPO under-gathers (asks 2.76→1.40, 1-ask 0.91; its high-clot supp 1.00 is a degenerate *exploit*). λ=0.03
under-corrects, λ=0.2 over-corrects (asks 4.10 > oracle 2.77). Calibrated **λ=0.1** (predicted by the per-ask law
`λ·ΔH_ask ≳ c`, ΔH≈0.33 nats, c=0.03) lands asks near oracle (2.67), no 1-ask collapse, high-clot support 1.00.
**But overall value stays inconclusive (−0.036)** — small BC-gap means even calibrated gathering has little
value to convert. *The coefficient controls the procedure; the task's headroom controls whether it pays off.*

**Refill triage — a saturated SFT warmup leaves little headroom.** RL runs from a light SFT warmup that already
reaches high-risk val +0.195 / support 1.00. Both plain and credit-assignment GRPO then *fall* to high-risk val
+0.032 / +0.043 and trim asks to ~1.3–1.6 (the retained "1.00 support" is a k-sampling artifact over an
under-gathering policy — flagged **exploit**). Credit-assignment holds high-risk reward at the warmup level at
the *early* checkpoint (+0.205 vs control −0.007) but the benefit doesn't persist. *A small/negative RL gain can
mean SFT already saturated the behavior — not that RL is useless.*

**SFT coverage repair — latent support is necessary but NOT sufficient.** A biased SFT that under-samples
high-risk cases looks strong on aggregate (overall +0.337, support 0.93) but hides a hole (high-risk val
−0.285). Even in the mild-bias case where the high-risk action `emergent_tx` is still sampled at 21% (Gate 2
passes), **neither** credit-assignment GRPO nor decoupled GRPO repairs it — both erode the SFT's strong
aggregate behavior and fail to restore high-risk support. *Sampling the action isn't enough; coverage must be
fixed in the data/reward.*

## Naming map (paper term ↔ code)

- **plain GRPO** = the control: `grpo_disagg.py --infogain-coef 0.0`, pure `env.score` reward.
- **credit-assignment GRPO** = per-ask info-gain shaping: `--infogain-coef 0.2` (0.1 for titration). Reward
  `R_train = env.score + λ·Σ_t ΔH_t`, where `ΔH_t` = entropy reduction of the posterior `risk_dist(fam,
  revealed)` per *new, relevant* ask. Potential-based-shaping spirit; repeated/irrelevant asks get 0 credit.
- **decoupled GRPO** = DREAM-inspired decoupled explore/exploit: `--dream`. One network + a mode token;
  explore phase (ask-only) rewarded for making the latent state recoverable, exploit phase (commit-only, sees
  only gathered findings) rewarded by pure verifier. A hard guard forbids the commit phase from seeing
  privileged μ.
- **oracle-grade** = realized reward within EPS=0.03 of the exact adaptive-oracle value.
- **support@k** = fraction of cases where ≥1 of k=12 rollouts reaches oracle-grade behavior.

## The gated workflow — gates are NEGATIVE SCREENS

The paper's framing: the two gates tell you when RL is *unlikely* to help; **passing them does NOT prove plain
GRPO will be safe** (diagnostics passes both and still exploits). The controllable factor is reward geometry.

```
Gate 1 (CPU, analytic): demonstrability — BC-gap = V_oracle − max_a E_μ U[μ][a] ≥ ~0.10
                                                                        → framework/gates/bc_gap.py
Gate 2 (GPU inference): latent support — support@k of the INIT policy (base or SFT), adaptive not blind
                                                                        → framework/gates/support_probe.py
   ↓ only if both pass (RL has room + behavior to reinforce)
bounded GRPO  →  framework/rl/grpo_disagg.py   (plain / --infogain-coef / --dream)
   ↓
guarded report  →  framework/report.py   (audits the PROCEDURE, not just the score)
```

**Guarded report** flags a run as **exploit** even when mean reward rises: it tracks 1-ask collapse, blind
commits, committing-before-enough-gathered, high-risk regression, and over-gathering. Four-way verdict:
THESIS-POSITIVE | ASK-COST EXPLOITATION | OPTIMIZATION-ONLY | INCONCLUSIVE. **Never judge by mean reward alone.**

Key invariant: **the env / gates / eval stay pure.** All shaping (info-gain credit, DREAM recoverability) lives
ONLY in the GRPO `rollout_func`, added to the trajectory reward. `env.score`/`U`/`bc_gap`/`support_probe` are
never touched, so gates need no re-run and the guarded verdict (does *true* task reward / high-risk support
improve?) stays honest.

## The three tasks (`tasks/`)

All instantiate the same ask-to-commit POMDP: hidden state μ, queryable binary findings, per-ask cost c=0.03,
one terminal commit, verifiable reward `R = U[μ][a] − c·n_asks`, exact adaptive oracle.

| task | findings | commit actions | families | latent state | turn cap | BC-gap |
|---|---|---|---|---|---|---|
| **Refill triage** (`rx_triage`) | 12 | 5 | 2 (ACEi/ARB, NSAID) | severity none/mod/high | 10 | +0.105 |
| **Diagnostics** (`diagnostics`, deephedge) | 18 | 5 | 2 (chest, abdominal) | severity none/mod/high | 12 | +0.161 |
| **Titration** (`titration`) | 6 | 3 (ordered) | 1 (anticoag) | joint clot×bleed (9 states) | 10 | +0.075 |

Diagnostics "deephedge" is deliberately built to expose *hedging*: the high-stakes action needs multiple
supporting findings, so a shortcut can score acceptably on common cases while failing rare emergencies. Select
the variant via `DX_VARIANT=deephedge`; `tasks/diagnostics/env_rv.py` is the revealed-set entry point used by
`--dream`. `tasks/longterm_care/` is an abandoned scaffold (titration replaced it as Task 3).

## Repo layout

```
framework/            # general-purpose, env-agnostic
  envs/base.py            # Env interface every task implements
  parsing.py              # hardened ask/commit JSON parser
  gates/bc_gap.py         # Gate 1 (demonstrability)
  gates/support_probe.py  # Gate 2 (latent support @k); dream_probe.py for the composed decoupled policy
  gates/reward_calibration.py, temp_sweep.py
  rl/sft_warmup.py        # generic multi-turn SFT (LoRA)
  rl/grpo_disagg.py       # disaggregated 2-node GRPO; --infogain-coef, --dream, --max-turns live here
  rl/merge_adapter.py
  infra/serve_b.sh (rollout-node vLLM), prep_text_base.py (drop MedGemma vision tower)
  report.py, plotting.py
tasks/  rx_triage/  diagnostics/ (+env_rv.py)  titration/  longterm_care/(scaffold)
docs/   research_thesis.md, framework.md, results.md, rx_triage_case_study.md, lessons_learned.md,
        eval_runs/ (per-run reports), scope + design docs (dream_*, titration_scope)
scripts/  run_gate1/2, run_dx_gate2, run_sft_gate2, plot_* (dx_4arm_seeds, dream_*, titration_dial, ...)
results/figs/   # NEW runs only (gitignored); paper figures regenerate here
checkpoints/    # LoRA adapters + intra-run ckpts: dh_* (diag), dx_* (diag SFT/repair), ti_* (titration),
                #   grpo_rx_*/sft_rx_* (refill)
```

> Note: some older `docs/` files (esp. `research_thesis.md`, `results.md`, the case study) predate the paper's
> final safety/procedure framing and still phrase the thesis as "when RL helps over SFT" with "reward geometry
> as a third factor." That's the same finding; **this file's framing supersedes them.** Don't rewrite those docs
> unless asked — they're the historical lab notebook.

## Timeline / how it evolved (dates matter)

- **carried-over lessons (pre-repo)** — `docs/lessons_learned.md`: fully-demonstrable tasks make RL flat (the
  anchor); measure BC-gap a priori; latent support is a separate gate; a verifiable reward can still be gamed.
- **2026-06-18 → 06-23** — refill triage (Task 1): both gates pass, SFT frontier v2–v5 + GRPO #1/#2 both
  collapse to 1-ask; temperature diagnostic proves under-gathering is a *learned policy*. Case study frozen.
- **2026-06-20 → 06-22** — credit-assignment plan; first positive result on diagnostics deephedge via
  single-episode info-gain shaping (`rl-thesis-positive-v1`).
- **2026-06-22 → 06-26** — decoupled explore/exploit (DREAM-inspired) on diagnostics (qualified win: max
  high-risk support, over-gathers low-acuity). Titration scoped; Gate 0 CPU design gate passes; λ-sweep
  establishes the safety-dial result.
- **2026-06-28 → 07-07** — paper completeness: related work (SPIRAL, OPSD/self-distillation, DoctorAgent-RL),
  master results table, appendices, multi-seed training curves, **3-seed error bars** on the diagnostics
  plain-vs-credit-assignment comparison + 4-arm seed figure; SFT-coverage repair experiment.
- **2026-07-11 (latest)** — planned **Claude process-supervision** extension (below). The one experiment the
  author *may still run*; **not part of the current paper**. Everything else is done.

## Possible next experiment (beyond the paper): Claude process supervision

Full plan: [docs/claude_process_supervision_experiment.md](docs/claude_process_supervision_experiment.md)
(opened in the IDE). **Planned, not implemented, and not in the current paper.** It adds an *empirical* process
signal alongside the existing *analytic* credit-assignment (info-gain) one: after each rollout, **Claude Opus**
(`claude-opus-4-8`, temp 0) scores each asked turn against a structured JSON rubric; scores reduce to a scalar
process reward added to the GRPO reward. Four conditions, identical except the shaping term: **C1** plain GRPO ·
**C2** GRPO+InfoGain (= the paper's credit-assignment GRPO) · **C3** GRPO+Claude (new) · **C4** both. The central
deliverable is the **Claude-vs-InfoGain analytical comparison** — *does an external LLM's structured judgment
approximate, exceed, or diverge from the analytic information-gain we compute for free?* Boundaries: **not**
distillation, **not** EMA/self-distillation, **not** an SFT/policy-init step — Claude is an off-policy *scorer*
only, folded into the same reward GRPO already consumes.

> That doc references `docs/project-context.md` (§7/§9), which does **not** exist under that name — the
> equivalents are `docs/framework.md` (gates/verdict) and `docs/rx_triage_case_study.md` (reward definition).
> Don't create a stub for it without asking.

## Environment & conventions

- **Python env:** reuse `~/spark1/.venv` (torch / vllm / trl / peft / transformers). Do **not** create a new
  venv — activate that one for any run.
- **Base model:** text-only **`medgemma-4b-it`** (vision tower dropped via `framework.infra.prep_text_base`),
  LoRA adapters. Supplied at runtime via `--base <path-or-hf-id>`; **no weights live in this repo.**
- **Compute:** disaggregated 2-node GRPO — trainer on node A (`192.168.100.1`), vLLM rollout server on node B
  (`192.168.100.2`, `framework/infra/serve_b.sh`, typically `:8001`). `env_mask` excludes simulator tokens from
  the policy gradient. A 60-step bounded run ≈ 7h; DREAM two-phase ≈ 1.7–2×. GRPO group size 8 rollouts/prompt.
- **⚠️ 2-node NCCL weight-sync gotcha (cost hours):** the trainer↔server NCCL collective only establishes on a
  **first `init_communicator` against a freshly-started, fully-ready node B server with node A's GPU free**. The
  *first* training of a session works; the *2nd+* crashes with `NCCL error: unhandled system error` after a
  server **restart** or an eval→train transition before node A's GPU releases. Run each training as a first-init
  against a fresh server (free node A GPU → fresh tmux server → wait health 307 + buffer → train directly); do
  NOT reuse/restart one server across runs. Stuck vLLM procs survive `pkill` (kill by PID; reboot node B if NCCL
  stays broken). Full writeup + working scripts: `framework/procreward/README.md` (Troubleshooting section).
- **Data:** cases generated on demand from the simulator — no external corpus, no human labeling. RL trains on
  300 cases/task (hidden, ρ=0, gen seed 12345); held-out eval is a disjoint draw (seed 100), n=60/condition
  (n=200 for the SFT-coverage experiment). SFT warmups: 800 oracle/checklist trajectories (seed 7). The env's
  analytic `U`/`risk_dist` *is* ground truth — that's what makes reward verifiable and gates CPU-computable.
- **Seeds:** most cells are single-seed (seed 1); the **diagnostics plain-vs-credit-assignment comparison is
  3-seed** (mean±std) — the paper's headline error bars. Single-seed cells are flagged as a limitation.
- **Determinism:** an oracle-grade threshold once wobbled ±0.03 via `PYTHONHASHSEED`-dependent set ordering;
  verify reproducibility across `PYTHONHASHSEED` when touching oracle / `rel_items` paths.
- `results/figs/` and run logs are **gitignored** — new runs only; prior-project numbers are not results of this repo.

## Working norms (from lessons_learned + the paper's discipline)

- **Gate-first / eval-first:** no SFT/GRPO until both gates pass — but remember gates are *negative screens*, not
  a safety guarantee.
- **Judge RL as procedure learning:** a reward gain is only real if it comes from *adaptive gathering*, verified
  by subset + trajectory audits (high-risk value/support, over-escalation, ask distribution). Report the
  high-risk subset separately, always.
- **Don't weaken the SFT baseline** to manufacture an RL win; compare against a strong warmup.
- **Fix reward tensions structurally** (widen the env), not by softening safety penalties (which collapses the BC-gap).
- One **bounded** run first; inspect KL (must move, ≳1e-3) and rollout diversity (`frac_reward_zero_std`) before
  scaling steps/LR. Commit results before interpreting; pre-register success/stop/falsification criteria.

## Common commands

```bash
# Gate 1 (CPU) — regenerate BC-gap for a task
python -m framework.gates.bc_gap --env tasks.diagnostics.env

# Gate 2 (GPU) — latent support probe
bash scripts/run_gate2.sh              # refill; scripts/run_dx_gate2.sh for diagnostics

# SFT warmup + Gate 2
bash scripts/run_sft_gate2.sh

# GRPO (disaggregated): start vLLM on node B, then the trainer on node A
bash framework/infra/serve_b.sh
python -m framework.rl.grpo_disagg --base <model> --max-steps 60 --lr 1e-5 --beta 0.02 \
    --rollout-temp 1.0 --save-steps 20 --n-cases 300 --seed 1 --max-turns 12 \
    --infogain-coef 0.2          # 0.0 = plain control; 0.1 for titration; --dream = decoupled explore/exploit
# DX_VARIANT=deephedge for the diagnostics deephedge variant

# Guarded verdict + paper figures
python -m framework.report ...
python scripts/plot_dx_4arm_seeds.py       # diagnostics 4-arm seed figure (fig:dx4arm)
python scripts/plot_titration_dial.py      # titration λ safety-dial (fig:tidial)
python scripts/make_learning_plots.py      # → results/figs/learning_dashboard/
```

Per-run reports live under [docs/eval_runs/](docs/eval_runs/) — check there before re-running anything; most
experiments are already recorded.
