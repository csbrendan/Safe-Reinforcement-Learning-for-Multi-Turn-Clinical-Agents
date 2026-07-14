# Design doc — full two-episode DREAM (decoupled explore/exploit) for adaptive gathering

**Status: DESIGN ONLY. Not implemented, not run.** Gate to build: the single-episode info-gain proxy is
already THESIS-POSITIVE (`docs/eval_runs/diagnostics_infogain_credit_assignment.md`,
tag `rl-thesis-positive-v1`). This doc scopes whether full DREAM is worth the extra machinery, measured
against that result.

## 0. Baseline we are comparing against (single-episode info-gain)

One policy does ask→commit in a single episode. Dense shaping `R = env.score + λ·infogain`, where `infogain`
is the **deterministic** entropy reduction of `env.risk_dist(fam, revealed)` (a hand-derived VOI proxy, not
learned). Result on deephedge: held HR support 0.50 / HR reward ≈base, asks ~2.2 (no 1-ask collapse), moderate
support 0.26→0.77. Two known limits: (a) the proxy needs an analytic `risk_dist` — it doesn't scale to envs
where the posterior isn't closed-form; (b) it gives **no** misleading-anchor resistance (anchor condition flat).

## 1. What new hypothesis DREAM tests (vs single-episode)

DREAM (Liu et al. 2021) decouples a separate **exploration** policy π_exp from an **exploitation** policy
π_task, tied by a learned task encoder/decoder `q(z | trajectory)`. The single-episode result already showed
*dense credit for asking* prevents collapse. The genuinely new hypotheses DREAM tests:

- **H1 (decoupling removes gradient interference).** In one-policy GRPO, the "learn to ask" and "learn to
  commit" gradients share parameters and fight; the policy settles at ~2.2 asks, well under the oracle VOI
  (~3.9). Hypothesis: training π_exp purely on an intrinsic recoverability reward — with π_task's commit
  gradient *not* flowing into it — lets exploration push toward oracle-VOI gathering without the commit head
  dragging it back toward shortcuts.
- **H2 (a learned decoder generalizes past the analytic proxy).** Replace the deterministic
  `H(risk_dist)` with a learned `q(μ | revealed findings)`. Hypothesis: a learned decoder matches the proxy
  on deephedge AND extends to envs without a closed-form posterior (where the single-episode method can't be
  applied at all) — i.e. it tests whether the *credit-assignment mechanism* is the load-bearing part, not the
  hand-derived entropy.
- **H3 (does decoupling buy anything on anchor?).** Weak/secondary: a privileged-z exploitation policy might
  expose whether anchor failure is an *exploration* failure (never gathers the disconfirming finding) or an
  *exploitation* failure (gathers it but commits to the anchor anyway). Diagnostic value even if DREAM doesn't
  fix it.

If H1/H2 don't beat the single-episode numbers, DREAM is not worth the cost — the proxy already wins.

## 2. Does it need separate exploration/exploitation episodes?

**Yes — that is the defining change.** Per case μ:
- **Exploration episode.** π_exp runs the ask dialogue (no commit), producing revealed findings τ_exp.
  Reward = `log q(μ | τ_exp)` (or MI estimate) — how recoverable the true task identity is. No `env.score`.
- **Exploitation episode.** π_task is re-instantiated on the *same case* with τ_exp's findings pre-supplied
  (or conditioned on `z = encode(τ_exp)`), and must commit. Reward = `env.score` (pure clinical reward).
  During training π_task may be conditioned on a **privileged** z derived from the full assignment, which is
  exactly how DREAM avoids the chicken-and-egg; at eval it is conditioned only on what π_exp actually gathered.

This is strictly more than the single-episode version, which never replays the case and never separates the
two reward signals.

## 3. Code changes required

| Area | Single-episode (today) | Two-episode DREAM (new) |
|---|---|---|
| Policies | 1 LoRA adapter | 2 adapters (π_exp, π_task), or 1 policy + mode token |
| Reward | `env.score + λ·traj_infogain` (deterministic) | π_exp: `log q(μ\|τ_exp)`; π_task: `env.score` |
| Decoder | none (uses `env.risk_dist`) | learned `q(μ\|revealed)` head + its own optimizer/loss |
| Env | ask→commit in one pass | **new entry point**: replay a case with a pre-seeded `revealed` set, commit-only prompt |
| Rollout | single pass in `make_rollout` | two-pass: explore pass collects τ_exp, exploit pass commits given τ_exp; tag each trajectory's reward route |
| Trainer | `reward_echo`/`traj_reward` scalar plumbing | per-episode reward routing + decoder train step interleaved with GRPO |

Concretely: (a) add `env_rv` support for "commit given a fixed revealed-set" (the memoized
`risk_dist(fam, revealed)` already exists; need the prompt that presents gathered findings + a commit-only
turn cap); (b) a small decoder module `q(μ|revealed)` (μ = true severity / disposition class — a few-way
classifier) trained on (revealed, μ) pairs from rollouts; (c) two-pass `make_rollout` that returns explore
trajectories scored by the decoder and exploit trajectories scored by `env.score`; (d) keep all shaping/decoder
training **out of `env.score` and the gates** (same purity rule that let us skip Gate re-runs for info-gain).

## 4. How evaluation stays comparable to the existing guards

The deployed system is still a single ask→commit dialogue: run π_exp to gather, then π_task to commit, on one
case. So **the existing probe and guards apply unchanged**:
- Same `support_probe` (k=12, n=60, conditions hidden,anchor, max-turns 12, **pure env.score**).
- Same `framework.report` guards: ask-collapse (`one_ask≥0.85 or asks≤1.3`), HR reward regression (>0.05),
  HR support collapse (>0.15). THESIS-POSITIVE bar = same as single-episode.
- Decoder + intrinsic reward are **training-only**, exactly like `traj_infogain`, so `env.score`/Gate 1/Gate 2
  stay pure and need no re-run.
- **Critical eval rule:** evaluate the *composed* π_exp→π_task policy on the gathered findings only. Never
  probe the privileged-z exploitation policy in isolation — it has seen true μ and would post inflated,
  non-deployable support. Add a guard/assert that the eval path supplies π_task only π_exp's revealed set.

Comparability is therefore 1:1 with the info-gain result: identical probe, identical guards, identical
deephedge env; the only difference is how the policy was trained.

## 5. Expected cost / runtime

- **Rollout cost ~1.7–2×** the single-episode run (explore pass + exploit pass per case). Single-episode was
  ~7h / 60 steps on the 2-node Spark setup → **~12–16h / 60 steps** for DREAM, plus a cheap decoder train
  step interleaved (small classifier, negligible GPU vs the 4B policy).
- **Memory:** two LoRA adapters ≈ one extra r32 adapter (small); the 4B base is shared. Fits the same A/B
  split; vLLM server on B serves the base for both passes.
- **Dev cost:** ~1–2 days. Non-trivial new surface: env replay entry point, decoder + its loss, two-pass
  reward routing.
- **New risks to budget for:** (i) reward-scale mismatch between `log q(μ|·)` and `env.score` (needs
  normalization/tuning, like λ for info-gain); (ii) distribution shift — π_task trains on privileged z but
  deploys on π_exp's imperfect gathering; (iii) decoder collapse / shortcut (decodes μ from spurious cues) —
  needs an information-bottleneck regularizer, the part of DREAM most likely to misbehave.

## 6. Recommendation

Build only if we want to test **H1 (decoupling > shared-parameter shaping)** or **H2 (learned decoder
generalizes beyond analytic `risk_dist`)** — e.g. before porting credit assignment to an env without a
closed-form posterior, or to chase oracle-VOI gathering the single-episode policy plateaued below. If the
near-term goal is just "fix the deephedge exploit," that is **already done** at `rl-thesis-positive-v1` and
DREAM adds cost without a new claim. Decision point: pick H1 or H2 as the primary question first; the env-replay
+ decoder scaffolding is the same either way and is the gating work.
