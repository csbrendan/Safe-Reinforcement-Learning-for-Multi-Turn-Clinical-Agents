# Lessons learned — why this repo is structured this way

Conceptual carry-over from the prior project (`clinical-rx-rl`). **No data, weights, logs, or result files are
carried** — only the durable findings that shaped this design. Representative figures are quoted as motivation.

## 1. A controlled, rule-labeled task is *fully demonstrable* → RL is flat (and that's the anchor)
Single-turn refill triage with deterministic labels and exact-match scoring: SFT strong, RL ≈ flat across
GRPO / RLOO. Not "RL failed" — when labels and reward encode the same rule, the optimal policy is fully
specified by demonstrations, so SFT captures it and RL has no slack. This is the **high-demonstrability
anchor**, and it is *why* the project pivoted from "does RL help?" to "**when** does RL help?"

## 2. Label-noise is the wrong instrument for non-demonstrability
Engineering non-demonstrability by sampling noisy SFT labels from an acceptable set (a `g00→g60` sweep)
produced no reliable signal: across 3 seeds the maximally-demonstrable anchor wobbled as much as the gray
conditions (anchor +0.008 vs gray ~+0.01; slope of gray-advantage vs gray-fraction ≈ −0.005, n.s.). The
reward still had a unique maximizer → demonstrable in principle. **Lesson:** non-demonstrability must come
from *task structure* (hidden state, value-of-information, competing objectives), not injected label noise.

## 3. Measure demonstrability a priori (BC-gap), or you reason in a circle
Don't infer "non-demonstrable" *after* RL helps. Measure it before training: **BC-gap = oracle − best
demonstrable policy**. A first POMDP attempt looked non-demonstrable but its oracle just "asked every lab"
— a *fixed* protocol — so the true BC-gap vs the best fixed checklist was **+0.000**: demonstrable. Caught on
CPU before any GPU. **Lesson:** the demonstrable baseline must be the strongest *fixed* policy, and
adaptivity must beat it.

## 4. Latent support is a second, separate gate (pass@k), measured before RL
RL elicits capability that exists. A flat RL result is ambiguous between "task demonstrable" and "model
lacked support." Gate them separately: BC-gap (demonstrability) **and** pass@k (support). A flat result is
only thesis-relevant when BC-gap is large AND support is present AND the policy actually moved.

## 5. A verifiable reward can be gamed — reward gain ≠ intended behavior
The first bounded GRPO on a genuinely non-demonstrable env *raised mean reward (+0.083) but via fewer asks*:
oracle-grade support didn't rise and the **high-risk safety subset regressed (support 0.286 → 0.000)** —
**ask-cost exploitation / under-gathering**, not adaptive gathering. The 0-ask blind-rate metric missed it
(the failure was *1-ask* under-gathering); the subset + trajectory audit caught it. **Lesson:** judge RL by
*how* it earns reward (subset/audit), never by mean reward alone.

## 6. Reward calibration trades off against demonstrability — fix it structurally
Penalizing under-gathering (steep high-risk penalties + lower ask-cost) fixed the exploit but made a thorough
*fixed* checklist nearly oracle-competitive, **collapsing the BC-gap (+0.112 → +0.050)** — a real
safety↔demonstrability tension. **Resolution: widen the env** (more candidate items, larger rel-union per
family, disjoint per-branch modules) so a safe fixed checklist must over-ask (cost-crushed) or stay short and
miss branches (penalized), while the adaptive oracle walks a short path. This **restored BC-gap to +0.105
under the steep reward, with both failure modes priced out** (under-gathering penalized on high; over-
escalation penalized on benign) — the Task-1 env carried into this repo.

## 7. Process discipline that carries over
Eval-first / gate-first (detect headroom before spending compute); commit results before interpreting; tmux +
intra-run checkpoints for long runs; **proactively verify a long run is actually progressing** (a silent
rsync failure once stalled a run unnoticed for hours); and pre-register success/stop/falsification criteria.
