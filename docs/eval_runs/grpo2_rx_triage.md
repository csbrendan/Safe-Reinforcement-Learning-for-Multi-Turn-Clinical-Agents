# Guarded GRPO #2 (from v3) — rx_triage final report

**Date:** 2026-06-19. Init = v3 (the only SFT gatherer), KL **beta=0.02**, rollout temp **1.0**, max_steps 60,
lr 1e-5, n_cases 300, MAX_TURNS=10, same widened env/reward, eos[1,106], IS off. Wall ~6h. Held-out probe
k=12/n=60 hidden; intra-run checkpoints (20/40) probed for best-checkpoint selection. Guards in `framework.report`.

## Held-out trajectory (hidden)
| stage | reward | support@12 | mean asks | 1-ask | parse | no-commit | high-risk reward | high-risk acceptable |
|---|---|---|---|---|---|---|---|---|
| v3 (init) | +0.318 | 0.97 | **2.78** | 0.539 | 0.950 | 0.196 | **+0.195** | **0.50** |
| step 20 | +0.384 | 0.95 | 1.81 | 0.753 | 0.986 | ~0.01 | −0.007 | 0.39 |
| step 40 | +0.437 | 0.97 | 1.38 | 0.863 | 0.998 | ~0.01 | +0.095 | 0.46 |
| **step 60 (final)** | **+0.431** | **0.97** | **1.31** | **0.874** | **0.998** | **0.004** | **+0.032** | **0.43** |

Analytic refs: oracle 0.854, best-fixed 0.749. Training rollouts (temp 1.0, training cases) read ~2.4 asks late,
but held-out checkpoints show the converged policy gathers far less — the rollout/held-out gap is real.

## Verdict: EXPLOIT / under-gathering — NOT thesis-positive (guarded `framework.report`)
`d_value +0.113, d_support 0.000, d_asks −1.47; high-risk reward 0.195→0.032 (regressed); one_ask 0.874,
mean_asks 1.31 (ask_collapse)`. RL improved **aggregate reward, parse, and termination** over v3, but by
**trimming asks + fixing validity**, NOT by adaptive gathering: gathering collapsed (2.78→1.31), high-risk
acceptable-decisions fell (0.50→0.43), high-risk reward regressed (+0.195→+0.032).

**vs GRPO #1** (no anchor, from a 1-ask-ish init): the KL anchor + gatherer init **mitigated** the collapse
(high-risk reward +0.032 vs −0.77; asks 1.31 vs 1.10) but did **not** prevent it. There was **no mid-run peak** —
asks declined monotonically from step 0; no checkpoint preserved v3's gathering.

## Interpretation (the result)
With a gatherer init, a KL anchor, temperature ruled out, and **both gates passing** (Gate 1 BC-gap +0.105,
Gate 2 latent support), a bounded GRPO **still** collapsed toward the 1-ask shortcut under the ask-cost. So
**BC-gap + latent support are necessary but not sufficient** for bounded RL to realize the gain here: the
reward's credit assignment — certain −0.03 per ask vs a diffuse benefit concentrated in the rare (12%)
high-risk subset — makes ~1-ask a strong attractor for both SFT and RL. Realizing the adaptive-gathering gain
likely needs the gathering benefit made less diffuse (reward design), not more RL steps.
