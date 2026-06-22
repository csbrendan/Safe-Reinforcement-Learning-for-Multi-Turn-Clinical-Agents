# rx_triage — info-gain credit assignment (single-episode DREAM proxy)

**Question.** Port the diagnostics info-gain shaping to rx_triage and test whether the same positive impact
appears. Same method (`R = env.score + λ·traj_infogain`, λ=0.2, RL-only; `env.score`/gates untouched), same
guards (`framework.report`).

**Setup (matched to the canonical control GRPO #2).** Init = `sft_rx_v3_merged` (the v3 SFT gatherer), 2-node
disagg GRPO (trainer A / vLLM B:8001), beta=0.02, rollout-temp 1.0, max_steps 60, lr 1e-5, n_cases 300,
MAX_TURNS=10, save 20, seed 1. Control = `checkpoints/grpo_rx_v3` (coef 0, the existing run); info-gain =
`checkpoints/grpo_rx_infogain` (coef 0.2). Held-out probe k=12/n=60, conditions hidden,anchor, max-turns 10,
**pure env.score**. Proxy validated drop-in on rx_triage (oracle full-path infogain 1.04; 1-ask / irrelevant = 0).

## Verdict: PARTIAL — info-gain beats the control but is NOT thesis-positive on rx_triage.

Unlike diagnostics (THESIS-POSITIVE), no info-gain checkpoint clears the strict bar. **Why the tasks differ:**
diagnostics trained from the **raw base** (high-risk had headroom — info-gain *improved* it); rx_triage trains
from an **already-strong SFT gatherer** (SFT-v3 high-risk reward +0.195, support 1.00, asks 2.8) — so there is no
high-risk headroom to *improve*, only to *preserve*, and `framework.report`'s thesis-positive verdict requires
high-risk improvement. The honest result is in the **checkpoint trajectory**, not the final endpoint.

## Checkpoint trajectory (held-out hidden) — `results/figs/infogain_rx_checkpoint_trajectory.png`

| step | ctrl HR-reward | info-gain HR-reward | ctrl asks | info-gain asks | ctrl 1-ask | info-gain 1-ask |
|---|---|---|---|---|---|---|
| 0 (SFT-v3) | +0.195 | +0.195 | 2.78 | 2.78 | 0.54 | 0.54 |
| 20 | **−0.007** | **+0.205** | 1.82 | 2.17 | 0.75 | 0.66 |
| 40 | +0.095 | +0.121 | 1.38 | 1.57 | 0.86 | 0.79 |
| 60 (final) | +0.032 | +0.043 | 1.31 | 1.60 | 0.87 | 0.79 |

- **Info-gain dominates the control at every checkpoint** on gathering: more asks (1.6 vs 1.31 final) and a lower
  1-ask shortcut rate (0.79 vs 0.87 final).
- **Step-20 is the high-water mark:** info-gain holds high-risk reward at the SFT level (+0.205) while the control
  has already crashed it to −0.007. `framework.report` verdict at IG step-20 = **TERMINATION-ONLY / no exploit**
  (no_commit 0.196→0.036, no ask-collapse, no high-risk regression); control step-20 already regresses high-risk.
- **The preservation erodes with training:** by step 40/60 info-gain also trims asks and regresses high-risk
  (always ≥ control, but enough to trip `HR_REWARD_REG` → EXPLOIT at IG s40/final).

## Four-arm endpoint comparison (final ckpts) — `results/figs/infogain_rx_base_sft_ctrl_ig.png`

| arm | overall | high val | high supp@12 | high asks | moderate val | 1-ask |
|---|---|---|---|---|---|---|
| base | +0.386 | +0.151 | 0.00 | 3.04 | +0.160 | — |
| SFT-v3 | +0.318 | +0.195 | 1.00 | 3.37 | +0.172 | 0.54 |
| GRPO control (final) | +0.431 | +0.032 | 1.00 | 1.77 | +0.356 | 0.87 |
| GRPO info-gain (final) | +0.415 | +0.043 | 1.00 | 2.21 | +0.372 | 0.79 |

At final-vs-final the two GRPO arms look similar overall (both lift moderate, both erode high-risk) — info-gain is
marginally better on high-risk asks (2.21 vs 1.77) and 1-ask rate. The endpoint plot alone under-tells the story;
read it with the trajectory.

## Takeaway (no overclaim)

The diagnostics thesis-positive result **does not cleanly replicate** on rx_triage. Info-gain credit assignment
**consistently reduces the 1-ask shortcut and preserves more gathering + high-risk performance than the plain-reward
control**, and at the comparable early checkpoint it fully preserves high-risk reward — but the benefit is
**transient** and no checkpoint reaches THESIS-POSITIVE. The decisive structural factor is the **init**: with an
already-saturated SFT gatherer there is no high-risk headroom for shaping to convert into a measured win, and
prolonged bounded GRPO erodes gathering regardless. This is consistent with the broader thesis (reward geometry is
necessary but interacts with the init / available headroom), and motivates either (a) selecting the early checkpoint,
(b) a larger λ or potential-based variant tuned to hold high-risk, or (c) testing info-gain from the **raw base** on
rx_triage (as in diagnostics) to isolate the headroom factor.
