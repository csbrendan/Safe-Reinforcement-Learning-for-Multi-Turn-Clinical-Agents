# Improved-SFT (curriculum) Gate 2 — rx_triage

**Date:** 2026-06-18. **What:** retrain SFT on the curriculum data (generator `cb3bce0`: 100% committing,
19% commit-now, 33% short 1–3 ask, fixed-checklist gathering) to fix the smoke-SFT non-termination, then re-run
Gate 2 only. Base = text-only medgemma-4b-it; SFT = 2 ep / 240 steps; probe k=12, n=60, MAX_TURNS=10,
deterministic oracle. Base probe reused (`probe_base_t10`, unchanged). Outputs: `results/probe_sft_v2_t10.*`;
raw 4-way table `sft_v2_gate2_rx_triage.txt`.

## Comparison (hidden) — all policies vs analytic baselines
| policy | reward | support@12 | mean asks | 1-ask rate | no-commit | parse | high reward | high supp | mod reward | benign over-esc |
|---|---|---|---|---|---|---|---|---|---|---|
| base | +0.386 | 0.58 | 2.19 | 0.73 | 0.085 | 0.994 | +0.151 | 0.00 | +0.160 | 0.181 |
| smoke-SFT | +0.168 | 0.60 | 4.48 | 0.44 | 0.304 | 0.985 | −0.428 | 0.14 | −0.060 | 0.022 |
| GRPO #1 | +0.261 | 0.85 | 1.10 | 0.98 | 0.004 | 1.000 | −0.770 | 0.43 | +0.130 | 0.029 |
| **improved-SFT** | **+0.409** | **0.93** | **1.33** | **0.90** | **0.010** | **0.998** | **−0.080** | **1.00** | **+0.498** | **0.13** |
| best-fixed (analytic) | +0.749 | — | ~4–6 | — | — | — | — | — | — | — |
| oracle (analytic) | +0.854 | — | 2.94 | — | — | — | — | — | — | — |

(improved-SFT high-risk oracle-grade ask-hist: 0.58 @1 / 0.15 @2 / **0.27 @3+** — i.e. it *does* gather more on
some high-risk cases, unlike GRPO #1's flat 1-ask.)

## What improved (vs smoke-SFT) — substantial
- **Termination fixed:** no-commit 0.304 → **0.010**, parse 0.985 → **0.998**.
- **Reward up to +0.409** — best of all learned/base policies (> base 0.386, GRPO 0.261, smoke 0.168).
- **High-risk reward −0.428 → −0.080** (best of the learned policies; GRPO was −0.770) and high-risk support → 1.00.
- moderate +0.498, support broadly ↑.

## What's wrong — over-corrected into under-gathering (FAILS Gate 2)
**Gate 2 result for improved-SFT: FAIL** — on "no collapse to 1-ask shortcut" (one_ask **0.90** ≥ 0.6; mean asks
**1.33**). The curriculum's commit-now + short emphasis (60% of demos had ≤3 asks) pushed the model to **ask ~1
then commit even on hidden (rho=0) cases**, where the demonstrable policy should gather the full checklist. Tell:
**improved-SFT reward 0.409 < best-fixed 0.749** — it did *not* clone the gathering policy it was trained on; it
under-fit the asks. (`framework.report` smoke→improved also flags EXPLOIT via the new ask-collapse guard.)

So we traded one SFT artifact (non-termination, over-asking) for another (under-gathering, 1-ask-dominant).
It is a *better policy* than smoke-SFT/GRPO on reward and high-risk safety, but it is **not a clean baseline**:
it fails the anti-shortcut guard and sits well below the demonstrable best-fixed value.

## Interpretation vs the pre-registered rules
- (a) "fixes commit-rate AND avoids 1-ask collapse, preserving high-risk support" → **partial**: commit-rate and
  high-risk fixed, but it did **not** avoid the 1-ask collapse (Gate-2 FAIL).
- (b) "closes most headroom" → **no** (0.409 ≪ best-fixed 0.749 / oracle 0.854).
- (c) "improves termination, leaves a real gap with latent support → propose (not launch) GRPO #2" → the gap +
  latent support are real, **but** the residual 1-ask under-gathering means a naive GRPO #2 (ask-cost-penalized)
  would most likely *amplify* the shortcut, repeating GRPO #1. So GRPO #2 is **not** the right next move yet.

## Recommendation: refine the curriculum (v3) before any RL — do NOT propose GRPO #2 yet
Root cause is the demo mix, not the idea. **Curriculum v3:** raise the rho=0 full-checklist-then-commit share
(e.g. ~70%) and cut commit-now/short to the minimum that still teaches reliable termination, so the SFT learns
to **gather the checklist AND commit** (target: reward ≈ best-fixed 0.749, mean asks ~4–6, pass the anti-shortcut
guard, keep high-risk support). Re-run Gate 2. Only once the baseline both *gathers* and *commits* (a clean Gate-2
pass) does proposing a bounded GRPO make sense — otherwise RL is again just fixing an avoidable SFT artifact.

**Status: no GRPO #2.** Awaiting review of this committed improved-SFT Gate 2 before the v3 refinement.
