# Curriculum v3 SFT Gate 2 — rx_triage

**Date:** 2026-06-18. **What:** SFT on curriculum v3 (generator `ac277ea`: ~70% rho=0 full-checklist, 18%
rho=0.4, 12% commit-now; drops the 1-ask-prone rho=0.7 bucket) to fix the v2 under-gathering, then re-run Gate 2.
Base = text-only medgemma-4b-it; SFT 2 ep / 300 steps; probe k=12, n=60, MAX_TURNS=10, deterministic oracle.
Outputs `results/probe_sft_v3_t10.*`; raw 5-way table `sft_v3_gate2_rx_triage.txt`.

## Comparison (hidden) — value / support@12 / mean asks, + key behavior
| policy | reward | support@12 | mean asks | 1-ask | no-commit | parse | high reward | high supp | high asks | benign over-esc |
|---|---|---|---|---|---|---|---|---|---|---|
| base | +0.386 | 0.58 | 2.19 | 0.73 | 0.085 | 0.994 | +0.151 | 0.00 | 3.0 | 0.181 |
| smoke-SFT | +0.168 | 0.60 | 4.48 | 0.44 | 0.304 | 0.985 | −0.428 | 0.14 | 4.7 | 0.022 |
| GRPO #1 | +0.261 | 0.85 | 1.10 | 0.98 | 0.004 | 1.000 | −0.770 | 0.43 | 1.1 | 0.029 |
| improved-SFT (v2) | +0.409 | 0.93 | 1.33 | 0.90 | 0.010 | 0.998 | −0.080 | 1.00 | 1.4 | 0.13 |
| **v3-SFT** | **+0.318** | **0.97** | **2.78** | **0.539** | **0.196** | **0.9499** | **+0.195** | **1.00** | **3.37** | **0.101** |
| best-fixed (analytic) | +0.749 | — | ~4–6 | — | — | — | — | — | — | — |
| oracle (analytic) | +0.854 | — | 2.94 | — | — | — | — | — | — | — |

v3 oracle-grade ask-hist (hidden): overall 0.74@1 / 0.18@3+; **high 0.35@1 / 0.53@3+** → on high-risk, the
majority of oracle-grade successes gather 3+ asks. v3 mean asks are **risk-graded** (high 3.37 > moderate 2.91 >
benign 2.43) — genuine adaptive selection, not a fixed length.

## What v3 fixed (the v2 problem) — and it's the thesis-relevant behavior
- **Ask-collapse gone:** mean asks 1.33 → **2.78** (near oracle 2.94), 1-ask 0.90 → **0.539** (passes the <0.6
  anti-shortcut guard), 41% of rollouts ask 3+ (vs ~9% for v2/GRPO).
- **Adaptive, risk-graded gathering:** asks scale with risk (3.37 high / 2.91 mod / 2.43 benign).
- **Best high-risk safety of any policy:** high-risk reward −0.08 → **+0.195** (now positive, beats even base
  +0.151 and obviously GRPO −0.77), high-risk support 1.0, benign over-escalation 0.101.

## What v3 broke — termination/parse regressed (so Gate 2 FAILS)
- **no-commit 0.010 → 0.196**, **parse 0.998 → 0.9499** → **Gate 2 FAILS** the parse≥0.95 check (barely).
- Mechanism: v3's longer trajectories (mean 2.78, high 3.37) compound the ~5% per-turn invalid rate —
  P(≥1 invalid turn) ≈ 1−0.95^3.5 ≈ 0.16–0.20, matching the 0.196 no-commit. Each invalid turn ends the rollout
  before a commit (reward 0).
- Consequence: overall reward 0.409 → **0.318** (down, *not* toward best-fixed 0.749) — the extra ask-cost plus
  the 20% zero-reward no-commits outweigh the better per-commit quality.

## Verdict: NOT a clean pass → do NOT propose GRPO. The SFT is oscillating on the termination↔gathering axis.
| v3 pass criterion | result |
|---|---|
| parse near 1.0 | ✗ 0.9499 |
| no-commit near 0 | ✗ 0.196 |
| non-trivial support@12 | ✓ 0.97 |
| high-risk support preserved/improved | ✓ 1.00 |
| benign over-escalation controlled | ✓ 0.101 |
| 1-ask shortcut check passes | ✓ 0.539 |
| mean asks not collapsed / not ask-all | ✓ 2.78 |
| reward meaningfully toward best-fixed | ✗ 0.318 (down from 0.409) |
| oracle-grade incl. selective multi-ask (hi/mod) | ✓ high 53% 3+ |

**Arc:** smoke-SFT (asks, won't commit) → improved-SFT (commits, won't gather) → v3 (gathers adaptively + best
safety, but commit reliability slips). Each fixes the prior artifact and exposes the other; the SFT is sensitive
to the curriculum's commit-vs-gather balance, **bottlenecked by per-turn output validity** (~95%) over longer
trajectories.

## Recommendation (no GRPO): v4 — hold BOTH gather and commit
Keep v3's gathering depth/risk-grading (it's the behavior we want) and restore termination by attacking per-turn
validity + commit discipline, e.g.: (a) raise per-turn parse reliability so long trajectories still commit —
more training and/or demos that only ever emit valid item names; (b) add explicit "after the checklist, COMMIT"
reinforcement without shortening gathering; (c) consider more epochs. Target: no-commit ≈ 0 AND parse ≈ 1.0 AND
mean asks ~2.8–4 (risk-graded) AND reward toward best-fixed 0.749. Re-run Gate 2; propose a bounded GRPO only
once a baseline both *gathers* and *commits* cleanly. **No GRPO until then.**
