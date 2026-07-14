# Live process-reward integration smoke — 2-node Spark (node A trainer / node B vLLM), deephedge

Config: 3 GRPO steps, 8 cases, `--claude-alpha 0.1`, `--claude-norm zscore_clip`, seed 1. Node A = 192.168.100.1
(trainer, `--base checkpoints/medgemma4b_text`), node B = 192.168.100.2 (`trl vllm-serve` on :8001). 96 rollouts
(32/step). **All checks PASS.**

## Completion & health
- **All 3 training steps completed.** `[grpo] DONE steps=3 wall=1271s`. No NaN. Reward-component log written (96 rows).
- **No trainer/server synchronization errors** — the 2-node weight-sync handshake worked (the single-GPU
  "same CUDA device" error does not occur across two nodes). No server-side errors.
- **GPU:** node B server launched at `--gpu-memory-utilization 0.55`; node A trainer used its own GPU; both
  returned to 0% after the run. Peak memory not queryable on GB10 (unified memory → `nvidia-smi` reports N/A).

## Claude judge calls (per step)
| step | scored | api_calls | cache_hits | retries | parse_fail | api latency/call | score wall |
|---|---|---|---|---|---|---|---|
| 1 | 32/32 | 10 | 22/32 | 0 | 0 | 4.54s | 45.4s |
| 2 | 32/32 | 12 | 20/32 | 0 | 0 | 5.73s | 68.8s |
| 3 | 32/32 | 8 | 24/32 | 0 | 0 | 5.22s | 41.8s |

- **0 retries, 0 invalid responses** across 96 scorings; **cache hit rate 66/96 = 69%** (warm cache from the
  validation phase — a fresh run starts colder).
- Judge scoring is sequential (~5s/call) → **~52s of Claude latency per GRPO step**.

## Wall time & cost for a 60-step run (extrapolated)
- **Wall:** 423.7s/step → **≈ 7.1h for 60 steps** (matches the repo's baseline ~7h/60). Claude adds ~52s/step
  (**≈12% wall overhead**, ~0.9h).
- **Cost:** the smoke made ~10 API calls/step against a warm cache; a fresh run is colder. Bounding: ≤32
  calls/step × 60 = 1920 calls worst case; with dedup/cache realistically ~800–1600 calls. At Opus rates
  (~1.5–2k input + ~300–500 output tokens/call) ≈ **$30–60 for one 60-step run** (C3 only; C1/C2 need no Claude).

## Reward distributions (n=96)
| quantity | mean | sd | min | max |
|---|---|---|---|---|
| analytic reward | +0.357 | 0.266 | −0.26 | +0.97 |
| Claude raw score | +0.042 | 1.952 | −4.0 | +5.0 |
| Claude normalized (within-case) | +0.000 | 0.957 | −2.65 | +2.65 |
| α × normalized | −0.000 | 0.096 | −0.26 | +0.26 |
| **total combined** | +0.357 | 0.277 | −0.43 | +1.08 |

- **Magnitude:** mean|α·norm| = 0.067 = **18.6%** of mean|analytic| (0.362). The process term perturbs, does
  not dominate — bounded, as required.
- **Nonzero process advantage:** 88/96 rollouts (**92%**); per step nonzero_adv_groups 3/4, 4/4, 4/4.
- **Claude score vs trajectory length:** corr(raw, n_asks) = **+0.19** (mild positive — longer work-ups score
  slightly higher, consistent with "more gathering ≈ better process"; not a strong length bias).

## Does a second normalization erase or amplify the process signal?
- The within-case normalization is **mean-preserving**: mean(total) − mean(analytic) = **−1e-5**. The process
  reward redistributes *within* each case group; it never shifts the group mean, so it cannot replace or inflate
  the analytic reward level. ✓
- It is **not erased**: adding α·norm changed the **within-group ordering in 7 of 12 groups**, i.e. the process
  term actively reshapes which rollouts look better inside a group. GRPO's own group-relative advantage then
  normalizes the *combined* reward, folding this reshaped signal into the advantages (rescaled, not removed).
- Net: the process signal survives GRPO's second normalization at a controlled magnitude (~19% of analytic),
  neither washed out nor amplified.

## Worked example (the intended effect)
`dx-5`: a 1-ask commit with high analytic reward (0.97, a case where one ask happened to suffice) — Claude
scored the *process* −4.0 (under-gathering), normalized to −0.64, so total 0.97 → **0.906**: the reward nudges
a lucky under-gathering rollout *down* relative to its group, without overturning its positive analytic reward.

## Verdict
The reward integration runs end-to-end on the two-node setup with no NaN, no sync errors, and clean, correctly
-scaled, mean-preserving process shaping. **Stopping here — the validation alpha sweep is NOT started, pending
review.**
