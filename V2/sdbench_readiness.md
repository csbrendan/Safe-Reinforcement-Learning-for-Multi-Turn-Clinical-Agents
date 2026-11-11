# V2 SD-Bench baseline and RL readiness (MedGemma 4B)

> **Update 2026-11-11:** V2 was refocused on a three-arm comparison with no V1 gate reproduction. Sections 4 and 7 are superseded by [`sdbench_execution_plan.md`](sdbench_execution_plan.md); the infrastructure findings below still hold.

*Audit dated 2026-11-11. No GPU jobs ran, no data was downloaded, and no external API was called. Sources: `V2/benchmark_smoke_report.md`, its logs and scripts, `~/v2_bench/` (SDBench commit `304ce96`) and the server logs in `~/v2_bench/logs/`.*

## Bottom line

**The software is ready to start, but the data is not.** MedGemma 4B runs SD-Bench end to end on the Sparks with local models. However, the only cases on disk are SDBench's 5 synthetic CPC cases, and their README says they are "**not** suitable for reporting actual diagnostic performance". Five cases also can't be divided into train, validation and test sets. So no credible baseline, gate measurement or GRPO run is possible until real cases arrive, from MIMIC-IV-Ext-SDBench or NEJM.

Two further problems need solving before any GRPO run:
- **The reward isn't verifiable.** SD-Bench grades with an LLM judge.
- **V1's Gate 1 and info-gain credit can't be transferred exactly.** SD-Bench has no hidden-state model, utility matrix or exact oracle.

Both can be handled with labeled approximations, set out below.

## 1. Infrastructure readiness

| Component | Status | Evidence |
|---|---|---|
| MedGemma 4B inference (text-only `checkpoints/medgemma4b_text`) | ✅ Verified | Smoke runs on 2026-11-08 |
| Stop token `[1, 106]` | ✅ Verified for SD-Bench | `vllm_doctor2/3.log` served with `--generation-config ~/v2_bench/genconfig_medgemma`. `~/v2_bench/medgemma4b_stopfix` is a symlinked copy with the same config |
| Multi-turn interaction | ✅ Verified, 3 cases | Benchmark-only template `medgemma_merge_roles.jinja` fixes consecutive user messages |
| Local gatekeeper, judge and costing | ✅ Plumbing works | Qwen3-8B on Spark 2, 32k context |
| Judge correctness | ❌ **Unvalidated** | Local judge gave a false positive (Hyperadrenergic Hypertension vs pheochromocytoma/MEN2A) |
| Reward calculation | ⚠️ Partial | Judge score and costs compute, but costs came from a local LLM estimate, not the grounded charge master. No RL reward is defined yet |
| Procedure metrics | ⚠️ Raw data available, not yet extracted | Result JSON has per-turn test and question lists and per-result `is_synthetic` flags. No extractor exists yet |
| Evaluation splits | ❌ Blocked | 5 synthetic cases, all in `train` |
| SD-Bench SFT initialization | ❌ None exists | All adapters in `checkpoints/` were trained on V1's JSON ask/commit tasks |
| GRPO integration | ❌ Not built | `grpo_disagg.py` has no SD-Bench environment adapter. The gatekeeper would run inside `rollout_func` |
| Node stability | ⚠️ Improving | Spark 1 has run 2 days without a crash since 11-09. The cause of its 3 earlier crashes is still unknown |
| V1 stop-token question | ⚠️ Open | Doesn't block SD-Bench, but blocks any V1-vs-V2 comparison |

**Known MedGemma behaviors that need handling:**
- It mixes `<test>` and `<question>` tags in one turn, which triggers a notice and wastes the turn.
- It sometimes fails to diagnose ("Unable to provide diagnosis").
- The doctor server uses `max_model_len` 8192, which a 12-turn transcript can reach.

**Checkpoint to use:** `checkpoints/medgemma4b_text` (text-only `google/medgemma-4b-it`), served with the `[1, 106]` generation config and the merge-roles template.

Don't call any V1 adapter an SD-Bench baseline. If we evaluate V1 adapters on SD-Bench at all, label it "V1 adapter, zero-shot transfer" and report it separately.

## 2. Scientific and implementation blockers

1. **Data (hard blocker).** Two routes:
   - **MIMIC-IV-Ext-SDBench:** 488 cases with official 163/254/71 splits. It needs PhysioNet credentialing for *each person who accesses it*, so Agustina's credential doesn't cover Brendan running jobs on these machines. LLM use must be local; cloud LLMs are allowed only with verified zero retention, which rules out an unapproved Claude or OpenAI judge. Our all-local setup fits.
   - **NEJM:** 304 cases, licensed through elicensing@nejm.org. The terms for training, local storage and who may run jobs are unknown. Confirm them before any file is moved.
2. **Unverifiable reward.** Pass is a judge score of 4 or more on a 5-point scale. The official judge is GPT-5 or o3, which costs API money and is barred for MIMIC unless retention is verified zero. A local judge needs validation against clinician labels before it scores evaluations, and certainly before it becomes an RL reward. Under RL the policy can learn to exploit the judge, so the audit has to check for that.
3. **Gatekeeper non-determinism.** Tests not in the case record get LLM-synthesized results. Across GRPO rollouts that adds reward noise, and it can make two answers to the same test disagree. Fix it with temperature 0 and a cache keyed on case and normalized test name.
4. **No high-risk labels.** The case schema has `final_diagnosis`, `tests`, `initial_presentation` and `full_text`, but no acuity field. A pre-registered high-risk definition is needed **before** any baseline, for example "can't-miss" diagnoses labeled by a clinician blind to model outputs.
5. **Gate 1 and info-gain credit lack exact analogues** (section 4).
6. **Throughput.** Each episode takes minutes, not seconds (section 5). That limits k, the number of cases and the number of GRPO steps.
7. **Turn cap.** The official cap is 70 turns; the smoke tests used 12. A cap below 70 is a deviation from the protocol, so we need to choose one and report it.

## 3. Recommended next steps, in priority order

1. **Choose the data route (your decision; nothing scientific proceeds without it).** I recommend MIMIC-IV-Ext-SDBench through Brendan's own PhysioNet credential: it has official splits and requires local models, which we already use. In parallel, have Agustina ask NEJM whether their terms allow training and local storage.
2. **While waiting, pre-register the protocol** (no data or GPU needed): splits, turn cap, stop config, template, gatekeeper and judge models, temperatures, metric definitions, the high-risk labeling procedure, gate thresholds and stopping rules.
3. **Build the cheap tools** (CPU work, in `V2/`):
   - an extractor for procedure metrics from SD-Bench result JSON;
   - a gatekeeper cache;
   - fixed-strategy agents for Gate 1-E (commit from vignette, fixed test panel, full-information ceiling);
   - a k-sample wrapper for Gate 2-E.

   Test each on the 5 synthetic cases, for plumbing only.
4. **Settle the judge:** choose a local judge and plan its validation, with clinician labels on a stratified sample of validation cases and agreement measured before use.
5. **Resolve the V1 stop-token question** with `stop_token_v1_support_probe.sh` on Spark 2, about one evening. Do this before any claim compares V1 with V2.
6. **Once data arrives:** run the validation baseline, then Gates 1-E and 2-E, then a go/no-go decision on the GRPO diagnostic.

## 4. Gate transfer: what is exact and what isn't

| V1 measurement | Transfers exactly? | Proposed SD-Bench version (label in all reporting) |
|---|---|---|
| **Gate 1, BC-gap** (`V_oracle − max_a E_μ U[μ][a]`) | **No.** There is no μ, U or exact oracle | **Gate 1-E (empirical headroom proxy, not a BC-gap):** score accuracy minus normalized cost for (i) commit from vignette, (ii) a fixed test panel, (iii) the adaptive base policy and (iv) a full-information ceiling (model reads `full_text`). Headroom ≈ ceiling − best fixed strategy; adaptive value ≈ adaptive − best fixed. Measures this model and judge, not an oracle |
| **Gate 2, support@k** | **Mostly.** Swap "oracle-grade" for "judge pass and gather-then-commit" | **Gate 2-E:** fraction of train cases where at least 1 of k rollouts passes the judge after ordering at least 1 test that is in the case record and committing validly. Also report how often the commit-from-vignette strategy passes on those same cases, so success that needed no gathering isn't counted |
| **Info-gain credit** ΔH of `risk_dist` | **No.** There is no analytic posterior | **Case-grounded ask credit (approximate):** credit asks whose result is in the case record (`is_synthetic=False`), new and non-repeated. It measures relevance, not entropy reduction. An entropy alternative from the agent's self-reported differential can be gamed, so I'd avoid it |
| High-risk subset | **No labels** | Clinician-labeled can't-miss subset, pre-registered |
| Guarded report verdicts | **Yes, after redefining the inputs** | Premature commit = diagnosis before any record test is ordered. Unnecessary = repeated tests, or tests not in the record and synthesized. Over-gathering = cost well above the case's documented workup |

## 5. Proposed baseline configuration

| Setting | Proposal |
|---|---|
| Policy | `medgemma4b_text`, stop `[1, 106]`, merge-roles template, `max_model_len` 16384, `max_tokens` 400 |
| Decoding | Baseline at temperature 0 (k=1). Gate 2-E at temperature 1.0 (k=8) |
| Gatekeeper and judge | One fixed local model for every arm, temperature 0, gatekeeper results cached. Official prompt versions (judge `v4_reasoning_with_tests`, to confirm) |
| Turn cap | 12, plus a 70-turn check on 20 cases to show how often 12 binds |
| Splits | Official train/validation/test. **Test stays sealed** until final arms are frozen; no published-style pooling of validation and test |
| Baseline set | All validation cases (MIMIC: 254), or a pre-registered 60-case stratified pilot first |
| Metrics | Judge pass rate (with clinician-validated agreement), mean judge score, cost per case, tests and questions per case, % record-grounded tests, premature-commit rate, unnecessary-test rate, failed or invalid commits, turns used; all reported separately for the high-risk subset |
| Uncertainty | Bootstrap 95% CIs over cases |

## 6. GPU time and memory (rough)

Based on smoke runs: 3.5–9.5 min per case, one worker, 12-turn cap. Aggregate throughput was about 3× single-stream at 8 concurrent requests. These are estimates, so measure real throughput in the first pilot.

| Job | Episodes | Estimated wall time | Nodes and memory |
|---|---|---|---|
| Baseline pilot, 60 validation cases, k=1 | 60 | ≈ 2 h | Spark 1: MedGemma server ≈ 45% (weights ≈ 8 GB). Spark 2: judge and gatekeeper (Qwen3-8B ≈ 16 GB weights, 32k context) ≈ 45% |
| Full validation baseline, k=1 | 254 | ≈ 8–9 h (overnight) | Same |
| Gate 1-E (3 fixed strategies, 60 cases) | 180 (shorter) | ≈ 2–3 h | Same |
| Gate 2-E (30 train cases, k=8) | 240 | ≈ 8 h | Same |
| GRPO diagnostic, 2 arms × 20 steps × 32 episodes | 1,280 | ≈ 1–2 days per arm | Spark 1 trainer: LoRA with gradient checkpointing, likely 40–70 GB with 8–16k-token transcripts (V1's were about 2k). Spark 2: rollout server and gatekeeper/judge together ≈ 70–90 GB |

The GRPO line is the most uncertain. A 60-step V1-sized run (1,920 episodes) would take roughly 2–3 days per arm here, against about 7 h in V1.

## 7. GRPO diagnostic (proposal only, conditional on both gates passing)

- **Arms:**
  - C1, outcome only: judge pass, or judge score, minus λ_c × normalized cost.
  - C2: C1 plus case-grounded ask credit.
  - Both arms share the init, data, seeds, gatekeeper cache, judge and evaluation.
- **Size:** 20 steps, 4 cases × 8 rollouts, seed 1, LR 1e-5, β 0.02, as in V1. Checkpoints every 10 steps.
- **Evaluation:** on validation only, with the guarded report, broken down by high-risk subset, premature commits, unnecessary tests and cost, and a check for judge exploitation (a sample re-scored by a second judge or a clinician).
- **Stop rules:**
  - KL stays below 1e-3 or `frac_reward_zero_std` stays above 0.8 (no learning signal).
  - Any arm drops high-risk pass rate by more than 10 points from base.
  - A judge-exploit signature appears.
- **Requires your approval before launch.** The integration work, an SD-Bench adapter in `rollout_func` with `env_mask` on gatekeeper text, is about 1–2 days of engineering.
