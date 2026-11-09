# V2 external benchmark smoke-test report

*Run 2026-11-08 on the local DGX Sparks. This is an integration feasibility study. **It is not a scientific evaluation.** Every sample is 1–5 cases, so no number here is a performance estimate or a comparison between models. No training ran, no paid API was called, and no restricted data was downloaded.*

## Executive summary

**All four benchmarks ran end to end on our Sparks with local models, but only Medmarks' verifiable tasks produced official scores.** AgentClinic and HealthBench can run a 50-case evaluation after small fixes. SDBench's software works, but it cannot produce a reportable evaluation without NEJM-licensed or PhysioNet-credentialed cases.

**Recommended first V2 evaluation:** AgentClinic-MedQA, with Medmarks as a quick supporting baseline. It is the only one of the four that is interactive, needs no credentials, and has enough public cases.

**The morning's looping failure is explained and fixed.** Our text-only MedGemma `generation_config.json` stops only at token 1, not at `<end_of_turn>` (106), so the model kept writing after finishing its turn. A one-line server-side configuration fixes it without changing any checkpoint.

**Open question for V1: unresolved.** V1's servers used the same stop configuration, and that behavior could affect V1's gathering measurements. My attempts to measure the effect were invalid or incomplete, and the final run, which uses V1's own evaluation code, crashed with Spark 1. **This does not show V1 is wrong, and it does not clear it.** No V1 result, checkpoint or manuscript file was changed. See "Stop-token effect on V1".

**Three findings to carry forward:**

1. **Local judges are not validated.** The local judge passed "Hyperadrenergic Hypertension" as correct when the answer was pheochromocytoma from MEN2A. Local judge scores here test plumbing only.
2. **Spark 1 ended abruptly three times** on 2026-11-08 and 11-09, with no shutdown record, each under vLLM load. Spark 2 did not. The cause is unknown; heat is one candidate.
3. **Answer leakage is real.** The correct diagnosis appears in `Test_Results` for 27 of the 215 extended MedQA cases. Exclude or mask them before freezing a split.

## Readiness comparison

| | AgentClinic-MedQA | SDBench | HealthBench | Medmarks (verifiable subset) |
|---|---|---|---|---|
| **Public subset tested** | MedQA extended (215 cases) | 5 synthetic cases only | Main file (5,000 examples) | MedQA, MedMCQA, MedCalc-Bench |
| **Dataset access** | Public | Synthetic public; **NEJM licensed; MIMIC credentialed** | Public | Public |
| **Install** | ✅ Own harness, no install | ✅ `uv sync` | ✅ Own venv; repo is `openai/simple-evals` | ✅ `uv sync` plus each environment |
| **Dataset loading** | ✅ | ✅ | ✅ Via `curl`; official loader failed | ✅ |
| **Model inference** | ✅ Base and 2 adapters | ✅ Base and credit | ✅ Base | ✅ Base and 2 adapters |
| **Complete interaction** | ✅ Chat format; 4 of 5 runs stated a diagnosis | ✅ 2 of 3 cases gave a real diagnosis | N/A (single response) | N/A (single response) |
| **Official scoring** | ⚠️ Protocol copied; **judge is local and unvalidated** | ⚠️ Official code; **substitute models, unofficial** | ⚠️ Official code; **grader is local and unofficial** | ✅ Deterministic scorers |
| **External API needed?** | No (optional second judge) | Only for official models | **Yes for the official grader** | No for these tasks |
| **Runs on** | Spark 1 + Spark 2 | Spark 1 + Spark 2 | Spark 1 + Spark 2 | Spark 1 |
| **Runtime per case** | ≈ 1–2 min | ≈ 3.5–9.5 min (12-turn cap) | ≈ 1.5 min | ≈ 25–45 s |
| **Work to a reproducible 50 cases** | 1–2 days | Software ≈ 1 day; data blocked | ≈ 0.5–1 day | ≈ 0.5 day |
| **Major blockers** | Judge validation; leak handling; token-cap truncation | **No usable public cases** | Grader needs approved OpenAI spend | None for verifiable tasks |
| **Recommendation** | **Needs small fixes** | **Blocked** for evaluation | **Needs small fixes** | **Ready** |

Runtimes are rough. Several were measured while other tests shared the server, and SDBench was capped at 12 turns (its default is 70). Effort figures are estimates.

## Preflight

| Item | Spark 1 (`spark-b4e7`, 192.168.100.1) | Spark 2 (`spark-04af`, 192.168.100.2) |
|---|---|---|
| Hardware | NVIDIA GB10, 121 GB unified memory | Same |
| Disk free | 3.3 TB | 3.4 TB |
| Running at start | Nothing | Nothing |
| Stack | vLLM 0.20.1, torch 2.11.0, transformers 5.8.0, verifiers 0.1.12 | Same |
| Models | Base MedGemma-4B and all adapters | Qwen3-8B copied or downloaded for this test |

- **Isolation:** the shared `~/spark1/.venv` was not modified. SDBench, Medmarks and simple-evals have their own venvs under `~/v2_bench/`. Medmarks needs verifiers ≥0.1.14, so it could not share that environment anyway.
- **`spark2` hostname:** it does not resolve here, and the `spark-b` alias fails host key verification. SSH by IP works.
- **Docker:** my user has no Docker permission on either node. It wasn't needed here, but MedAgentBench requires it.

| Benchmark | Repository | Commit | License |
|---|---|---|---|
| AgentClinic | `SamuelSchmidgall/AgentClinic` | `b6fbe22` | MIT (code) |
| SDBench | `microsoft/sdbench` | `304ce96` | MIT (code) |
| HealthBench | `openai/simple-evals` | `652c89d` | MIT (code). `openai/healthbench` is not a public repo |
| Medmarks | `MedARC-AI/medmarks` | `154710b` | MIT (code); dataset licenses vary |

## AgentClinic

### Why it looped

A controlled reproduction used one case with MedGemma in every role.

| Condition | Doctor output | Finish reason |
|---|---|---|
| No stop token | 200 tokens per turn, with repeated `<end_of_turn>` markers and invented turns | `length` (token cap) |
| Stop tokens `[1, 106]` | ≈35 tokens per turn | `stop`, on token 106 |

With special tokens hidden, as in the morning's run, those markers vanish, leaving stray text like "model" and "title:". The original Google config lists `[1, 106]`; our text-only copy lists only `1`.

Pointing vLLM at a separate `generation_config` with `[1, 106]` fixes this server-wide, with no checkpoint changes. A direct test returned `finish=stop` on token 106.

### Fixing it took more than the stop token

| Run | Change | Result on 3 cases |
|---|---|---|
| A1 | Stop token; Qwen3-8B-**Base** as environment | Base model adds "Assistant: Patient:", and the doctor copies that transcript style |
| A2 | Also role stop strings and label cleanup | 0 of 3 diagnosed; base model echoes or returns empty replies |
| A3 | Instruction-tuned Qwen3-8B; AgentClinic's single-message prompt | 1 of 3 diagnosed; 14 empty environment replies, caused by my own stop string |
| **A4** | **Same, with real multi-turn chat format** | **0 repeats, 0 empty replies; 2 valid commits, 1 cut off** |

### Clean results (chat format, Qwen3-8B environment)

| Doctor | Case | Doctor turns | Questions | Tests | Secs | Stated diagnosis | Local judge | Correct answer |
|---|---|---|---|---|---|---|---|---|
| Base | 0 | 10 | 6 | 3 | 97 | Myasthenia gravis | Yes | Myasthenia gravis |
| Base | 3 | 9 | 7 | 1 | 131 | **None: cut off at `DIAGNOSIS READY`** | No | Diffuse large B-cell lymphoma |
| Base | 4 | 12 | 7 | 4 | 63 | Pyelonephritis, hedged as "unlikely" | No | Acute interstitial nephritis |
| Credit s1 | 0 | 12 | 11 | 0 | 79 | Multiple sclerosis | No | Myasthenia gravis |
| Plain s1 | 0 | 5 | 3 | 1 | 46 | Myasthenia gravis | Yes | Myasthenia gravis |

**Do not read the last two rows as a comparison.** They show that adapters can operate the interface in free text. One case cannot say which adapter is better, and the plan's zero-shot pilot exists to answer that.

### Checks you asked for

- **Role separation:** doctor on Spark 1; patient, measurement and judge on Spark 2, a different model family. Working.
- **Test requests:** `REQUEST TEST:` parsed and routed correctly. Doctors sometimes request several at once.
- **Diagnosis commits:** `DIAGNOSIS READY:` works, but the 200-token cap can cut off a reply right at the marker. My harness first counted that as diagnosed, and now records it as `invalid_commit`. That fix compiles but hasn't been rerun on a live case.
- **Remaining doctor weakness:** turns can run long and pack in many questions, and 2–4 turns per run still hit the token cap in some cases.
- **Answer leakage:** 27 of 215 cases contain the diagnosis verbatim in `Test_Results` after normalizing case and punctuation (case IDs in `agentclinic/a1.stdout`). This morning's cruder string check counted 28, so finalize the list with one agreed rule. I tested only clean cases, and the doctor never sees that field directly. One leak was flagged in run A1, from a measurement model inventing a biopsy result, not from the case data. The audit only catches verbatim text, so paraphrases and abbreviations can slip through.
- **Correction to the plan doc:** I did not reproduce the "final-turn notice is dropped" bug listed in `V2/agentclinic_experiment_plan.md`; on reading the code, the notice does reach the doctor on the last turn. Worth rechecking before relying on that entry.

## SDBench

**Local connection works.** Models named `local-*` route to any OpenAI-compatible server, and per-role model arguments let each role use a different server. All roles ran without external APIs.

| Run | Case | Diagnosis | Judge (local) | Tests / all requests | Secs |
|---|---|---|---|---|---|
| Base | 1 | Hyperadrenergic hypertension | **4/5, counted correct; a false positive** | 5 / 25 | ≈ 400 |
| Base (after template fix) | 0 | "Unable to provide diagnosis" | 1/5 | 0 / 0 | 218 |
| Credit s1 (after template fix) | 1 | Secondary hyperaldosteronism | 1/5 | 15 / 36 | 570 |

**What it exposes:** test and question lists, test rounds, per-result `is_synthetic` flags, judge scores with reasoning, and estimated costs. That is well suited to procedure analysis.

**Problems found, in order:**

1. The gatekeeper sends the full case, which overflowed an 8,192-token server. A 32,768-token window fixed it.
2. Calls have no default output cap, so generation ran very long at about 10 tokens/s. I set `max_tokens` through model arguments.
3. MedGemma's template rejects two consecutive user messages, which the baseline agent sends when turns run out. A benchmark-only template (`benchmark_smoke_scripts/medgemma_merge_roles.jinja`, with `--chat-template-content-format string`) fixed it without SDBench changes.
4. Non-fatal tracebacks from an optional costing cache (`unhashable type: 'dict'`) appear because of the dict-valued model arguments. Costs were still computed.
5. MedGemma mixes `<question>` and `<test>` tags, which triggers turn-type notices and wastes turns.

**Official configuration:** the `default` config uses GPT-5 for gatekeeper and judge, and `science` uses GPT-5 with `o4-mini` for some datasets. Costing defaults to `gpt-5-mini`, and grounded costing uses a public hospital price file that I did not download. Costs here come from a local LLM estimate, so they are not official. Running the official configuration needs OpenRouter or Azure credentials.

**Blocker:** the 5 public cases are explicitly not for reporting performance. Reportable results need `nejm` (license from elicensing@nejm.org; terms unknown) or `mimic` (PhysioNet credentialing; local models only). I did not attempt to get either.

## HealthBench

- **Location:** the official code is `healthbench_eval.py` in `openai/simple-evals`, which OpenAI no longer updates for new models or results but keeps as a reference implementation.
- **Data:** the public file has 5,000 examples, with a mean of 11.4 physician rubric items per example, and 42% are multi-turn. About 30% of sampled rubric items carry negative points. Its themes include `emergency_referrals`, `hedging` and `context_seeking`, which relate closely to our safety question.
- **Loader:** the official path failed through `blobfile` with a request for Azure credentials, though `curl` fetched the file without one. The evaluator accepts a local `input_path`, so I used that, with the official grading code unchanged.
- **Run:** 2 examples through official `HealthBenchEval`, 185 seconds, 0 grading parse failures.
- **Grading:** each rubric item is judged separately by an LLM returning `criteria_met` as JSON. The official grader is `gpt-4.1-2025-04-14`.
- **Status: official grader untested.** I used Qwen3-8B as a stand-in to test the plumbing. Its 0.43 score on 2 examples must **not** be reported as a HealthBench score.
- **Cost estimate for the official grader:** about 570 grader calls per 50-case arm, or roughly 0.5M input and 0.06M output tokens. That is on the order of **$1–2 per arm** at GPT-4.1 prices as I remember them ($2/$8 per million); check current pricing before relying on this. This needs your approval.
- **Validating a local grader:** the eval has a physician-completion mode that could measure a local grader's agreement before trusting it. I have not tried it.

## Medmarks

**Our stack is directly supported:** the CLI's `--api-base-url` reached our vLLM server, with no adapter. It's built on `verifiers`, the framework you've already used, but needs ≥0.1.14 versus 0.1.12 in the shared venv, so keep its venv separate. Its endpoint registry already lists `medgemma-4b-it`.

| Task | Model | Examples | Score | Secs | License (per Medmarks README) |
|---|---|---|---|---|---|
| MedQA | Base | 5 | 2/5 | 213 | CC-BY-4.0 |
| MedQA | Credit s1 | 3 | 0/3 | 123 | |
| MedQA | Plain s1 | 3 | 0/3 | 115 | |
| MedMCQA | Base | 5 | 4/5 | 126 | Apache-2.0 |
| MedCalc-Bench | Base | 5 | 0/5 | 198 | CC-BY-SA-4.0 |

All 3 tasks parsed answers and scored with their own scorers, with 0 request errors. MedCalc's 0/5 looks low, so I confirmed its 11 scorer tests pass; the model's arithmetic answers really were wrong. Answers are in `medmarks/summary_*.txt`.

**Base model's first 3 MedQA answers scored 1/3, the same order as the adapters' 0/3. That's far too few to mean anything.**

**Repeatable arm comparisons:** yes. Changing `-m` is the only difference between arms, and results save to structured files, with `process` and `winrate` commands to compare them. Note that adding a new adapter means serving it under a name first.

**Not yet tested:** Medmarks' own `agentclinic` and `healthbench` environments, which use LLM judges (`openai/gpt-5-mini` by default), and its experimental Medmarks-T training environments. Since these tasks are single-response, they measure knowledge and calculation, **not** information gathering.

## Stop-token effect on V1

**Status: unresolved. Treat every number in this section as direction only.**

**Established:**

- `checkpoints/medgemma4b_text`, `dx_grpo_init` and `rx_grpo_init` all have `generation_config` stopping only at token 1. No code in the repo sets `stop_token_ids` for the rollout server. The GRPO trainer sets `[1, 106]` on its own model only.
- On a current server using V1's prompt and 128-token limit, base, plain and credit all kept writing past `<end_of_turn>` to the token cap: 5 of 5 cases per arm. The run-on text can contain further JSON actions (`stop_token/v1_stop_probe.txt`).
- `framework/parsing.py` takes the first valid JSON, so the first action in those samples matched the clean output. However, `support_probe.py` and `grpo_disagg.py` append the full reply to the conversation, so run-on text would enter later turns' history.
- Five V1 training arms logged `clipped_ratio` 1.0 and `max_terminated_length` 0 on all 60 steps. That fits this behavior, but the metric could have other causes under our custom rollout function.

**Not established:** whether V1's servers behaved the same way (raw V1 replies weren't saved), and whether it changes any published number.

**Measurement attempts, none usable as a result:**

| Attempt | Problem | Location |
|---|---|---|
| 1 | Two copies wrote to the same files, because a killed session's process kept running | `stop_token/wrong_env_tasks_diagnostics_env/corrupted_two_writers/` |
| 2 | **Wrong environment:** `tasks/diagnostics/env.py`, not V1's `env_rv.py` deephedge | `stop_token/wrong_env_tasks_diagnostics_env/` |
| 3 | Right environment (all 60 cases match V1), but correlated random seeds across cases; credit and base did not reproduce their V1 records | `stop_token/wrong_seed_design/` |
| 4 | V1's own `support_probe`, k=12, changing only the stop config: **Spark 1 crashed before any arm finished** | `benchmark_smoke_scripts/stop_token_v1_support_probe.sh` |

Attempt 3 is the closest evidence, on 60 correlated rollouts per condition:

| Arm | Mean asks, no stop → stop | ≤1 ask, no stop → stop | Reward, no stop → stop | V1 record (asks, ≤1 ask, reward) |
|---|---|---|---|---|
| Base | 4.80 → 5.03 | 0.50 → 0.52 | 0.227 → 0.222 | 2.54, 0.66, 0.251 |
| Plain s1 | 1.90 → 3.97 | 0.73 → 0.52 | 0.302 → 0.260 | 1.92, 0.76, 0.255 |
| Credit s1 | 1.00 → 3.72 | 1.00 → 0.52 | 0.446 → 0.307 | 2.23, 0.67, 0.334 |

**Why it matters enough to finish:** if this direction holds, the stop setting moves adapter ask counts substantially, and V1's central claims concern asking too little. Attempt 3's mismatch with V1's records and its seed flaw mean it can't support that claim yet. The attempt 2 numbers came from a different environment and should be disregarded.

**To finish:** run `stop_token_v1_support_probe.sh`, which includes a no-stop control to confirm today's stack reproduces V1 before comparing. That comparison decides whether the arXiv paper needs a note or correction.

## Stability

- **Abrupt ends:** Spark 1's system journal stops at 16:45 and 19:57 UTC on 11-08 with no shutdown record, followed by reboots at 16:55 and 20:11. Both happened while vLLM was serving. The journal has no OOM, GPU driver or thermal messages. I can't tell whether the cause was hardware or software.
- **Since then:** a third abrupt end came at about 00:21 UTC on 2026-11-09, roughly 4 minutes into the offline `support_probe` run. Telemetry's last sample showed the GPU at 88 °C, 96% utilization and 98 GB memory in use, up from about 56 GB during server runs. Spark 2 has stayed up since 09:12 PST on 11-08, though it carried less load.
- **What the evidence does and doesn't show:** this crash came under a heavier load than the earlier hours that ran without trouble, and an abrupt power loss from a thermal or power fault wouldn't necessarily be logged. But nothing confirms that, and the first two crashes have no temperature data. Check Spark 1's airflow and power, and run the next heavy job with Spark 2 as a comparison.
- **Precaution for real runs:** run servers in tmux, write results incrementally, and keep telemetry on.

## Next pilot for each benchmark

| Rank | Benchmark | Relevance | Access friction | Effort | Evaluation value | Next pilot |
|---|---|---|---|---|---|---|
| **1** | AgentClinic-MedQA | High: free-form gathering | None | Low | High | Plan's P0–P2: freeze split without leaky cases, validate judge, then 50 validation cases across base, plain and credit |
| **2** | SDBench | **Highest**: gathering with costs | **High** | Low for software, data blocked | Highest if data arrives | Ask NEJM about terms; start PhysioNet credentialing; no results from synthetic cases |
| **3** | HealthBench | Medium-high: safety and escalation, but not interactive | Low; grader costs money | Low | Medium-high | With your approval, 50 examples per arm with the official grader |
| **4** | Medmarks | Medium: knowledge baseline | None | Lowest | Medium | Fixed 50-example MedQA, MedMCQA and MedCalc run across all arms to check adapters didn't damage general ability |

## Answers to your five questions

**1. Which can we run successfully right now?**

Medmarks' verifiable tasks are fully working, with official scoring. AgentClinic-MedQA and HealthBench work end to end, but with local judges that don't count as official scoring. SDBench's software also works, on synthetic cases that can't be used for results.

**2. Which are blocked, and by what?**

| Blocker type | What's blocked |
|---|---|
| **Data access** | SDBench's real cases (NEJM license or PhysioNet); AgentClinic-MIMIC-IV (PhysioNet plus author request) |
| **External API** | Official HealthBench grader (OpenAI); official SDBench configuration (OpenRouter or Azure) |
| **Model compatibility** | Fixed: stop token and strict role template, both server-side. Still open: long doctor turns and MedGemma mixing tags in SDBench |
| **Infrastructure** | No confirmed block, but Spark 1's three abrupt ends are an unresolved risk to long runs |

**3. Can both Sparks support meaningful external evaluation without interfering with ongoing RL?**

Evaluation yes, at the same time as RL no. This pilot's scale fits, using about 45% of memory for each server. But our GRPO setup uses both Sparks and needs node A's GPU free for its first weight-sync, so evaluations should run **between** training runs. Throughput is the other limit: I saw roughly 10–20 tokens/s per stream, and about 44 tokens/s across 8 concurrent requests on the adapter-loaded server. So 50 AgentClinic cases at 1–2 minutes each take about 1–2 hours per arm if run one at a time. Also, no RL was running today, so I didn't test interference directly.

**4. Which is the strongest candidate for the first V2 evaluation?**

AgentClinic-MedQA. SDBench is scientifically closer, but it can't produce reportable results without gated data.

**5. What is the smallest set of engineering changes to get there?**

1. Serve MedGemma with stop tokens `[1, 106]` through a separate `generation_config`, and record the decision (see plan milestone M2).
2. Use the chat-format harness in `benchmark_smoke_scripts/agentclinic_smoke.py`, adding a higher doctor token cap and a live test of `invalid_commit`.
3. Freeze the case split, excluding or masking the 27 leaky cases.
4. Validate the judge on a sample using a second judge and clinician spot-checks, before trusting any accuracy number.
5. Keep Qwen3-8B or a chosen replacement fixed as the environment across all arms, and check patient quality.
6. Run concurrently at a modest level, writing results incrementally.

Whichever stop setting is chosen, resolve the V1 question first, because it decides how V2 results compare with V1.

## Recommended next steps

1. **Check Spark 1 airflow and power, then resolve the V1 stop-token question** with `stop_token_v1_support_probe.sh`, ideally running it on Spark 2 after copying the base model and adapters. If Spark 2 is fine under the same load, that points toward Spark 1 itself. Hold any V1 correction decision until then.
2. Decide whether to approve about $1–2 per arm for the official HealthBench grader.
3. Start PhysioNet credentialing and ask NEJM about SDBench terms, including whether they permit training. Those are dependencies of the plan's Experiment A (called Experiment B when this report was written).
4. Assess MediQ next. It wasn't part of this campaign and remains an untested candidate.
5. Build the AgentClinic pilot from the steps above.

## Files

- `benchmark_smoke_logs/preflight/`: node state
- `benchmark_smoke_logs/agentclinic/representative_traces.md`: readable traces. Raw `.jsonl` traces are excluded by the repo's `.gitignore` and stay local
- `benchmark_smoke_logs/sdbench/`: commands, run outputs and failed attempts
- `benchmark_smoke_logs/healthbench/` and `medmarks/`: runs and scorer tests
- `benchmark_smoke_logs/stop_token/`: V1 probe outputs
- `benchmark_smoke_logs/stability/`: boot records and telemetry
- `benchmark_smoke_scripts/`: harness, probes and benchmark-only template

Outside the repo, `~/v2_bench/` holds the clones, venvs and generation config, and Spark 2 holds the Qwen models.
