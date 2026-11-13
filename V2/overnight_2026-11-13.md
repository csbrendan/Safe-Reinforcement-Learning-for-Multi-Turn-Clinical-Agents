# V2 overnight report — 2026-11-13

Evidence:
- `V2/hw_stability/PROGRESS.md` (hardware log, written incrementally);
- `results/v2_hw/` (telemetry CSVs and trainer benchmarks; Spark 2 copies in `~/v2_hw/`);
- `results/v2_format/` and `results/v2_sdbench/`.

Nothing was sent to Azure. No GRPO training was run. V1 and the paper are untouched.

## 1. Spark stability, temperatures, power, shutdown evidence

**Shutdown evidence (Spark 1).** All 7 earlier boots end abruptly: no shutdown sequence, no kernel panic, thermal,
Xid, OOM or MCE line, and pstore is empty. Spark 2's boots also all end without a shutdown sequence, so an abrupt end
alone can't separate a fault from a pulled plug.

Both machines' firmware sets `critical = 104.8 °C` on every ACPI zone, and the GPU slowdown point is ≈ 95–97 °C.
Hitting the firmware trip would have produced a logged, orderly shutdown. So **Spark 1's power-offs at a GPU
reading of ~88 °C were not the OS thermal trip.** The cut came from something outside the OS: the power brick or
PSU, an embedded-controller cut-off, or the supply. The logs can't say which.

**Idle after the electrical change (both rebooted 03:43 UTC, both reachable, P8 state, ~5 W GPU):**

| | GPU | ACPI max | NVMe | NIC |
|---|---|---|---|---|
| Spark 1 | 55–57 °C | 59–61 °C | 55 °C | 64 °C |
| Spark 2 | 43–44 °C | 45 °C | 43 °C | 50 °C |

- Spark 1 idles about 16 °C hotter on every sensor, NVMe and NIC included, so the whole chassis is warmer.
- No throttle bits on either machine.
- Driver: Spark 1 580.126.09, Spark 2 580.159.03.

**Spark 1 ramp (supervised, 03:53 UTC): stopped in stage 1.**
- At ~50 W GPU (the SW power cap; the "25% duty" stage still ran at the cap), ACPI went 60.9 → 85.3 °C in 55 s and
  was still rising.
- The independent watchdog killed the load. No crash, no kernel error, no throttle bit.
- Per your rule I did **not** re-stress Spark 1 and skipped its 20–30 min realistic test. Spark 1 did no GPU work
  for the rest of the night.

**Spark 2 control (same stage, 120 s): plateau.** ACPI 44.8 → 70.3 °C, levelling off.

| t from load start | Spark 1 ACPI | Spark 2 ACPI |
|---|---|---|
| idle | 60.9 | 44.8 |
| +15 s | 74.6 | 58.9 |
| +30 s | 78.1 | 63.1 |
| +55 s | 85.3 (stopped) | 68.8 |
| +115 s | – | 70.3 |

The rise over the first minute is the same on both (+24 °C). The difference is Spark 1's 16 °C higher starting point.

**Spark 2 under real V2 work (Qwen gatekeeper and judge, MedGemma rollouts, trainer benchmark).**
- **Fine at sustained ~40–50 W:** ACPI 70–80 °C.
- **85 W bursts are the problem.** Gatekeeper and judge calls, and training steps, spike to ~85–92 W. At that
  power the SoC hot spot climbs ~2.7 °C/s, from 66 to 85 °C in ~7 s.
- **Reaching the 85 °C kill.** The watchdog reached its 85 °C hard limit 4 times (05:58 and 06:15 serving, 06:40
  and 06:46 in the trainer benchmark at 16k tokens), after 5 earlier trips at my lower limits (83 °C three times,
  82 °C twice). One of those (04:37) was a CPU burst from grammar compilation. Another (05:24, when I let two jobs
  overlap; my error) read ACPI 88.8 °C for ~3 s while the servers were exiting.
- **Pausing helps, but late.** A pause/resume governor (SIGSTOP/SIGCONT of the server group) takes 2–7 s to cut
  power, because the GPU keeps running queued work.
- **Final setting:** pause at 68 °C, resume at 62 °C, kill at 85 °C. Since 06:50, peaks stayed at 81–82 °C with no
  further trip, at ~30 W average and roughly 10–35% of normal
  throughput.
- **History:** Spark 2's 11-11 log shows sustained GPU 83–84 °C at 79–85 W with no fault, so these temperatures
  are inside what it has run before. I kept your 85 °C limit regardless.
- **No crash or hardware error on either machine tonight.** The only kernel errors were NVRM out-of-memory
  allocation failures, caused by my co-residence memory test (section 5). Every watchdog trip was a limit I set.

Harness: `V2/hw_stability/` holds `telemetry.py` (1 Hz, mirrored live to Spark 2), `watchdog.py` (NVML, fail-safe,
optional governor), `run_supervised.sh` and `serve_spark2.sh`.

Bug found and fixed: the first watchdog let a GPU reading of 80–81 °C pass for 8 s. nvidia-smi stalls under load,
so it now reads NVML in-process. The ACPI limit had stopped that run anyway.

## 2. Did the electrical change help?

Not shown either way.
- Spark 1 did not power off tonight, but its test stopped at 85 °C, below the ~88 °C where it shut down before, and
  ran for only about a minute.
- Absence of a crash in that window says nothing about brownouts or supply quality.
- What the data does show: Spark 1's problem at load is thermal headroom that starts 16 °C behind Spark 2's. That
  is a placement, airflow or fan issue, independent of the power strip.
- Next steps, your call:
  1. check Spark 1's position, vents and fan, and whether it is drawing in Spark 2's exhaust;
  2. swap the two machines' positions (and power bricks) to see whether the offset follows the machine or the
     location.

## 3. MedGemma formatting: fixed, or is format SFT needed?

**Status: not fixed. Constrained decoding removes the turn-rejection failure, but the base model never ends a list,
so full episodes overflow the context before diagnosing. A turn-length control is needed: SD-Bench's own per-turn
request caps first (cheap, not yet tested), and a narrow format SFT if that isn't enough.**

**Audit of the raw sampled outputs** (saved runs at T = 0.7 and 1.0):
- Mixed turns are long lists: about 12 questions and 7 tests per turn. Often each test is paired with a question
  about it ("What is the patient's CBC showing?" next to `<test>CBC</test>`).
- After SD-Bench's "turn type violation" notice the model apologises and usually repeats the mix (P = 0.79–0.86).

**Template, parsing, stopping.**
- The chat template is standard Gemma 3: the system prompt goes into the first user turn, and roles alternate
  correctly.
- The parser and the turn rule are SD-Bench's own (`test_runner_enforce_single_turn_type=True`, kept).
- Stopping at `<end_of_turn>` (106) was already fixed. One new issue: grammar-constrained decoding can only end on
  an EOS id, and stop ids don't count. MedGemma is now served from `medgemma4b_text_v2stop` (hard links to the same
  weights; `generation_config` lists 106 as EOS), which is equivalent for unconstrained decoding.

**Interventions** (`tasks/v2_sdbench/turn_format.py`, config key `policy_format`, default `official` = unchanged).
First-turn probe, 5 synthetic cases × 16 samples per cell (`results/v2_format/first_turn.json.parts.jsonl`):

| variant | T | mixed (rejected) | question-only | hit 400-token cap | requests per turn | diversity (Jaccard) |
|---|---|---|---|---|---|---|
| official | 0.7 | 0.54 | 0.46 | 0.65 | 22.0 | 0.94 |
| official | 1.0 | **0.79** | 0.21 | 0.42 | 18.5 | 0.95 |
| + one prompt sentence | 1.0 | 0.76 | 0.24 | 0.33 | 18.9 | 0.94 |
| constrained decoding | 0.7 | **0.00** | 0.99 | 1.00 | 23.5 | 0.97 |
| constrained decoding | 1.0 | **0.00** | 1.00 | 1.00 | 21.1 | 0.98 |
| prompt + constrained | 1.0 | 0.00 | 1.00 | 0.60 | 17.8 | 0.98 |

- **The prompt sentence does nothing** (0.76 vs 0.79 mixed).
- **Constrained decoding** (a vLLM structured-output regex: all `<question>`, all `<test>`, or one `<diagnosis>`)
  removes mixing completely. Diversity and question topics are unchanged (history 13→16%, medications 7→8%,
  lab-type questions 16→10% because tests can no longer be mixed in).
- **But the base model doesn't end a question list on its own.** Even unconstrained, 82–100% of question-only first
  turns hit the 400-token cap; switching to tests is how it usually ends a list. Under the constraint every opening
  turn becomes ~21 questions cut at the cap. That's a valid SD-Bench turn (complete tags are parsed), but it costs
  ~20 gatekeeper calls per turn.
- Plumbing: TRL's `VLLMClient.chat(structured_outputs_regex=…)` is passed through in the GRPO rollout, so training
  and evaluation use one interface. Use an unbounded regex: counted repeats cost 30× more grammar-compile CPU and
  caused a thermal trip.

**Full episodes** (constrained, T = 1.0, guarded condition, 2 at a time, under the governor; 07:19–09:40 UTC). One
episode finished in 2.3 h (case 0): every turn single-type and valid, **every turn cut at 400 tokens**:

| turn | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| requests | 17 questions | 33 tests | 25 tests | 26 tests | 28 tests | 28 tests |

- It ordered the decisive studies: bone-marrow biopsy, then a repeat biopsy with CD20-positive large atypical cells.
- With 136 tests and their results, the context passed 16,384 tokens at turn 7. The episode ended on **overflow
  with no diagnosis**: judge score 1, cost $95,360, 17 repeated requests.
- I stopped the second episode and the remaining runs at 09:40. The answer was already clear, and each further
  episode took ~2 h of thermally cycled serving.
- The official-vs-constrained greedy comparison (T = 0) did not run.

**What this means.**
- At sampling temperature the base model's "list everything" habit is the real format problem; tag mixing was only
  how it showed up.
- Constrained decoding is still needed (it is the only thing that removes rejection), but it isn't sufficient.
- Cheapest next test: constrained decoding plus SD-Bench's own per-turn limits (`max_questions_per_turn` /
  `max_tests_per_turn`, already implemented, opt-in, e.g. 5 each). That bounds results per turn and so context
  growth, without new weights. It's a task change applied to all arms and the guarded evaluation, as the config
  comment says.
- If episodes then still don't reach a diagnosis, format SFT.

**Format SFT: designed, not built.** The SFT data has to come from well-formed trajectories. The only cases
available are the 5 public synthetic ones, which are also the plumbing and evaluation cases, so SFT on them would
contaminate. The MIMIC training split isn't available. Plan once there is data: LoRA SFT on base temperature-0
trajectories filtered for validity, with long question lists split into turns of at most 8 requests and one short
`<diagnosis>`. That gives one shared initialisation for arms A/B/C (arm A = that checkpoint). Constrained decoding
needs no new checkpoint: every arm uses the base weights plus the same decoding interface.

## 4. Is the Qwen judge stable enough for GRPO?

**Short answer: usable only with mitigations; noisy enough to matter in low-spread groups.** No new paid calls.

The calibration from 11-12 scored each saved episode four times with the same Qwen3-8B and official prompt (logged,
reference re-score, two fresh-cache re-scores). I rebuilt the full GRPO groups from it, giving duplicates their
episode's scores (`V2/judge_stability_from_calibration.py`, metrics in `tasks/v2_sdbench/judge_stability.py`).
Data: 4 groups, 24 episodes, 19 judged; synthetic cases 0–1; T = 0.7 and 1.0; base policy.

| Pairwise over the 4 scorings | Value |
|---|---|
| Judged episodes with the same score all 4 times | 58% |
| GRPO advantage sign agrees (zero counted as its own sign) | 82% |
| Advantage sign strictly flips | 5.6% |
| Within-group Kendall τ of rewards | 0.81 |
| Cosine between the groups' advantage vectors | 0.84 |
| Groups that gain or lose all spread between scorings | 12.5% |
| Judge-noise variance ÷ within-group reward variance | 0.15 |

- **Groups with real spread are stable.** Case 1 at T = 1.0 has std ≈ 0.3, and its advantage vector matches the
  consensus with cosine 0.92–0.99.
- **Groups without real spread are noise.** Case 1 at T = 0.7 has a std of 0 to 0.125 depending on the scoring, with
  consensus 0.04. Its advantage direction is set by the judge. **Standard GRPO divides by the group std, so judge
  noise in such a group becomes a full-size random update.**
- **Caveats.** The data is small and mostly format-invalid episodes (the format penalty is constant within a group,
  so it doesn't mask judge noise, but scores sit mostly at 1–2). Re-measure on format-valid rollouts.
- **Recommended before GRPO** (both are training-side, all arms alike):
  1. Don't divide advantages by the group std: TRL `scale_rewards="batch"` or `"none"`. Tiny, noise-only spreads
     then give tiny updates.
  2. Score each unique training trajectory with the median of 3 judge calls (noise variance ÷ ~3). The cache keeps
     the median fixed. At the measured ~1 min per governed judge call this costs time, not money.
- **The cache doesn't remove the noise.** First-write-wins caching makes rewards reproducible, but each unique
  trajectory still gets one noisy draw.

## 5. Two-node training configuration (measured)

**Memory.** Trainer microbenchmark (`V2/hw_stability/trainer_mem_bench.py`): MedGemma-4B text, LoRA r 32 on all
projections, gradient checkpointing, reference forward plus policy forward/backward/AdamW, one sequence. Runs:
standalone on Spark 2, and next to idle Qwen (0.45) + MedGemma (0.20) servers.

| tokens | peak, standard loss | peak, Liger fused loss | time (ref + update), governed |
|---|---|---|---|
| 4,096 | 12.4 GiB | 13.3 GiB | 27–42 s |
| 8,192 | 17.8 GiB | 14.5 GiB | 69–96 s |
| 12,288 | 22.8 GiB | 15.4 GiB | 117–162 s |

- **Co-resident with both servers:** the Liger run completed 4k–12k at 15.4 GiB peak.
- **Without Liger, at the 26.8 GiB cap,** the first attempt failed with kernel `NVRM: Out of memory` errors. That run
  also lacked checkpointing (my benchmark bug, now fixed). The point it shows: unified memory had only ~9–10 GiB truly
  free and ~25 GiB of page cache, which the driver does not reclaim fast enough, and clearing it needs root.
- Added `--liger` to `framework/rl/grpo_disagg.py` (TRL `use_liger_kernel`; it honours `env_mask`). Off by default.
- Transcript lengths in saved episodes: median 4–6k tokens, tail to 12–17k (the context guard stops at 14k).

**Thermal and throughput.**
- **Spark 1:** no GPU role until its airflow is fixed. It reaches 85 °C ACPI in under a minute at 50 W, and every
  GPU role here (Qwen bursts, training steps) draws 85–92 W.
- **Spark 2:** carries everything. Under the 85 °C ceiling the governor holds it at ~30 W average.
- Measured throughput under the governor:
  - Qwen ~6 tokens/s across 3–4 streams;
  - one judge call ≈ 1–2 min;
  - one guarded full episode ≈ 30–60+ min.
- Unthrottled reference: MedGemma decode ~150 tokens/s at 32 streams; 16 concurrent episodes in 40 min (11-12).

**GRPO step-time estimate** (16 episodes per step = 2 prompts × 8):

| | Rollouts | Update (measured per-episode times × median 5–6k, tail to 12k) | Step |
|---|---|---|---|
| Unthrottled (11-12 throughput) | 40–80 min | 12–20 min | ≈ 1–1.7 h |
| Under the 85 °C governor | ~2–6 h (≈ 10–35% throughput) | 15–25 min | ≈ 2.5–6.5 h |

A 3-step smoke therefore takes about 3–5 h if Spark 2 may run unthrottled (it ran 83–84 °C GPU for hours on 11-11
without a fault), or about 8–20 h under the 85 °C ceiling.

**Best configuration.**
- **Now:** all GPU work on Spark 2 — Qwen3-8B :8101 (0.45), MedGemma rollout server via `trl vllm-serve` (0.20),
  and the trainer with `--liger --grad-ckpt`. That totals ~94 GiB, measured to fit.
  - Trainer and rollout server on one node also avoids the V1 two-node NCCL weight-sync problem.
  - Spark 1 runs only CPU work: orchestration, evaluation clients, analysis, monitoring.
- **After Spark 1's airflow is fixed:** trainer + MedGemma server on Spark 2, Qwen on Spark 1. GRPO alternates
  phases, so Spark 1's load (gatekeeper and judge) falls only in the rollout phase and it cools during updates.
  Each node then carries roughly half the heat.
- **Never two long RL runs at once:** neither node has the thermal or memory headroom.

## 6. SD-Bench leak routes (Priority 5)

Probe (`V2/leak_probe/v2_leak_routes.py`). The same 16 generic requests for every case, nothing case-specific, sent
through the training runner with the guard on. Qwen3-8B as gatekeeper, the 5 public synthetic cases, a separate
cache. Results: `results/v2_sdbench/leak_probe2/routes.jsonl`; summary: `V2/leak_probe/summarize_routes.py`.

| Route | Requests (5 cases) | Gatekeeper revealed diagnosis or course | Reached the agent with the probe-time guard | With tonight's final detector |
|---|---|---|---|---|
| Documents ordered as `<test>` | 35 | 24 | 0 (all refused) | 0 |
| Treatment questions | 15 | 8 | 0 detected; 7 passed as info | 3 more withheld |
| Timeline questions | 15 | 5 | 0 detected; 10 passed as info | (included above) |
| Interpretation questions | 15 | 11 | 0 (dx_leak) | 0 |

Per document request, cases naming the diagnosis: discharge summary 5/5, consultant's impression 5/5, assessment and
plan 5/5, pathology report final diagnosis 3/5, problem list 2/5, hospital course 1/5, final diagnosis 0/5 (refused).

**Worst route (new, test channel, open in the official condition):** "Discharge summary", "Consultant's
impression" and "Assessment and plan" named the reference diagnosis in **5 of 5 cases** (e.g. "Primary Diagnosis:
Intravascular large B-cell lymphoma…", "…most likely a pheochromocytoma"). Only "Final diagnosis" was refused,
5/5. Before tonight the guard only covered questions, so this route was open in training too.

Changes (training environment only; the official condition is unchanged and still reports these as `channel_kind`):
- **`meta_request`:** a `<test>` naming a document or conclusion is answered with a refusal that carries
  SD-Bench's own rejection phrase, so it is never charged (`channel.is_meta_request`).
- **`course_leak`:** a question answer revealing the post-diagnosis course, in-hospital management or planned
  procedures is withheld (`channel.reveals_course`).
  - It catches "fevers resolved within the first week of therapy", "scheduled procedure tomorrow is a random skin
    biopsy", "upcoming laparoscopic right adrenalectomy" and "later managed with phenoxybenzamine … alpha-blockade".
  - It does not match medication history or past surgical history: 2 of the 486 logged answers match, both real
    leaks.
- New metrics `n_course_leak_q` and `n_meta_requests`, with regression tests. All 117 V2 tests pass (72 at the start
  of the night).

**Remaining routes not caught:**
- Bare in-hospital drug names: "What medications were you given in the hospital?" → "phenoxybenzamine". The
  detector reads wording, not pharmacology.
- Gatekeeper interpretation with no diagnosis words ("a complex multisystem illness…"); low content.
- In-admission course without post-diagnosis wording (case 4: "congestive symptoms have stabilized with meticulous
  diuretic and blood-pressure management"; "admitted for decompensated heart failure… IV diuresis limited by
  orthostatic hypotension"). I did not add patterns for these, to avoid overfitting to individual answers.
- Everything in the official condition, by design.

Caveats:
- The detectors were tuned on these 5 public synthetic cases (not held-out data), so expect misses out of sample.
- RL will search for gaps, so the stop criterion stays: rising `guarded`, `dx_leak_q`, `course_leak_q` or
  `meta_requests` across steps.
- Whether SD-Bench's official GPT-5 gatekeeper refuses document requests is unknown (needs Azure).

## 7. Blockers before a 3-step GRPO smoke

1. **Thermal limit (decision needed).** Under your 85 °C rule, Spark 2 runs the V2 stack at ~10–35% throughput, so
   a 3-step smoke is ≈ 8–20 h. Options:
   - (a) check and fix Spark 1's placement and airflow, then split the load across both nodes (section 5);
   - (b) allow Spark 2 a higher hard limit, e.g. 90 °C ACPI, with the governor kept. It ran GPU 83–84 °C for hours
     on 11-11; firmware critical is 104.8 °C;
   - (c) accept a slow governed smoke.
2. **Turn format (decision needed).**
   - Constrained decoding alone yields valid turns but overflowing, diagnosis-less episodes (1 full episode: 136
     tests, overflow at turn 7).
   - For a plumbing smoke that only tests NCCL, masks, logging and checkpoints, this is tolerable.
   - For any learning signal, first test constrained decoding plus per-turn caps. If that fails, a format SFT, which
     needs non-synthetic training trajectories (MIMIC train split).
3. **Trainer configuration untested end to end.** Next run: `--liger --grad-ckpt --scale-rewards batch` with
   `trl vllm-serve` for MedGemma on Spark 2, same node as the trainer (avoids the two-node NCCL issue). Memory
   co-residence is measured (section 5); NCCL weight sync on one node is not.
4. **Judge noise.** Use `--scale-rewards batch` (or `none`) and `judge.repeats: 3` for training (both opt-in,
   implemented tonight). Re-measure stability on format-valid rollouts.
5. **Still open from before:**
   - Azure credentials and spend approval (the official GPT-5 gatekeeper and judge, and the leak re-check);
   - MIMIC data access and the human-review opt-out;
   - `lam_cost` set from arm-A development statistics.
   None of these blocks a synthetic plumbing smoke.
