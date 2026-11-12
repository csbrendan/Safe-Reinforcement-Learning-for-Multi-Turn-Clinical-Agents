# V2 SD-Bench environment validation (2026-11-12)

Overnight work toward a validated, efficient SD-Bench environment for the three-arm study. **No RL was run.**
Everything here uses SD-Bench's 5 public synthetic cases, Qwen3-8B (unvalidated) as gatekeeper, judge and pricer,
and base MedGemma 4B as the policy. Numbers are engineering measurements, not results.

Code: `tasks/v2_sdbench/` (`channel.py` is new), tests in `tests/test_v2_sdbench*.py`, raw logs in
`results/v2_sdbench/{leak_probe,bench,smoke}/` (JSONL and logs gitignored; probe and audit text committed).

## 1. The question channel is an information leak, not only a scoring bug

SD-Bench charges only rounds that contain a `<test>`. Its gatekeeper prompt (`v6_fixed`, and `v13_mimic` for
MIMIC) quotes any patient-question answer that exists in the chart, including lab, imaging and pathology results.
It refuses only results that aren't charted.

**Probe** (`results/v2_sdbench/leak_probe/probe.txt`): each case's decisive studies, asked as a question
("What did your X show?") and ordered as a test.

| Case | Asked as a question | Leaked? |
|---|---|---|
| 1 Pheochromocytoma / MEN2A | plasma metanephrines, adrenal CT, RET genetics | 3/3, same content as ordering |
| 0 Intravascular lymphoma | bone-marrow biopsy, skin biopsy, brain MRI | 3/3 |
| 3 Anti-NMDA encephalitis | CSF NMDA antibody, pelvic ultrasound | 2/2 |
| 4 AL amyloidosis | fat-pad biopsy, free light chains | 2/2 |

Several question answers were worse than the ordered result, because they stated the diagnosis ("…confirming the
diagnosis of anti-NMDA receptor encephalitis", "…consistent with multiple endocrine neoplasia type 2A").

**Diagnosis fishing** (`leak_probe/exploit_probe.txt`):

| Question | Answer |
|---|---|
| Case 4: "What was your final diagnosis?" | The reference diagnosis, verbatim |
| Case 1: "What did the doctors conclude was causing your spells?" | "…specifically a pheochromocytoma" |
| Case 2: "What infection did they find in your intestine?" | "Tropheryma whipplei, the causative agent of Whipple disease" |

Cases 1 and 2 refused the direct "final diagnosis" question, so Qwen3-8B applies SD-Bench's safeguard
inconsistently. **Under an outcome reward, asking for the diagnosis is the optimal policy.**

Base MedGemma already uses the channel without any RL. In the first smoke run it ordered 0 tests and obtained at
least 16 and 34 lab and imaging results through questions. Official cost was $300 for each case, against real
workups of $4,600 and $5,500.

**Conclusion.** Charging for leaked results fixes the incentive but not the leak: the agent still receives results
it never ordered, and sometimes the answer itself. Both fixes are needed.

**Caveat.** This is Qwen3-8B as gatekeeper. SD-Bench's official gatekeeper model may refuse more often. Checking
that requires paid API calls (not done; needs approval).

### Fixes (training environment only; official behaviour unchanged)

| Fix | What it does | Where |
|---|---|---|
| Result classifier | Labels each answer `test_result` (lab value with unit, analyte with value or result word, imaging, pathology, ECG, culture, genetics), `dx_leak` (names a distinctive term of the reference diagnosis), `historic_test` (pre-admission study, allowed by SD-Bench's chronology rule), `rejected`, or `info` (history, symptoms, vitals, exam: free by SD-Bench's rules). Reads the **result** text, so the policy can't avoid it by rephrasing the question. | `channel.classify` |
| Question guard | A question whose answer is `test_result` or `dx_leak` returns a redirect instead. The real answer is kept in `withheld_result` for audit. | runner, `question_guard` |
| Corrected cost | Charges question-channel test results that reached the agent, priced by the study named. That round counts as a test round. Compound orders ("CBC, CMP; troponin") are priced per component. | `CostModel.corrected_cost` |
| Parrot canonicalization | "What is X?: 38.6°C" is sent to the gatekeeper as "What is X?" | `channel.canonical_request` |
| Grounding fix | A gatekeeper refusal ("not available in the patient's current record") no longer counts as `record_grounded`. Found by the new tests. | runner |

**Classifier check.**

| Set | Result |
|---|---|
| Smoke questions (82 unique, hand-reviewed) | 14/14 leaks flagged, 0 false positives |
| Probe studies | 10/10 flagged |
| Diagnosis-fishing probes | 5/5 flagged |
| Diagnosis-term false positives | 1 of 82 ("endocrine"; fixed by the generic-word list) |

This is in-sample: I tuned the rules on these answers. Section 3 has the out-of-sample check.

Known gaps:
- Paraphrased results with no study, analyte or unit word ("the kidneys are working normally") pass.
- A diagnosis term in a legitimate history answer is withheld ("no history of encephalitis").
- Ordered tests that echo a guessed diagnosis ("Confirmatory testing for pheochromocytoma") are counted
  (`n_dx_in_test_results`), not blocked, because a targeted order is legitimate.

### Evaluation conditions

- **Official** (primary, for benchmark comparability): SD-Bench as is. Report the official cost and, next to it,
  the corrected cost and the leak counts (`q_test_results`, `dx_leak_q`) as exploitation audits.
- **Guarded** (secondary): the training environment.
- Run all arms in both conditions. An arm trained in the guarded environment can't have learned to fish. If its
  leak counts rise in the official condition, that's an exploit flag.

## 2. Throughput

All runs use the same servers on Spark 2: Qwen3-8B (bf16, `:8101`) for gatekeeper, judge and pricing, and MedGemma
4B (`:8100`) as the policy. Logs: `results/v2_sdbench/bench/`. Cold caches in every run.

**Single episodes, temperature 0, synthetic cases 2 and 3**, run one at a time:

| Run | Change | Case 2 wall | Case 3 wall | Gatekeeper per call |
|---|---|---|---|---|
| before | official code: requests sent one at a time, prices one at a time | 2,180 s | 2,386 s | 37.5 s |
| after1 | gatekeeper requests within a turn in parallel; price lookups in parallel | 700 s | 782 s* | 8.5 s |
| after2 | after1 plus the policy stop-token fix (section 3) | 576 s | 490 s | 7.1 s |

\* Case 3 diverged in after1 (12 truncated turns, forced answer), so it isn't a like-for-like comparison. On case 2,
which made the same 39 requests in before and after1, parallel requests cut the episode 3.1×: gatekeeper time fell
from 1,462 s to 330 s and pricing from 320 s to 80 s. after1 and after2 differ in more than the stop fix, because
trajectories diverged (n = 2 per run), so the after2 speedup is not attributed to it.

Neither change alters what the gatekeeper is asked, the prompts, or the reward. A test checks that parallel and serial
runs give identical logs and costs.

**16 concurrent episodes, temperature 1.0** (cases 0 and 1, 8 samples each, guarded condition, `conc16_guarded.jsonl`):

| Measure | Value |
|---|---|
| Wall time, all 16 | 2,387 s (≈ 149 s per episode amortized) |
| Policy turns / output tokens | 176 turns / 42,860 tokens |
| Gatekeeper calls | 233 |
| Judge time per episode (one call) | 195–598 s under load (≈ 500 output tokens each) |
| Spark 2 GPU peak | 83 °C (watchdog limit 85 °C; no kill) |

**Bottleneck: Qwen decode.** Spark 2's GB10 decodes an 8B bf16 model at about 15 tokens/s per stream. One judge
reply (≈ 500 tokens of reasoning) is 35 s alone and several minutes when it shares the server with gatekeeper calls.
At temperature 1.0 the policy also spends most turns on rejected output (section 3), which inflates policy time.

Options not taken, because they change models or prompts and need your approval:
- FP8 Qwen roughly doubles decode speed, but it's a different judge and gatekeeper model.
- A shorter judge prompt, or a token cap on the judge, changes the reward signal.
- A second Qwen instance for the judge would need memory that GRPO needs on Spark 2 (section 4).

## 3. Validation

Tests: 58 pass (`tests/test_v2_sdbench*.py`), all deterministic and offline with a fake gatekeeper. They cover:
- question-channel abuse, charged in the corrected cost while the official cost is unchanged;
- duplicate and parroted requests, charged once;
- legitimate history questions, free;
- unavailable and refused results, free and not record-grounded;
- the guard, which withholds results from questions and never touches ordered tests;
- diagnosis fishing;
- compound-order pricing;
- parallel runs equal to serial;
- grounded-credit rules;
- the stop-token cleaner;
- the opt-in per-turn caps.

### Bugs found and fixed

| Bug | Effect | Fix |
|---|---|---|
| Policy didn't stop at `<end_of_turn>` | The text-only MedGemma's `generation_config.json` lists only `eos_token_id: 1`. vLLM generated past the turn end, with invented "user" turns and repeated `<diagnosis>` tags, until the 400-token cap. Inflated tokens and malformed answers. | Policy requests pass `stop_token_ids: [106]` (GRPO rollout and eval); `episode.clean_reply` cuts at the first turn marker as a backstop. In after2, `n_runon` = 0, so the server stops correctly. |
| Refusals counted as record-grounded | Arm C's gathering credit would pay for gatekeeper refusals ("not available in the patient's current record") | Runner: a `rejected` or guarded answer is never grounded |
| Parroted questions | "What is X?: 38.6°C" (the policy pasting an earlier answer) defeated the cache and repeat detection | Canonicalized before sending |

### Classifier, out of sample

The 16-episode run produced 208 unique questions the rules had never seen. I read all of them
(`results/v2_sdbench/bench/audit_conc16.txt`, labels from before the fix below):

| Label | n | Errors |
|---|---|---|
| `test_result` (withheld) | 56 | 1 false positive: an exam finding, "pathologic livedo" |
| `dx_leak` (withheld) | 7 | 0. All are "what is the significance of / your impression" questions, where the gatekeeper volunteered interpretation |
| `info` (passed) | 143 | 2 misses: "Order a chest CT" asked as a question ("…chest CT performed… did not reveal…"), and "significance of the elevated ESR and CRP" (a confirming interpretation) |
| `rejected` | 2 | 0 |

So roughly 96% of withheld answers were real results, and about 2% of free answers leaked something. The CT miss
and the livedo false positive are now fixed, with tests, and that makes those three items in-sample. Re-classifying
all 516 logged answers changed only those three labels.

New leak routes the classifier doesn't model:
- **Treatment course.** "Has there been any improvement?" got "fevers resolved within the first week of therapy…
  over the first two cycles of treatment", a post-diagnosis hint that chemotherapy was given. The gatekeeper answers
  from the whole chart, including events after the diagnosis.
- **Timeline probing.** "What is performed tomorrow?" got "a random skin biopsy", which was the diagnostic test for
  case 0. It was caught here only because it named a biopsy.
- **Gatekeeper interpretation.** "What is the significance of …" questions get clinical reasoning back. Caught here
  by diagnosis terms or result words, but not guaranteed.
- **Test orders inside question tags.** "Order a CBC" sent as a question got the result. The guard withheld it, but
  in the official condition it's free.

These are gatekeeper-prompt problems, not classifier problems. SD-Bench's prompt answers anything charted. A guard
that reads results can only catch what it recognises.

### Policy behaviour

Each finding's evidence is in brackets.
- **Repeated requests:**
  - rare at temperature 0 (0 repeats in after2);
  - at temperature 1.0, 2 episodes repeated 19 and 23 requests [conc16].
  - The cache serves repeats free, the corrected cost charges once, and the gathering credit ignores them.
- **Forced answers and malformed output (main finding, see below):**
  - at temperature 0, 0 of 4 episodes forced [after2, earlier smoke];
  - at temperature 1.0, 13 of 16 forced and 15 of 16 invalid [conc16].
- **Judge leniency:** the judge gave score 5 to a hedged paragraph ("likely a variant of autoimmune
  encephalitis") in before. The training judge is not validated (section 5).
- **Non-reproducibility:**
  - vLLM output isn't batch-invariant, so gatekeeper answers differ between cold caches.
  - Within a run the cache (first write wins) keeps them fixed.
  - Arms must share one gatekeeper cache, as the execution plan already says.

**Main validation finding: at temperature 1.0 the base policy usually breaks SD-Bench's turn format.**
- 139 of 194 turns mixed `<test>` and `<question>` tags, which SD-Bench rejects whole.
- 6 of 16 episodes got no valid request in 12 turns.
- The same cases at temperature 0 had 0 violations.
- 10 of 16 final answers were hedged, over-long paragraphs.

Rewards collapse with it:
- Judge scores were 1 for 10 of 16 episodes.
- Case 0's group of 8 had 7 identical rewards (std 0.08).

GRPO rollouts use temperature 1.0 (V1 setting). As things stand, most of the early learning signal would be about
format, not diagnosis. That is the execution plan's risk 4 (learning signal), showing up as a format problem.

**Lower temperature doesn't fix it** (`t07_guarded.jsonl`, cases 0 and 1, 4 samples each, temperature 0.7):
- 70 of 94 turns mixed tags.
- All 8 episodes were invalid; 6 were forced.
- Group reward std was 0.11 for both cases.

The failure is sticky:

| Run | Episodes whose first turn mixes | P(mix \| previous turn mixed) | P(mix \| previous turn clean) |
|---|---|---|---|
| T = 1.0 (conc16) | 12 / 16 | 107 / 136 (79%) | 17 / 42 (40%) |
| T = 0.7 | 4 / 8 | 60 / 70 (86%) | 6 / 16 (38%) |
| T = 0 (after2 + smoke) | 0 / 4 | – | 0 / 23 |

Greedy decoding happened to start clean on these cases. Any sampling temperature usable for GRPO doesn't. After the
"turn type violation" notice, the policy apologises and usually repeats the mixed turn.

## 4. RL readiness

**Verdict: not ready for a learning run. Ready for a plumbing smoke once the trainer topology is decided.**

### Blockers, in order

1. **Format failure under sampling (section 3).** With the base checkpoint, GRPO would mostly be learning SD-Bench's
   turn format, and groups like case 0's (7 of 8 rewards identical) carry almost no diagnostic signal. Options, your
   call, since each changes the design:
   - **(a) Shared format SFT** for all arms (the execution plan's own fallback for risk 4). Short multi-turn SFT on
     well-formed trajectories (questions-only or tests-only turns, one clean `<diagnosis>`). Arm A becomes this
     checkpoint. Environment and reward unchanged. *Recommended.* Trajectory source needs deciding: base-model
     temperature-0 trajectories filtered for validity are the cheapest and add no outside knowledge.
   - **(b) Training-environment relaxation:** split a mixed turn and answer it, or answer only the first tag type,
     instead of rejecting it. Same status as the question guard: a training deviation, applied to all arms, official
     evaluation unchanged. Faster to build, but the policy never learns the official rule, so it would keep losing
     turns in the official evaluation.
   - **(c) Lower rollout temperature:** tested, doesn't work (0.7 ≈ 1.0).
2. **Trainer placement.**
   - **Spark 2's budget:**
     - Qwen at gpu-mem 0.45 (≈ 54 GB) plus MedGemma at 0.20 (≈ 24 GB) leaves ≈ 40 GB, before the OS.
     - The GRPO trainer for 4B with sequences up to 14k tokens and gradient checkpointing is estimated at 30–45 GB,
       with the 262k-vocabulary logits dominating. This is an estimate, not measured.
     - It probably doesn't fit without shrinking Qwen, which slows the gatekeeper.
   - **V1 topology:** trainer on Spark 1, servers on Spark 2. But Spark 1 has had 4 thermal shutdowns under load
     and is still on the old driver and firmware.
   - Decide one of: fix Spark 1 cooling, or shrink Qwen's share on Spark 2 and measure.
3. **Training judge not validated.**
   - It gave score 5 to a hedged answer.
   - The κ ≥ 0.6 check with Agustina hasn't happened.
4. **`lam_cost` still 0.0.** By design until arm-A development statistics exist. Needs MIMIC data.
5. **MIMIC-IV-Ext-SDBench access** (credentialing) gates everything past synthetic plumbing.

### Can training and inference run concurrently?

Not within a run.
- TRL's GRPO alternates: each step generates all rollouts through the vLLM server, then trains.
- The rollout function batches all episodes of a step turn by turn, so generation itself is concurrent (16–32
  episodes in flight).
- The trainer GPU idles during rollouts, and the servers idle during the update.
- Running arms B and C at the same time would need two trainers and two MedGemma servers. That doesn't fit on two
  Sparks alongside Qwen.

### Duration estimate (measured rollouts, estimated update)

| Component | Basis | Per step (4 prompts × 8 = 32 episodes) |
|---|---|---|
| Rollouts | 16 concurrent episodes took 2,387 s (temperature 1.0). 32 in flight: assume 1.3–2× | 50–80 min |
| Policy update | Not measured. V1: 60 steps ≈ 7 h including rollouts; sequences here are ≈ 3–10× longer | 5–15 min |
| **Step** | | **≈ 1–1.5 h** |

- **One arm:** 30 steps ≈ 30–45 h.
- **B and C one after the other:** ≈ 2.5–4 days.
- This is at or above the execution plan's 1–1.5 days per arm. It doesn't include checkpoint selection (324
  episodes) or final evaluation (≈ 810 episodes).
- At the measured ≈ 150 s per episode amortized, those two take ≈ 13 h and ≈ 34 h.
- Episodes with a valid format would end sooner (fewer wasted turns) but make more gatekeeper calls, so the rollout
  figure is rough.
- Biggest available speedup: FP8 Qwen, which needs approval because it changes the judge and gatekeeper model.

### Recommended smoke (after blocker 2; plumbing only)

| Setting | Value |
|---|---|
| Arm, cases | B, the 5 synthetic cases (no MIMIC data touched) |
| Flags | `--max-steps 3 --per-device-batch 1 --grad-accum 16 --num-generations 8 --grad-ckpt --lr 1e-5 --beta 0.02 --rollout-temp 1.0 --save-steps 3` |
| Per step | 2 prompts × 8 rollouts = 16 episodes |
| Expected duration | ≈ 40–60 min per step, ≈ 2–3 h total |
| Environment | Guarded condition, corrected cost, stop fix, shared gatekeeper cache |

What the smoke must show:
- the NCCL weight sync works (a fresh server, first initialisation; see the V1 gotcha);
- the environment mask covers gatekeeper tokens only;
- rewards and audit metrics are logged per step;
- a checkpoint saves and reloads.

**Stop immediately if:**
- NCCL error or any crash;
- Spark GPU ≥ 85 °C, or an unexpected Spark 1 power-off;
- per-step wall time > 2× the estimate (> 2 h);
- KL > 0.5, or NaN loss;
- question-channel attempts (`guarded` + `dx_leak_q`) rise across steps. That's the policy learning to fish
  against the guard.

**Don't read the smoke as a learning result.** Expect `frac_reward_zero_std` to be high and KL tiny. With the
base checkpoint, invalid ≈ 90% is expected, not a stop criterion. The learning-signal check (within-group reward
std > 0 on most groups, invalid < 50%) belongs after blocker 1 is fixed, on development cases.

### A/B/C consistency

- **Shared across arms:**
  - every change in this document: guard, corrected cost, canonicalization, stop fix, parallel gatekeeper;
  - the format fix, if (a) or (b) is chosen.
- **Arm C's gathering term:** stays the record-grounded relevance proxy,
  `r_gather = min(new grounded ordered tests, 10)/10`, λ = 0.1. It is a proxy for relevance, not information gain.
  It pays for any new test with a charted result, relevant or not.
- **Φ:** the contrastive Φ reward failed its pre-registered check and isn't used. It stays available as an audit
  only (`log_phi`, off by default).

## 5. Scientific risks

| Risk | Status |
|---|---|
| Gatekeeper and judge are unvalidated Qwen3-8B | Judge leniency seen. The gatekeeper contradicts the chart (case 1: "near-syncope… a brief loss of consciousness", where the record says he denied syncope) and contradicts itself ("denies alcohol… drinks rarely"). Qwen applies the diagnosis safeguard inconsistently |
| Leak guard is a regex classifier | ≈ 2% misses out of sample, plus leak routes it doesn't model (treatment course, timeline probing). RL will search for gaps. Mitigation: the `guarded`/`dx_leak_q` trend is a stop criterion, and both evaluation conditions are reported |
| Guarded environment ≠ official SD-Bench | A deviation that must be reported. Official remains the primary evaluation |
| The official SD-Bench gatekeeper may leak less than Qwen | Not checked (paid API) |
| Base checkpoint can't hold the format under sampling | Blocker 1 |
| Gatekeeper answers depend on the cache | Shared cache across arms. Report that answers come from the cache, not a fresh call |
| Development and evaluation sets | Untouched. Everything here used the 5 public synthetic cases |
