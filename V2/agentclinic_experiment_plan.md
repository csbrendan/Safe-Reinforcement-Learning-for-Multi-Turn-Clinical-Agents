# V2 plan: AgentClinic as an external interactive clinical benchmark

*Drafted 2026-11-07. Planning only: nothing has been trained, installed, or downloaded beyond a read-only
clone of the AgentClinic repository. This expands workstream **W5, Option A** of [`V2/PLAN.md`](PLAN.md).*

*Update 2026-11-09: AgentClinic is now the preferred out-of-distribution target for checkpoints from the plan's Experiment A (see Experiment C). The P1–P2 pilot still validates the harness, on validation cases only. The §6 split must be redone before use, because it predates the finding that 27 extended MedQA cases leak the diagnosis and cases used in smoke tests must stay out of the test subset.*

**How to read this document.** Every statement is tagged as one of:
- **[VERIFIED]**: checked against the AgentClinic code, the paper, or our own repository.
- **[PROPOSED]**: an experiment or design we intend to run.
- **[OPEN]**: an open question or something not yet checked.

**References.**
- AgentClinic code was read at commit `b6fbe22` (main, 2026-10-03). No code was executed.
- Paper: Schmidgall et al., *npj Digital Medicine* 9, 499 (2026), DOI 10.1038/s41746-026-02674-7.
- An earlier arXiv version exists: 2405.07960.

---

## 1. Executive summary and motivation

**Research question.** Does the information-gathering behaviour that RL taught our models in the controlled
simulators carry over to an interactive clinical environment that someone else built? The alternative is that
the improvement is specific to our simulator's rules and reward.

**Why AgentClinic.** It is the closest public match to our setup **[VERIFIED]**:
- A doctor agent must gather information from a simulated patient (and a measurement agent for test results)
  before stating a diagnosis.
- It was independently constructed.
- Its code is MIT-licensed.

**The central difficulty.** Our policies learned one narrow skill **[VERIFIED]**:
- They pick from 18 fixed findings, using a JSON ask/commit format.
- They commit to one of five *dispositions* (discharge … emergency treatment).

AgentClinic asks for a free-text *diagnosis* among hundreds of conditions and judges it with an LLM
**[VERIFIED]**. A transfer study therefore mostly tests whether the learned *procedure* survives a change of
format and domain: how many questions, when to stop, and whether to commit prematurely. Diagnostic accuracy
will mainly reflect the shared MedGemma base model, so **procedure metrics are primary and accuracy is
secondary**.

**Recommendation.** Run a small **zero-shot pilot** (§10, step P0–P2) first:
- one dataset, 50 validation cases;
- the diagnostics checkpoints we already have;
- a local patient simulator.

Commit to the full integration only if the trained models can operate the interface and differ from base on
procedure metrics.

---

## 2. Verified AgentClinic capabilities and limitations

| Topic | Finding | Evidence |
|---|---|---|
| Public data | Only **MedQA** (107 cases; *extended*: 215) and **NEJM** (15; *extended*: 120) are in the repo **[VERIFIED]** | repo file listing; record counts parsed |
| Not distributed | MIMIC-IV (200), Spec (260, from MedMCQA), Lang (749), and all tool code (CoT, RAG, notebook) are absent **[VERIFIED]** | loader opens `agentclinic_mimiciv.jsonl` (`agentclinic.py:288`), but the file does not exist |
| MIMIC access | "Can be requested by the authors upon completion of PhysioNet training" **[VERIFIED]**. PhysioNet's DUA notice forbids sending MIMIC data to e.g. the OpenAI API **[VERIFIED]** | Nature paper; README; PhysioNet notice 2023-04-18 |
| Splits | **None.** Every loader evaluates the whole file in order. The MedQA cases were generated from MedQA's *test* split **[VERIFIED]** | `agentclinic.py:601`; `generate_cases/gen_medqa_tutorial.py:8` |
| Case structure (MedQA) | `OSCE_Examination{Objective_for_Doctor, Patient_Actor, Physical_Examination_Findings, Test_Results, Correct_Diagnosis}`. Test results are nested key/value pairs (median ≈5 tests, ≈8 exam findings) **[VERIFIED]** | parsed JSONL |
| Case structure (NEJM) | Free-text `patient_info` / `physical_exams`, 5 answer options, image URL (linked, not stored) **[VERIFIED]** | parsed JSONL |
| Doctor interface | Free text, 1–3 sentences. `REQUEST TEST: […]` goes to the measurement agent, `DIAGNOSIS READY: […]` ends the episode, anything else goes to the patient **[VERIFIED]** | `agentclinic.py:524,626,639,646` |
| Turn limit | Default 20 doctor inferences. No diagnosis by then = incorrect **[VERIFIED]** | `:676`, `:639-644` |
| Scoring | Moderator LLM answers "Are these the same? Yes/No". Accuracy is the only metric in code **[VERIFIED]** | `:566,640` |
| Backends | Doctor, patient, measurement and moderator default to OpenAI `gpt-4-turbo-preview` via `openai==0.28`. No configurable `base_url`. The HuggingFace path is broken (`raise Exception("Sorry, fixing TODO :3")`) **[VERIFIED]** | `agentclinic.py:21-172,669-672`; `requirements.txt` |
| Local vLLM | Likely possible through `OPENAI_API_BASE` with openai 0.28, or a ~10-line patch **[OPEN: untested]** | — |
| Known bugs | (i) Images never reach the answer (`if` instead of `elif`, `:67`). (ii) `NEJM_Ext` never sets the image flag (`:624`). (iii) The final-turn notice is dropped (`:631/647/655`). (iv) Exceptions are retried silently 30× with 20 s sleeps **[VERIFIED: by reading the code]** | as cited |
| Licence | Code MIT; data "open under an MIT license" per the paper **[VERIFIED]**. Upstream MedQA and NEJM content licences **[OPEN]**; NEJM Image Challenge text is NEJM copyright | `LICENSE.txt`; paper |
| Contamination | High. MedQA test items and NEJM challenges are public; the paper concedes models may have seen MedQA test **[VERIFIED]**. Whether MedGemma saw MedQA material **[OPEN]** | paper |
| Not available | No oracle, no question cost, no severity or high-risk labels, no structured patient state beyond the case dict **[VERIFIED]** | — |

**Consequence [PROPOSED]:** we should **not run AgentClinic's script as-is**, for four reasons:
- it pins `openai==0.28`, which conflicts with our shared environment, and CLAUDE.md says not to create a new venv;
- it hard-codes GPT-4;
- it swallows errors;
- it has the bugs above.

Instead, we would write a thin harness of our own that reproduces AgentClinic's protocol (its prompts, cases,
turn limit and moderator question). The MIT licence permits this. We would validate the harness by reproducing
the paper's reported accuracy for one reference model before using it (§10, P1).

---

## 3. Our V1 simulator compared with AgentClinic

| Dimension | V1 diagnostics (deephedge) | AgentClinic-MedQA |
|---|---|---|
| Case source | Generated on demand from hand-designed priors/trees (see `docs/provenance_audit_2026-11-07.md`) | GPT-4-generated OSCE cases from MedQA test questions, manually checked |
| Patient | Deterministic lookup of binary findings | LLM role-playing a patient from a case dict |
| Question space | 18 fixed findings (`intent` keys) | Open-ended questions, or `REQUEST TEST: <any test>` |
| Terminal action | 5 dispositions | Free-text diagnosis (open vocabulary) |
| Reward / score | Exact verifier `U[μ][a] − 0.03·n_asks` | Binary LLM-judge match; no cost for questions |
| Oracle | Exact adaptive oracle | None |
| High-risk subgroup | Defined by the latent state | Not labelled |
| Horizon | 12 turns (≈3.9 oracle asks) | 20 inferences |
| Determinism | Fully deterministic | Stochastic LLM patient, measurement and judge |

**Implication.** Of the V1 metrics, only behavioural ones transfer directly: questions asked, stopping, and
commitment timing. Oracle-grade support, information gain, and high-risk value do not. §7 defines replacements.

---

## 4. Experimental matrix and controls

### Checkpoints available today [VERIFIED]
- **Base:** `checkpoints/medgemma4b_text`, text-only MedGemma-4B.
- **Outcome-only GRPO** (diagnostics deephedge), 3 seeds: `dh_grpo_base`, `dh_grpo_base_s2`, `dh_grpo_base_s3`.
  Each has `checkpoint-20/40/60` and `final`.
- **Credit-assignment GRPO** (λ=0.2), 3 seeds: `dh_grpo_infogain`, `dh_grpo_ig_s2`, `dh_grpo_ig_s3`.
- **Decoupled GRPO**, 1 seed: `dream_dh`. Step 40 was the held-out-selected checkpoint.
- **Biased-SFT family:** `dx_sft_biased30` with RL repairs `dx_repair_grpo` (credit) and `dx_repair_dream`.
- `dx_sft_v3` is an older diagnostics SFT that over-gathers, never commits 71% of the time, and was never an RL
  start point.
- All adapters are LoRA (r=32, α=64, all projection layers) on the same base.

**Important:** the main diagnostics RL arms started from **base, not SFT**. A clean "SFT-only" arm matched to
plain and credit GRPO **does not exist**.

**Options for the SFT arm [PROPOSED]:**
- **(a) Primary.** Evaluate `dx_sft_biased30` only together with its repair arms, as a separate,
  clearly-labelled family.
- **(b) Optional.** Train a new oracle-trajectory SFT on deephedge, then plain and credit GRPO from it. This
  needs new 4B runs (≈3 × 7 h per seed), so it belongs to V2 compute, not the pilot.

### Phase 1 — Zero-shot transfer [PROPOSED]
| Arm | Checkpoint(s) | Seeds |
|---|---|---|
| Base | `medgemma4b_text` | — |
| Outcome-only GRPO | `dh_grpo_base*`, `final` | 3 |
| Credit-assignment GRPO | `dh_grpo_ig*` / `dh_grpo_infogain`, `final` | 3 |
| Decoupled GRPO (optional) | `dream_dh`, `final` *and* step 40 (labelled as selected-on-held-out) | 1 |
| Biased-SFT family (optional) | `dx_sft_biased30`, `dx_repair_grpo`, `dx_repair_dream` | 1 |

**Controls [PROPOSED]:**
1. Identical prompts, cases, patient, measurement and judge models, decoding settings and turn limit for every arm.
2. `final` checkpoints only, since no checkpoint selection is possible without a validation split. Selection, if
   used, happens on the §6 validation split.
3. A **format-only control**: base model given the same few-shot interface examples. This separates "RL taught
   procedure" from "RL taught the JSON format".
4. A **re-test on our own simulator** to confirm the loaded checkpoints reproduce V1 numbers. This guards against
   loading or merging errors.

**Diagnostics only.** Refill and titration are excluded, because their tasks and action spaces have no
AgentClinic counterpart.

### Phase 2 — Interface-adapted transfer [PROPOSED]
1. Give every arm **identical** supervised interface training, so the model learns to ask open questions, use
   `REQUEST TEST`, and state a diagnosis.
2. Then compare whether credit-assignment GRPO keeps an advantage over outcome-only GRPO and over base+adaptation.

**Adaptation data must not come from AgentClinic evaluation cases.** Candidates:
- **(i)** AgentClinic-style cases generated from the MedQA **train** split with AgentClinic's own generator
  template. This is disjoint from the AgentClinic cases, which come from MedQA test.
- **(ii)** Our deephedge trajectories re-rendered as free-text dialogue.
- **(iii)** Both.

Trajectories should be *format* demonstrations. Their question counts should not be pushed toward any arm's
behaviour, for example by sampling from varied procedures rather than from an oracle **[OPEN: how best to keep
the adaptation data procedure-neutral]**.

**The overwriting problem.** SFT on top of RL may erase the learned procedure. Controls:
1. Keep every pre-adaptation checkpoint (do not overwrite adapters).
2. Use a short, low-learning-rate adaptation with the **same data, steps and seed for every arm**.
3. Measure each adapted model on **our own simulator** before and after adaptation, to quantify how much RL
   behaviour survived.
4. Include a **base+adaptation** arm as the no-RL control.
5. Try a **parameter-isolated variant**: a separate interface LoRA stacked on the frozen RL adapter.

### Phase 3 — Optional RL inside AgentClinic [PROPOSED, feasibility only]

**Outcome reward:** the binary judge-correct signal, minus a question cost that we would add ourselves.
- AgentClinic has no question cost, so the cost value is our choice **[OPEN]**.
- The signal is noisy: an LLM judge, a stochastic patient, and an unexplored judge sensitivity.

**Process reward:** exact posterior information gain is **unavailable**, because there is no explicit hidden
state or likelihood model. Possible substitutes:
- **(a)** A heuristic relevance credit: a requested test matches a key in the case's `Test_Results` (non-matches
  return "NORMAL READINGS", `:554`).
- **(b)** A novelty penalty on repeated questions.
- **(c)** An LLM-judged evidence-sufficiency score. Our earlier Claude process-reward
  alpha sweep was negative (`framework/procreward/`), so treat this cautiously.

All of these are approximations; claims must say so.

**Common evaluation:** the same §7 protocol, the same test split, checkpoint selection on validation only.

**Feasibility blockers [OPEN]:**
- Only 215 MedQA cases exist, and some must be held out, so in-environment RL would train on an AgentClinic-style
  set generated from MedQA-train.
- Every episode needs ≤ 40 patient, measurement and judge calls inside the GRPO `rollout_func`.
- Rollout latency on a two-node Spark setup is unknown.

---

## 5. Interface adaptation architecture [PROPOSED]

```
policy (MedGemma-4B + LoRA, vLLM)            fixed environment models (local vLLM, temp ≈0)
  │ JSON or native text                       ┌─ patient agent   (role-plays Patient_Actor)
  ▼                                           ├─ measurement     (returns Test_Results / "NORMAL READINGS")
adapter: parse → route ───────────────────────┤
  ask  {"intent": free-text question}  → patient
  ask  {"intent": "test: <name>"}      → measurement   (= REQUEST TEST)
  commit {"decision": <diagnosis>,
          "disposition": one of 5}     → moderator (diagnosis) + logged disposition
  ▼
episode log (JSONL, one record per turn) → metric scripts (§7)
```

**Two prompt formats, both run for every arm:**
- **JSON (primary):** our ask/commit schema with an *open* `intent` and a free-text `decision`. Our parser
  (`framework/parsing.py`) already accepts any string intent **[VERIFIED]**. This is closest to the training
  format.
- **Native:** AgentClinic's original free-text doctor prompt.

Reporting both shows how much of any difference depends on format.

**Fairness:** base receives the same JSON instructions and examples, so the RL models have no format advantage.
Format compliance (valid action rate) is reported per arm.

**The `disposition` field** lets us reuse the decision our models were actually trained on. It is scored against
acuity labels we would add (§7). It is *not* part of AgentClinic's scoring.

**Environment models.** The patient, measurement and moderator must be **fixed and different from the policy**.

| Choice | Advantages | Drawbacks |
|---|---|---|
| Local open model on the second Spark node | Free; MIMIC-safe | **[OPEN: which model; fidelity check needed]** |
| API model (Claude) for public MedQA/NEJM | Higher fidelity | Cost; not allowed for MIMIC |

Proposed default: a local model, with a Claude-API moderator cross-check on a subsample to estimate judge
disagreement.

---

## 6. Data access and split strategy [PROPOSED]

- **Usable now:** MedQA-extended (215, text-only) and NEJM-extended (120). For NEJM, text only, because images
  are external links and broken in the code.
- **Later, only if justified:** MIMIC-IV cases need PhysioNet credentialing, an author request, and local-only
  inference.
- **Split (fixed once, seeded, stratified by specialty if derivable):**

  | Split | Cases | Use |
  |---|---|---|
  | Validation | 65 MedQA + 30 NEJM | prompt tuning, checkpoint selection |
  | **Test (untouched)** | 150 MedQA + 90 NEJM | evaluated once per final model |

  Record case IDs in the repo before any evaluation runs.
- **Contamination:** absolute accuracy may be inflated for all arms because of the shared base model. Treat
  arm-versus-base *differences* as the quantity of interest.
- **[OPEN]** Check the MedGemma technical report for MedQA training use and state it.

---

## 7. Evaluation and clinician-review protocol [PROPOSED]

| Metric | Definition | Automatic? |
|---|---|---|
| Diagnostic accuracy | Moderator "same?" = yes | Yes (LLM judge). Report judge agreement on a subsample with a second judge, and clinician spot-checks |
| Valid-action rate | Turns parsed as ask/test/commit | Yes |
| Questions / tests per case | Count of patient questions and `REQUEST TEST` | Yes |
| Premature commitment | Diagnosis after ≤1 question, or before any test when the case has tests | Yes (heuristic) |
| Relevant vs unnecessary tests | Requested test matches a `Test_Results` key vs returns "NORMAL READINGS" | Yes (fuzzy match; validate on a sample) |
| Relevant vs unnecessary questions (patient) | Whether the answer carried case-specific content | Partly (LLM-assisted); clinician subsample |
| Evidence sufficiency at commitment | Were the findings needed to support the diagnosis obtained? | **Clinician** (LLM pre-screen optional) |
| Appropriate escalation | Logged `disposition` vs a per-case acuity label | Automatic once labels exist; **labels need a clinician** |
| High-risk subgroup | Cases labelled emergent/high-acuity | Needs clinician acuity labels |
| Unsafe or unsupported escalation | Clinician judgement on sampled trajectories | **Clinician** |

**Clinician review (W6, Dr. Saenz):**
- **Sample:** 60–100 test trajectories, stratified by arm and by correct/incorrect.
- **Blinding:** arm identity hidden and order randomised.
- **Rubric:** evidence sufficiency, premature commitment, unnecessary questioning, unsupported or unsafe
  escalation, and an overall procedure rating.
- **Reliability:** a second rater or repeated ratings on a subset to estimate κ.
- **Acuity labels:** she would also label every test case's acuity (three levels) to define the high-risk subgroup.

**Statistics:**
- Per-case paired comparisons across arms, since all arms see the same cases.
- Bootstrap CIs over cases. Seeds are reported separately and pooled where available.
- k = 3 samples per case at the V1 rollout temperature, plus a greedy run.

---

## 8. Compute and feasibility

**Calls per case [VERIFIED]:** at most 20 doctor turns, each followed by one patient or measurement call, plus
one moderator call. That is ≤ 41 calls per episode, and the prompt grows every turn.

**Zero-shot pilot (P2) [PROPOSED]:**
- 50 validation cases × ~8 policies × 1 sample ≈ 400 episodes ≈ ≤ 16k calls.
- All local on the Sparks: a policy server on one node and an environment-model server on the other.
- Estimated at hours, not days **[OPEN: measure]**.

**Full Phase 1 test:**
- 240 cases × ~11 policies × (3 samples + greedy) ≈ 10.5k episodes ≈ ≤ 430k calls.
- Local throughput decides wall-clock. API cost would apply only to cross-check subsamples.

**Phase 2:** one short LoRA adaptation per arm (≈ 9–11 runs), each far cheaper than a 60-step GRPO run.

**Phase 3:** multi-turn GRPO with LLM-in-the-loop environment calls. Expect several times the V1 cost of ≈ 7 h
per 60-step run; needs a pilot.

**27B (W4):** run the same harness on Azure after the 4B pilot. Environment-model calls dominate, not policy size.

**Infrastructure:** our own harness needs no new heavy dependencies; it reuses vLLM, our parser, and plain HTTP
calls. AgentClinic's `openai==0.28` pin is avoided.

---

## 9. Key risks and expected failure modes

1. **Format collapse.** RL-trained adapters may produce only our 18 intents or 5 dispositions and fail at open
   diagnosis. Mitigations: the format-only control, the valid-action rate, and Phase 2.
2. **Accuracy dominated by the base model.** Arms may tie on accuracy, so the result rests on procedure metrics.
   Declare this before running.
3. **Procedure doesn't transfer.** Credit GRPO's ask-more tendency may be tied to our intent vocabulary. This is
   a legitimate negative result and worth reporting.
4. **Over-gathering looks good.** AgentClinic has no question cost, so arms that ask more may score higher. Report
   burden alongside accuracy, and add a cost-adjusted score.
5. **Environment-model noise.** A stochastic patient and LLM judge add variance. Fix the temperature, use the
   same models for all arms, and measure judge agreement.
6. **Phase 2 overwrites RL behaviour.** Covered by the before/after own-simulator tests and the stacked-adapter
   variant.
7. **Contamination** inflates absolute accuracy (§6).
8. **Harness drift from AgentClinic.** Reproduce a published number before trusting results (P1).
9. **Licensing.** Do not redistribute NEJM text; keep it in the cloned repo only.
10. **Small n.** 240 test cases, and the high-risk subgroup will be smaller still. Use CIs, paired tests, and
    pre-registered primary metrics.

---

## 10. Prioritized roadmap

| Step | Work | Gate to continue |
|---|---|---|
| **P0** | Freeze the case split (validation/test IDs committed). Ask Dr. Saenz about acuity labelling. Check the MedGemma report for MedQA use | Split recorded |
| **P1** | Write a minimal harness (AgentClinic prompts plus our adapter). Reproduce AgentClinic's MedQA accuracy for one reference model within the paper's reported range, on a 30–50 case subset | Reproduction within reasonable tolerance |
| **P2: pilot** | **Zero-shot, 50 validation cases, JSON format.** Arms: base, base+format control, outcome-only s1–s3, credit s1–s3. Metrics: valid-action rate, questions/tests per case, premature-commitment rate, accuracy | **Continue if** trained arms reach ≥ 80% valid actions **and** at least one procedure metric differs from base beyond seed spread. **Stop or revise if** RL arms mostly fail the format (go to Phase 2 directly) or are indistinguishable from base |
| P3 | Full Phase 1 on the test split (both formats, decoupled and biased-SFT optional). Clinician review batch 1 | Pre-registered analysis |
| P4 | Phase 2 adaptation with overwrite controls | Before/after own-simulator check |
| P5 | Phase 3 feasibility pilot (≤ 20 GRPO steps) | Only if P3/P4 show a transferable procedure effect |
| P6 | Repeat P3 (and P4 if run) with 27B (W4) | — |

**Smallest scientifically meaningful pilot:** P1 + P2. One dataset, 50 validation cases, seven existing
checkpoints plus a format control, local models only, and no new training. It answers the gating question:
*do RL-trained policies behave differently from base on an independent interactive benchmark, in a way our
procedure metrics can detect?*

---

## Open research questions (summary)
- How do we make Phase 2 adaptation data procedure-neutral?
- What question cost, if any, should be added for cost-adjusted scoring and Phase 3?
- Which local model is a faithful enough patient, measurement and judge? How sensitive is accuracy to the judge?
- Can heuristic test-relevance credit stand in for information gain? Does it correlate with clinician
  evidence-sufficiency ratings?
- Is MIMIC-IV access worth the credentialing and local-only constraints, compared with MedQA and NEJM alone?
