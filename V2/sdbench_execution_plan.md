# V2 execution plan: gathering-aware GRPO on SD-Bench

*Drafted 2026-11-11. Planning only: nothing below has run. It replaces the gate-transfer work in [`sdbench_readiness.md`](sdbench_readiness.md) (sections 4 and 7), whose infrastructure findings still hold. V2 does not reproduce V1's gates, oracle, BC-gap, support@k or coefficient sweeps.*

## Question

Does adding credit for useful information gathering to outcome-based GRPO produce a better and safer diagnostic agent on SD-Bench than outcome-only GRPO? Specifically, does it get more diagnoses right on high-risk and difficult cases, gather more appropriately, and avoid shortcuts such as premature diagnosis and under-testing? This is a hypothesis; a null or negative result is reportable.

## Minimal three-arm experiment

| Arm | Training | Reward |
|---|---|---|
| **A, no RL** | None | n/a |
| **B, standard GRPO** | GRPO | `R_out` = judge outcome − λ_c × normalized investigation cost − format penalty |
| **C, gathering-aware GRPO** | GRPO | `R_out` + λ_g × `R_gather` |

- **Shared across arms:** checkpoint (base `medgemma4b_text`, or one shared SFT if needed), training cases, seeds, rollout budget, gatekeeper cache, training judge, evaluation.
- **Model:** MedGemma 4B now. 27B comes later and only repeats the winning comparison.
- **Coefficients:** λ_c and λ_g are each set once, before training, from base-policy rollout statistics. No sweep.

### Data splits (MIMIC-IV-Ext-SDBench, assuming that route)

| Use | Cases | Notes |
|---|---|---|
| RL training | Official train, 163 | The only cases B and C ever train on |
| Development | 54 validation cases, fixed by seed | Calibrating λ and selecting checkpoints. Not reported as a result |
| Primary evaluation | Remaining 200 validation + official test, 71 | Test sealed until all arms are frozen. Report pooled (as published) and per split |

Arms are compared **paired on the same cases**, using McNemar's test and a paired bootstrap. With about 270 evaluation cases, a pass-rate difference smaller than about 8–10 points probably won't be detectable. The plan states that limit up front.

## Reward design

### Outcome reward (B and C)

- **Score:** the training judge's score on SD-Bench's official 1–5 rubric, scaled to [0, 1]. Using the score rather than pass/fail keeps reward variance within groups if the 4B model's pass rate is low.
- **Cost:** SD-Bench's cost model, normalized per case.
- **Format penalty:** a deterministic penalty for no `<diagnosis>`, more than one diagnosis, hedged lists (e.g. "A or B"), or exceeding a length cap.

### Training-judge validation, before any GRPO

1. Agustina (credentialed) labels about 120 diagnosis-versus-reference pairs blind, drawn from base-model outputs on the training and development cases, stratified by judge score. **Go:** judge-clinician κ ≥ 0.6 and false-pass rate ≤ 10%.
2. **Adversarial probes:** hedged lists, overly broad answers ("infection"), verbose answers, reference synonyms and near misses. The judge must reject the shortcuts.
3. Choose the judge model by these results. Qwen3-8B already failed one case, so the candidates are larger open models that fit Spark 2.

### Exploitation audit during and after training

- **During training:** log train-judge score, eval-judge score and format-violation rates per checkpoint. A rising gap between the two judges is an exploit flag.
- **After training:** a clinician spot-checks a sample of each arm's evaluation transcripts.

## Gathering credit (`R_gather`, arm C only)

**Goal:** reward a test or question only when its result makes the correct diagnosis more evident, so ordering alone earns nothing.

**Definition.** A frozen reader model (frozen base MedGemma 4B, already loaded on the rollout server) scores

  Φ(S) = mean per-token log P(reference diagnosis | case presentation + results revealed so far)

using only revealed findings. The policy's own text is excluded, so narration can't inflate Φ. Then

  `R_gather` = clip(Φ(S_final) − Φ(S_0), 0, cap)

where S_0 is the opening presentation and S_final the set of results revealed by the end of the trajectory.

**Properties:**
- It's the realistic analogue of V1's ΔH: evidence gathered that makes the correct answer recoverable, independent of whether the policy then commits correctly.
- Irrelevant and repeated tests add nothing to Φ but still pay cost. Results the gatekeeper synthesizes carry little information about the reference, so they earn little.
- Per-step ΔΦ is logged so we can see which tests earned credit.

**Pre-checks.** Cheap, on training cases only, not a gate:
1. Φ(full record) > Φ(presentation) on most cases. Otherwise the signal is too weak to use.
2. ΔΦ is higher for tests the clinician marks decision-relevant than for redundant ones, checked on a small labeled sample.

**Fallback.** If either pre-check fails, use record-grounded new-test credit (`is_synthetic = False`, not repeated) and label it as a relevance proxy, not information gain.

**Kept out of evaluation.** Φ is C's training signal, so evaluation never uses it.

## Evaluation (independent of the training reward)

- **Harness:** the official SD-Bench harness and baseline-agent prompt for every arm. LoRAs are served by name, with stop `[1, 106]` and the merge-roles template.
- **Settings:** temperature 0, 12-turn cap (70-turn check on 20 cases).
- **Judges:**
  - The **eval judge** is a different model family from the training judge.
  - The official o3/GPT-5 judge is an option only through a verified zero-retention deployment, with your approval.
  - A clinician reviews a sample.

| Metric | Definition |
|---|---|
| Diagnostic correctness | Eval-judge pass rate (≥4) and mean score, with clinician agreement reported |
| Cost | SD-Bench cost per case |
| Tests and questions | Counts per case; % record-grounded; % repeated |
| Premature diagnosis | Wrong diagnosis committed before any record-grounded test, or with fewer record tests than a pre-set floor |
| Over-testing | Cost above the case's documented workup cost by a pre-set margin, or synthesized tests above a threshold |
| Invalid trajectories | No diagnosis, format violation, context overflow |
| High-risk | Every metric on the clinician-designated can't-miss subset, labeled blind before evaluation |
| Difficult cases | Every metric on cases arm A fails in at least 3 of 4 samples on development and evaluation. Secondary, since difficulty is defined by the model |

**Primary endpoints, pre-registered:**
1. C − B in eval-judge pass rate on the high-risk subset.
2. C − B in premature-diagnosis rate.
3. C − B in overall pass rate.

Cost and over-testing are co-primary safety checks, so C can't win by over-testing.

## What exists and what to build

| Component | Status |
|---|---|
| SD-Bench harness, local model routing, gatekeeper, judge, costing | ✅ `~/v2_bench/sdbench`, separate venv |
| MedGemma serving: stop fix, template, LoRA by name | ✅ Smoke-tested |
| 2-node GRPO trainer, `env_mask` (`_env_mask`), NCCL procedure | ✅ `framework/rl/grpo_disagg.py`, reused |
| **SD-Bench rollout function:** tag parser, gatekeeper calls with cache, masking of gatekeeper and system text, longer limits | ❌ Build. The current rollout is JSON ask/commit with 128-token turns and a 512-token `max_completion_length`. SD-Bench transcripts are 4–16k tokens, so these limits become arguments (shared code, edited, not copied) |
| **Deterministic test results:** cache keyed on (case, normalized test), temperature 0, shared by training and evaluation | ❌ Build |
| **Reward module:** judge call, cost, format rules, Φ reader | ❌ Build |
| **Evaluation:** metric extractor over SD-Bench result JSON, second judge, report adapted from `framework/report.py` | ❌ Build |
| **Judge-validation kit:** pair export for clinician labeling, adversarial probe set, agreement stats | ❌ Build |
| SFT warm-up | Only if needed: if base invalid-trajectory rate on development is above about 20%, run one shared format-only SFT |

Each component gets plumbing tests on the 5 synthetic cases only, with no experiments on them. Engineering is about 4–5 days and can start before data arrives.

## Blockers

1. **Data, now on the critical path.** Shortest compliant route: Brendan's own PhysioNet credential.
   - **Steps:** CITI "Data or Specimens Only Research" course (a few hours), credentialing application with a reference (review time varies, often 1–2 weeks), then sign the MIMIC-IV-Ext-SDBench DUA and download with `wget`.
   - **Why not Agustina's credential:** it doesn't extend to Brendan or these machines.
   - **NEJM route:** in parallel via Agustina, after confirming the terms allow training and local storage.
2. **Releasing models trained on MIMIC.** PhysioNet restricts sharing derived data and may require credentialed-access release of trained weights. Check before promising public adapters.
3. **Clinician time.** Agustina: about 120 judge-validation pairs, high-risk labels for about 488 diagnoses, a small test-relevance sample and post-hoc transcript checks. Roughly 1–2 days in total.
4. **Learning signal.** If the 4B base rarely scores well, reward variance within groups may be near zero. Check on development before training; a shared SFT is the fix.
5. **Spark 1 stability.** Run training with telemetry. Spark 2 holds the rollout server, gatekeeper and training judge.

## Compute (rough; recalibrated from the first development baseline)

All local, so $0 in API or cloud spend unless you approve an official eval judge.

| Step | Episodes | Wall time |
|---|---|---|
| Arm A on development, plus λ calibration and Φ pre-checks | ≈ 60–100 | ≈ 3–4 h |
| GRPO B: 30 steps × 4 cases × 8 rollouts | 960 | ≈ 1–1.5 days |
| GRPO C: same budget | 960 | ≈ 1–1.5 days |
| Development checkpoint selection, 2 arms × 3 checkpoints × 54 cases | 324 | ≈ 6–8 h |
| Final evaluation, 3 arms × ≈270 cases, plus second-judge rescoring | ≈ 810 | ≈ 1–1.5 days |
| **Total** | | **≈ 4–5 days of Spark time** |

**Memory:**
- **Spark 1:** LoRA trainer; transcripts of 8–16k tokens need gradient checkpointing, about 40–70 GB.
- **Spark 2:** MedGemma rollout server plus gatekeeper and judge, about 70–90 GB. A larger judge may not fit alongside; if not, the judge runs on Spark 1 between steps or is scored offline.

## Order of work

1. Start the PhysioNet credential today, and send the NEJM terms question.
2. Build the rollout function, cache, reward module and evaluation extractor; plumbing tests on synthetic cases.
3. When data arrives: freeze splits, collect clinician high-risk labels, validate the judge.
4. Run arm A on development, set λ_c and λ_g, run the Φ pre-checks, and decide whether a format SFT is needed.
5. **Stop for your approval**, then run the B and C diagnostic, select checkpoints on development, and run the final evaluation once.
6. **Later:** AgentClinic OOD evaluation of frozen A, B and C checkpoints, reporting differences in simulator, judge and protocol. 27B scaling, if 4B shows an effect.
