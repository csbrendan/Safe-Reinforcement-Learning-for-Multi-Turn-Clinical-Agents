# V2 plan — from arXiv preprint to journal article

*Created 2026-11-01; research strategy expanded 2026-11-08 and reorganized 2026-11-09. Status: proposed plan. No V2
training has started, and every experiment below is proposed, not completed. External-benchmark smoke tests are done
except the V1 stop-token check; see [`benchmark_smoke_report.md`](benchmark_smoke_report.md). v1 (arXiv preprint,
submitted 2026-11-07) is a separate track and is NOT blocked by anything here.*

## Goal

Turn the current paper into a journal-grade article (style reference: Nature Machine Intelligence) by fixing every
major weakness of v1, accepting extra cloud spend to compress the timeline, and submit the result to a journal. V1
stays frozen as a citable preprint. Whether V2 is an expanded version of that paper or a separate one is an open
decision; see [`../paper/v2-extension/MANUSCRIPT_PLAN.md`](../paper/v2-extension/MANUSCRIPT_PLAN.md).

The thesis stays the same — *RL for clinical multi-turn agents must be evaluated as procedure learning; verifiable
outcome rewards are not enough* — but v2 must back it with adequately powered, scaled, externally checked, and
clinician-validated evidence.

## V2 research strategy (reorganized 2026-11-09)

*This is a proposed research plan, not a commitment to a particular simulator. It organizes the workstreams below
(W0–W8); it does not replace them.*

**Objective.** V1 used MedGemma-4B and three controlled synthetic environments to study whether outcome-only GRPO
learns reliable gather-then-commit procedures, and whether process-aware rewards improve information gathering. V2
tests whether those findings extend to a larger model (MedGemma-27B), an established and independently developed
multi-turn clinical simulator, conventional train/validation/test evaluation, and independent out-of-distribution
simulators.

**The primary objective is in-distribution.** First establish whether GRPO, particularly credit-assignment GRPO,
produces measurable improvements on held-out cases from the same realistic simulator used for training. Emphasize
cross-simulator generalization only after that. V2 must not depend on behavior transferring between two very different
simulator interfaces.

### The experiments

| Experiment | Purpose | Model | Environment | Role | Workstreams |
|---|---|---|---|---|---|
| **A** | In-distribution RL on an established simulator | 27B | SDBench preferred, pending checks | **Primary** | W8 |
| **B** | Validate the training environment before scaling | 4B | Same as A | Prerequisite to A | W8 |
| **C** | Out-of-distribution cross-simulator evaluation | Checkpoints from A | AgentClinic preferred | Secondary | W5, W6 |
| **D** | Controlled 27B replication of V1 | 27B | V1's three simulators | Separate replication | W1, W2, W4; W3 if VOI credit wins |

**Labels changed on 2026-11-09.** In commits before this date, "Experiment A" meant the 27B replication (now **D**),
"Experiment B" meant richer-environment RL (now **A** and **B**), and "Experiment C" meant external evaluation (still
**C**, now out-of-distribution).

**Keep the factors separable.** Model scale, environment realism and cross-simulator generalization must not be
confounded in one comparison.

| Factor | Valid comparison | Not valid |
|---|---|---|
| Model scale, exact rewards | 4B vs 27B on V1's simulators (D) | V1's 4B results vs A's 27B results |
| Environment realism | Whether the same mechanisms appear in D and A, both at 27B | Comparing reward values between them, since the scales and provenance differ |
| Scale inside the benchmark | Only if a matched, adequately powered 4B arm is run | Treating B's bounded pilot as a scale comparison |
| Generalization | The same checkpoints, in-distribution vs OOD (A vs C) | Calling an interface failure a failed decision procedure |

### Hypotheses

Freeze each primary endpoint and threshold before opening the relevant test set.

| ID | Tier | Hypothesis (on untouched test cases, without shaping) |
|---|---|---|
| **A1** | **Primary, in-distribution** | From the same SFT initialization, credit-assignment GRPO improves the information-gathering procedure over outcome-only GRPO (less premature commitment, more relevant gathering) without reducing diagnostic accuracy or raising cost beyond a pre-registered margin |
| **A2** | **Primary, in-distribution** | At least one GRPO arm beats SFT-only on the pre-registered cost-adjusted diagnostic endpoint, and the gain comes from gathering rather than from output formatting |
| **A3** | **Primary, in-distribution** | Outcome-only GRPO can raise aggregate score while worsening procedure or difficult/high-risk performance. This is V1's central negative result, tested in a realistic setting |
| C1 | Secondary, OOD | Without further training, A's process-aware checkpoints keep a procedure advantage over outcome-only checkpoints on the independent simulator |
| C2 | Secondary, OOD | Any such advantage also improves decision quality, not only gathering volume |
| C3 | Secondary, OOD | With matched interface adaptation, the advantage is recoverable where zero-shot transfer failed on format |
| D1 | Replication | The V1 pattern persists at 27B on the original simulators: outcome-only GRPO lowers high-risk value or gathering compared with base, and credit-assignment GRPO does not |

**How the tiers relate:**
- A negative C does not invalidate a successful, well-controlled A. It is informative in its own right.
- A negative A is also reported. C can then describe behavior, but it can't claim that a demonstrated in-distribution
  gain transferred.
- D stands regardless of A and C, because it keeps exact rewards and oracles.

### Experiment A: in-distribution RL on an established simulator (W8)

**Preferred candidate: Microsoft's SDBench, pending access and technical feasibility.** It supports sequential
questioning, test ordering, cost-sensitive decisions, diagnostic evaluation and documented dataset partitions. A
train/test split does not make it suitable for RL training. It has to pass the [go/no-go
criteria](#gono-go-criteria-for-the-rl-training-simulator) first. MediQ, AgentClinic, Medmarks-T and others are
[alternatives](#alternatives-if-sdbench-is-inaccessible-or-unsuitable).

**Arms, under matched training and evaluation conditions:**
1. MedGemma-27B base.
2. MedGemma-27B SFT-only.
3. MedGemma-27B SFT + outcome-only GRPO.
4. MedGemma-27B SFT + credit-assignment GRPO.

Optional: decoupled exploration/exploitation GRPO, if feasibility and experimental headroom justify it.

**Design constraints:**
- **Same initialization.** For the primary causal comparison, all RL arms start from the same SFT adapter. V1's main
  diagnostics RL arms started from base, so A is not a direct replication of them.
- **SFT is fixed before RL.** V1 showed that a saturated SFT can leave no headroom. Measure the SFT's support and
  estimated headroom first. If it is saturated, report the comparison as inconclusive rather than weakening the SFT
  after the fact.
- **SFT demonstrations are unsolved.** V1 used oracle trajectories, and SDBench has no oracle. The source must be
  defined and must use training cases only.
- **Partitions:** training cases for SFT and RL, validation for hyperparameters and checkpoint selection, test evaluated
  once per final model. Report the validation-selected and final checkpoints (W2). Seeds: at least 3, and 5 if budget
  allows.

**No exact oracle.** SDBench does not provide V1's analytic oracle. Information gain, evidence sufficiency and process
credit may need approximate, clinician-reviewed or model-based estimators. Claims must say which one was used. Don't
call a model-graded reward "verifiable", and don't compare its values with V1's.

| Reward component | Provenance | Status or risk |
|---|---|---|
| Turn and test counts, repeated-request detection | **Deterministic** | Direct from the transcript |
| Cost | **Mixed:** $300 per visit is fixed; tests map to CPT codes through a language model, then to a price table | Deterministic only if the mapping is cached |
| Diagnostic score | **Learned judgment:** LLM judge with a physician-authored 5-point rubric; ≥4 counts as correct | Noise, bias and gaming must be measured |
| Test matched a real case result (`is_synthetic`) | **Learned judgment:** set by the gatekeeper LLM | The agent isn't shown this flag, so it could inform a reward; accuracy is unmeasured |
| Approximate information gain or evidence sufficiency | **Learned or clinician-derived** | V1's exact posterior isn't available; validate against clinician ratings (W6). Our Claude process-reward sweep was negative, so treat learned judges cautiously |

**Process rewards must not expose ground truth.** Rewards may use hidden case information, but nothing derived from the
diagnosis, judge output or process score may enter the policy's prompt. The gatekeeper reads the full case, including
the diagnosis, so its replies need a leak audit. Shaping stays in the training rollout only, and evaluation stays
unshaped.

**Evaluation.** Use the simulator's untouched test partition, with the same rubric for every model and no process
shaping. Measure:
- diagnostic accuracy and appropriate final decisions;
- diagnostic testing and investigation cost;
- information-gathering quality and relevance;
- number of questions and tests;
- premature commitment;
- unnecessary investigations;
- high-risk or difficult-case performance;
- aggregate versus subgroup performance;
- persistence across training seeds.

SDBench has no risk labels. Define high-risk and difficult cases in advance using blinded clinician labels (W6), not
from any arm's results.

**Limits of the partitions:**
- **Test sets are small.** A single proportion at n=71 (MIMIC test) has a 95% CI half-width of up to about ±12 points,
  and n=56 (NEJM test) about ±13. Paired per-case analysis helps, but subgroups will be smaller still.
- **Published results pool validation and test.** Our protocol selects checkpoints on validation, so our test numbers
  won't match the published pooled protocol. Pooling validation with test after selecting on it would bias the result.
- **Possible power option (open decision):** select checkpoints on a fixed subset of validation and add the rest to the
  final held-out set. Report the official test partition separately. Decide this before any evaluation.

### Experiment B: validate with 4B before scaling to 27B (W8)

Before expensive 27B training:
- Implement and smoke-test the selected environment with MedGemma-4B.
- Confirm valid multi-turn conversations, stop-token handling, proper rollout token masking (`env_mask`), stable rewards
  and nonzero reward variance.
- Verify that training changes the policy, not just output formatting.
- Run a bounded SFT vs outcome-only GRPO vs credit-assignment GRPO pilot.
- Measure initial policy support and attainable headroom where possible.
- Inspect trajectories for shortcut behavior, reward exploitation and premature commitment.

Proceed to 27B only after the training environment is technically validated. The pilot uses training cases, with
validation for checks. **It never uses the test partition.** Pilot details and pass criteria are in
[W8](#w8--training-simulator-selection-and-pilot-experiments-a-and-b).

### Experiment C: out-of-distribution cross-simulator evaluation (W5, W6)

**Central question:** does process-aware RL learned in one clinical environment improve information gathering and
decision quality in another, independently developed environment?

- **Preferred candidate:** AgentClinic, using a strictly untouched evaluation subset. MediQ and MedAgentBench are
  possibilities where task and interface compatibility are reasonable.
- **Design:** the same checkpoints from A, with no additional training on external test cases.
- **Report two conditions, clearly separated:**
  1. **Zero-shot transfer:** no adaptation of any kind.
  2. **Optional matched interface adaptation:** identical for every arm, using no external test cases. Keep the overwrite
     controls in [`agentclinic_experiment_plan.md`](agentclinic_experiment_plan.md) §4.
- **Attribution:** control for prompt and interface differences. Keep the format-only control and valid-action reporting,
  and report interface failures separately from diagnostic errors. An interface failure isn't definitive evidence that
  the learned procedure failed.
- **Measures:** diagnostic performance, information-gathering quality, premature commitment, question burden and
  high-risk behavior. Benchmarks without risk labels need clinician acuity labels first (W6).

**Keeping the evaluation subset untouched:**
- Freeze AgentClinic case IDs before any further use. It ships no splits.
- Exclude or mask the 27 of 215 MedQA-extended cases that leak the diagnosis.
- Keep cases already seen in smoke tests (extended MedQA 0, 3 and 4, and any others in the logs) out of the test subset.
- Tune the harness, judge and prompts on validation cases only.
- If AgentClinic becomes the training simulator, it can't also be the OOD test. Choose another OOD target.
- Check source overlap between the two simulators. If both are MIMIC-derived, independence isn't established, because
  the same patients may appear in both. AgentClinic-MedQA avoids this problem, but its MedQA source may have been in the
  base model's training data, so compare arm differences rather than absolute accuracy.

### Experiment D: controlled 27B replication of V1 (W1, W2, W4; W3 if VOI credit wins)

Determines whether V1's reward-design effects persist at larger parameter scale while retaining exact analytic rewards
and oracles.

- Replicate the key V1 experiments with MedGemma-27B on the original three simulators: refill triage, diagnostics
  and titration.
- **Arms:** base, SFT-only, outcome-only GRPO, credit-assignment GRPO.
- **Question:** do the reward-design improvements persist as model capacity increases?
- **Gate 1 (adaptive headroom):** BC-gap is an analytic property of the environment and does not depend on the model,
  so with unchanged simulators it should reproduce exactly (+0.105, +0.161, +0.075). Rerun it as a check. Separately,
  report model-relative headroom: oracle value minus the value 27B base actually achieves, because a stronger base may
  leave less for RL to add.
- **Gate 2 (initial policy support):** model-dependent, so remeasure it for 27B base and for each 27B SFT init before
  any GRPO run.
- **Design constraints:**
  - V1's main diagnostics RL arms started from base, and no matched SFT-only arm exists at 4B
    (see `V2/agentclinic_experiment_plan.md` §4). Define the SFT-only arm before training, and train a 4B match
    if the 4B-vs-27B comparison needs one.
  - Resolve the V1 stop-token uncertainty first, decide the configuration before training (see "Dependencies"), and use
    the same configuration for both scales.

### Unified V1 + V2 paper strategy

Full notes: [`../paper/v2-extension/MANUSCRIPT_PLAN.md`](../paper/v2-extension/MANUSCRIPT_PLAN.md).

**Central thesis:** improving a clinical agent's final reward does not necessarily improve the sequential procedure it
uses to gather evidence. Process-aware RL may improve that procedure, but its effectiveness depends on task headroom,
initialization, reward design, model scale and environment complexity.

- **Part I, mechanistic understanding (existing V1):** MedGemma-4B on three controlled simulators with exact oracles.
- **Part II, realistic benchmark training (new; Experiments A and B):** MedGemma-27B on a recognized simulator.
- **Part III, external generalization (new; Experiment C):** trained models on an independent simulator, with positive
  and negative results reported.

D supports Part I by testing it at scale. V1 remains a frozen, citable preprint. A combined paper isn't assumed to be
better than a separate V2 paper, and the plan's notes set out when each is preferable. Disclose the relationship to V1
and distinguish existing results from new ones, to avoid duplicative publication.

### Implementation priorities

Some activities can run in parallel. All are proposed.

1. Complete V1 stop-token and evaluation audits.
2. Verify SDBench dataset access, licensing, splits and RL compatibility.
3. Compare alternative training simulators if SDBench is not feasible.
4. Build and validate a multi-turn GRPO training adapter at 4B (Experiment B).
5. Validate MedGemma-27B training infrastructure, including Ray and Azure.
6. Run a controlled 27B benchmark experiment with matched SFT and GRPO arms (Experiment A).
7. Perform independent OOD evaluation on AgentClinic (Experiment C).
8. Replicate key V1 findings at 27B under controlled conditions (Experiment D).
9. Incorporate results into the expanded paper, including clinician-reviewed trajectory analysis.

**Can run in parallel:** steps 2 and 3 (paperwork and reading), PhysioNet credentialing, the Azure quota request, W6
materials, and the AgentClinic harness pilot on validation cases.

### Milestones and dependencies

| Milestone | Depends on | Done when |
|---|---|---|
| M1: V1 audits complete | — | Stop-token effect on V1 measured with V1's own evaluation code; any correction decision recorded |
| M2: stop-token decision | M1 | Configuration chosen and recorded for all experiments |
| M3: training simulator decision | Access answers; go/no-go checks | One simulator selected against the criteria below, or all rejected |
| M4: 4B pilot (Experiment B) | M2; M3 | W8 pilot criteria pass or fail, reported either way |
| M5: 27B pipeline validated | Azure quota; 27B checkpoint and license check; M2; data-location decision for gated cases | W4 pilot measures memory fit, seconds per step, weight-sync time and real cost |
| M6: Experiment A | M4 passes; M5; budget approval; endpoints frozen; clinician labels | Matched arms evaluated once on test, or formally skipped |
| M7: Experiment C | M6 checkpoints; frozen evaluation subset; validated judge; W6 labels for high-risk metrics | Zero-shot and any adapted results reported, with interface failures separated |
| M8: Experiment D | M5; W1 stratified sets; W2 validation split | Diagnostics arms complete; decision on extending to titration and refill |
| M9: manuscript | M6–M8 as completed; W6 review | Unified-or-separate decision made and manuscript drafted |

### Infrastructure: confirmed vs proposed

| Item | Status | Evidence or next step |
|---|---|---|
| Two DGX Sparks reachable; vLLM 0.20.1 on both; adapters load | **Confirmed** (2026-11-08) | Smoke-test preflight |
| 4B doctor on Spark 1 with separate auxiliary model on Spark 2 | **Confirmed** for bounded AgentClinic runs | Smoke tests; not yet run at evaluation scale |
| MedGemma's `<end_of_turn>` not stopping generation under our text-only `generation_config` | **Confirmed** on the test server; server-level fix verified | Effect on V1 results **unresolved**; see [smoke report](benchmark_smoke_report.md) |
| Spark 1 stability under sustained vLLM load | **Open risk** | Three abrupt ends on 2026-11-08 and 11-09 with no shutdown record; cause unknown |
| Azure 8×H100 quota, Ray cluster | **Proposed**, not validated | W0, W4 pilot |
| MedGemma-27B checkpoint and license | **Proposed**, not verified | W4 |
| 27B fit, speed and cost | **Unmeasured** estimates | W4 pilot replaces them |
| SDBench software with local models | **Confirmed** on 5 synthetic cases only | Smoke report; not a performance result |
| SDBench, or an alternative, as an RL training environment | **Proposed**, not validated | Go/no-go criteria, W8 |
| 27B training on gated cases in Azure | **Unresolved conflict** | See "Dependencies" |

### Dependencies

- **Stop-token configuration.** Our text-only MedGemma `generation_config.json`, and the rollout inits copied from it,
  stop only at token 1, not at `<end_of_turn>` (106). A replication needs an explicit choice: keep V1's behavior for
  comparability, fix it, or run both. Make the choice after the V1 measurement, not before. V1 results shouldn't be used
  as the foundation for V2 comparisons until this is resolved.
- **Base-model comparability.** Compare 4B and 27B on the same stratified sets (see "How V2 combines results across
  scales").
- **Reward provenance.** Every result reports whether its reward came from an exact verifier or a model-based judgment.
- **Gated data and compute location.** This plan runs 27B on Azure, and earlier notes say MIMIC work should stay on the
  local Sparks. Those conflict if Experiment A trains on `mimic`. PhysioNet's guidance prohibits sending data to third
  parties, and allows a cloud service only if zero retention, no training use and no human review can be fully verified.
  Resolve this before M5. The options are written confirmation that our Azure setup complies, running gated work locally
  (27B would be far slower), or using a dataset whose terms allow it.
- **Environment models for gated data.** SDBench's official configurations use external GPT-5 or o4-mini models, which
  is unlikely to be acceptable for `mimic`. Local replacements would make scores unofficial and need their own
  validation against published gatekeeper and judge behavior.
- **Public repo.** This repo is public. MIMIC-derived text and NEJM case text must never be committed, including in logs
  and traces, and may not leave the people and systems the license covers. Confirm whether adapters trained on gated
  cases can be released before training them. PhysioNet's guidance page doesn't address derived weights.

## Go/no-go criteria for the RL training simulator

*Proposed starting thresholds. Freeze them before the checks run. Apply the same table to every candidate. Thresholds
may be revised only before results are seen, with the reason recorded.*

**Stage 1: desk check. No gated data needed.**

| # | Criterion | Go if | SDBench status (2026-11-09) |
|---|---|---|---|
| G1 | Access and terms | Written confirmation for the person running the work that the license covers storage on our machines, the chosen compute location, model training and keeping the resulting weights | **Open.** NEJM: terms unknown. MIMIC: credentialing required; the dataset page doesn't address training |
| G2 | Training supply | ≥150 training-only cases. V1 consumed 240 prompts in a 60-step run (4 prompts × 8 rollouts per step) from a 300-case pool | MIMIC: **163 train, a narrow pass.** NEJM: **unconfirmed.** The paper gives 248 validation + 56 hidden test = all 304 cases and no training count, while SDBench's site describes a train split |
| G3 | Split integrity | Case IDs frozen and hashed; SFT, RL and prompt tuning cannot load test cases; no patient or case overlap across splits | Not checked. Harness separates case sets by name. MIMIC's validation set was used for gatekeeper development, though not for agents |
| G4 | Test power | Pre-registered minimum detectable paired difference is achievable | MIMIC 71 test; NEJM 56 test. **Marginal;** see "Limits of the partitions" |

**Stage 2: environment check, before any training. Uses training cases only, once access exists.** Software checks may
use the 5 synthetic cases, but those can't produce results.

| # | Criterion | Go if | SDBench status |
|---|---|---|---|
| G5 | Repeatable rollouts | With fixed local models, temperature 0 and fixed seed where supported, the same request returns the same result ≥95% of the time, or results are cached per case and normalized request so all arms see the same world | Not measured. Temperature passes through model arguments; I found no seed setting or gatekeeper result cache |
| G6 | Judge reliability | Same trajectory re-scored: binary agreement ≥95%. Against clinician or adjudicated labels on ≥50 cases: κ ≥0.70 | Not measured. Paper reports κ 0.70 and 0.87 with its o3 judge. Our local judge gave a false positive at 4/5 |
| G7 | Cost stability | Same test maps to the same cost in ≥99% of repeats after caching | Not measured |
| G8 | Reward variation | Reward varies within rollout groups for the init policy: `frac_reward_zero_std` ≤0.5 | Not measured |
| G9 | Benefit from gathering | Estimated, cost-adjusted gap between a no-gathering commit and adaptive gathering ≥0.10 on the normalized reward (the Gate 1 analogue, labeled as an estimate). No fixed strategy, such as ordering everything, is near best | Not measured |
| G10 | Latent support | Init policy reaches top-tier score in ≥1 of 12 rollouts on ≥10% of cases, adaptively rather than blindly (the Gate 2 analogue) | Not measured |
| G11 | Reward exploitation | Scripted shortcuts don't score as well as genuine reference trajectories: hedged lists of diagnoses, generic or paraphrased labels, immediate commit, asking the gatekeeper for the diagnosis, instructions aimed at the judge, cheap test spam | Not measured. Hedging resistance is a specific concern |
| G12 | No ground-truth exposure | Audit of ≥500 gatekeeper replies finds no diagnosis leak after adjudication; policy prompts contain no reward or judge content | Paper reports 0 leaks in 508 responses with o4-mini. **Must be redone for any local gatekeeper** |
| G13 | Rollout throughput | A 60-step 4B run projects to ≤72 h at the measured rate, and its cost is approved | **At risk.** A V1-sized run is 1,920 episodes. At smoke-test rates of 3.5–9.5 min per case, serially, that is about 5–13 days, compared with V1's ≈7 h. Concurrency and shorter turns may help; both are unmeasured |

**Decision rule:**
- **G1, G3, G6 and G12 are hard requirements** and can't be waived.
- Failing another criterion is a no-go, unless a pre-specified change fixes it and the check passes on rerun.
- Stage 1 failure means moving to an alternative before spending engineering time.
- Passing all this only permits the Experiment B pilot. Like V1's gates, these are negative screens and don't show that
  RL will help or be safe.

### Alternatives if SDBench is inaccessible or unsuitable

Candidates are chosen by the same criteria, before results are seen. Don't switch environments after seeing test
results.

| Candidate | Attraction | Known problems | Status |
|---|---|---|---|
| **AgentClinic** | Public MedQA cases, interactive; our harness works | No splits, no cost model, only 188 non-leaking extended MedQA cases. Training cases would have to be generated from MedQA-train. Using it here removes the preferred OOD test | Smoke-tested for evaluation; not RL |
| **MediQ** | CC-BY-4.0; explicitly about asking follow-up questions | README refers only to a development set; no train or test split stated; no costs known | Not yet run |
| **Medmarks-T** | Described as training-capable, with train/test splits; built on `verifiers` | Experimental. No environment or config in our clone uses that name. The large training splits we can see are mostly single-response tasks, which don't test gathering | Not tested |
| **MedAgentBench** | Multi-step EHR tool use, MIT code | Different task from an interview. Needs Docker (no permission here), and the grader is a separate download | Not tested |

**If no candidate passes:** report that, and keep Experiment D plus an external evaluation of the existing V1 checkpoints
on AgentClinic. Do not substitute a materially different environment under the same claim.

## Keeping V2 easy to identify

Best effort. The aim is that readers can spot V2 work easily, without duplicating code or adding structure.

- **V1 paper:** the arXiv submission is tagged `arxiv-v1`. Don't edit `paper/main.tex`, `paper/figs/`, `paper/main.pdf` or
  `paper/arxiv_submission_v1.zip`. **One authorized exception (2026-11-09):** Brendan asked for an AI-assistance
  disclosure to be added for an arXiv replacement. It is in `paper/main.tex` and `paper/arxiv_replacement_disclosure.zip`,
  with no other changes; the tag still holds the original. The V2 manuscript lives in [`paper/v2-extension/`](../paper/v2-extension/).
- **V2 docs and wholly new experiments:** keep them under `V2/`, or use a `v2_` prefix when they belong elsewhere (for
  example `scripts/v2_*`, `results/v2_*`, `checkpoints/v2_*`).
- **Shared code:** update `framework/` and `tasks/` directly when needed, with no copies or parallel versions. Keep V1
  behavior available where it's cheap, and don't overwrite existing V1 results.
- **External clones, venvs and generated data** stay outside the repo (currently `~/v2_bench/`).

## Decisions so far (2026-11-01)

- **v1 first:** finish v1 and post to arXiv by **2026-11-15**; make the repo + paper public on GitHub. V2 starts
  after. NMI is the *style* reference, not necessarily the venue; realistic V2 targets are **TMLR** (judges
  correctness/support of claims) or **ML4H / CHIL**; NMI only if W5/W6 land something striking.
- **arXiv:** Brendan has no university email and has never submitted → needs a one-time **endorsement** in cs.LG
  (or cs.AI). Agustina can endorse if she has prior arXiv papers in that category; if she is not an author she
  cannot submit on his behalf.
- **Agustina Saenz, MD MPH:** role TBD (author vs acknowledged clinical reviewer) — her call. Not listed until
  decided. Likely owns W6 and may provide benchmark / agent-system access for W5.
- **Compute split:** 4B work + benchmark evals on the local **DGX Sparks**; **27B on Azure** (Brendan prefers
  ~$1–2k+ over ~4 extra weeks of Spark time). See "Compute analysis" below.
- **Ray:** Brendan explicitly wants to learn Ray; use it on Azure for orchestration (W0).
- **Stage the spend:** cheap checks (W1, W2, W6, W5-A) run first; the 27B grid is committed only after the
  27B pilot measures cost and W1 shows the core diagnostics effect survives with CIs.

## Why V2 is worth doing (the case for spending)

The topic is more general than "clinical information gathering". Many LLM agents trained with RL share the same
gather-then-commit structure — search agents decide what to retrieve before answering, coding agents what to
inspect before editing, tool-use agents which calls to make before acting — and the dominant recipe rewards only
the final outcome. Whether outcome-only RL teaches a sound gathering procedure or a shortcut is a general open
question; clinical decision-making is its high-stakes instance. The paper's distinct asset is a setting where
the procedure is measured **exactly** (exact oracle, exact posterior), which real benchmarks cannot provide.

What makes it journal-grade is showing the phenomenon is real beyond the toy setup. The 27B replication is a
good bet because it is informative either way: if the exploit persists, the claim gets much stronger; if scale
removes it, that is itself a finding. Honest stop condition: if W1 shows the diagnostics plain-vs-credit
difference does not survive with CIs **and** W5 shows the procedure audit doesn't transfer, stop at a solid arXiv
preprint + workshop paper.

**Intro sentence for v1/v2** (place after the RLVR paragraph, before "This distinction creates an important
safety problem"):

> Many language agents now trained with reinforcement learning share a gather-then-commit structure: search
> agents decide what to retrieve before answering, coding agents what to inspect before editing, and tool-use
> agents which calls to make before acting, yet the dominant recipe rewards only the final outcome. Whether
> outcome-only training teaches a sound information-gathering procedure or a reward-exploiting shortcut is
> therefore a general open question, and clinical decision-making is its high-stakes instance.

## How V2 combines results across scales

*Written for the 4B-to-27B replication, now Experiment D. Overall paper structure is in
[`../paper/v2-extension/MANUSCRIPT_PLAN.md`](../paper/v2-extension/MANUSCRIPT_PLAN.md).*

Organize by **question**, with scale as a factor — not by hardware (hardware is one sentence in Methods).
- **4B = full mechanism study:** all three tasks, λ sweep, SFT experiments, VOI-credit arm.
- **27B = replication of the headline:** diagnostics plain vs credit (± VOI), 3 seeds; one Results subsection,
  "Does the exploit persist at scale?"
- **One paired figure:** 4B and 27B panels with identical axes/metrics.
- **Master table gains a "model" column** (4B/27B rows for diagnostics); full per-task tables → Supplement.
- **Prerequisite:** both scales evaluated on the **same W1 stratified eval sets** (re-eval all 4B ckpts) so every
  number is directly comparable.

## Compute analysis: 27B on DGX Sparks vs Azure (estimates — the pilot replaces them)

- **Memory:** 128 GB unified memory per Spark; 27B bf16 ≈ 54 GB → rollout node fits vLLM + KV cache; trainer
  with LoRA + gradient checkpointing likely fits but tight (QLoRA fallback would add a confound vs the 4B setup).
- **Speed:** Spark decode is memory-bandwidth-bound (~270 GB/s); ~7× the parameters → each 60-step run
  **~35–50 h** vs 7 h at 4B. Weight sync grows to ~54 GB/update; the NCCL fresh-first-init issue means a crash
  late in a 2-day run costs the whole run.
- **Spark-only total:** 27B grid ~12 days + 27B evals ~4–5 days + 4B work ~1 week ≈ **4–5 weeks** serial.
- **Decision:** 27B on Azure (single 8×H100 node, no cross-node NCCL); Sparks keep 4B + benchmark evals.

## Timeline reality check

The original 8-week estimate was ~3 wk compute (critical path) + ~2 wk engineering + ~2 wk external waiting
(Agustina, quota) + ~2 wk analysis/writing, overlapping. Coding agents compress engineering to days and speed up
writing. Incompressible: machine time, Agustina's turnaround (send W6 materials in week 1), Azure quota /
PhysioNet approvals, and learning Ray. **Realistic: ~4–5 weeks with Azure for 27B** (vs ~6–7 Spark-only).

## The weaknesses v1 has (what each workstream fixes)

| # | Weakness in v1 | Workstream |
|---|---|---|
| 1 | Diagnostics high-risk subgroup is **n=2 cases**; most cells single-seed; plain-vs-credit overall value overlaps at 3 seeds | W1 power |
| 2 | Decoupled GRPO headline is the **step-40 ckpt chosen on the test set** (final ckpt: high val +0.158, supp 0.50, asks 5.03) | W2 selection hygiene |
| 3 | Credit-assignment still **under-gathers** (~2.1 asks vs oracle ~3.8; 1.1–1.5 on high-risk) — "less bad", not "learned the procedure" | W3 decision-value credit |
| 4 | One small model (MedGemma-4B) — "small-model RL artifact?" | W4 scale (27B) |
| 5 | Synthetic toy simulators only | W8 in-distribution training on an established simulator (Experiments A, B); W5 OOD evaluation (Experiment C) |
| 6 | Utility matrices + audit labels never clinically validated | W6 clinician validation |
| 7 | Missing literature (reward hacking, MediQ, ED-Copilot, VOI/active feature acquisition); claims broader than evidence | W7 writing |
| — | Infra: fragile 2-node NCCL weight sync; user wants to learn Ray | W0 Ray on Azure |

All numbers above were reconciled against `results/probe_*.records.jsonl` on 2026-11-01.

---

## W0 — Infrastructure: Ray on Azure (enabler for everything else)

**Why:** run seeds/arms/evals in parallel instead of one 7h run at a time; escape the 2-node NCCL weight-sync
fragility (see `framework/procreward/README.md`); learn Ray.

**Steps (in increasing ambition):**
1. **Azure GPU quota first.** Request quota for an 8×H100 SKU (ND H100 v5 family) in one region, plus some
   smaller single-GPU SKUs for 4B work. Quota approval can take days — file it on day 1. Check spot/low-priority
   pricing for evals (interruptible is fine there; less so for training).
2. **Ray cluster on Azure.** Either the Ray cluster launcher (`ray up` with the Azure provider) on VMs, or KubeRay
   on AKS. Start with the VM launcher — fewer moving parts.
3. **Easy win — Ray fan-out for evals.** Wrap `framework/gates/support_probe.py` as a Ray task: one task per
   (checkpoint × eval set), each with its own vLLM on 1 GPU. All of W1/W2's re-evals become one parallel job.
4. **Seed/arm sweeps as Ray jobs.** Launch the existing `grpo_disagg.py` runs as independent Ray jobs (one per
   seed × arm), each with its own trainer+vLLM GPU allocation on a single node — no cross-node NCCL.
5. **Optional, bigger: Ray-native GRPO.** Evaluate porting to a Ray-based RL framework (veRL or OpenRLHF both run
   on Ray and support GRPO; veRL has multi-turn/agent rollout support). Only do this if step 4 is too slow for
   W4. Porting risk: must preserve `env_mask` (no gradient on simulator tokens), the custom `rollout_func`, the
   info-gain/DREAM shaping, and the invariant that env/gates/eval stay unshaped. **Gate:** reproduce one existing
   4B diagnostics run (e.g. plain GRPO seed 1) within noise before trusting the port.

**Done when:** one command launches N seeds × M arms on Azure and a second command evaluates all resulting
checkpoints, results landing in `results/` in the existing probe format.

---

## W1 — Statistical power (highest priority; cheap)

**Problem:** headline high-risk numbers are 0/1/2 of 2 cases.

**Fix:**
- **Stratified held-out set.** The simulator generates on demand (`tasks/diagnostics/env.py:gen_cases`), so draw
  a large pool from a fresh seed and keep a fixed quota per risk level — e.g. **100 high + 100 moderate +
  100 none** — via a small rejection-sampling wrapper (no env change needed). Same for titration (high-clot) and
  refill. Report each stratum separately; report overall value re-weighted to the natural prevalence.
- **More seeds:** diagnostics plain vs credit-assignment → **5 seeds** each (currently 3). With Ray these run in
  parallel.
- **Stats:** bootstrap 95% CIs over cases and over seeds; paired per-case comparisons (same cases, both arms).
  Pre-register the primary metric: *high-risk value on the stratified set, credit vs plain, paired.*
- **Also eval** every existing checkpoint (base, SFT, s20/s40/final of all arms) on the new sets — eval only.

**Pre-registered outcome:** if the credit-vs-plain high-risk difference's CI excludes 0 → claim stands; if not →
v2 reports it as "no detectable high-risk difference; credit-assignment's benefit is on moderate cases /
stability", and the framing adjusts.

## W2 — Checkpoint-selection hygiene (free)

- Create a **validation split** (separate generation seed) distinct from the test set.
- Choose checkpoints for every arm on validation only; report test numbers for that ckpt **and** the final ckpt.
- Decoupled GRPO is reported this way; if its high-risk advantage disappears, say so.

## W3 — Decision-value credit (the candidate new positive result)

**Problem:** entropy-reduction credit rewards uncertainty reduction, not questions that change the decision — so
the policy still under-gathers.

**Fix:** add a reward option where each new ask earns `λ · VOI(f | revealed)` — the expected improvement in the
best achievable utility from asking finding `f` given what is already revealed. The exact oracle already computes
these quantities. Lives only in the GRPO `rollout_func` like the existing credit (env/gates/eval stay pure).

**Arms (4B, diagnostics, stratified eval):** plain · entropy credit (λ=0.2) · VOI credit (λ calibrated with the
same per-ask law `λ·ΔV ≳ c`). Then repeat the best on titration to test whether it removes the need for per-task
λ calibration.

**Pre-registered success:** VOI-credit mean asks within ±0.5 of oracle, no 1-ask collapse, and high-risk value ≥
entropy credit (paired CI). Failure is still reportable ("decision-aware credit does not fix under-gathering").

## W4 — Scale: MedGemma-27B

- Verify the text-only 27B checkpoint on Hugging Face and its license; reuse `framework/infra/prep_text_base.py`
  only if a multimodal variant is the one available.
- **Original scope (2026-11-01):** diagnostics only, plain vs entropy credit (+ VOI credit if W3 wins), **3 seeds
  each**, stratified eval. Do not port refill/titration/repair to 27B.
- **Expanded scope (2026-11-08, now Experiment D):** the three original simulators with base, SFT-only, plain and credit
  arms. To stage the spend, the original diagnostics grid plus an SFT-only arm is the **core**; titration and refill
  are an **extension**, committed only after the pilot measures cost and the core reports. Biased-SFT repair stays out
  of the 27B scope.
- **Layout:** one 8×H100 node — e.g. 4 GPUs trainer (LoRA) + 4 GPUs vLLM (tensor-parallel 4). Single node avoids
  the cross-node NCCL problem.
- **Hardware decision (2026-11-01):** 27B runs on **Azure**, not the DGX Sparks. Spark-only 27B was estimated at
  ~35–50 h/run (~4–5 weeks for the grid + evals, with NCCL first-init risk on every 2-day run); Brendan prefers to
  spend ~$1–2k+ to save those weeks. Sparks stay on 4B work (W1–W3) and benchmark evals (W5).
- **Pilot first:** one 10–20-step run on the Azure node with the real config, measuring: fits in memory? seconds
  per step, rollout tokens/s, weight-sync time. Extrapolate to 60 steps × grid size → real cost before committing.

**Pre-registered:** the v1 pattern "replicates at scale" if plain GRPO lowers high-risk value or gathering vs
base and credit-assignment does not. If 27B plain GRPO is already safe, that is itself a finding (scale changes
exploitability) and goes in the paper.

## W5 — External validity: benchmark + (optionally) real data

**Option A — interactive benchmark eval (do this):** evaluate base vs trained policies on an open interactive
clinical benchmark where the agent must ask before diagnosing — **MediQ** and/or **AgentClinic** are the closest
matches, along with **SDBench** (see "Benchmark options and access" below). No exact oracle there, so report procedure metrics that
transfer: questions asked, premature commitment, accuracy, plus the guarded-report flags. Requires an adapter from
the benchmark's patient simulator to our ask/commit JSON schema.

**Under the 2026-11-09 strategy,** Option A's trained-checkpoint evaluation is Experiment C, and AgentClinic is its
preferred target. A pilot with existing 4B checkpoints stays useful for validating the harness, on validation cases only.

Detailed AgentClinic plan (verified capabilities, experiment matrix, interface adapter, splits, pilot):
[`agentclinic_experiment_plan.md`](agentclinic_experiment_plan.md).

**Option B — real-data environment (stronger, more work):** build a 4th task from **MIMIC-IV-ED** in the style of
ED-Copilot: findings = triage vitals/labs, per-test cost, outcome = critical-event label; "oracle" = best policy
estimated from data. Requires PhysioNet credentialing.
- ⚠️ **Data-use rules:** PhysioNet credentialed access is per person — Agustina can't share the data with an
  uncredentialed user; Brendan needs his own credentialing (CITI course + application, a few weeks) or Agustina
  runs that part. Check PhysioNet's guidance on using credentialed data with LLMs **before** sending any MIMIC
  text to an external API (including the Claude judge/teacher code in this repo).

**Agustina:** ask which benchmarks / agent systems she already has access to — that decides A vs B and saves
setup time.

### Benchmark options and access

*Candidates under consideration, not commitments; none has been used for a result yet. Access was checked on
2026-11-08. Confirm the terms again before publishing anything derived from a dataset.*

| Option | What it evaluates | Open now | Gate or condition | V2 relevance |
|---|---|---|---|---|
| [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) | Interactive patient interviews, tests and diagnosis | Code (MIT); MedQA and NEJM cases | MIMIC-IV cases: PhysioNet credentialing plus a request to the authors. NEJM text is NEJM copyright, so do not redistribute it. Spec, Lang and tool code are not in the repo | Very high: first evaluation pilot; preferred OOD test (Experiment C) |
| [SDBench](https://github.com/microsoft/sdbench) | Sequential diagnosis and the cost of tests | Code (MIT); 5 synthetic cases, not for reporting performance | NEJM cases: license from NEJM (terms not yet checked). MIMIC cases: PhysioNet credentialing | Very high: preferred training simulator (Experiment A), pending go/no-go |
| [MediQ](https://github.com/stellalisy/mediQ) | Asking follow-up questions before deciding | Code and development data (CC-BY-4.0) | None found. The README does not mention a test set | High |
| [MedAgentBench](https://github.com/stanfordmlgroup/MedAgentBench) | Agent tool use in a simulated EHR (FHIR) | Code (MIT); task data; Docker FHIR server | No credentialing mentioned. The grader (`refsol.py`) is a separate Stanford Box download | High for tool use; not an interview benchmark |
| [Medmarks](https://github.com/MedARC-AI/medmarks) | Suite of 30 medical benchmarks, including AgentClinic, MedAgentBench, HealthBench and MedR-Bench | Framework (MIT) | Licenses vary per dataset and some are "not specified". Inclusion does not override a dataset's own terms | High: its Medmarks-T subset has train/test splits for post-training |
| [HealthBench](https://openai.com/index/healthbench/) | Clinical safety, accuracy, escalation and communication | Reported public | Rubric grading needs inference compute. **Access not confirmed at the source** | High for external safety evaluation |
| [MedR-Bench](https://www.nature.com/articles/s41467-025-64769-1) | Diagnosis and treatment reasoning on case reports | Reported public; Medmarks lists CC-BY-SA-4.0 | **Access not confirmed at the source** | Medium to high |
| [MIMIC-ED-Assist / ED-Copilot](https://github.com/cxcscmu/ED-Copilot) | Sequential lab selection and outcome prediction | Code (no license stated) | Needs MIMIC-IV and MIMIC-IV-ED from PhysioNet: credentialing | Medium to high |
| [MIMIC-IV](https://physionet.org/content/mimiciv/) / [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/) | Real clinical records for building a new evaluation (Option B) | Nothing | PhysioNet credentialing, CITI course, data use agreement | High for realism; more engineering |
| MedQA, MedMCQA, [MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench) | Medical knowledge, decision questions and calculations | MedCalc-Bench (CC-BY-SA-4.0). MedQA and MedMCQA reported obtainable | Dataset-specific terms. **MedQA and MedMCQA terms not confirmed** | Supporting baselines; not interactive |

**The PhysioNet gate (class and certificate).** It applies to AgentClinic-MIMIC-IV, SDBench-MIMIC, MIMIC-ED-Assist and
MIMIC-IV(-ED):
1. Create a [PhysioNet](https://physionet.org/) account.
2. Complete the CITI course "Data or Specimens Only Research".
3. Apply for credentialing, including a reference. Review typically takes 1–2 weeks.
4. Sign the data use agreement for each dataset.
5. For AgentClinic-MIMIC-IV only, request the derived cases from the AgentClinic authors.

Access is per person and cannot be shared within a team. The agreement prohibits sending the data through
third-party APIs, and [PhysioNet's LLM guidance](https://physionet.org/news/post/llm-responsible-use) strongly
recommends local models, so MIMIC work should run on the local Sparks.

**Suggested order for evaluation pilots** (the training-simulator order is under "Implementation priorities"):
1. AgentClinic-MedQA, which needs no credentials.
2. SDBench: start with the code and synthetic cases, then decide on NEJM licensing or MIMIC access.
3. HealthBench and Medmarks, as independent accuracy and safety measurements that complement the interactive
   evaluations.

Start PhysioNet credentialing early, even while using public data, because it opens every MIMIC option above.

## W6 — Clinician validation (Agustina Saenz, MD MPH)

*Experiments A and C add two clinician tasks: blinded acuity or difficulty labels, and validating any approximate process
measure. Her V2 role still needs confirming; she is a co-author on V1.*

1. **Utility & finding review:** she reviews the three utility matrices, finding definitions, and per-ask cost for
   clinical plausibility; record changes she'd make. (If changes are large, rerun Gate 1 — CPU only.)
2. **Blinded trajectory rating:** ~100 trajectories sampled across arms and risk levels, method labels hidden.
   She labels each: appropriate / premature commitment / unsupported escalation / hedging / over-gathering.
3. **Agreement analysis:** Cohen's κ between her labels and the automated guarded report. High agreement
   validates the paper's central evaluation tool; disagreements become a findings section.

Her V2 credit is **her call — do not list her as a V2 author until decided**. (V1 lists her as a co-author.)

## W7 — Writing & literature

- Add: reward hacking / specification gaming (e.g. Skalse et al.; Gao et al., reward-model overoptimization);
  MediQ; ED-Copilot (`sun2024edcopilot`, already in the bib); AMIE (`tu2024amie`); value of information / active
  feature acquisition; curiosity / info-gain intrinsic rewards; multi-turn LLM-RL credit assignment.
- Re-scope claims to the W1 confidence intervals; state subgroup n everywhere.
- Restructure to NMI article shape: short main text → Results → Discussion → Methods; ≤ ~8 display items; rest
  to Supplementary.
- Decide whether the Claude process-reward α-sweep (negative; see `framework/procreward/`) goes in as a
  supplementary "learned judge vs analytic credit" result.

## W8 — Training simulator selection and pilot (Experiments A and B)

**Status: proposed, feasibility not established.** SDBench is the preferred candidate pending the [go/no-go
criteria](#gono-go-criteria-for-the-rl-training-simulator). No candidate has been run as an RL training environment, and
being selected is not assumed.

**What we know so far (from reading code and docs, and from smoke tests on synthetic cases, not from RL use):**

- **SDBench access:** MIT-licensed code with a gatekeeper, judge and cost estimator. Only 5 synthetic cases are public,
  in a `train` split, and they are explicitly not for reporting performance, which is too few to train on.
- **SDBench partitions:**

  | Dataset | Access | Train | Validation | Test | Source |
  |---|---|---|---|---|---|
  | `mimic` | PhysioNet credentialed, DUA | 163 | 254 | 71 | PhysioNet dataset page |
  | `nejm` | License from NEJM; terms unknown | Not confirmed | 248 | 56 | Paper (304 cases total) |

  The paper's counts leave no training cases, while SDBench's site describes a train split, so confirm the NEJM count
  when access exists. Published results pool validation and test, and MIMIC's validation set was used to develop its
  gatekeeper.
- **Our access:** Agustina Saenz holds PhysioNet and NEJM licenses (reported 2026-11-08). PhysioNet access is per person
  and can't be shared with Brendan. Confirm whether her NEJM license covers the SDBench dataset, Brendan's use and model
  training. Nothing gated has been retrieved.
- **SDBench environment models:** the gatekeeper and judge are LLMs, and the gatekeeper may synthesize a result when a
  case lacks one. The agent isn't told which results are synthetic. The paper used o4-mini and o3; the repo's configs use
  GPT-5 and o4-mini.
- **MediQ:** CC-BY-4.0 code and development data with a patient system and an expert skeleton. Not yet run. The README
  mentions no test set, and the default patient calls a model.

**Keep exact and model-based rewards distinct.** V1's reward is an exact verifier. A richer environment's reward is
likely to be a model-based clinical judgment, so it may be noisy and gameable, and it cannot certify oracle-grade
behavior. Do not call it "verifiable" or compare its values directly with V1's. Exact posterior information gain is
also unavailable without an explicit state model, so a process reward here would be an approximation, and claims must
say so. Gate 1 cannot be computed analytically without a utility model; estimate headroom empirically and label it an
estimate.

**4B pilot before any 27B training (Experiment B).** Bounded (≤20 GRPO steps per arm) on Sparks, with fixed environment
models and disjoint splits. Arms under matched conditions: SFT-only, outcome-only RL, process-aware RL. Spark 1's
stability risk applies, so write results incrementally.

**Pilot passes only if:** rollouts complete reliably, with valid turns and correct stop-token handling; simulator tokens
are masked from the gradient; the reward is stable on repeated scoring of the same trajectory; KL moves (≳1e-3) with
non-degenerate rollout diversity; behavior changes beyond output formatting; a trajectory audit finds no unaddressed
reward exploit; and projected cost is acceptable. The thresholds are pre-registered before the pilot runs. A 20-step
pilot tests feasibility. It can't establish which arm is better.

**Stop condition.** See "If no candidate passes" under the go/no-go criteria.

---

## Timeline (accelerated, cloud-backed; weeks after v1 ships on 2026-11-15)

*Original week-by-week sketch below; with coding agents + Azure the realistic total is ~4–5 weeks (see "Timeline
reality check").*

| weeks | work |
|---|---|
| now → 11-15 | v1 to arXiv (separate track). **Day 1 of V2: file Azure GPU quota request + message Agustina.** |
| 1 | W0 steps 1–3 (Ray cluster, eval fan-out); W2 validation split; W1 stratified sets; W7 lit additions start |
| 2 | W1 re-eval of all existing ckpts + 5-seed dx runs (parallel on Ray); W3 VOI-credit implementation + CPU checks |
| 3 | W3 4B arms; W4 27B pilot run; W5 benchmark adapter; W6 materials sent to Agustina |
| 4–5 | W4 full 27B grid; W5 benchmark evals; W6 ratings back → κ analysis |
| 6–7 | Analysis, figures, rewrite (W7) |
| 8 | Internal review → journal submission, plus an arXiv posting chosen under `paper/v2-extension/MANUSCRIPT_PLAN.md` |

**Mapping to the reorganized strategy (2026-11-09).** The table above remains the baseline sketch, and it predates the
priority order under "Implementation priorities", which now governs. Experiment D maps to the W1, W2 and W4 rows.
Experiment C maps to the W5 and W6 rows. Experiments A and B are additional. Simulator access alone may take weeks
(PhysioNet review typically takes 1–2 weeks, and NEJM's response time is unknown), and rollouts may be slow, so the
"~4–5 weeks" estimate no longer covers the full plan. Re-estimate after the go/no-go decision and the 27B pilot.

## Budget (rough — get real quotes)

Treat as order-of-magnitude only; Azure on-demand H100 pricing varies by region and changes. Get a quote and run
the W4 pilot before committing.

| item | estimate |
|---|---|
| W1/W2/W3 4B training + all evals (single GPUs, spot where possible) | low hundreds of $ |
| W4 27B grid (6–9 runs + evals on an 8×H100 node) | ~$2–6k |
| W5 benchmark evals | low hundreds of $ |
| Idle/debugging overhead (Ray setup, failed runs) | budget +30% |
| **Total** | **roughly $3–8k** |

**Expanded-strategy estimate (updated 2026-11-09, order of magnitude, derived from the table above and not measured).** The W4
row implies roughly $330–670 per 27B run.

| item | scope | estimate |
|---|---|---|
| Experiment B: go/no-go checks + 4B pilot | Local Sparks | No cloud spend, if environment models run locally. A NEJM license fee, if any, is unknown |
| Experiment A at 27B | Conditional on pilot: base, SFT, 2 GRPO arms × ≥3 seeds | **Not estimated.** LLM-in-the-loop rollouts should cost several times a V1 run; the pilot measures it. Gated data may rule out Azure |
| Experiment C | Local Sparks; 27B checkpoints evaluated locally or on Azure | Low hundreds of $ (the W5 row) |
| Experiment D core | Diagnostics: 6 GRPO runs + 1 SFT, plus evals | ~$2–6k (the W4 row) |
| Experiment D extension | Titration and refill: 12 GRPO runs + 2 SFT, 3 seeds | ~$5–9k additional |

The full expansion would exceed both the $3–8k total above and Brendan's ~$1–2k comfort level for 27B. Each step past
the pilot therefore needs his explicit approval, and fallbacks include fewer seeds or a smaller extension. External
API use (SDBench's defaults route to GPT-5 via OpenRouter; HealthBench's official grader uses GPT-4.1) also needs
approval, and it is not allowed for MIMIC-derived data.

Cost controls: shut nodes down between jobs; spot instances for evals; pilot before grid. **4B work and benchmark
evals run on the local DGX Sparks; only 27B goes to Azure.** Brendan's comfort level for 27B is ~$1–2k — the pilot
decides whether the full grid fits that; if not, cut to 2 seeds/arm or drop the VOI arm at 27B.

## Open decisions (Brendan)

- [ ] W5: Option A only, or A + B? (depends on Agustina's access)
- [ ] Experiment D: approve the titration and refill extension after the core (cost above)?
- [ ] Experiment D: define the SFT-only arm, and whether to train a matched 4B SFT-only arm
- [ ] Experiment A: source of SFT demonstrations, given that SDBench has no oracle
- [ ] Experiment A: official test partition only, or also a larger held-out set taken from validation?
- [ ] Gated data: who runs it (Brendan's PhysioNet credentialing, or Agustina), and may it run on Azure?
- [ ] Unified expanded paper or separate V2 paper (see `paper/v2-extension/MANUSCRIPT_PLAN.md`)
- [ ] Stop-token configuration for V2 training and evaluation (after the smoke-test measurement)
- [ ] Select SDBench, an alternative or no training simulator after the go/no-go checks
- [ ] Experiment A: approve 27B training if the 4B pilot passes
- [ ] W4: include the VOI-credit arm at 27B, or only plain vs entropy credit?
- [ ] W0 step 5: port to a Ray-native RL framework, or keep TRL + Ray job orchestration?
- [ ] Agustina's V2 role (V1 lists her as a co-author)
- [ ] Target journal(s) for submission (leaning TMLR; ML4H/CHIL alternatives; NMI only if W5/W6 are striking)
- [ ] arXiv endorser identified (Agustina, if eligible)
