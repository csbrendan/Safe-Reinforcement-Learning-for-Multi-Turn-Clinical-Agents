# V2 plan — from arXiv preprint to journal article

*Created 2026-11-01; research strategy expanded 2026-11-08 and reorganized 2026-11-09; compute, infrastructure and budget
revised 2026-11-09. Status: proposed plan. No V2
training has started, and every experiment below is proposed, not completed. External-benchmark smoke tests are done
except the V1 stop-token check; see [`benchmark_smoke_report.md`](benchmark_smoke_report.md). v1 (arXiv preprint,
submitted 2026-11-07) is a separate track and is NOT blocked by anything here.*

## Goal

Turn the current paper into a journal-grade article (style reference: Nature Machine Intelligence) by fixing every
major weakness of v1, within a hard ceiling of $10,000 in incremental spending, and submit the result to a journal. V1
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

*Revised 2026-11-09 for the compute strategy. The Sparks come first, and cloud is used only if measured costs justify it.*

1. Resolve the V1 stop-token evaluation issue.
2. Validate the selected external simulator and its train/validation/test splits. This includes the SDBench access
   checks and the alternatives.
3. Run the initial 4B multi-turn GRPO pilot locally (Experiment B).
4. Benchmark Ray orchestration and local 27B inference.
5. Test local 27B LoRA-GRPO feasibility.
6. Benchmark affordable cloud training alternatives, if necessary.
7. Run the main 27B in-distribution SFT/GRPO comparisons (Experiment A).
8. Run independent OOD evaluations on the Sparks (Experiment C).
9. Expand with additional seeds or reward methods only as the budget permits.

**Experiment D** isn't in this order. Its simulators are deterministic and cheap to roll out, so it can train on the
Sparks once step 5 has measured local 27B speed. Schedule it then, within the training cap. Writing the paper, including
clinician-reviewed trajectory analysis, follows the results.

**Can run in parallel:** steps 2 and 4; PhysioNet credentialing and NEJM inquiries; W6 materials; and the AgentClinic
harness pilot on validation cases. A cloud quota request has no cost and can be filed early, but no cloud GPU is
provisioned before BC1.

### Milestones and dependencies

| Milestone | Depends on | Done when |
|---|---|---|
| M1: V1 audits complete | — | Stop-token effect on V1 measured with V1's own evaluation code; any correction decision recorded |
| M2: stop-token decision | M1 | Configuration chosen and recorded for all experiments |
| M3: training simulator decision | Access answers; go/no-go checks | One simulator selected against the criteria below, or all rejected |
| M4: 4B pilot (Experiment B) | M2; M3 | W8 pilot criteria pass or fail, reported either way |
| M5: 27B pipeline validated | 27B checkpoint and license check; M2; Ray benchmark; data-location decision for gated cases | Local 27B pilot (and the BC1 cloud pilot, if needed) measures memory fit, seconds per step, rollout throughput, sync time and cost per step |
| M6: Experiment A | Scaling criteria S1–S9; BC2 budget checkpoint; clinician labels | Matched arms evaluated once on test, or formally skipped |
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
| Cloud GPUs (1× or 2× H100 candidates); Ray | **Proposed.** No quotes; Ray not installed | "Compute, infrastructure and budget" |
| MedGemma-27B checkpoint and license | **Proposed**, not verified | W4 |
| 27B fit, speed and cost, on Sparks or cloud | **Unmeasured** estimates | Local pilot, then BC1 if needed |
| SDBench software with local models | **Confirmed** on 5 synthetic cases only | Smoke report; not a performance result |
| SDBench, or an alternative, as an RL training environment | **Proposed**, not validated | Go/no-go criteria, W8 |
| 27B training on gated cases in the cloud | **Unresolved** | See "Dependencies" |

### Dependencies

- **Stop-token configuration.** Our text-only MedGemma `generation_config.json`, and the rollout inits copied from it,
  stop only at token 1, not at `<end_of_turn>` (106). A replication needs an explicit choice: keep V1's behavior for
  comparability, fix it, or run both. Make the choice after the V1 measurement, not before. V1 results shouldn't be used
  as the foundation for V2 comparisons until this is resolved.
- **Base-model comparability.** Compare 4B and 27B on the same stratified sets (see "How V2 combines results across
  scales").
- **Reward provenance.** Every result reports whether its reward came from an exact verifier or a model-based judgment.
- **Gated data and compute location.** The Sparks are now the default for training and evaluation, which suits gated
  data. The conflict returns if 27B training on `mimic` or NEJM cases moves to the cloud. PhysioNet's guidance prohibits
  sending data to third parties, and allows a cloud service only if zero retention, no training use and no human review
  can be fully verified. Resolve this before any cloud training on gated cases. The options are written confirmation that
  the cloud setup complies, keeping gated training local, or using a dataset whose terms allow it.
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

## Compute, infrastructure and budget (revised 2026-11-09)

*Infrastructure and budgeting only. The research objectives, hypotheses and benchmark plans above are unchanged. Every
dollar figure is a provisional allocation or an estimate, not a price quote. Every throughput figure for 27B is
unmeasured until the pilots below run.*

### Hard budget ceiling

**Total incremental V2 spending must stay below $10,000 USD.** This ceiling covers:
- cloud GPU training and inference;
- external LLM API calls for patient simulators or judges;
- benchmark grading;
- infrastructure pilots, debugging and failed experiments;
- storage, networking and other cloud costs.

**Excluded:** the two DGX Sparks, which we already own. Their electricity and running costs aren't zero. Estimate and log
them separately. Whether they count against the ceiling is an open decision, and they don't count by default.

**Target $5,000–$7,000 of planned spending,** leaving $3,000–$5,000 for contingencies and additional experiments.

| Category | Provisional cap |
|---|---|
| 27B LoRA-GRPO training (Experiments A and D) | $4,000 |
| 27B SFT and initial evaluations | $1,000 |
| Benchmark grading and external judge APIs | $1,000 |
| Infrastructure pilots and debugging | $1,000 |
| Contingency | $3,000 |
| **Total of caps** | **$10,000** |

The caps add up to exactly the ceiling, so they are limits, not targets. Actual spending must stay below the total.
Revise the caps after measuring real throughput and getting actual prices. No category may be raised without lowering
another.

**What the caps imply (arithmetic, not a forecast):**
- $4,000 for training must cover Experiment A's two GRPO arms at 3 seeds (6 runs). If Experiment D's diagnostics core
  also goes to the cloud, that is 12 runs, or about $330 per run. That matches the low end of the earlier
  order-of-magnitude estimate, which was for a larger cloud node.
- Experiment D runs on V1's deterministic simulators, with no LLM patient or judge, so it is the natural candidate for
  local Spark training. Its titration and refill extension is outside the planned spend and would come from contingency.

### Budget checkpoints

Each checkpoint needs Brendan's explicit approval and a written projection before spending.

| Checkpoint | Before | Requirement |
|---|---|---|
| BC0 | Any paid spend | A spend ledger exists, recording date, category, provider, amount, purpose and running total. Cloud budget alerts and automatic shutdown are configured |
| BC1 | First cloud training pilot | Spending is capped at **$200–$300**, and the pilot's measurement goals are written down |
| BC2 | Any full training grid | The measured cost per optimizer step projects all required arms, plus evaluation, to fit within the training cap, and the $3,000 contingency stays intact |
| BC3 | Each further run | The previous run's actual cost was within 25% of its projection. Otherwise stop and re-project |
| BC4 | Any use of contingency | Stated purpose and amount, with the remaining headroom shown |
| BC5 | Any external API batch | Estimated tokens and cost, and confirmation that no restricted data is included |

### 27B training approach

**Use parameter-efficient LoRA or QLoRA, not full-parameter fine-tuning.** The primary comparisons are unchanged:
1. MedGemma-27B base.
2. MedGemma-27B SFT-only.
3. MedGemma-27B SFT + outcome-only GRPO.
4. MedGemma-27B SFT + credit-assignment GRPO.

Both RL arms start from the same SFT checkpoint. Add reward methods (decoupled GRPO, VOI credit) or extra seeds only after
the core results exist and the budget allows.

**8×H100 is not the default or a commitment.** Candidate configurations, cheapest first:
1. Local 2× DGX Spark (trainer on one, rollout server on the other, as in V1).
2. Cloud 1× H100 80GB, only if a measured test shows it fits.
3. Cloud 2× H100 80GB.
4. Larger configurations, only if measured performance and total projected cost justify them.

**What LoRA does and doesn't save.** LoRA cuts trainable parameters and optimizer memory. The full 27B base model still
has to run forward and backward passes. QLoRA may cut memory further, but may slow training and cause compatibility
problems. The pilot must account for each memory component:

| Component | Rough size or behavior (estimate) | Notes |
|---|---|---|
| Base weights | ≈54 GB in bf16; ≈14–17 GB in 4-bit (QLoRA) | Needed on both the trainer and the rollout server |
| LoRA parameters and optimizer state | About 0.2B trainable parameters at V1's rank 32 on all projections, so a few GB with Adam | Estimated from the published model shape; confirm by counting |
| Activations | Grow with sequence length and micro-batch size; gradient checkpointing is required | Multi-turn episodes are long, and SDBench cases are longer than V1's |
| Reference policy | With LoRA, the base with the adapter disabled serves as the reference, so there is no second copy of the weights. It costs an extra forward pass | Applies when the KL coefficient is above 0 |
| KV cache (rollout server) | Grows with concurrent episodes × context length | Limits rollout batch size |
| Rollout batch | V1 used 4 prompts × 8 rollouts per step | Larger batches need more KV cache on the server |

**QLoRA cautions:**
- If the trainer uses a 4-bit base but the rollout server samples from bf16, the policy being trained differs slightly
  from the policy generating data. Either match the precision or measure the gap.
- 4-bit kernel support on the Sparks' GB10 GPUs (ARM) is untested.

**Do not assume single-H100 GRPO is feasible without a measured test.**

### Two-Spark utilization strategy

The two DGX Sparks are the **default infrastructure**. Rent GPUs only for training that the Sparks can't do in acceptable
time, and avoid renting for evaluation. Each Spark has about 128 GB of unified CPU/GPU memory (121 GB visible), with a
GB10 GPU.

Default uses of the Sparks:
- MedGemma-4B experimentation;
- simulator implementation and debugging;
- reward and verifier development;
- 27B inference feasibility testing;
- local benchmark evaluations;
- small 27B LoRA-GRPO feasibility pilots;
- patient simulator and judge model hosting;
- Ray experimentation and distributed episode scheduling.

**Proposed evaluation architecture:**

| Node | Role |
|---|---|
| Spark 1 | MedGemma-27B on vLLM or another compatible server, loading base, SFT or GRPO adapters as needed. Evaluates checkpoints under matched inference settings |
| Spark 2 | Patient-simulation and grading models when feasible; benchmark orchestration and simulator components; extra evaluation workers when resources allow |

Possible local evaluation targets: AgentClinic, SDBench, MediQ, Medmarks, MedAgentBench, HealthBench response
generation, and other feasible benchmarks from "Benchmark options and access".

**Operational constraints:**
- Spark 1 ended abruptly three times under vLLM load. Write results incrementally, keep telemetry on, and make long jobs
  resumable.
- Evaluation and two-node GRPO can't share the Sparks at the same time. The NCCL weight sync needs a fresh server and a
  free trainer GPU.
- Swapping Spark 1 and Spark 2 roles remains an option if Spark 1's stability doesn't improve.

**Generating responses is not the same as official scoring.** Some benchmarks require specific judge models, external
APIs or proprietary grading. Results with substitute local judges are reported separately, never as official, and are
validated against official scoring where possible.

| Benchmark | Local response generation | Official scoring needs | Local scoring status |
|---|---|---|---|
| AgentClinic | Works (smoke test) | The paper's protocol used GPT-4 as moderator | Unofficial; local judge unvalidated |
| SDBench | Works on synthetic cases | GPT-5 / o4-mini configurations | Unofficial. For `mimic`, local or verified zero-retention models only |
| MediQ | Not run | Not yet checked | Unknown |
| Medmarks verifiable tasks | Works | Deterministic scorers, run locally | **Official** |
| Medmarks open-ended tasks | Not run | LLM judge (default `openai/gpt-5-mini`) | Unofficial if substituted |
| MedAgentBench | Not run; needs Docker permission | Grader downloaded separately | Unknown |
| HealthBench | Works | GPT-4.1 grader (≈$1–2 per 50-example 4B arm, estimated) | Unofficial with a local grader. Validate with the physician-completion mode |

**Restricted data.** For PhysioNet/MIMIC-derived cases, comply with dataset access and data-use restrictions. Don't send
restricted patient records to unauthorized external APIs. Running locally on the Sparks is the default for gated data.

### Ray orchestration evaluation

Ray is an orchestration and scheduling layer. It may improve resource use and concurrency. **It doesn't make the GPUs
compute any faster.** Adopt it only where it measurably helps.

**Possible applications:**
- **A. Parallel benchmark evaluation:**
  - independent encounters run concurrently;
  - lightweight simulator workers on CPU;
  - GPU-backed inference servers shared, rather than one model per worker;
  - structured trajectories, scores and error logs;
  - results resumable after a node failure.
- **B. Multi-turn GRPO rollout orchestration:**
  - concurrent patient episodes;
  - model calls routed to shared vLLM endpoints, with batching where possible;
  - each rollout tagged with the model version that generated it, plus its rewards;
  - on-policy integrity preserved through controlled adapter or weight synchronization, with stale rollouts rejected.
- **C. Multi-model simulation:**
  - doctor-policy inference and patient/judge inference as separate services;
  - workloads scheduled across the Sparks without duplicating model weights.
- **D. Integration with RL infrastructure:**
  - investigate Ray-compatible training, such as veRL or OpenRLHF;
  - don't rewrite the working TRL/GRPO loop unless a measured benefit justifies it;
  - compare a Ray design with the existing two-node trainer/rollout architecture;
  - any port must preserve `env_mask`, the custom `rollout_func`, the shaping, and the unshaped evaluation path.

**Benchmark protocol:**
- **Workload:** a fixed set of public validation cases, such as AgentClinic-MedQA validation, never test cases. Use the
  same models, decoding settings and turn limits throughout. Start with 4B, and repeat with 27B once it is served.
- **Conditions:** serial baseline; 4, 8 and 16 concurrent episodes. Each concurrency level is also run with a plain
  thread or async pool, so that any gain from Ray is separated from the effect of concurrency itself.
- **Measures:**
  - completed episodes per hour;
  - generated tokens per second;
  - GPU utilization and memory;
  - CPU utilization;
  - time waiting for inference;
  - patient and judge inference overhead;
  - episode latency distributions;
  - failure and timeout rates.
- **Decision (proposed threshold):** adopt Ray for evaluation if it gives at least 1.3× the completed episodes per hour
  of the best non-Ray pool, at an equal or lower failure rate. Also adopt it if it provides fault tolerance or resumability
  we need and wouldn't otherwise have. More concurrency is not assumed to help. If throughput flattens because the GPU is
  saturated, more episodes only add latency.
- **Isolation:** install Ray in its own environment, not in the shared `~/spark1/.venv`, until it's proven compatible.

### Local 27B LoRA-GRPO feasibility pilot

Run this tightly bounded local experiment before renting significant cloud infrastructure. It resembles V1's
disaggregated architecture.

| Node | Role |
|---|---|
| Spark 1: trainer | MedGemma-27B LoRA or QLoRA training; gradient checkpointing and memory-conscious batch sizes; GRPO optimizer and reference-policy handling |
| Spark 2: rollout server | MedGemma-27B inference; multi-turn episode generation; vLLM continuous batching where supported; simulator interaction and rollout logging |

The pilot runs on a V1 simulator or the selected simulator's training cases, never test cases.

**Steps:**
1. Verify 27B inference on one Spark.
2. Verify a 27B LoRA or QLoRA forward and backward training step.
3. Benchmark multi-turn rollout generation.
4. Run 5–10 genuine GRPO optimizer steps.
5. Validate model updates, gradient masking, reward variation, adapter synchronization and checkpoint integrity.
6. Record peak memory, GPU utilization, wall-clock time per optimizer step, rollout throughput and synchronization
   overhead.

Extrapolate the time for full experiments from these measurements, not from assumptions.

**Synchronization risk.** V1 synced full merged weights over NCCL, about 54 GB per update at 27B. That path is also prone
to the NCCL problem on any initialization after the first. Syncing only the adapter, by hot-loading LoRA weights into
vLLM, would move far less data. That option is **untested in our pipeline**.

**If the two-Spark setup is too slow,** keep the Sparks for evaluation and move only the expensive training stage to
cloud GPUs. "Too slow" means the projected wall-clock for the required arms doesn't fit the timeline Brendan accepts. The
earlier estimate for a V1-sized 27B run on the Sparks was about 35–50 h, before LLM-in-the-loop rollouts.

### Cloud-compute decision framework

Use cloud compute only when local benchmarks show a significant throughput advantage relative to the incremental cost.

**Before committing to cloud training:**
1. Price 1× H100 and 2× H100 configurations from current quotes, including region availability and quota.
2. Establish memory feasibility for the chosen configuration.
3. Benchmark a representative 27B LoRA-GRPO workload, with long multi-turn rollouts and realistic sequence lengths.
4. Compare the hourly price with optimizer steps completed per hour, giving **cost per optimizer step** and **projected
   cost per complete run**.
5. Include overheads: setup and idle time, storage, data egress and failed runs.

**First cloud experiment:** cap spending at about $200–$300 (BC1). Its only goal is to measure cost per step and projected
cost per run. **No full training grid** starts until the projected cost of all required arms fits within the budget
(BC2).

**Hybrid option** (local rollouts, cloud training). This may save money, but evaluate four things first:
- network latency and bandwidth for weight or adapter sync;
- synchronization cost per step;
- security and data-use terms, especially for gated cases;
- policy-version staleness, and whether it breaks on-policy GRPO.

Don't adopt a hybrid design without measuring these.

### Go/no-go criteria for scaling from 4B to 27B

All must hold before Experiment A's 27B arms launch:

| # | Criterion |
|---|---|
| S1 | The V1 stop-token question is resolved, and V2's configuration is recorded (M1, M2) |
| S2 | The training simulator passed its go/no-go criteria, and its splits are frozen (M3) |
| S3 | The 4B pilot (Experiment B) passed its pre-registered criteria (M4) |
| S4 | 27B inference measured on a Spark; the evaluation pipeline validated end to end at 4B |
| S5 | The local 27B pilot completed, or was ruled out for a measured reason |
| S6 | Where cloud is needed, the BC1 pilot gave a measured cost per step, and BC2 is satisfied |
| S7 | Data location and model-release terms are confirmed for the training data |
| S8 | Endpoints, checkpoint-selection rules and subgroup definitions are frozen before any test evaluation |
| S9 | Brendan explicitly approves the projected spend |

### Scientific safeguards (unchanged by budget pressure)

- The primary objective stays the same: does credit-assignment GRPO improve clinical information gathering and decision
  quality beyond a strong SFT baseline?
- Keep train, validation and test strictly separate.
- Use one shared, unshaped evaluation metric across SFT, outcome-only GRPO and credit-assignment GRPO.
- Track safety-relevant metrics separately from aggregate reward.
- Don't choose metrics or checkpoints based on favorable test results.
- If the budget runs short, cut seeds or optional arms first, and say so. Don't cut validation, the SFT baseline or
  test-set hygiene.

### Confirmed vs assumed

| Item | Status |
|---|---|
| Two Sparks, about 128 GB each, vLLM 0.20.1; 4B doctor plus Qwen3-8B environment across both | **Confirmed** (smoke tests) |
| 4B two-node GRPO on the Sparks (V1) | **Confirmed**, with the NCCL fresh-server caveat |
| 27B inference on a Spark | **Untested** |
| 27B LoRA or QLoRA training step on a Spark | **Untested** |
| QLoRA on GB10/ARM | **Untested** |
| Adapter-only sync to vLLM | **Untested** in our pipeline |
| Ray on the Sparks | **Untested**; not installed |
| 1× H100 27B GRPO feasibility | **Untested**. Don't assume it |
| Cloud prices, quota and cost per step | **Unknown**. No quotes yet |
| Spark 1 stability under heavy load | **Open risk** |

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
  ~$1–2k+ over ~4 extra weeks of Spark time). See "Compute analysis" below. *Superseded 2026-11-09: the Sparks are the
  default for everything, cloud training is optional, and the $10,000 ceiling applies.*
- **Ray:** Brendan explicitly wants to learn Ray; use it on Azure for orchestration (W0). *Superseded 2026-11-09: Ray is
  evaluated on the Sparks first.*
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

*2026-11-01 estimates, kept for reference. The decision below is superseded by "Compute, infrastructure and budget".*


- **Memory:** 128 GB unified memory per Spark; 27B bf16 ≈ 54 GB → rollout node fits vLLM + KV cache; trainer
  with LoRA + gradient checkpointing likely fits but tight (QLoRA fallback would add a confound vs the 4B setup).
- **Speed:** Spark decode is memory-bandwidth-bound (~270 GB/s); ~7× the parameters → each 60-step run
  **~35–50 h** vs 7 h at 4B. Weight sync grows to ~54 GB/update; the NCCL fresh-first-init issue means a crash
  late in a 2-day run costs the whole run.
- **Spark-only total:** 27B grid ~12 days + 27B evals ~4–5 days + 4B work ~1 week ≈ **4–5 weeks** serial.
- **Former decision (2026-11-01, superseded):** 27B on Azure, on a single cloud node. The revised plan doesn't assume
  8×H100. It tests the two Sparks first, then 1× or 2× H100 only if measured costs justify it.

## Timeline reality check

The original 8-week estimate was ~3 wk compute (critical path) + ~2 wk engineering + ~2 wk external waiting
(Agustina, quota) + ~2 wk analysis/writing, overlapping. Coding agents compress engineering to days and speed up
writing. Incompressible: machine time, Agustina's turnaround (send W6 materials in week 1), Azure quota /
PhysioNet approvals, and learning Ray. **Realistic: ~4–5 weeks with Azure for 27B** (vs ~6–7 Spark-only). *Superseded:
see "Timeline".*

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
| — | Infra: fragile 2-node NCCL weight sync; user wants to learn Ray | W0 Ray orchestration (Sparks first) |

All numbers above were reconciled against `results/probe_*.records.jsonl` on 2026-11-01.

---

## W0 — Infrastructure: Ray orchestration (Sparks first; cloud optional)

**Why:** run episodes, arms and evaluations concurrently on hardware we own, make long jobs resumable given Spark 1's
abrupt ends, and learn Ray. The 2-node NCCL weight-sync fragility (see `framework/procreward/README.md`) remains a
separate problem. Ray doesn't fix it by itself. The full evaluation plan and adoption threshold are in "Ray orchestration
evaluation".

**Steps (in increasing ambition):**
1. **Ray on the two Sparks,** in its own environment.
2. **Ray fan-out for evaluations.** Wrap episode-level evaluation, and `framework/gates/support_probe.py`, as Ray tasks
   that share the GPU-backed vLLM servers, instead of loading a model per worker. Benchmark it against a plain pool at
   serial, 4, 8 and 16 concurrent episodes.
3. **Seed and arm sweeps as Ray jobs.** Queue the existing `grpo_disagg.py` runs, one at a time on the Sparks, honoring
   the fresh-server-first-init rule.
4. **Optional, bigger: Ray-native GRPO.** Evaluate porting to a Ray-based RL framework (veRL or OpenRLHF both run on Ray
   and support GRPO; veRL has multi-turn/agent rollout support). Do this only if a measured benefit justifies replacing
   the working TRL loop. Porting risk: it must preserve `env_mask` (no gradient on simulator tokens), the custom
   `rollout_func`, the info-gain/DREAM shaping, and the invariant that env/gates/eval stay unshaped. **Gate:** reproduce
   one existing 4B diagnostics run (e.g. plain GRPO seed 1) within noise before trusting the port.
5. **Cloud, only if needed:** the same Ray jobs on a 1× or 2× H100 node, after BC1. Request quota early, since requesting
   it is free.

**Done when:** one command evaluates a list of checkpoints on a benchmark across both Sparks, with resumable per-episode
results in the existing probe format, and the Ray-vs-pool benchmark is recorded.

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
- **Layout (revised 2026-11-09):** LoRA or QLoRA only. First, the two Sparks as trainer and rollout server, as in V1.
  Cloud 1× or 2× H100 only if the measured cost per step justifies it. 8×H100 is not assumed. The earlier plan of one
  8×H100 node (4 trainer + 4 vLLM GPUs) on Azure is superseded.
- **Earlier estimate:** Spark-only 27B at ~35–50 h per run, with NCCL first-init risk on every 2-day run. The local
  pilot replaces this with measurements.
- **Pilot first:** the local 27B pilot (5–10 real GRPO steps), then BC1 if cloud is considered. Measure memory fit,
  seconds per step, rollout tokens/s and sync time. Extrapolate to 60 steps × the required arms to get real time and cost
  before committing.

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

## Timeline (revised 2026-11-09)

*No reliable total exists until the pilots measure throughput, and data access may take weeks. The earlier "~4–5 weeks
with Azure" estimate no longer applies. Phases overlap where noted.*

| Phase | Work | Machines | Exit |
|---|---|---|---|
| 0 | V1 stop-token rerun; PhysioNet credentialing and NEJM inquiry; Ray environment; W6 materials to Agustina | Sparks (preferably Spark 2) | M1, M2 |
| 1 | Simulator desk and environment checks; AgentClinic harness on validation; Ray-vs-pool benchmark at 4B | Sparks | M3; Ray decision |
| 2 | 4B adapter and bounded GRPO pilot (Experiment B); W1/W2 sets; W3 4B arms | Sparks | M4 |
| 3 | 27B inference on a Spark; local 27B LoRA-GRPO pilot | Sparks | Local measurement of cost and time |
| 4 | Only if needed: price 1×/2× H100 and run a cloud pilot (≤$200–300) | Cloud | BC1, BC2 |
| 5 | Experiment A core: 27B SFT, outcome-only and credit-assignment GRPO, 3 seeds | Sparks and/or cloud | M6 |
| 5′ | Experiment D diagnostics core, in parallel where hardware allows | Sparks preferred | M8 |
| 6 | Experiment C OOD evaluation; clinician review | Sparks | M7 |
| 7 | Expansions as budget permits; analysis; manuscript (see `paper/v2-extension/MANUSCRIPT_PLAN.md`) | — | M9 |

**Known delays:** PhysioNet review typically takes 1–2 weeks after the CITI course, and NEJM's response time is unknown.
Training on the Sparks is serial and slow at 27B. If cloud is used, quota approval can take days. Re-estimate the
timeline after phase 3.

## Budget (revised 2026-11-09)

**The ceiling, caps and checkpoints are in "Compute, infrastructure and budget".** The table below maps experiments to
those caps. Every figure is a placeholder until measured.

| Item | Default machines | Cap it draws on | Estimate |
|---|---|---|---|
| W1/W2/W3 4B training and all 4B evals | Sparks | — | No cloud spend; electricity logged separately |
| Experiment B: go/no-go checks and 4B pilot | Sparks | — | No cloud spend if environment models run locally. A NEJM license fee, if any, is unknown and isn't covered by any cap yet |
| Ray benchmark; local 27B pilot | Sparks | Infrastructure pilots ($1,000) | No cloud spend |
| Cloud pilot, if needed | 1× or 2× H100 | Infrastructure pilots | ≤$200–300 (BC1) |
| Experiment A: 27B SFT and initial evals | Sparks, or cloud for SFT | SFT and evaluations ($1,000) | Measured at BC2 |
| Experiment A: 2 GRPO arms × 3 seeds | Sparks and/or cloud | Training ($4,000) | Measured at BC2. LLM-in-the-loop rollouts should cost several times a V1 run |
| Experiment D core: diagnostics, 2 arms × 3 seeds + SFT | Sparks preferred | Training | Cloud would draw roughly half the training cap at about $330 per run |
| Experiment D extension: titration and refill | Sparks | Contingency | Only if headroom remains |
| Experiment C and other benchmark evaluations | Sparks | Grading and APIs ($1,000), for official graders only | HealthBench official grader ≈$1–2 per 50-example 4B arm (estimate); SDBench official models not priced |
| Extra seeds and reward methods (decoupled, VOI) | Sparks first | Contingency ($3,000) | After core results, by BC4 |

**Earlier estimates (superseded).** These assumed an 8×H100 node: a 27B grid of ~$2–6k, roughly $3–8k in total, and
~$330–670 per 27B run, against Brendan's earlier ~$1–2k comfort level for 27B. Keep them only as a sense of scale.

**Cost controls:**
- Shut cloud nodes down between jobs, and set budget alerts and automatic shutdown.
- Use spot capacity only for interruptible evaluation, never for training that can't resume.
- Pilot before any grid.
- Cut seeds or optional arms before cutting validation or baselines.
- External APIs need approval for each batch (BC5), and are never used for MIMIC-derived data unless verified compliant.

## Open decisions (Brendan)

- [ ] W5: Option A only, or A + B? (depends on Agustina's access)
- [ ] Experiment D: approve the titration and refill extension after the core (cost above)?
- [ ] Experiment D: define the SFT-only arm, and whether to train a matched 4B SFT-only arm
- [ ] Experiment A: source of SFT demonstrations, given that SDBench has no oracle
- [ ] Experiment A: official test partition only, or also a larger held-out set taken from validation?
- [ ] Gated data: who runs it (Brendan's PhysioNet credentialing, or Agustina), and may it ever run in the cloud?
- [ ] Do Spark electricity and running costs count against the $10,000 ceiling? (Default: no; logged separately)
- [ ] After the local 27B pilot: keep 27B training on the Sparks, or go to the BC1 cloud pilot?
- [ ] Cloud configuration, if any: 1× or 2× H100, chosen on measured cost per step
- [ ] LoRA or QLoRA at 27B (and matching precision on the rollout server)
- [ ] Adopt Ray for evaluation after the Ray-vs-pool benchmark?
- [ ] Unified expanded paper or separate V2 paper (see `paper/v2-extension/MANUSCRIPT_PLAN.md`)
- [ ] Stop-token configuration for V2 training and evaluation (after the smoke-test measurement)
- [ ] Select SDBench, an alternative or no training simulator after the go/no-go checks
- [ ] Experiment A: approve 27B training if the 4B pilot passes
- [ ] W4: include the VOI-credit arm at 27B, or only plain vs entropy credit?
- [ ] W0 step 4: port to a Ray-native RL framework, or keep TRL + Ray job orchestration?
- [ ] Agustina's V2 role (V1 lists her as a co-author)
- [ ] Target journal(s) for submission (leaning TMLR; ML4H/CHIL alternatives; NMI only if W5/W6 are striking)
- [ ] arXiv endorser identified (Agustina, if eligible)
