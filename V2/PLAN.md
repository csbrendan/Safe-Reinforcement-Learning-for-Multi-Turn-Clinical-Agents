# V2 plan — from arXiv preprint to journal article

*Created 2026-11-01. Status: planned, nothing started. v1 (arXiv preprint, target 2026-11-15) is a separate
track and is NOT blocked by anything here.*

## Goal

Turn the current paper into a journal-grade article (style reference: Nature Machine Intelligence) by fixing every
major weakness of v1, accepting extra cloud spend to compress the timeline. Post the result as arXiv **v2** and
submit to a journal.

The thesis stays the same — *RL for clinical multi-turn agents must be evaluated as procedure learning; verifiable
outcome rewards are not enough* — but v2 must back it with adequately powered, scaled, externally checked, and
clinician-validated evidence.

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
| 5 | Synthetic toy simulators only | W5 external benchmark / real data |
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
- **Scope:** diagnostics only, plain vs entropy credit (+ VOI credit if W3 wins), **3 seeds each**, stratified
  eval. Do not port refill/titration/repair to 27B.
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

Detailed AgentClinic plan (verified capabilities, experiment matrix, interface adapter, splits, pilot):
[`docs/v2_agentclinic_experiment_plan.md`](../docs/v2_agentclinic_experiment_plan.md).

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
| [AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) | Interactive patient interviews, tests and diagnosis | Code (MIT); MedQA and NEJM cases | MIMIC-IV cases: PhysioNet credentialing plus a request to the authors. NEJM text is NEJM copyright, so do not redistribute it. Spec, Lang and tool code are not in the repo | Very high: first pilot |
| [SDBench](https://github.com/microsoft/sdbench) | Sequential diagnosis and the cost of tests | Code (MIT); 5 synthetic cases, not for reporting performance | NEJM cases: license from NEJM (terms not yet checked). MIMIC cases: PhysioNet credentialing | Very high: closest match to the RL question |
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

**Suggested order:**
1. AgentClinic-MedQA, which needs no credentials.
2. SDBench: start with the code and synthetic cases, then decide on NEJM licensing or MIMIC access.
3. HealthBench and Medmarks, as independent accuracy and safety measurements that complement the interactive
   evaluations.

Start PhysioNet credentialing early, even while using public data, because it opens every MIMIC option above.

## W6 — Clinician validation (Agustina Saenz, MD MPH — role TBD)

1. **Utility & finding review:** she reviews the three utility matrices, finding definitions, and per-ask cost for
   clinical plausibility; record changes she'd make. (If changes are large, rerun Gate 1 — CPU only.)
2. **Blinded trajectory rating:** ~100 trajectories sampled across arms and risk levels, method labels hidden.
   She labels each: appropriate / premature commitment / unsupported escalation / hedging / over-gathering.
3. **Agreement analysis:** Cohen's κ between her labels and the automated guarded report. High agreement
   validates the paper's central evaluation tool; disagreements become a findings section.

Her credit (author vs acknowledged clinical reviewer) is **her call — do not list her as an author until decided**.

## W7 — Writing & literature

- Add: reward hacking / specification gaming (e.g. Skalse et al.; Gao et al., reward-model overoptimization);
  MediQ; ED-Copilot (`sun2024edcopilot`, already in the bib); AMIE (`tu2024amie`); value of information / active
  feature acquisition; curiosity / info-gain intrinsic rewards; multi-turn LLM-RL credit assignment.
- Re-scope claims to the W1 confidence intervals; state subgroup n everywhere.
- Restructure to NMI article shape: short main text → Results → Discussion → Methods; ≤ ~8 display items; rest
  to Supplementary.
- Decide whether the Claude process-reward α-sweep (negative; see `framework/procreward/`) goes in as a
  supplementary "learned judge vs analytic credit" result.

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
| 8 | Internal review → arXiv v2 + journal submission |

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

Cost controls: shut nodes down between jobs; spot instances for evals; pilot before grid. **4B work and benchmark
evals run on the local DGX Sparks; only 27B goes to Azure.** Brendan's comfort level for 27B is ~$1–2k — the pilot
decides whether the full grid fits that; if not, cut to 2 seeds/arm or drop the VOI arm at 27B.

## Open decisions (Brendan)

- [ ] W5: Option A only, or A + B? (depends on Agustina's access)
- [ ] W4: include the VOI-credit arm at 27B, or only plain vs entropy credit?
- [ ] W0 step 5: port to a Ray-native RL framework, or keep TRL + Ray job orchestration?
- [ ] Agustina's role (author vs clinical reviewer)
- [ ] Target journal(s) for submission (leaning TMLR; ML4H/CHIL alternatives; NMI only if W5/W6 are striking)
- [ ] arXiv endorser identified (Agustina, if eligible)
