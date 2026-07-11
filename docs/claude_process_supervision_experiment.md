# Experiment Extension — Claude Process Supervision for Multi-Turn Clinical GRPO

> **Status:** planned workshop-style extension
> **Owner:** _tbd_
> **Depends on:** existing GRPO training stack (MedGemma-4B-Instruct, synthetic multi-turn clinical environment) and the existing InfoGain reward path.

---

## 1. Purpose and scope

The core project post-trains **MedGemma-4B-Instruct** with **GRPO** in a **synthetic multi-turn clinical environment** for prescription and medication-refill reasoning. Our main result compares:

- **Baseline GRPO** — verifier-based task reward only.
- **GRPO + InfoGain** — the same task reward plus an **analytically computed** per-turn information-gain shaping term.

This document specifies **one additional experiment** that adds an *empirical* process-supervision signal alongside the existing analytic one. After each rollout, a strong external LLM (**Claude Opus**) reads the trajectory and emits a **structured, turn-level score for each question the model asks**. Those scores are reduced to a **scalar process reward** and added to the existing reward. Training is otherwise **identical** — same environment, same optimizer, same task reward, same hyperparameters, same eval.

This is a **complement to the main story, not a replacement.** The main story remains "how much does RL post-training help, and does analytic InfoGain shaping help further." This extension asks a narrower, orthogonal question:

> **Does an external-LLM process signal add anything beyond the task reward, and how does it compare — head to head and analytically — with the InfoGain term we already compute?**

### What this experiment is NOT

To avoid confusion with adjacent techniques, we state the boundaries explicitly:

- **Not token-level distillation.** We do not train MedGemma toward Claude's token distribution, logits, or generated text. Claude never provides a target sequence.
- **Not EMA / self-distillation.** There is no exponential-moving-average teacher, no target network derived from the policy, and no bootstrap from the model's own past checkpoints.
- **Not a policy-init or SFT step.** Claude's outputs never enter the SFT data or warm-start.

It is **only** LLM-based **process supervision via structured scores**: Claude acts as an off-policy *scorer* of intermediate turns, and its structured score is folded into the scalar reward that GRPO already consumes.

---

## 2. Process-supervision lineage (direct ancestors)

This experiment sits in the process-supervision line of work. We cite three direct ancestors and summarize how each maps onto our design.

### Uesato et al., 2022 — *Solving math word problems with process- and outcome-based feedback*
Introduced the explicit contrast between **outcome-based** feedback (reward the final answer) and **process-based** feedback (reward the intermediate reasoning steps), and showed that process-based feedback can improve not just final accuracy but the **trace-level correctness** of reasoning. **Mapping to us:** our baseline GRPO task reward is largely *outcome-based* (did the final structured decision satisfy the verifiers). Claude's turn-level scores are a *process-based* signal layered on top.

### Lightman et al., 2024 — *Let's Verify Step by Step*
Trained **process reward models (PRMs)** on dense, per-step human judgments (the PRM800K effort) and demonstrated that **step-level supervision outperforms outcome-level supervision** on hard reasoning, and that PRMs generalize as verifiers. **Mapping to us:** Claude is used as a *zero-training, prompted* stand-in for a PRM. Instead of a learned PRM over math steps, we elicit structured per-turn scores over clinical dialogue turns. We inherit the "score the steps, not just the answer" thesis without training a separate reward model.

### TIPS, 2026 — *Turn-level / Interactive Process Supervision*
Extends process supervision from single-shot chains of thought to **multi-turn, interactive trajectories**, where the unit of supervision is a **turn** (e.g., a clinician-style question or action) rather than a reasoning token span. **Mapping to us:** this is the closest ancestor — our environment is multi-turn, and our supervision unit is exactly "each question the model asks." We adopt the turn-as-step framing directly.

> _Citation note: confirm the exact TIPS venue/author list and page reference before submission; the summary above reflects the turn-level-process-supervision framing under which it is cited here._

**Positioning.** Our contribution is **not a new algorithm.** It is a **controlled-environment case study**: holding the environment, optimizer, and evaluation fixed, we measure what an external-LLM process signal buys us, and we compare it — empirically and analytically — against an analytic process signal (InfoGain) computed on the same turns. The value is the controlled comparison, not a novel training method.

---

## 3. Experimental conditions

Four conditions, identical in every respect except the shaping term added to the task reward.

| # | Condition | Reward = task reward + … |
|---|-----------|--------------------------|
| **C1** | **GRPO** (baseline) | nothing (task reward only) |
| **C2** | **GRPO + InfoGain** | `λ_ig · InfoGain_turn` (analytic, existing) |
| **C3** | **GRPO + Claude Process Supervision** | `λ_cl · ClaudeScore_turn` (external LLM, new) |
| **C4** | **GRPO + InfoGain + Claude** (combination) | `λ_ig · InfoGain_turn + λ_cl · ClaudeScore_turn` |

- **C1** and **C2** already exist in the codebase; they are re-run under the frozen config so all four share seeds, data, and eval.
- **C3** is the new path.
- **C4** tests whether the analytic and empirical process signals are **complementary or redundant** — the most interesting single comparison in the study.

All four use the **same** GRPO group size, KL coefficient, learning rate, rollout budget, environment seeds, and the **unchanged** evaluation set (§7).

---

## 4. Reward integration

Let a rollout be a trajectory of turns `t = 1..T`. The existing per-trajectory task reward `R_task` (verifier-based: schema validity, action correctness, dosing, safety penalties, escalation calibration — see `docs/project-context.md` §7) is unchanged.

We define a **scalar process reward** aggregated from per-turn scores:

```
ClaudeProc(traj)  = Σ_t  w_t · s_t          # s_t = Claude's normalized turn score in [0,1]
InfoGainProc(traj)= Σ_t  g_t                # g_t = analytic info-gain for turn t (existing)
```

and the total reward fed to GRPO in each condition:

```
C1:  R = R_task
C2:  R = R_task + λ_ig · InfoGainProc
C3:  R = R_task + λ_cl · ClaudeProc
C4:  R = R_task + λ_ig · InfoGainProc + λ_cl · ClaudeProc
```

Design decisions:

- **Scalar, not per-token.** `ClaudeProc` is a single scalar added to the trajectory reward before GRPO's group-relative advantage normalization. It never becomes a per-token target. This is what keeps the method "process supervision via structured scores" and not distillation.
- **Turn weighting `w_t`.** Default `w_t = 1/T` (mean of turn scores) so trajectory length does not inflate reward. A "last-turn-weighted" variant is an ablation, not the default.
- **Normalization.** Claude scores are mapped to `[0,1]` per rubric dimension, averaged to a turn score `s_t`, then the trajectory sum enters reward. `λ_cl` is tuned on a small dev sweep so `ClaudeProc`'s contribution is comparable in magnitude to `InfoGainProc` under C2 — this is required for a fair C2-vs-C3 comparison.
- **Off-policy, post-rollout.** Claude scores **after** rollouts are generated. It never influences sampling, so it adds no on-policy latency to generation; it adds a scoring pass before the reward is finalized (see §6 on throughput and caching).

---

## 5. The Claude scoring rubric (structured JSON)

Claude receives, per rollout: the case context (patient state), the full dialogue so far, and **one target turn** (the model's question/action). It returns a strict JSON object. Scoring is **per turn**; a rollout of `T` turns yields `T` such objects.

### Prompt contract
- **System:** "You are a clinical-reasoning process evaluator. Score only the *target turn* in the context of the trajectory so far. Do not reward the final answer; reward whether this turn was a good next step. Output strict JSON matching the schema. Do not include prose outside the JSON."
- **Inputs:** `{case_context, dialogue_history, target_turn_index, target_turn_text, available_findings}`.
- **Output:** the JSON below.

### Rubric schema

```json
{
  "turn_index": 3,
  "dimensions": {
    "information_value": {
      "score": 2,
      "max": 3,
      "justification": "Asks for renal function, which is decision-relevant and not yet known."
    },
    "clinical_appropriateness": {
      "score": 3,
      "max": 3,
      "justification": "Guideline-consistent next question for a renally-cleared drug."
    },
    "safety_sensitivity": {
      "score": 2,
      "max": 3,
      "justification": "Probes a plausible contraindication path; did not yet check allergies."
    },
    "progress_toward_decision": {
      "score": 2,
      "max": 3,
      "justification": "Moves the case toward a dose decision rather than stalling."
    },
    "non_redundancy": {
      "score": 3,
      "max": 3,
      "justification": "Information not already present in the history."
    },
    "calibration": {
      "score": 2,
      "max": 3,
      "justification": "Reasonable to ask rather than escalate immediately; not over-cautious."
    }
  },
  "turn_score_normalized": 0.78,
  "would_a_careful_clinician_ask_this_now": true,
  "redundant_with_prior_turn": false,
  "flags": ["none"],
  "confidence": 0.7
}
```

Field notes:
- Each **dimension** is an integer `score` out of `max`, with a one-line `justification` (justifications are for auditing, not fed into reward).
- `turn_score_normalized ∈ [0,1]` is the reduction used as `s_t` (default: mean of dimension `score/max`). We compute it ourselves from the dimensions rather than trusting a free-form overall number, to keep the mapping deterministic and auditable.
- `flags` ∈ `{"none","unsafe_suggestion","hallucinated_finding","redundant","off_task"}`. Any safety flag can optionally gate a penalty in an ablation; in the default it only sets `s_t = 0` for that turn.
- `confidence` is logged for analysis (does agreement with InfoGain rise with Claude's stated confidence?) but is **not** used in reward by default.

### Robustness rules
- **Strict JSON parse**; on parse failure, retry once with a "return valid JSON only" reminder, then treat as `s_t = neutral (0.5)` and log a parse-failure counter. Parse-failure rate is a reported diagnostic.
- **Rubric versioning.** The rubric text and schema are versioned (`rubric_vX.json`); the version id is stored on every score record so runs are reproducible.
- **Determinism.** Score with temperature 0 and a fixed system prompt; log the exact model id (`claude-opus-4-8`) and rubric version with every record.

---

## 6. Claude vs InfoGain — the analytical comparison

This is a first-class deliverable, not a side note. Both signals score the **same turns**, so we can compare them directly.

For every turn in a held-out set of trajectories, we log the pair `(g_t, s_t)` = (analytic InfoGain, Claude turn score) plus scenario tags (renal_dosing, allergy, missing_labs, needs_escalation, …). We then report:

1. **Correlation** — Spearman rank and Pearson correlation between `g_t` and `s_t`, overall and **per scenario bucket**. Where do the analytic and LLM signals agree, and where do they diverge?
2. **Agreement on "high-value" turns** — treat top-quartile `g_t` and top-quartile `s_t` as "valuable"; report precision/recall/κ between the two notions of value.
3. **Disagreement analysis** — sample turns where the signals most disagree (high InfoGain / low Claude and vice-versa) and characterize them qualitatively. E.g., does InfoGain reward information-theoretically useful but clinically pointless questions? Does Claude reward "sounds thorough" turns with low actual information value?
4. **Redundancy in reward space** — does C4 (both) beat C2 and C3 on eval, or is the improvement subadditive (signals redundant)? This is the empirical test of whether the LLM signal carries anything the analytic signal does not.
5. **Cost-adjusted view** — InfoGain is ~free to compute; Claude scoring has API cost/latency. Report eval gain **per unit cost** so the practical tradeoff is explicit.

The headline analytic claim we want to support or refute: *does the external LLM's structured judgment approximate, exceed, or diverge from the analytic information-gain quantity we already have for free?*

---

## 7. Evaluation (unchanged)

Evaluation is **identical** to the main study — same held-out eval set, same structured/safety/calibration/edge-case metrics, same reporting (see `docs/project-context.md` §9). No eval metric is added or removed for this extension; the four conditions are compared on the existing ladder so results are directly commensurable with the main GRPO-vs-InfoGain result.

The **only** new measurements are training-time diagnostics and the Claude-vs-InfoGain analysis in §6 — none of which touch the eval protocol.

---

## 8. Implementation steps

1. **Freeze the shared config.** Pin seeds, GRPO hyperparameters, environment version, and eval set so C1–C4 differ *only* in the shaping term. Re-run C1 and C2 under this frozen config to serve as clean references.
2. **Trajectory logger.** Ensure each rollout persists: case context, per-turn text, turn indices, existing per-turn InfoGain `g_t`, and `R_task`. This is the input to Claude scoring and the §6 analysis.
3. **Claude scoring service.** Build a batched, cached client (`claude-opus-4-8`, temp 0) that takes a trajectory + target turn and returns the §5 JSON. Include: strict-JSON validation, one retry, parse-failure fallback, rubric-version stamping, and a response cache keyed on `(rubric_version, case_id, turn_hash)` so re-runs and C3/C4 sharing the same rollouts don't re-score.
4. **Reward assembler.** Implement `ClaudeProc(traj)` reduction (§4) and wire the four reward expressions behind a single config switch (`shaping ∈ {none, infogain, claude, both}`).
5. **λ calibration sweep.** Small dev sweep to set `λ_cl` so Claude's reward magnitude is comparable to InfoGain's under C2. Record the chosen value; use it for both C3 and C4.
6. **Run C1–C4.** Same compute budget each. Log training curves, reward-component magnitudes, and the paired `(g_t, s_t)` records.
7. **Evaluate on the frozen eval set.** Produce the standard A/B tables for all four conditions.
8. **Analysis pass (§6).** Correlation, agreement, disagreement case studies, redundancy (C4 vs C2/C3), cost-adjusted gains.
9. **Write-up.** Workshop-style short paper: method boundaries (§1), lineage (§2), the four-way result, and the Claude-vs-InfoGain analysis as the central figure.

---

## 9. Milestones

| Milestone | Definition of done |
|-----------|--------------------|
| **M0 — Config frozen** | C1/C2 re-run reproduces the existing main-study numbers within noise. |
| **M1 — Scoring service** | Claude scorer returns schema-valid JSON on ≥99% of turns (post-retry) on a 200-turn smoke set; cache hit-path verified. |
| **M2 — Reward wired** | All four `shaping` modes train end-to-end for a short run without NaNs; reward-component magnitudes logged and comparable. |
| **M3 — λ calibrated** | `λ_cl` chosen; Claude and InfoGain contributions within the same order of magnitude under matched conditions. |
| **M4 — Full runs done** | C1–C4 trained to the same budget; checkpoints and logs archived. |
| **M5 — Eval complete** | Four-condition tables on the unchanged eval set. |
| **M6 — Analysis complete** | §6 correlation/agreement/disagreement/redundancy/cost figures produced. |
| **M7 — Write-up drafted** | Workshop extension draft with method, lineage, results, and the Claude-vs-InfoGain figure. |

---

## 10. Risks and mitigations

- **Scorer noise / prompt sensitivity.** Fix rubric version, temp 0, deterministic reduction; report parse-failure and inter-run stability on a re-scored subset.
- **Reward hacking toward "Claude-pleasing" turns.** Watch for verbosity or thoroughness-signaling that raises `s_t` without helping eval; the unchanged eval set is the guard, and disagreement analysis (§6.3) surfaces it.
- **Magnitude imbalance.** If `λ_cl` is mis-set, C3/C4 either ignore or over-weight the signal; M3 calibration is the mitigation.
- **Cost/latency.** Off-policy post-rollout scoring + caching keeps Claude off the generation hot path; §6.5 reports the cost-adjusted tradeoff honestly.
- **Confounded comparison.** Any drift in seeds/data/eval between conditions invalidates the four-way claim; M0 config-freeze is the safeguard.

---

## 11. Future work (out of scope here)

The natural next step — **explicitly out of scope for this workshop extension** — is **true teacher–student distillation with an open-weight teacher**: replacing the prompted external scorer with a trained, open-weight process reward model (or a teacher policy), and distilling its judgments into MedGemma, potentially at the token or step level. That would trade the current method's simplicity and auditability for a self-contained, reproducible, no-external-API pipeline, and would let us study distillation dynamics directly. It is a plausible and appealing follow-up, but it is a **different experiment with a different contribution** and is intentionally not attempted here. This extension deliberately stays at "external LLM provides structured process scores added to the reward," so the controlled comparison against analytic InfoGain remains clean.
