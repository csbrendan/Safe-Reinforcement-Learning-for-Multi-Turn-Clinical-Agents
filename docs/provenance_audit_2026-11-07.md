# Provenance audit of the three synthetic clinical environments

*Audit date: 2026-11-07. Read-only investigation; no code, results, or manuscript changed.*

**Sources examined**
- This repo (`rl-gains-in-clinical-tasks`, 107 commits from 2026-06-17).
- The prior project `~/clinical-rx-rl` (172 commits from 2026-04-17), where refill triage was built.
- `docs/`, the `design.md` files, and the commit messages in both repos.
- The surviving Claude Code transcripts.
- `paper/main.tex` (current version).

**Bottom line**
- **No parameter in any environment was estimated from a dataset, and none is taken from a guideline or published study.**
- Finding names and drug and score pairings are clinically flavoured. Every number is a hand-chosen experimental value.
- Many numbers were explicitly revised to make Gate 1 pass, or to stop GRPO from exploiting the question cost.

**Limits of the evidence**
- No conversation transcript from the June 2026 design period survives. Rationale is therefore known only where commit messages, docs or code comments record it.
- Where no rationale is recorded, the source is marked *unknown* rather than inferred.

**Classification key.**
- **Empirical**: derived from data.
- **Literature**: taken from a guideline or published study.
- **Clinically informed, hand-designed**: clinical concept or naming, values set by hand.
- **Arbitrary experimental choice**: a value chosen or tuned for experimental behaviour (gates, headroom, exploit control).
- **Unknown**: no recorded source.

---

## 1. Medication refill triage (`tasks/rx_triage/env.py`)

**Lineage, all in `~/clinical-rx-rl/scripts/phase2b/`:**
1. `sim_renal.py`, a 2-variable prototype (17e1df7, 2026-06-16).
2. `sim_workup.py`, with 8 findings and the trees (3184ab4, 06-16).
3. 12 findings and rebalanced priors (9610921, 06-17).
4. A reward-recalibration sweep (c1407f5, 06-17).
5. steepHigh + tightMod with c=0.03 (6224e1b, 06-17).
6. Widened trees, with hold_for_labs on high set to 0.40 (e7cfeee, 06-17).

The result was copied unchanged into this repo in b75d4c9 (2026-06-18); only the comments differ. Later, 4879cf2 changed only the oracle's tie-break order.

| Parameter / rule | Current value | Earliest origin | Evidence | Classification | Conf. |
|---|---|---|---|---|---|
| 12 findings | `env.py:20-21` (reveal text `:181-194`) | 8 in 3184ab4 (06-16); +4 (acute_symptoms, egfr_severe, sodium_low, gi_bleed_risk) in 9610921 | 9610921: "VOI-structural changes … lft/a1c remain distractor+trap" | Clinically informed, hand-designed (added to widen adaptive gap) | High |
| Finding priors P1 | `env.py:23` | 3184ab4; current values 9610921 | "screen priors rebalanced ~0.5 so deep branches are common". Findings independent. Prototype comment "prior grounded in rough real co-occurrence" (`sim_renal.py:29`) cites no source and is not in the final P1 | Arbitrary experimental choice | High |
| Severity trees | `env.py:45-74` | 3184ab4; current topology e7cfeee | "widened to 3-level disjoint-module trees … so a safe fixed checklist must over-ask" | Clinically flavoured pairings (ACEi to K/eGFR; NSAID to GI bleed/renal), structure hand-designed for gates | High |
| Latent severity distribution | Induced by P1 plus trees (`risk_dist`, `env.py:112`); ACEi .29/.57/.14, NSAID .40/.46/.14 | 9610921 / e7cfeee | Never targeted or justified | Consequence of arbitrary choices | High |
| Utility matrix U | `env.py:36-40` | 3184ab4 ("soft" version); steepHigh+tightMod 6224e1b; U[high][hold_for_labs] 0.85 to 0.40 e7cfeee | c1407f5 `docs/phase2b_task1_reward_recal.md`: candidate U tables swept against SFT/GRPO rollouts to "fix ask-cost exploitation". e7cfeee "removes the moderate/high hedge" | Arbitrary experimental choice (safety-ordered shape, values tuned against gates) | High |
| Question cost c = 0.03 | `env.py:30` | 0.04 (d6c598d), 0.06 (3184ab4), 0.03 (6224e1b) | "gathering cheaper; the oracle's ~2.5 asks aren't over-penalized; 1-ask's cost edge shrinks" | Arbitrary experimental choice | High |
| 5 actions | `env.py:28` | 17e1df7 (06-16); earlier v0 schema had approve/modify/deny/escalate | No rationale beyond "multiple defensible options" | Clinically informed, hand-designed | Med-high |
| Families ACEi/ARB, NSAID | `env.py:25` | 4 families in 3184ab4; this pair chosen in 9610921 | Selected as "deepest trees, largest adaptive gap"; drug names are labels | Hand-designed (selected for gap) | High |
| Oracle | `env.py:128-160` | Belief-state DP 17e1df7; `oracle_rollout`, EPS 50b9455; tie-break 4879cf2 | Exact enumeration under the simulator's own U and P1 | Algorithmic (exact for the model; no clinical validity implied) | High |

---

## 2. Acute diagnostics (`tasks/diagnostics/env_rv.py`, `DX_VARIANT=deephedge`)

**Lineage:**
1. A scaffold in b75d4c9 (06-18).
2. The base environment, 8a757f1 (06-19), Gate 1 BC-gap +0.117.
3. The reward-ablation screen, 6afe739 (06-20): "hedge is load-bearing for BOTH the gate and the exploit".
4. deephedge, 746da27 (06-20): "Co-designed deephedge variant: Gate 1 PASS (BC-gap +0.161, 1-ask insufficient, capped hedge)".

The 06-18 scaffold `design.md` proposed "mine realistic test/finding co-occurrence + dangerous-miss rates to set priors and U". There is no evidence this was done.

| Parameter / rule | Current value | Earliest origin | Evidence | Classification | Conf. |
|---|---|---|---|---|---|
| 18 findings | `env_rv.py:19-24` | 14 in 8a757f1; +4 (renal_impair, age_high, anticoag_use, copd) in 746da27 | The 4 added for "forcing a fixed checklist to over-order -> bigger BC-gap" | Clinically informed names; extras exist to shape the gate | High |
| Base priors P1 | `env_rv.py:28` (0.30–0.50) | 8a757f1 | No source; round values mirroring rx_triage | Arbitrary experimental choice | High |
| deephedge priors (0.52 on emergent findings) | `env_rv.py:36-41` | 746da27 | "raise those findings' prevalence so emergent stays ~10%"; realised ≈6.9% | Arbitrary experimental choice (tuned) | High |
| Severity trees; emergent = 4-finding conjunction | `env_rv.py:116-148` | 8a757f1 / 746da27 | "emergent is engineered rare"; "no single finding diagnostic". (Stale comment at L108 still says 3-finding.) | Hand-designed to expose hedging and pass gates | High |
| Latent severity distribution | Induced by P1 plus trees (`risk_dist`, L184); emergent ≈6.9% | 746da27 | Follows from the rows above | Consequence of arbitrary choices | High |
| Utility U (base) | `env_rv.py:67-73` | Copied from rx_triage U (b75d4c9) | design.md: "identical utility matrix under canonical severity keys"; comment maps discharge to continue … emergent_tx to escalate | Arbitrary (inherits rx_triage's tuned U) | High |
| deephedge `admit` cap (0.86 to 0.45; 0.40 to 0.20) | `env_rv.py:70,72` | 746da27 | "capped hedge … tuned to pass Gate 1"; antihedge (−0.30) "BREAKS Gate 1" | Arbitrary experimental choice (tuned to pass Gate 1) | High |
| Question cost 0.03 | `env_rv.py:52` | From rx_triage | lowcost variant (0.015) cut BC-gap to +0.059 | Arbitrary (kept for consistency) | High |
| EPS 0.03, EPS_TIE 0.05 | `env_rv.py:53,50` | b75d4c9 / rx_triage | No rationale | Arbitrary | High |
| Turn cap 12 | CLI flag (default 10) | 12 first used cedc383 (06-21) | No recorded rationale (oracle averages ~3.9 asks) | Arbitrary | Medium |
| Oracle | `env_rv.py:200-232` | 8a757f1 | Exact Bellman recursion under P1 | Algorithmic | High |

---

## 3. Dosing titration (`tasks/titration/env.py`)

**Lineage:** every value comes from a single commit, c92f4e0 (2026-06-23), which passed "Gate 0". The only earlier document is the scope doc, `docs/titration_scope.md` (1715229, same day). No value has changed since.

The docstring says (L13): "Stylized, guideline-INSPIRED, NOT clinically authoritative".

| Parameter / rule | Current value | Earliest origin | Evidence | Classification | Conf. |
|---|---|---|---|---|---|
| 6 risk factors | `env.py:26` | c92f4e0 | Scope proposed 12 (diabetes, vascular, renal, labile INR …); 6 dropped without recorded reason (orphan `_REVEAL` strings L189-196) | Clinically informed subset, hand-picked | High |
| Priors P1 (0.35–0.50) | `env.py:29` | c92f4e0 | No source | Arbitrary | High |
| Clot weights {prior_clot 3, age_high 2, chf 1, htn 1} | `env.py:33` | c92f4e0 | "stylized CHA2DS2-VASc … PIVOTAL high-weight factors so a single ask can decisively move a bucket" | Loosely inspired by CHA2DS2-VASc; weights set for value of information (see below) | High |
| Bleed weights {bleed_history 3, fall_risk 2, age_high 1, htn 1} | `env.py:34` | c92f4e0 | Same; "stylized … HAS-BLED" | Loosely inspired by HAS-BLED; largely invented | High |
| Risk buckets (low ≤1, mod 2–4, high ≥5) | `env.py:90-91` | c92f4e0 | No source; match neither score's published cut-offs | Arbitrary | High |
| Utility terms EFF, EXP, CLOT_VAL, BLEED_VAL | `env.py:53-56` | c92f4e0 | `design_gate.py` instructs to "iterate U" until Gate 0 passes; only final values recorded | Arbitrary (possible unrecorded tuning) | High / tuning unknown |
| 3 ordered actions | `env.py:40` | c92f4e0 | Scope proposed 4 (incl. "standard"); cut without recorded reason | Arbitrary | High |
| Question cost 0.03, EPS 0.03, EPS_TIE 0.05 | `env.py:43-45` | c92f4e0 (copied) | Shared with other tasks | Arbitrary | High |
| Gate 0 pass threshold BC-gap > 0.02 | `design_gate.py:64` | c92f4e0 | Far below the project's ~0.10 bar; realised BC-gap +0.075 | Arbitrary (lowered bar) | High |
| Oracle | `env.py:137-164` | c92f4e0 | Exact recursion | Algorithmic | High |

**Comparison with the published scores.**
- **CHA2DS2-VASc:**
  - Matches: CHF 1, hypertension 1, age ≥75 = 2.
  - Mismatch: prior clot is weighted 3 here; stroke/TIA is 2 in the published score.
  - Missing: diabetes, vascular disease, age 65–74, and sex.
- **HAS-BLED:**
  - Matches: only hypertension 1.
  - Mismatch: bleeding history is 3 here; it is 1 in the published score.
  - Not in HAS-BLED at all: fall risk.
  - Differs: "elderly" means >65 in HAS-BLED; age_high here is ≥75.
  - Missing: renal/liver function, stroke, labile INR, and drugs/alcohol.

The scores are therefore named as inspiration, not implemented.

---

## 4. Datasets and guidelines: inspiration versus derivation

**In the prior repo, `~/clinical-rx-rl`:**
- `DATASETS_README.md` (4d47eb5, 2026-05-05) calls itself a "ranked candidate list of datasets we considered". It lists MedQA, MedMCQA, Medmarks, MIMIC-IV, MIMIC-ED-Assist, PMC-Patients and others.
- The datasets that *were* loaded (PMC-Patients, MD-authored seed cases, telemedicine refill sets, OOD sets) fed an earlier single-turn refill SFT project. No module in `scripts/phase2b/` reads them.
- The simulator modules import only `itertools`, `statistics`, `json` and `pathlib`. The JSON they read is the model's own rollout records.
- The Phase 2B design doc (17e1df7) planned distributions "grounded in real data (MIMIC-IV / MedCaseReasoning co-occurrence + real event rates)". This was never implemented.
- Guideline citations (JNC 8, KDIGO, ADA, ACC-AHA) appear only in the May single-turn seed cases, never in the POMDP simulators.

**In this repo:**
- MedQA, MIMIC-IV-ED, AgentClinic, SDBench and ED-Copilot appear only as related work in `main.tex` or as plans in `V2/PLAN.md`.
- HEART, Alvarado, DDXPlus, MedMCQA and ACC/AHA do not appear.

**Clinical review:** none has occurred. Dr. Saenz's only recorded input concerns Figure 2's labelling. Clinical review of utilities and trajectories is planned for the ongoing extension.

---

## 5. Answers

1. **Did we use any real clinical dataset to construct the simulator's probabilities or rules?**
   No. No code path reads clinical data into the priors, trees or utilities. Grounding in MIMIC-IV co-occurrence was planned in the prior project and never implemented.

2. **Were any parameters derived from documented clinical guidelines or published studies?**
   No.
   - Titration's factor weights are loosely *inspired by* CHA2DS2-VASc and HAS-BLED. They share some factors and a few weights, but diverge substantially and add fall risk.
   - Finding names and drug and finding pairings reflect general clinical knowledge. No specific source is recorded for them.

3. **Which elements were designed to create controlled experiments with meaningful information-gathering trade-offs?**
   Essentially all numeric parameters:
   - priors;
   - tree topologies and depth;
   - the 4-finding emergent conjunction in diagnostics;
   - the added "over-ordering" findings;
   - the capped hedge actions (rx `hold_for_labs`, dx `admit`);
   - all utility values;
   - the 0.03 question cost;
   - turn caps and the oracle-grade tolerance;
   - titration's weights and buckets.

   Several were explicitly revised until Gate 1 passed, or to remove an ask-cost exploit (rx: c1407f5, 6224e1b, e7cfeee; dx: 746da27). The titration Gate 0 bar was set at BC-gap > 0.02.

4. **Paper claims to clarify or correct.**
   The manuscript does not overclaim realism. It already says the environments are "synthetic, stylized, and labelled by deterministic rules", that utilities are "experimental design choices" "not validated by clinical experts", that no external dataset is used, and that diagnostics is "deliberately constructed". Three items deserve attention:
   - **(a) Gate framing (most important).** The paper presents Gates 1–2 as screens that determine *whether* RL is likely to help. In practice the environments were iteratively redesigned (findings added, trees widened, hedges capped, utilities recalibrated) until Gate 1 passed. This is legitimate for a controlled study but should be stated. Otherwise readers may assume the gates were applied to fixed, independently specified tasks.
   - **(b) Titration.** If the text or a figure mentions CHA2DS2-VASc or HAS-BLED (the current `main.tex` does not), it should say "loosely inspired by", not "based on". The weights and buckets do not implement either score.
   - **(c) Minor.** The refill description lists "adverse effects" among the askable findings. The closest finding is `acute_symptoms`; worth checking that the wording matches the finding list.

5. **Recommended Methods wording (defensible as written):**

   > All three environments are hand-designed. Their findings, finding prevalences, severity rules, utility matrices, and question cost were specified by the authors to create controlled information-gathering trade-offs, not estimated from clinical data, guidelines, or published event rates. Clinical concepts informed the choice and naming of findings and actions (for example, renal function and potassium for ACE-inhibitor refills, and stylized clot- and bleeding-risk factors loosely inspired by CHA2DS2-VASc and HAS-BLED for titration), but no parameter reproduces a published score or estimate. Environment parameters were revised during development so that each task offered adaptive headroom (Gate 1) and did not reward trivial shortcuts; the gates are therefore properties of the designed tasks rather than independent tests of naturally occurring ones. The parameters have not been reviewed by clinicians.

   A shorter version for the "Scope and clinical interpretation" paragraph: "Finding prevalences, severity rules, and utilities were hand-specified and tuned during development to create adaptive headroom; they are not derived from clinical data or guidelines."
