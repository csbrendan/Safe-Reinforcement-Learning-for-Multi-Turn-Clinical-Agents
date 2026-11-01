# Figure 1 (v4) — design notes

Companion to `scripts/make_figure1_v4.py` → `figure1_overview_v4.{svg,pdf,png}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

v4 is a **refinement of v3** (not a redesign): same 3-panel story (A incomplete information → B same-patient
procedures → C quantitative confirmation) and the same verified data. Clean export has **no title**
(`FIG1_TITLE=1` gives a titled preview).

---

## QA checklist (all confirmed)

1. **Trajectory orders re-verified** against the same records used in v3 (this session):
   outcome-only `probe_dh_grpo_final` (rigidity→admit, +0.17); credit `probe_dh_ig_final`
   (rigidity→guarding→rebound→CT free air→emergent_tx, +0.83, oracle_grade); decoupled `probe_dream_s40`
   (rigidity→rebound→guarding→radiating→dyspnea→BP→WBC→ECG→emergent_tx, +0.71, acceptable_hit); oracle
   `env_rv.oracle_rollout` (rigidity→rebound→CT free air→lactate→emergent_tx, +0.83). Rewards recomputed via
   `env_rv.score()` and matched to stored `reward`.
2. **All plotted numbers verified** (Panel C table below).
3. **Oracle and credit both = 4 asks** on the depth scale: gold vertical line (oracle) at position 4; green credit
   marker at position 4 (offset slightly right so both are visible); numbers "4"/"4" printed. Red = 1, blue = 8.
4. **Base and Decoupled are visibly single-seed:** x-labels `Base†` / `Decoupled†`, bars hatched, and the
   footnote defines `†`. Plain/Credit carry SD error bars; Base/Decoupled have none.
5. **Panel A is substantially smaller than v3** (see height reduction below).
6. **No label smaller than needed:** finding-card text 6.4 pt, chip status 5.1 pt, annotations 7.2–8.4 pt, method
   names 10.5 pt, panel titles 13.5 pt, chart value labels 7.2 pt (bold) — all legible at print size.

---

## 1. What changed from v3

- **Panel A → compact horizontal strip.** Removed the 6 large finding cards, the big patient area, and the large
  severity box. Now: tiny patient icon + "Adult with acute abdominal pain" / "severity hidden", → 3 small hidden
  chips (`? Rigidity`, `? CT free air`, `? Lactate`) → one revealed chip (`Abdominal rigidity — revealed: PRESENT`),
  teaching UNKNOWN → QUERY → REVEALED in one line, with a single one-line annotation.
- **Panel B (hero) kept**, with three refinements (below): oracle-vs-learned labels sharpened; decoupled
  non-decisive findings dimmed + bracketed; depth scale re-anchored on the oracle.
- **Panel C polished:** replication status made visible (see §5); footnote shortened; left metric renamed to plain
  "High-risk value" (no "↑ safer", support@12 not shown).
- **Visual polish:** finding cards are smaller & **squarer** (rounding 0.035) while **action pills stay rounded
  (0.16) and tinted** — so evidence cards and decisions no longer share one uniform box style. Stronger type
  hierarchy; annotation grays darkened for readability (`#586170` / `#8b93a0` instead of ultra-faint). Rows share a
  fixed start x and chip grid, so trajectory length remains directly comparable.

## 2. Exact Panel A height reduction

Panel A spans from its heading to the divider below it.
- **v3:** 12.35 → 9.80 = **2.55 in** (≈ 20.2% of the 12.6-in canvas).
- **v4:** 12.35 → 10.73 = **1.62 in** (≈ 12.9% of the 12.6-in canvas).
- **Reduction: 0.93 in, ≈ 36% shorter**, now well under one Panel-B trajectory-row pitch (1.32 in row + notes).

## 3. How the oracle vs learned distinction was strengthened

- Oracle row label is now explicit: **"Reference: exact adaptive oracle"** with the secondary line **"not a learned
  policy"**, in the distinct **gold** treatment (gold accent bar, gold action pill, gold reward) used for no learned
  method.
- Credit row is labeled **"Credit-assignment GRPO"** with sub **"sampled learned trajectory · 4 findings"** and the
  concise caveat **"successful learned rollout; reward within oracle tolerance, evidence order differs from oracle."**
  It is never called oracle / optimal / calibrated.
- On the depth scale the **gold line is the anchor** for "appropriate depth"; the green credit marker merely sits
  near it (both at 4 asks) — so credit does not visually define the correct depth.

## 4. How non-decisive decoupled findings were marked

The decoupled row's first three findings (Abdominal rigidity, Rebound tenderness, Guarding) are drawn normally; the
remaining five (**Radiating pain, Severe dyspnea, Blood pressure, White cell count, ECG**) are drawn **dimmed**
(≈50% opacity, lighter border, muted text, present/normal dot suppressed) and enclosed by a **bracket** labeled
**"additional non-decisive findings."** This shows over-gathering (and that the extras are cardiac/systemic tests
irrelevant to the abdominal decision) without implying they are part of the oracle abdominal work-up. Kept: 8 vs
oracle 4, correct action, reward +0.71. (Data-faithful: BP, WBC, ECG actually returned *normal* for this patient.)

## 5. How replication status is communicated in Panel C

Three redundant cues so no reader assumes equal statistical support:
1. **x-axis labels** `Base†`, `Plain`, `Credit`, `Decoupled†`.
2. **Hatched bars** for the two single-seed references (Base, Decoupled); solid fill for the 3-seed arms.
3. **Error bars** (mean ± SD) only on Plain and Credit; none on Base/Decoupled.
Footnote: *"Held-out diagnostics; pure verifier at evaluation. High-risk value = rare emergent subset. † single-seed
reference; Plain / Credit = mean ± SD, n = 3 seeds."* High-risk value is described as the rare emergent subset, and
is **not** labeled "safety."

---

## Data used (unchanged from v3, re-verified)

**Panel B trajectories (diagnostics deephedge, `dx-13`, held-out, pure verifier).** Physician labels are
translations of code keys (`abd_rigidity`→Abdominal rigidity, `rebound_tender`→Rebound tenderness,
`ct_free_air`→CT free air, `lactate_high`→Lactate elevated, `guarding`→Guarding, `radiation_arm`→Radiating pain,
`dyspnea_severe`→Severe dyspnea, `hypotension`→Blood pressure, `wbc_high`→White cell count, `ecg_ischemia`→ECG,
`emergent_tx`→Emergency treatment, `admit`→Admit). Sources: oracle = `env_rv.oracle_rollout` (deterministic);
outcome-only = `results/probe_dh_grpo_final_t12.records.jsonl`; credit = `results/probe_dh_ig_final_t12.records.jsonl`;
decoupled = `results/probe_dream_s40_t12.records.jsonl` (all `case_id=dx-13`, `cond=hidden`).

**Panel C values** (CLAUDE.md / manuscript `tab:dx`):

| Method | High-risk value | Mean questions asked | Seeds |
|---|---|---|---|
| Base† | +0.160 | 2.50 | single |
| Plain (outcome-only) | +0.098 ± 0.037 | 1.82 ± 0.11 | n=3 |
| Credit-assignment | +0.157 ± 0.057 | 2.08 ± 0.30 | n=3 |
| Decoupled† | +0.248 | 4.40 | single |
| Oracle (asks reference) | — | ≈ 3.9 (dashed line) | — |

support@12 is intentionally **not** shown in Figure 1.

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_figure1_v4.py                 # clean, no title
FIG1_TITLE=1 python scripts/make_figure1_v4.py    # titled preview
```
Editable-text SVG (`svg.fonttype=none`); grouped by named components for Figma polishing.
