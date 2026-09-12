# Figure 1 (v2) — design notes

Companion to `scripts/make_figure1_v2.py` → `figure1_overview_v2.{svg,pdf,png}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

The figure's job (v2): **show, visually and clinically, that the same patient yields qualitatively
different multi-turn procedures under different reward designs, and that the procedure matters apart from
the final score.** Optimized for immediate interpretability by ML readers and physicians — not for completeness.

---

## 1. Exact data source for every trajectory

All trajectories are the diagnostics **deephedge** task, case **`dx-13`** (adult with acute abdominal pain,
hidden `true_risk = high`/emergent, `v_oracle = 0.83`, only acceptable action `emergent_tx`). Held-out draw
(`gen_cases(60, ρ=0.0, seed=100)`), evaluation with **pure `env.score`** (reward shaping OFF). Every reward below
was re-verified by recomputing `env_rv.score()` and by reading the stored probe records (`cond=="hidden"`,
`case_id=="dx-13"`).

| Row (panel) | Method | Source file (record) | Stored asks (verbatim order) | Decision | Reward | Notes |
|---|---|---|---|---|---|---|
| **B** | **Oracle** (exact adaptive) | reconstructed via `env_rv.oracle_rollout(fam, assign, 0.03)` — deterministic | `abd_rigidity → rebound_tender → ct_free_air → lactate_high` | `emergent_tx` | **0.83** | = `v_oracle`; the reference, not a learned policy |
| **C1** | **Outcome-only GRPO** | `results/probe_dh_grpo_final_t12.records.jsonl` (modal, ×6 of 12) | `abd_rigidity` | `admit` | **0.17** | 1-ask premature hedge |
| **C2** | **Credit-assignment GRPO** | `results/probe_dh_ig_final_t12.records.jsonl` (×1 of 12) | `abd_rigidity → guarding → rebound_tender → ct_free_air` | `emergent_tx` | **0.83** | `oracle_grade=True`; **not** the oracle evidence set (guarding, not lactate; different order) |
| **C3** | **Decoupled explore/exploit** | `results/probe_dream_s40_t12.records.jsonl` (×1 of 12) | `abd_rigidity → rebound_tender → guarding → radiation_arm → dyspnea_severe → hypotension → wbc_high → ecg_ischemia` | `emergent_tx` | **0.71** | `oracle_grade=False`, `acceptable_hit=True`; correct action, 8 asks vs oracle 4 |

Reward geometry for `dx-13` (`tasks/diagnostics/env_rv.py:67-73`, `_CAP=True`):
`U["high"] = {discharge −1.00, symptomatic_tx −0.50, observe −0.60, admit 0.20, emergent_tx 0.95}`, cost 0.03.
Checks: 0.95 − 4·0.03 = **0.83**; 0.20 − 1·0.03 = **0.17**; 0.95 − 8·0.03 = **0.71**. ✓

### Present / normal marker per finding (verified, `dx-13` `assign`)
From `env_rv.gen_cases(60,0.0,seed=100)` → `_REVEAL[finding][assign]`:
`abd_rigidity, rebound_tender, ct_free_air, lactate_high, guarding, radiation_arm, dyspnea_severe, fever_high` = **present (1)**;
`hypotension, wbc_high, ecg_ischemia` = **normal (0)**. The three "normal" chips in the decoupled row are
therefore **data-faithful** (the over-gatherer ordered non-informative tests that came back negative).

---

## 2. Exact values plotted in the Panel D anchor

Only two metrics (diagnostics, held-out, pure verifier, k=12). Values from CLAUDE.md / manuscript `tab:dx`
(3-seed mean±std for plain and credit; base and decoupled single-seed):

| Method | High-risk support | Mean questions asked |
|---|---|---|
| Base | 0.50 | 2.5 |
| Plain (outcome-only) GRPO | 0.33 ± 0.29 | 1.82 ± 0.11 |
| Credit-assignment GRPO | 0.50 ± 0.00 | 2.08 ± 0.30 |
| Decoupled GRPO | 1.00 | 4.4 |
| Oracle reference (asks) | — | ≈ 3.9 (dashed line) |

Error bars drawn only where a std exists (plain, credit); base/decoupled are single-seed (no bar). The
"oracle ≈ 3.9" line is the **task-average** oracle gather depth (manuscript `sec:results`), distinct from this
particular case's 4-ask oracle.

---

## 3. Labels that are translations of code keys

Physician-readable labels replace code keys everywhere in the visual. Mapping (`LAB`/`ACT` in the script):

`abd_rigidity`→Abdominal rigidity · `rebound_tender`→Rebound tenderness · `ct_free_air`→CT free air ·
`lactate_high`→Lactate elevated · `guarding`→Guarding · `radiation_arm`→Radiating pain ·
`dyspnea_severe`→Severe dyspnea · `hypotension`→Blood pressure · `wbc_high`→White cell count ·
`ecg_ischemia`→ECG · `fever_high`→Fever · `emergent_tx`→Emergency treatment · `admit`→Admit · `observe`→Observe.
Reward rule shown as **"R = clinical utility − 0.03 × questions asked"** (= `env.score` = `U[μ][a] − c·n_asks`).
"Underlying severity / μ" → "severity: hidden". "ΔH / info-gain" → "informative questions get intermediate
credit" (Panel D). Implementation terms (GRPO, LoRA, vLLM, group=8, env_rv) are **omitted** from the main visual
per the brief; "held-out, pure verifier (k=12)" appears once as a small anchor note.

---

## 4. Illustrative (non-data) elements vs. data

- **Patient glyph** (head + shoulders) — illustrative icon, not data.
- **Panel A hidden-finding cards** (6 face-down "?" cards: Abdominal rigidity, Rebound tenderness, CT free air,
  Lactate, Fever, WBC) — an **illustrative representative subset** of the 18-finding ask space, to convey
  "findings are hidden until asked / the agent chooses what to order." Not the full vocabulary; not case-specific.
- **Panel A loop** (Ask → Result → Commit → Reward with an "ask again" arc) — illustrative schematic of the
  turn mechanic, not a specific trajectory.
- **Panel C evidence-burden axis** (red–green–red track with "too few / appropriate / too many") — a **conceptual**
  axis; the three colored dots are placed at the *actual* ask counts (1, 4, 8) mapped onto the same column grid as
  the chips, and the gold tick marks the oracle depth (4 for this case). The green "appropriate" band (≈3–5) is a
  didactic region, not a measured threshold.
- Everything in **Panels B, C (trajectories/rewards/present-normal)** and the **Panel D bars** is real, verified data.

---

## 5. Detail omitted for readability (and why)

- **No per-turn model rationale / conversational prose** — the probe records store only parsed `asks`/`decision`;
  no rationale text exists, so none is shown (nothing invented).
- **Panel D uses only 2 metrics** (high-risk support, mean asks). Overall value, moderate support, BC-gap,
  SFT-coverage, titration dial, and Claude process-supervision are **deliberately excluded** from Figure 1
  (they belong to later figures).
- **Three tasks reduced to tiny footer labels**; diagnostics marked "running example." No environment-count panel.
- **Decoupled row shows all 8 findings** (no ellipsis) so trajectory length visibly encodes burden; a 9th "wrong
  final action" 12-ask trace exists (`probe_dream_final`, reward −0.16) but was **not** used — the 8-ask positive
  over-gather cleanly isolates "right answer, inefficient procedure" per the brief.
- **Credit-assignment is labeled "successful sampled trajectory · oracle-grade by reward," with the caveat "not
  identical to the oracle evidence sequence"** — it is *not* called "calibrated" or "oracle."

---

## 6. What changed from v1 and why

| v1 | v2 | Why |
|---|---|---|
| Schematic, box-heavy 2×2 (methods-slide feel) | Airy, journal-style vertical narrative; hairline dividers instead of panel borders | Brief: fewer boxes, more white space, read before caption |
| Code keys (`emergent_tx`, `env.score`, μ, ΔH) on the face | Physician-readable clinical labels; jargon dropped/tiny | Interpretable by physicians and ML readers |
| "Good" row = base or credit, labeled **calibrated** | **Oracle** is the explicit gold reference (Panel B); credit-assignment shown as a **sampled** oracle-grade trajectory, not "calibrated/oracle" | Scientific accuracy: reward-equivalent ≠ clinically calibrated / ≠ oracle evidence sequence |
| Ask counts shown as abstract squares | Real clinical findings as cards with **present/normal** results; **length encodes evidence burden**; explicit evidence-burden axis | Make under/appropriate/over-gathering visible pre-reading |
| Metrics table-ish, several numbers | One small 2-metric anchor (support, mean asks) + oracle line | Confirm the qualitative story without a results dump |
| Decoupled = 12-ask wrong-action (−0.16) | Decoupled = 8-ask correct-action (+0.71) | Isolates "right answer, inefficient" rather than conflating over-gather with a wrong final action |
| Env-count card panel for 3 tasks | Tiny footer one-liners, diagnostics flagged | Don't spend a panel on environment counts |

Color semantics kept from existing plots: base/reference **gray**, outcome-only **red**, credit-assignment
**green**, decoupled **blue**; the exact oracle uses a distinct **gold** treatment so it is never confused with a
learned policy.

---

## 7. Reproduce

```bash
source ~/spark1/.venv/bin/activate
python scripts/make_figure1_v2.py     # -> figure1_overview_v2.{svg,pdf,png}
```
SVG uses `svg.fonttype=none` (editable text) and the drawing is grouped panel-by-panel with named helper
components (`finding_chip`, `trajectory`, `action_pill`, `mini_bars`, …) for polishing in Figma/Illustrator.
