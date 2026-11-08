# Figure 1 (v3) — design notes

Companion to `scripts/make_figure1_v3.py` → `figure1_overview_v3.{svg,pdf,png}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

v3 is a **simplification of v2** (not a redesign): five visual sections collapsed to **three panels** —
A (the problem) → B (same patient, reference + learned procedures) → C (quantitative confirmation). Same
verified data as v2. The clean export has **no title** (starts at panel A); run with `FIG1_TITLE=1` for a titled
preview.

---

## 1. Exact trajectory source for each row (Panel B)

Diagnostics **deephedge**, case **`dx-13`** (adult with acute abdominal pain, hidden `true_risk=high`/emergent,
`v_oracle=0.83`, only acceptable action `emergent_tx`). Held-out draw `gen_cases(60, ρ=0.0, seed=100)`; evaluation
with **pure `env.score`** (shaping OFF). Reward geometry `U["high"] = {…, admit 0.20, emergent_tx 0.95}`, cost 0.03
(`tasks/diagnostics/env_rv.py:67-73,314-320`).

| Row | Method | Source | Asks (verbatim stored order) | Decision | Reward |
|---|---|---|---|---|---|
| 1 | **Exact adaptive oracle** | reconstructed `env_rv.oracle_rollout(fam, assign, 0.03)` (deterministic) | rigidity → rebound → CT free air → lactate | emergent_tx | **+0.83** (=v_oracle) |
| 2 | **Outcome-only GRPO** | `results/probe_dh_grpo_final_t12.records.jsonl` (modal rollout, ×6 of 12) | rigidity | admit | **+0.17** |
| 3 | **Credit-assignment GRPO** | `results/probe_dh_ig_final_t12.records.jsonl` (×1 of 12) | rigidity → guarding → rebound → CT free air | emergent_tx | **+0.83** (oracle_grade=True) |
| 4 | **Decoupled explore/exploit** | `results/probe_dream_s40_t12.records.jsonl` (×1 of 12) | rigidity → rebound → guarding → radiating pain → severe dyspnea → blood pressure → WBC → ECG | emergent_tx | **+0.71** (oracle_grade=False, acceptable_hit=True) |

Reward checks: 0.95−4·0.03=**0.83**; 0.20−1·0.03=**0.17**; 0.95−8·0.03=**0.71**. ✓

---

## 2. Exact values in both Panel C plots

Diagnostics, held-out hidden, pure verifier, k=12 (CLAUDE.md / manuscript `tab:dx`). **v3 uses high-risk VALUE**
(not support@12) per instruction, because support@12 is a sampled-capability metric that should not visually imply
deployment safety.

**Left — High-risk value** (ylim 0–0.31): Base **+0.160**; Plain (outcome-only) GRPO **+0.098 ± 0.037**;
Credit-assignment GRPO **+0.157 ± 0.057**; Decoupled **+0.248**.

**Right — Mean questions asked** (ylim 0–5.2): Base **2.50**; Plain **1.82 ± 0.11**; Credit **2.08 ± 0.30**;
Decoupled **4.40**; dashed reference **oracle ≈ 3.9** (task-average oracle gather depth).

Error bars drawn only where a std exists (Plain, Credit = 3-seed mean±SD); Base and Decoupled are single-seed
(no bar), stated in the note. Bar colors: Base gray, outcome-only red, credit-assignment green, decoupled blue.
support@12 is **not** shown in v3; if ever reinstated it must be labeled "High-risk oracle-grade support@12"
and never "↑ safer".

---

## 3. Is every displayed question order directly verified from records?

**Yes for the three learned rows (2–4):** the ask sequences are the verbatim stored `asks` lists from the probe
records above (re-read this session; each reward independently recomputed via `env_rv.score()` and matched to the
stored `reward`). The **present/normal** marker on every chip is the verified `assign` value for `dx-13`
(`_REVEAL[finding][assign]`): rigidity, rebound, CT free air, lactate, guarding, radiating pain, severe dyspnea,
fever = **present (1)**; blood pressure (hypotension), WBC, ECG = **normal (0)** — so the three "normal" chips in
the decoupled row are data-faithful (the over-gatherer ordered non-informative tests).

**Row 1 (oracle)** is not a sampled record but the deterministic exact-oracle rollout from `env_rv.oracle_rollout`
— labeled in the figure as a reference procedure, not a learned policy.

**Panel A** is illustrative (see §5): the six finding cards and the single "reveal" card are a representative
subset of the 18-finding ask space, not a specific trajectory.

---

## 4. Text translated from internal finding/action keys

Physician-readable labels replace all code keys (`LAB`/`ACT` in the script):
`abd_rigidity`→Abdominal rigidity · `rebound_tender`→Rebound tenderness · `ct_free_air`→CT free air ·
`lactate_high`→Lactate elevated · `guarding`→Guarding · `radiation_arm`→Radiating pain ·
`dyspnea_severe`→Severe dyspnea · `hypotension`→Blood pressure · `wbc_high`→White cell count ·
`ecg_ischemia`→ECG · `fever_high`→Fever · `emergent_tx`→Emergency treatment · `admit`→Admit.
"true_risk / μ" → "severity: hidden from agent". No JSON, `env.score`, GRPO/LoRA/vLLM, turn caps, or parse-error
text appears in the figure.

---

## 5. Information intentionally omitted from v2

Removed to reduce density and match the 3-panel brief:
- **Panel A right-hand Ask→Result→Commit loop diagram** — deleted; the single mid-reveal card now teaches "asking".
- **Reward equation** `R = clinical utility − 0.03 × questions asked` — removed from the visual (cost stated once in
  the Panel A sentence).
- **Panel D reward-geometry explanatory cards** (what outcome-only / credit / decoupled *mean*) — removed;
  Panel B's trajectories make the procedures self-explanatory.
- **Environment/infra content** — MedGemma / SFT / vLLM / group-size / LoRA; the three-task footer strip; BC-gap
  values; process supervision; SFT-coverage & repair experiments. (All belong to Methods / later figures / tables.)
- **Panel C metric switch** — v2 showed high-risk **support@12** ("↑ safer"); v3 shows high-risk **value** instead
  (support removed) to avoid implying deployment safety.
- **Paper title** — omitted from the clean publication export (present only in the `FIG1_TITLE=1` preview).
- The evidence-burden scale is **kept but slimmed** (thin bar, small dots) so it is not visually dominant.

Retained from v2: same verified `dx-13` trajectories and rewards; oracle as a distinct **gold** reference;
color semantics (base gray, outcome-only red, credit green, decoupled blue); "successful sampled trajectory ·
oracle-grade by reward · evidence sequence differs from oracle" wording for credit-assignment (never "calibrated"
/ "oracle" / "clinically optimal"); trajectory length encoding evidence burden (1 vs 4 vs 4 vs 8).

---

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_figure1_v3.py                 # clean, no title -> figure1_overview_v3.{svg,pdf,png}
FIG1_TITLE=1 python scripts/make_figure1_v3.py    # titled preview variant
```
SVG uses editable text (`svg.fonttype=none`); drawing grouped by named components (`finding_chip`, `trajectory`,
`reveal_card`, `mini_bars`, …) for Figma/Illustrator polishing.
