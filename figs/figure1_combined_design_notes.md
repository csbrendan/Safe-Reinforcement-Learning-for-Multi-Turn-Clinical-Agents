# Figure 1 (combined) — design notes

Companion to `scripts/make_figure1_combined.py` → `figs/figure1_combined.{png,svg,pdf}`
and `figs/figure1_combined_caption.tex`.

Paper: *Safe Reinforcement Learning for Clinical Multi-Turn Agents Requires More Than Outcome Rewards.*

Combines two previously developed graphics into one figure:

| source | contributed | script |
|---|---|---|
| `figs/experiment_overview_strip_v2.png` | Panel A (three-zone experimental setup) | `scripts/make_experiment_overview_strip_v2.py` |
| `figure1_overview_v4.png` | Panels B and C (trajectories, bar charts) | `scripts/make_figure1_v4.py` |

**Neither PNG is embedded.** Every element was re-drawn from vector primitives in a single script, on one
canvas, with one palette and one type scale. The SVG keeps live text (`svg.fonttype="none"`) for Figma/Illustrator
polishing; the PDF uses Type-42 fonts.

---

## 1. Harmonisation (what changed from the two originals)

The two sources had drifted apart. `experiment_overview_strip_v2` is the design reference, so its values won:

| token | strip_v2 | v4 | combined |
|---|---|---|---|
| INK (primary type) | `#22303c` | `#20303f` | **`#22303c`** |
| GOLD (oracle) | `#a9821f` | `#b0851f` | **`#a9821f`** |
| TEAL (policy / revealed) | `#1f7f8c` | `#0f8a7e` | **`#1f7f8c`** |
| secondary / tertiary type | `#7c848d` / `#a3aab2` | `#586170` / `#8b93a0` | **`#7c848d` / `#a3aab2`** |
| RED / GREEN / BLUE | identical | identical | unchanged |

Also unified: one type scale (`FS_*` constants, 5.1–15.5 pt) shared by all three panels; one set of
primitives (`rrect` / `arrow` / `dline` / `obj` / `badge`); one arrow vocabulary (1.6 lw for zone
transitions, 1.4 for local flow, 1.1 for trajectory steps, dashed `dline` for annotation links); panel
headings drawn by one `head()` function so A/B/C letters and titles are identical in size and placement.

Two deliberate design carries from A into B, so the figure reads as one system rather than two graphics:
the **pale zone-panel language** of Panel A is reused as the pale gold band behind the oracle reference row,
and Panel A's **pill** shape (used for the reward variants) is reused for Panel B's terminal action pills.

Method colour is consistent across all three panels (outcome-only red, credit assignment green, decoupled
blue, oracle gold, base grey). **Colour is never the only cue:** every method is named in text in Panel B
(row label) and Panel C (x-tick label), and the single-seed bars are hatched as well as daggered.

## 2. Proportions

Canvas 14.00 × 16.42 in. Panel A 4.64 in (28%), Panel B 7.47 in (46%), Panel C 3.45 in (21%) — within the
requested 25–30 / 45–50 / 20–25 envelope, with Panel B the largest single block.

## 3. Panel B trajectories — sources and verification

Case **`dx-13`** of the held-out diagnostics draw: `DX_VARIANT=deephedge`,
`tasks.diagnostics.env_rv.gen_cases(n=60, rho=0.0, seed=100)`, `cond=hidden`.
`family=abd_pain`, `true_risk=high`, `v_oracle=0.83`, `acceptable=['emergent_tx']`.

| row | trajectory (verified order) | action | reward | source | frequency in that condition |
|---|---|---|---|---|---|
| Exact adaptive oracle | rigidity → rebound → CT free air → lactate | `emergent_tx` | +0.83 | `env_rv` `oracle_asks` (deterministic) | n/a — not a learned policy |
| Outcome-only GRPO | rigidity | `admit` | +0.17 | `results/probe_dh_grpo_final_t12.records.jsonl` | **6 / 12** rollouts (modal; 9/12 ask only once) |
| Credit-assignment GRPO | rigidity → guarding → rebound → CT free air | `emergent_tx` | +0.83 | `results/probe_dh_ig_final_t12.records.jsonl` | **1 / 12** rollouts (`oracle_grade=true`) |
| Decoupled explore/exploit | rigidity → rebound → guarding → radiating pain → severe dyspnea → blood pressure → white cell count → ECG | `emergent_tx` | +0.71 | `results/probe_dream_s40_t12.records.jsonl` (checkpoint 40) | **1 / 12** rollouts (`acceptable_hit=true`, `oracle_grade=false`) |

Verification performed for this figure (not inherited from the v4 notes):

1. Each ordered ask list and reward was read back out of the stored `.records.jsonl` and matched exactly.
2. Each reward was **independently recomputed** with `env_rv.score(case, asks, decision)` from a freshly
   regenerated `dx-13`: +0.83 / +0.17 / +0.83 / +0.71 — all four match the stored values.
3. The present/normal state of every displayed finding was read from the regenerated
   `case['assign']`, not assumed.
4. Reproducibility across `PYTHONHASHSEED` ∈ {0, 1, 12345} confirmed (CLAUDE.md flags a historical
   set-ordering wobble on oracle / `rel_items` paths): case, oracle ask order, and all four rewards identical.

### Rollout-frequency caveat (important, and a change from v4)

The credit-assignment and decoupled rows are each **1 of 12** sampled rollouts, and outcome-only GRPO also
produced one +0.83 oracle-grade rollout on this case. The rows therefore illustrate *procedure classes that
each reward design produces*, **not** each method's typical behaviour on `dx-13`. The figure states this
explicitly ("single representative rollouts (k = 12 per condition); illustrative of procedure class, not
modal or aggregate behaviour") and the caption repeats it. The aggregate claim lives in Panel C and in the
support@12 numbers reported elsewhere in the paper — **not** in Panel B. Per-row counts are deliberately kept
off the figure: printing "6/12" next to "1/12" would invite a comparison that is not the paper's claim and
that the single-case view cannot support.

### Decoupled row: which findings are marked non-decisive

Findings 4–8 (radiating pain, severe dyspnea, blood pressure, white cell count, ECG) are drawn at ~55%
opacity inside a bracket labelled "additional non-decisive findings". They are not decisive for the
*abdominal* disposition, which is the point of the row.

**Correction to the v4 design notes,** which stated these extras "actually returned normal for this patient."
Only three did: `hypotension=0` (blood pressure stable), `wbc_high=0` (WBC normal), `ecg_ischemia=0` (ECG
without ischemia). **`radiation_arm=1` and `dyspnea_severe=1` were in fact present** (concerning). Because the
dimmed chips suppress the present/normal dot, the figure makes no claim either way — but the earlier note was
wrong and is not carried forward.

### Clinical label translations (code key → figure text)

`abd_rigidity`→Abdominal rigidity · `rebound_tender`→Rebound tenderness · `ct_free_air`→CT free air ·
`lactate_high`→Lactate elevated · `guarding`→Guarding · `radiation_arm`→Radiating pain ·
`dyspnea_severe`→Severe dyspnea · `hypotension`→Blood pressure · `wbc_high`→White cell count ·
`ecg_ischemia`→ECG · `emergent_tx`→Emergency treatment · `admit`→Admit.

Shortened for the chip grid: "Blood pressure" and "ECG" stand for the *tests ordered*, whose results here were
`blood pressure stable` and `ECG without ischemia`; "White cell count" likewise (result: `WBC normal`). No
internal variable name appears in the figure.

### Gathering-depth scale

Anchored on the oracle: the gold vertical line sits at 4 questions and carries the label "oracle depth"; the
pale green band spans 3–5. Credit assignment's marker is at 4 and is offset +0.17 units horizontally only so
both markers remain visible — it does **not** define the correct depth. Red = 1, blue = 8.

## 4. Panel C values — all recomputed from the probe records

Recomputed here directly from `*.records.jsonl` (`cond=hidden`), not copied from CLAUDE.md:

| method | high-risk value | mean questions | seeds | record files |
|---|---|---|---|---|
| Base† | +0.160 | 2.54 | 1 | `probe_dh_base_t12` |
| Outcome-only GRPO | +0.098 ± 0.037 | 1.82 ± 0.11 | 3 | `probe_dh_grpo_final_t12`, `probe_dh_ctl_s2_t12`, `probe_dh_ctl_s3_t12` |
| Credit-assignment GRPO | +0.157 ± 0.057 | 2.08 ± 0.30 | 3 | `probe_dh_ig_final_t12`, `probe_dh_ig_s2_t12`, `probe_dh_ig_s3_t12` |
| Decoupled GRPO† | +0.248 | 4.42 | 1 | `probe_dream_s40_t12` |
| Exact adaptive oracle | — | 3.78 (dashed line) | — | `env_rv.gen_cases(60, 0.0, seed=100)` |

High-risk value = mean verifier reward over the `true_risk=high` subset (2 cases × 12 rollouts = 24 records
per condition). SD is across seed means, n=3.

### Three numeric discrepancies against the brief (flagged, not silently absorbed)

1. **Base mean questions: 2.54, not 2.50.** The brief and the v4 figure show 2.50; the records give 2.54.
   The manuscript's one-decimal "2.5" is consistent with both. **2.54 is plotted.**
2. **Decoupled mean questions: 4.42, not 4.40.** Same situation; manuscript "4.4" is consistent.
   **4.42 is plotted.**
3. **Oracle reference line: 3.78, not 3.9.** The "≈3.9" quoted in `docs/eval_runs/` and in v4 is the oracle's
   mean ask count on the *Gate-1 / RL case distribution* (seed 12345: 3.85 at n=300, 3.848 at n=2000). On the
   **held-out seed-100 n=60 set that Panel C's bars are computed from**, it is 3.783. Since every bar in
   Panel C comes from that set, the internally consistent reference line is **3.78, plotted and labelled
   "oracle ≈ 3.8."** *This is worth checking in the manuscript text too, which currently says 3.9.*

Deliberately excluded from Panel C, per the brief: support@12, overall value, moderate-subset support,
BC-gap, titration, SFT-coverage and process-supervision results.

## 5. Scientific limitations introduced by the visual simplifications

- **One case, one rollout per row.** Panel B is a single held-out case (`dx-13`) and one rollout per
  condition. It demonstrates that these procedures are *producible* under each reward design; it is not
  evidence about frequency. See the rollout-frequency caveat above.
- **Reward values are case-specific.** +0.83 / +0.17 / +0.71 are this patient's rewards, not method means.
- **Panel A is conceptual.** Turn caps, the per-ask cost (0.03), the parser, group size (8 rollouts/prompt
  appears only inside the two-node inset), KL/β, prompt formats and the mode-token mechanics of the
  decoupled condition are all omitted; they belong in Methods. Panel A also does not show the per-ask cost
  at all — Panel B's reward values carry it implicitly.
- **The oracle is an analytic reference, not a clinical gold standard.** It is exact for *this rule-labelled
  simulator*. The figure never labels a learned rollout "oracle", "optimal" or "calibrated".
- **High-risk value is behavioural.** Stated on-figure and in the caption as *not* a direct measure of
  clinical safety.
- **Single-seed references.** Base and decoupled bars are one seed each: hatched, daggered, and no error bar.
- **Environments are synthetic and stylised.** Nothing in this figure is clinical validation or deployment
  evidence; the scope caveat still belongs in the manuscript text.

## 6. QA performed

| check | result |
|---|---|
| PDF renders and opens | pass (1022.4 × 1196.64 pt = 14.2 × 16.62 in with tight bbox) |
| text overlap / clipping | pass after six fixes (below) |
| trajectory labels readable | pass at the canvas's native size; **see §7 for print scale** |
| every plotted number matches verified source | pass — with the three discrepancies in §4 resolved toward the records |
| exact oracle vs sampled learned trajectories distinguished | pass — gold-only treatment, pale gold band, "reference procedure · not a learned policy", and the caveat line on the credit row |
| Panel A separates simulator / oracle / verifier / training machinery | pass — oracle and verifier are dashed-linked annotation badges on the simulator, not inline pipeline stages |
| evaluation reward shown as common and unshaped | pass — single "Same pure verifier" block (strongest outline on that side), "one terminal score, every method", red "reward shaping off" pill, and the Panel C subtitle |
| three panels legible when combined | pass at native size; §7 qualifies this for page-width printing |

### Defects found and fixed during QA

1. **Action pill overlapped the last finding chip** on every row — `apx = last_r + 0.63` with a
   1.32-wide pill put the pill's left edge *inside* the final chip. **This bug is inherited from
   `make_figure1_v4.py` and is visible in `figure1_overview_v4.png`.** Fixed: `apx = last_r + 0.30 + w/2`.
2. **Two-node inset** had three text tiers in 0.78 in: the "Disaggregated RL" title landed on top of the
   node boxes and the footer line fell outside the inset panel. Re-laid out with a 0.98-in budget.
3. **Panel C subtitle** collided with the left chart's "High-risk value" title. Chart block moved down 0.36 in.
4. **Oracle row label** ("Reference: exact adaptive oracle") was the longest string in the figure and ran into
   the first finding chip. Restructured to "Exact adaptive oracle" / "reference procedure · not a learned
   policy" / "4 targeted findings → emergency treatment".
5. **Oracle and verifier dashed links** originally pointed at the *clinical action* box, which read as a
   sequential scoring stage. Re-pointed at the patient simulator, per the brief.
6. **8-finding row overran the canvas.** Chip grid tightened (width 0.86→0.84, gap 0.20→0.17, track start
   3.42→3.05) so the longest row plus pill plus reward tag clears the right margin.

## 7. Print-scale finding (unresolved — needs a decision)

The canvas is 14.2 in wide, inherited from the two sources (v4 was 14 in, the strip 16 in). Included at a
7.2-in page width, every point size is multiplied by **0.507**:

| element | authored | at 7.2-in page width |
|---|---|---|
| panel letter | 15.5 pt | 7.9 pt |
| panel title | 13.0 pt | 6.6 pt |
| method name / object label | 10.2 pt | 5.2 pt |
| annotation | 7.8 pt | 4.0 pt |
| secondary annotation | 7.0 pt | 3.5 pt |
| clinical finding label | 6.4 pt | **3.2 pt** |
| per-chip present/normal tag | 5.1 pt | **2.6 pt** |

Most journals set a 5-pt floor, so the finding labels, annotations and inset text are **too small to print at
full-page width**, even though they are comfortable at the native canvas size. This is a property of both
source figures, not something introduced by combining them.

Scaling the existing geometry down does not help (fonts scale with it). Fixing it requires recomposing the
geometry at print scale — the box sizes are very generous relative to the type, so there is real slack — plus
trimming the finest text. A feasible print-scale budget for Panel B at 7.2 in:

```
label column 1.35 + chips 8 x 0.42 + gaps 7 x 0.08 + pill 0.62 + reward 0.45  ~=  6.6 in
```

which fits, with finding labels at 5 pt, if the **per-chip "present"/"normal" words are dropped** (the dot plus
the existing legend already carries them) and the three per-row caveat lines are shortened.

Options:

- **(a) Recompose at print scale** (~7.2 in wide): correct for a full-page portrait figure; costs the per-chip
  status words and shorter caveat lines.
- **(b) Rotated / full-page landscape placement** (~9.5 in wide): scale 0.669, finding labels 4.3 pt — better
  but still under the 5-pt floor.
- **(c) Keep as is** and place it as a wide foldout or a supplementary full-bleed figure.

Recommended: **(a)**.

## Reproduce

```bash
source ~/spark1/.venv/bin/activate
python scripts/make_figure1_combined.py      # writes figs/figure1_combined.{png,svg,pdf}
```

Verification commands used for §3–§4 are plain reads of `results/*.records.jsonl` plus
`DX_VARIANT=deephedge python -c "import tasks.diagnostics.env_rv as e; e.gen_cases(60, 0.0, seed=100)"`.
