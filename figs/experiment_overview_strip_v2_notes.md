# Experiment overview (v2, Nature-style pass) — notes

Companion to `scripts/make_experiment_overview_strip_v2.py` →
`figs/experiment_overview_strip_v2.{png,svg,pdf}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

Editorial reduction of the earlier v2 (fewer outlined boxes, more whitespace, stronger hierarchy). Same content;
restyled toward a journal opening figure rather than a lab schematic. Standalone; not Figure 1.

## Hierarchy chosen (three zones, unequal weight)

- **Zone A — Clinical multi-turn environment** (left, **pale tinted panel**, no border): the focal process. Model ⇄
  simulator with a curved ask/finding loop, a small light hidden-state pill above the simulator, a `commit → clinical
  action` box below the model, and the oracle/verifier as annotation badges along the bottom.
- **Zone B — Training** (center, **no panel**, just a labelled section): a single strongly-outlined **GRPO training**
  anchor; beneath it a clean **row of three reward pills** under a "training reward differs" label; a small
  subordinate two-node inset below.
- **Zone C — Held-out evaluation** (right, **pale panel**): a vertical stack — held-out cases → **Same pure verifier**
  (the anchor) → "reward shaping off" pill → metrics as light text.

One horizontal main flow ties them together: `environment → GRPO training → common evaluation`. The three reward
variants hang **below** GRPO (implying "reward differs during training"), while the single arrow into evaluation
("trained policies · common evaluation") carries the convergence — all methods are judged by one verifier.

## Why the environment is a pale container, not outlined boxes

Only three objects are strongly outlined — **MedGemma-4B-it**, **GRPO training**, **Same pure verifier** — so the eye
knows what matters. Everything else is a light fill with a thin or no border, and the environment/evaluation are pale
background regions rather than hard rectangles. This removes the "row of equal boxes" feel and gives an editorial
hierarchy: containers for context, outlines for the few key actors.

## How oracle and verifier are distinguished

They are **small annotation badges** (◆) along the bottom of the environment, tied to the simulator by thin **dashed**
connectors — read as analytic properties of the controlled simulator, not stages the model passes through. Distinct
roles/colors: **oracle = muted gold** ("reference procedure"); **verifier = violet-gray** ("outcome scorer"). The
verifier's violet reappears on the Zone-C "Same pure verifier" anchor so the reader sees the same scorer at eval.

## How the reward methods branch (and converge)

Under GRPO, a light fork (one line to a short bus, then three ticks) drops to a row of three **pills** — Outcome-only
(red), Credit assignment (green), Decoupled (blue) — with the "training reward differs" label above. No curved branch
arrows; alignment and the bracket imply the branching. Convergence is stated, not drawn as arcs: a single
"trained policies · common evaluation" arrow runs from GRPO into the evaluation panel, where one verifier scores every
method. This keeps the train-varies / eval-constant contrast without arc clutter.

## Two-node infrastructure inset

**Included**, deliberately subordinate (light gray, thin text, small — a technical footnote), below the pills and
dash-linked to the training column. It shows the **disaggregated two-node RL** architecture: **Trainer node**
(GRPO · LoRA) ⇄ **Rollout node** (vLLM · 8 rollouts), captioned "two DGX Spark nodes · updates ↔ rollouts." Restrained
styling keeps it from competing with the main story; delete the "disaggregated two-node inset" block in the script to
remove it entirely.

## Text reduction

~25% less text than the prior v2: "one terminal scorer, every method" → "common scorer"; "reward varies by method" is
now only the "training reward differs" label; "hidden diagnosis / severity / medication" → "hidden state · diagnosis ·
severity · dose"; "finding revealed" → "finding"; the "analytic properties of the controlled simulator" caption
removed (the dashed badges already say it); metrics are light two-column text, not a boxed panel.

## Typography & color

One sans family; **bold** only for the zone/section titles and the three key actors; medium for object titles; light
gray for subtitles. Restrained palette — teal (MedGemma), neutral gray (environment/simulator), gold (oracle),
violet-gray (verifier), red/green/blue for the three reward variants only, pale background fills.

## If it must fit above Figure 1 — remove first

1. The two-node inset. 2. The `commit → clinical action` box (the ask/finding loop conveys interaction). 3. Compress
metrics to two items. 4. Shrink the environment panel height. 5. De-duplicate with `train_eval_overview` — use only
one overview above Figure 1. Editable-text SVG (`svg.fonttype=none`) is provided for Figma.

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_experiment_overview_strip_v2.py   # -> figs/experiment_overview_strip_v2.{png,svg,pdf}
```
