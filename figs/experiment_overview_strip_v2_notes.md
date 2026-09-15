# Experiment overview (v2) — notes

Companion to `scripts/make_experiment_overview_strip_v2.py` →
`figs/experiment_overview_strip_v2.{png,svg,pdf}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

A substantial redesign of the v1 strip. Same content, rethought composition: instead of six equal-weight boxes in
one pipeline, it uses hierarchy and a **one → three → one** flow (one environment/model family → three training
reward variants → one common evaluation). Standalone; not Figure 1.

## Hierarchy chosen

Three zones with deliberately unequal visual weight:
1. **Zone 1 — Clinical multi-turn environment (largest, focal).** A big soft-shaded container holding the main
   process: **MedGemma-4B-it ⇄ Patient simulator** with a two-arrow ask/reveal loop, a red "hidden state" banner,
   and a `commit → clinical action` branch. This is the visual anchor because the interactive, partially-observed
   process is the conceptual heart of the paper.
2. **Zone 2 — GRPO training (medium).** A single neutral anchor ("GRPO training · reward varies by method") that
   **fans out** into three small reward **pills** (capsules), not full cards.
3. **Zone 3 — Common held-out evaluation (medium, emphatic endpoint).** The three pills **converge** back to one
   path: held-out cases → the prominent violet **Same pure verifier** box → "reward shaping off" → metrics.

Type hierarchy: zone titles (12.5 pt bold) > box titles (10–11 pt) > pill titles (9.4 pt) > secondary/italic
labels (6.8–7.4 pt). Two bracket call-outs carry the thesis contrast: **"training reward differs"** over the pills,
**"common evaluation — held constant"** over the eval column.

## Why the environment is a container, not a pipeline box

In v1 the environment was one box equal in size to "GRPO" and "evaluation", which implied the model *passes
through* it as a stage. Scientifically the environment is not a stage — it is the setting in which the whole
policy↔patient interaction happens, and the oracle/verifier are analytic properties *of* that setting. Making it a
large enclosing region with the interaction inside communicates "interactive RL environment," not "static dataset
step," and lets the ask/reveal/commit loop be the focal process rather than a caption.

## How oracle and verifier are distinguished

They are **small secondary badges** along the bottom of the environment container (never stages the model passes
through), each marked with a ◆ and tied to the simulator by a thin **dashed** connector labeled "analytic
properties of the controlled simulator." Roles and color are kept distinct: **oracle = muted gold**, "reference
procedure" (a reference adaptive policy); **verifier = violet**, "terminal outcome scorer" (a scorer). The verifier's
violet is reused on the Zone-3 "Same pure verifier" box so the reader sees the *same* scorer reappear at evaluation.

## How the reward methods branch and then converge

One arrow ("policy rollouts") leaves the environment into the single **GRPO training** anchor. From there, three
arrows **fan out** to three colored pills — Outcome-only (red, "terminal reward"), Credit assignment (green,
"+ information credit"), Decoupled (blue, "separate gather / commit") — under the "training reward differs" bracket.
The three pills then **fan in** (converging arrows) to a single merge point and one arrow into **Held-out simulator
cases**, under the "common evaluation — held constant" bracket. This diamond (1→3→1) is the core visual argument:
methods differ only in training reward, then are judged identically. The pills are capsules, not big cards, so they
read as variants of one method family rather than separate tasks/environments.

## Two-node infrastructure inset

**Included**, small and secondary (~10%), beneath the GRPO anchor: a light "Disaggregated RL" box with two tiny
nodes `trainer ⇄ rollout` and the note "LoRA updates · vLLM generation." It is low-contrast so it does not compete
with the experimental-design message; delete the "disaggregated-RL inset" block in the script to remove it.

## What I would remove first if this must fit above Figure 1

1. **The disaggregated-RL inset** (pure implementation; least essential).
2. **The `commit → clinical action` sub-branch** and the "ask again or commit" italics — the ask/reveal loop alone
   conveys the interaction.
3. **Compress the metrics** from four to two (high-risk value, question burden) to avoid duplicating Figure 1's
   quantitative panel, and drop "oracle-grade support" first.
4. **Shrink the environment container** height (tighten the vertical gap between the interaction and the
   oracle/verifier badges) to reduce total height.
5. **De-duplicate:** this figure and the separate `train_eval_overview` cover overlapping ground (shared-verifier
   control). Use only one above Figure 1 — this version if a single environment→train→eval narrative is wanted;
   the train/eval split if the train-vs-evaluate control is the point to foreground. Don't stack both.
Editable-text SVG (`svg.fonttype=none`) is provided for Figma/Illustrator.

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_experiment_overview_strip_v2.py   # -> figs/experiment_overview_strip_v2.{png,svg,pdf}
```
