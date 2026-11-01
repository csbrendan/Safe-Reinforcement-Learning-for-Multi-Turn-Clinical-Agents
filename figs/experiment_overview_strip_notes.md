# Experiment overview strip — notes

Companion to `scripts/make_experiment_overview_strip.py` →
`figs/experiment_overview_strip.{png,svg,pdf}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

## Design (one paragraph)

A compact, journal-style horizontal banner that introduces the experimental object at a glance, meant to sit
**above** Figure 1 (it is not Figure 1). It reads strictly left→right as a pipeline of rounded cards joined by
arrows: **MedGemma-4B-it (policy)** ⇄ **clinical multi-turn environment** → **patient simulator (hidden state)** →
**exact adaptive oracle + deterministic verifier** → **GRPO training** → **held-out evaluation**. The policy is
given a distinct oval-ish teal card and connected to the environment by a two-way arrow ("acts / observes") so the
setting reads as *interactive RL, not static QA*. The interactive mechanic is shown as a small `ask → finding →
commit` loop with an "ask again" back-arc beneath the simulator card, conveying that state is hidden and the agent
must gather before deciding. Scoring is a two-part stacked block — the **exact adaptive oracle** (reference policy,
gold) and the **deterministic verifier** (common outcome scorer) — and the verifier's gold is deliberately reused
on the **held-out evaluation** card, with the italic caption "the same deterministic verifier scores every method
at evaluation," to encode the key point that evaluation is shared and unshaped. Only the **GRPO training** card
carries method variation: three small chips (outcome-only red / credit-assignment green / decoupled blue) with the
note "reward varies by method (during training only)." The palette is restrained (neutral gray for generic parts,
one muted teal for the model, one gold for oracle/verifier/eval, and the three method colors used elsewhere in the
paper); type hierarchy is title → subtitle → tiny note.

## Source terminology (as used in the repo / paper)

- **MedGemma-4B-it**, **LoRA-adapted** — the text-only policy base (CLAUDE.md; `framework.infra.prep_text_base`).
- **Clinical multi-turn environment** / three tasks **refill triage · diagnostics · titration** (`tasks/`).
- **Patient simulator / hidden state**; **ask → finding → commit** — the ask-to-commit POMDP with hidden severity
  (`framework/envs/base.py`, task envs).
- **Exact adaptive oracle** ("reference policy") and **deterministic verifier** ("common outcome scorer") —
  `env.score` / oracle rollout; the verifier is the shared held-out scorer with **reward shaping off**
  (`framework/gates/support_probe.py`, `framework/report.py`).
- **GRPO training** with reward varying by method: **outcome-only**, **credit assignment**, **decoupled
  explore/exploit** (`framework/rl/grpo_disagg.py`).
- **Held-out evaluation · same pure verifier for all methods** (CLAUDE.md key invariant).

## Simplifications made (conceptual graphic, not a methods diagram)

- Omitted: rollout group size, GRPO batch/optimizer settings, equations, JSON schemas, parse logic, trainer
  internals, process supervision, per-task counts, and BC-gap/support numbers.
- The environment and patient simulator are shown as two adjacent cards (per the requested structure); in the code
  the simulator *is* the environment — the split emphasizes "interactive across 3 tasks" vs "hidden state +
  ask/commit mechanic."
- The oracle and verifier are drawn as one stacked block (two sub-cards) rather than two pipeline stages, since
  both are scoring components rather than sequential steps.
- The `ask → finding → commit` loop is illustrative (no specific case/values).

## Optional disaggregated-RL note

**Included**, as a deliberately secondary, tiny element beneath the GRPO card: a small two-square (2-GPU) glyph
with the text "disaggregated RL: trainer + rollout server." It is faint and low in the hierarchy so it does not
compete with the main flow. Remove it by deleting the "disaggregated-RL inset" block in the script if a cleaner
strip is preferred.

## How to integrate into the main Figure 1 later

- **Simplest:** place this strip as a full-width **row 0 (panel "A0" / top band)** above the current Figure 1 v4,
  and relabel v4's panels A→B, B→C, C→D. The strip answers "what is the experimental object?"; Figure 1 then
  answers "what do different reward designs do to the procedure?"
- **Visual continuity:** the strip already uses the paper's method colors (outcome-only red, credit green,
  decoupled blue) and the gold oracle/verifier treatment, so it will match v4 without recoloring. Keep the strip's
  MedGemma teal only here (v4 has no model card, so no conflict).
- **Width/scale:** the strip is 16:4.3; v4 is ~14:12.6. If combined into one figure, render both from a shared
  width (e.g. re-emit the strip at 14 wide) so column widths line up, or keep them as separate stacked figures
  (Figure 1a = strip, Figure 1b = the four-row comparison).
- **De-duplication:** if combined, the strip's `ask → finding → commit` mini-loop makes v4's Panel A partly
  redundant — consider shrinking or dropping v4 Panel A when the strip sits above it.
- Export is editable-text SVG (`svg.fonttype=none`) for Figma/Illustrator assembly.

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_experiment_overview_strip.py   # -> figs/experiment_overview_strip.{png,svg,pdf}
```
