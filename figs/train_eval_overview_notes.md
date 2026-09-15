# Train-vs-evaluate overview — notes

Companion to `scripts/make_train_eval_overview.py` → `figs/train_eval_overview.{png,svg,pdf}`.
Paper: *Reward Design Governs Information Gathering in Clinical Multi-Turn Agents.*

A standalone conceptual figure (not Figure 1) that may sit above it. Its single job: make it obvious that
**RL methods differ only in their training reward, while every method is evaluated on the same held-out
environments with the same unshaped deterministic verifier.**

## Visual hierarchy

Two shaded half-panels split by a dashed vertical divider — **Training** (left, richer) and **Held-out
evaluation** (right, simpler) — each with a large bold title. A single heavy black arrow ("freeze & select
policy · trained checkpoint") crosses the divider left→right, so the reading order is train → freeze → test and
evaluation never looks like continued learning. Within each half a clear tier order: panel title → main
boxes → connective arrows/labels → small captions. Color is the top signal: teal = policy, gray = environment/
simulator, gold = oracle, violet = verifier, and the three method accents (red/green/blue) mark only the training
reward cards. Right-half emphasis is deliberately concentrated on one large violet box ("Same deterministic
verifier") plus a red "reward shaping off" chip and the bold footer "Same pure verifier for all methods."

## What each element represents

**Training (left)**
- **MedGemma-4B-it / LoRA-adapted policy** (teal, rounded) — the agent, drawn outside the environment.
- **Clinical multi-turn environment** (container; "refill triage · diagnostics · titration") — the interactive
  setting (not a static dataset).
- **Patient simulator / hidden clinical state** with a red "state hidden from agent" chip — partial observability.
- **ask → / finding revealed ← / ask again or commit** arrows — the gather-then-commit interaction loop.
- **Exact adaptive oracle** (gold, "reference policy") and **Deterministic verifier** (violet, "terminal outcome
  scorer"), dash-linked to the simulator as analytic components of the known simulator — the oracle is a reference,
  the verifier is a scorer; they are deliberately *not* merged.
- **Training reward varies by method** → three cards fed by one branching arrow from the rollouts:
  Outcome-only ("terminal utility − question cost"), Credit-assignment ("terminal reward + information credit"),
  Decoupled explore/exploit ("separate gathering & commitment objectives"), under the caption "same task family —
  only the training reward differs" (so they are variants, not different environments or different eval rewards).
- Small **Disaggregated RL** inset (trainer node · GRPO + LoRA; rollout server · vLLM) — see below.

**Evaluation (right)** — a simple vertical flow: **Trained policy (frozen checkpoint)** → **Held-out clinical
cases (unseen simulated patients)** → **Same deterministic verifier (one scorer for every method)** with the
**reward shaping off · no auxiliary training rewards** chip → **Evaluation metrics** (overall value · high-risk
value · question burden · oracle-grade support). Footer: **Same pure verifier for all methods.**

## Why the training/evaluation distinction is emphasized

It is the paper's central experimental control: the thesis (reward design changes the *procedure*, not just the
score) is only credible if all methods are judged identically. The figure encodes "same verifier" three ways —
the verifier uses the **same violet** in both halves; the eval verifier box is the single largest, boldest object
on the right; and the "reward shaping off" chip + "Same pure verifier for all methods" footer state it in words.
The one-way "freeze & select policy" arrow makes clear that evaluation is on a frozen checkpoint, not more learning.

## Optional disaggregated-RL inset

**Included**, as a small, low-contrast box in the bottom-left of the Training half (~12% of the graphic): two tiny
2-square GPU glyphs labeled "trainer node · GRPO + LoRA" and "rollout server · vLLM generation." It is faint and
subordinate so it does not compete with the experimental-design message. Delete the "disaggregated-RL inset" block
in the script to remove it.

## If merged into Figure 1 later — what to remove / compress

- **Drop the disaggregated-RL inset** first; infrastructure is the least important part for a main Figure 1.
- **Compress the oracle+verifier** to a single small stacked pair (or move the oracle to the caption) if space is
  tight; keep the verifier, since the shared-verifier point is essential.
- **Collapse the three reward cards** to one-line chips (names + a 3–4 word reward phrase) to save vertical space.
- **Shorten the evaluation metric list** to two (high-risk value, question burden) to match whatever quantitative
  panel Figure 1 already carries, avoiding duplication.
- **De-duplicate with the existing experiment-overview strip**: that strip and this figure overlap (policy ↔
  environment, oracle/verifier, GRPO, held-out eval). Use *one* of them above Figure 1 — this train-vs-eval version
  if the shared-verifier control is the point to foreground; the strip if a simple left-to-right pipeline is
  preferred. Do not stack both.
- Editable-text SVG (`svg.fonttype=none`) is provided for Figma/Illustrator assembly.

## Reproduce
```bash
source ~/spark1/.venv/bin/activate
python scripts/make_train_eval_overview.py   # -> figs/train_eval_overview.{png,svg,pdf}
```
