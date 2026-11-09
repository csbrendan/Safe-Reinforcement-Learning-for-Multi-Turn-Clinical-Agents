# V2 manuscript plan

*Drafted 2026-11-09. Planning notes only: `main.tex` hasn't been restructured, and no V2 result exists yet. Parts II and III
describe proposed experiments. Experiment labels (A–D) and hypotheses are defined in [`../../V2/PLAN.md`](../../V2/PLAN.md).*

## Recommendation

Plan for one expanded paper, and make the final decision after Experiment B, when we'll know whether realistic-simulator
RL is feasible. Keep the option of a separate V2 paper open until then, because the outcome could change which is better.

## The combined story

**Central thesis:** improving a clinical agent's final reward does not necessarily improve the sequential procedure it
uses to gather evidence. Process-aware reinforcement learning may improve that procedure, but its effectiveness depends
on task headroom, initialization, reward design, model scale and environment complexity.

The three parts answer a progression of questions: *why* does this happen, does it still matter *in a realistic setting*,
and does any improvement *travel*?

### Part I: mechanistic understanding (existing V1)

MedGemma-4B on three controlled synthetic environments identifies:
- outcome-only GRPO shortcut behavior;
- credit-assignment benefits and limitations;
- over-gathering versus premature commitment;
- adaptive headroom and latent behavioral support;
- SFT saturation and rare-case coverage problems.

**Why the controlled simulators still matter:** they provide exact oracles and deterministic rewards, which realistic
benchmarks can't easily provide. That is what lets Part I measure whether a procedure is right, not only whether its
outcome scored well. Part II gives up that precision in exchange for realism, and the paper should say so openly.

**Experiment D belongs here.** It tests Part I's findings at 27B with exact rewards and oracles. That keeps model scale
separate from environment change.

### Part II: realistic benchmark training (new V2 core; Experiments A and B)

MedGemma-27B on a recognized clinical simulator such as SDBench, comparing base, SFT-only, outcome-only GRPO and
credit-assignment GRPO on conventional held-out benchmark cases.

**Question:** do the mechanisms from Part I remain relevant with more realistic cases, natural-language interaction,
model-based grading and larger model capacity?

Experiment B appears briefly as validation, not as a results section. If a simulator is rejected, report why, since
feasibility of RL on these simulators is itself useful to readers.

### Part III: external generalization (new V2; Experiment C)

The trained models on an independent simulator such as AgentClinic, assessing whether changes in information gathering
and clinical decision quality transfer beyond the training environment. Report zero-shot transfer and any interface
adaptation separately, and report positive and negative results transparently.

## Writing rules for the combined paper

- **Label reward provenance everywhere.** Part I has verifiable rewards. Part II likely has model-based judgments. Don't
  call Part II's rewards verifiable or plot their values on a shared axis with Part I's.
- **Compare mechanisms across parts, not numbers.** Model scale and environment change must not be presented as one effect.
- **Mark the origin of every result.** Every table and figure should show whether it is previously reported V1 work or new.
- **Keep the primary claim in-distribution.** The abstract's main result should be Part II. Part III is secondary and
  shouldn't be written as the test of whether the work succeeded.
- **Keep the scope caveat.** Part I's environments are synthetic and rule-labeled. Benchmark cases are still simulations,
  so nothing here is clinical validation or deployment evidence.
- **Resolve the stop-token question before reusing V1 numbers.** If it changes any V1 result, Part I must use corrected
  numbers and state what changed, and V1 may need its own correction note.
- **Plan for mixed outcomes:**

| Outcome | How the paper reads |
|---|---|
| A positive, C positive | Process-aware RL improves procedure in realistic training and transfers |
| A positive, C negative | Improvement is real in-distribution but environment-specific. Still a useful finding |
| A negative or null, D replicates | Mechanisms are robust in controlled settings but don't yet pay off in realistic ones, which limits the thesis |
| A infeasible | Part II becomes a feasibility analysis. A separate, smaller paper may be better (see below) |

## One paper or two

| | One expanded paper | Separate V2 paper |
|---|---|---|
| **Main advantage** | One coherent argument from mechanism to realism; easier for readers | Clean, focused contribution; no repeated material |
| **Main risk** | Long and hard to review; overlap with V1 must be handled carefully | Readers must connect two papers, and the second lacks Part I's exact-oracle evidence unless it cites it well |
| **Best when** | V2's question and method extend V1's, and the combined result supports one thesis | V2 has its own question, method or contribution |
| **V1 material** | Reused and clearly disclosed, with mainly new material for the journal | Summarized and cited only |
| **Arrangement on arXiv** | Could be a new version of the same record, or a new record that cites V1 | New record |

**A separate V2 paper is scientifically preferable if any of these holds:**
- Realistic-simulator RL needs a substantially different method, such as a new reward estimator or training framework.
- The main question changes, for example to the reliability of model-graded rewards.
- Part II is an independent contribution that would be obscured by V1's material.
- Part II's reward provenance makes a single thesis misleading.
- Experiment A is infeasible, leaving D and C as a smaller replication and evaluation paper.
- The target journal requires mostly unpublished material, or restricts substantial reuse of a preprint.
- Authorship or data-access terms differ enough that one paper is impractical.

**One paper remains appropriate if** credit-assignment GRPO is recognizably the same method, the primary questions are A1–A3
as planned, and the Part I findings are needed to interpret Part II.

**Decide** after Experiment B passes or fails, and revisit after Experiment A's test results.

## Publication strategy and avoiding duplication

- **V1 stays a frozen, citable research artifact.** It isn't replaced as a historical record. Cite a specific version, and
  keep tag `arxiv-v1` as the source of record.
- **V2 is a substantially expanded manuscript,** potentially suitable for journal submission.
- **Disclose the relationship to V1,** in the manuscript and cover letter: state that a preprint exists, cite it, and
  identify which results are reused and which are new.
- **Check journal policy before choosing.** Confirm the target journal's rules on preprints and how much new material an
  extension needs. Don't assume them.
- **Don't reuse V1 text or figures without labeling them.** Any reused figure or number should be identified as previously
  reported, even in a combined paper.
- **Avoid "arXiv v2" as shorthand.** "V2" in this repo means the extension. An arXiv replacement of V1, for example to add
  a disclosure, would also be numbered v2 on arXiv. Say which is meant.
- **Reflect authorship and contributions at submission.** V2 authorship, including Agustina Saenz's role, is still to be
  confirmed. The AI-assistance disclosure needs carrying over and updating to match V2's actual tool use.

## Rough section outline

No `main.tex` changes yet. This is the proposed shape once results exist.

1. **Introduction:** procedure versus outcome; the general gather-then-commit question; relationship to V1.
2. **Framework:** ask-to-commit processes; reward provenance (verifier versus judgment); gates as negative screens.
3. **Part I:** mechanistic results, marked as previously reported, plus the D replication at 27B.
4. **Part II:** simulator selection, Experiment B validation, Experiment A results on test.
5. **Part III:** zero-shot transfer, any adaptation, and interface failures reported separately.
6. **Clinician review:** trajectory ratings and agreement with automated audits (W6).
7. **Discussion:** what the factors explain, limitations, not deployment evidence.
8. **Methods and supplement:** full per-task tables and pre-registered endpoints.

`main.tex` in this folder still matches V1. Its future-work paragraph describes the earlier plan, so update it during drafting.
