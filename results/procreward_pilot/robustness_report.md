# Task 1 — Rubric robustness (paraphrased judge vs original)

Re-scored 64 Phase-0b trajectories with a fully reworded, meaning-preserving rubric.

## Score stability
- Pearson(original, paraphrase) overall_process_score: **0.983**
- Spearman(original, paraphrase): **0.956**
- within-case ranking agreement (pairwise): **90/107 = 84%**
- trajectories changing > 1 point: **0/64**
- step-flag drift (mean |Δ| summed sub-score): {'unsafe_step': 0.05, 'premature_step': 0.0, 'redundancy_step': 0.11}

## High-risk safety inversions under the paraphrase (§ must stay clean)
None — no unsafe high-risk trajectory scored at/above its case oracle under the paraphrase. ✓