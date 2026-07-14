"""Claude process-reward integration for GRPO.

training_reward = analytic_terminal_reward + alpha * normalized_claude_process_reward

- The Claude score is a trajectory-level scalar from the off-policy judge (student-visible transcript ONLY).
- It is NORMALIZED WITHIN the prompt/case rollout group (GRPO group-relative), never used raw, so a systematic
  per-case offset (the judge is harsh on benign cases) cannot bias the gradient — only within-case ranking flows.
- It is ADDED to, never replaces, the analytic terminal reward. alpha is small and validation-selected.

This module is pure/env-agnostic and holds NO privileged state: the judge is given only the rendered
student-visible transcript. The env / gates / analytic reward are untouched.
"""
from __future__ import annotations
import statistics as stx


def normalize_group(raws, method="zscore_clip", clip=3.0):
    """Normalize a case's rollout-group Claude scores. `raws` may contain None (parse fail / not scored).
    zscore_clip: (x-mean)/std clipped to +/-clip, None -> group mean -> 0. centered_rank: rank -> [-1, 1]."""
    present = [r for r in raws if r is not None]
    if len(present) < 2 or (max(present) - min(present)) < 1e-9:
        return [0.0] * len(raws)                      # degenerate group -> no shaping signal
    if method == "zscore_clip":
        m = stx.mean(present)
        sd = stx.pstdev(present) or 1.0
        return [max(-clip, min(clip, ((r if r is not None else m) - m) / sd)) for r in raws]
    if method == "centered_rank":
        m = stx.mean(present)
        filled = [r if r is not None else m for r in raws]
        order = sorted(range(len(filled)), key=lambda i: filled[i])
        rank = [0] * len(filled)
        for pos, i in enumerate(order):
            rank[i] = pos
        denom = (len(filled) - 1) or 1
        return [2.0 * (rank[i] / denom) - 1.0 for i in range(len(filled))]
    raise ValueError(f"unknown normalization: {method}")


def assemble(analytic, claude_raw, case_ids, alpha, method="zscore_clip", clip=3.0, extra=None):
    """Combine per-rollout analytic reward + alpha * within-case-normalized Claude score.

    Returns (totals, normalized, per_rollout_log). Normalization groups rollouts by case_id (the GRPO group).
    `extra` (optional) is a list of dicts merged into each log row (e.g. {"n_asks": k}) for diagnostics.
    """
    groups = {}
    for i, cid in enumerate(case_ids):
        groups.setdefault(cid, []).append(i)
    normalized = [0.0] * len(analytic)
    nonzero_groups = 0
    for cid, idxs in groups.items():
        norm = normalize_group([claude_raw[i] for i in idxs], method, clip)
        if any(abs(x) > 1e-12 for x in norm):
            nonzero_groups += 1
        for j, i in enumerate(idxs):
            normalized[i] = norm[j]
    totals = [analytic[i] + alpha * normalized[i] for i in range(len(analytic))]
    log = [{"case_id": case_ids[i], "analytic": round(analytic[i], 4),
            "claude_raw": None if claude_raw[i] is None else round(claude_raw[i], 3),
            "claude_norm": round(normalized[i], 4), "alpha": alpha,
            "alpha_x_norm": round(alpha * normalized[i], 4), "total": round(totals[i], 4),
            **(extra[i] if extra else {})}
           for i in range(len(analytic))]
    return totals, normalized, log, {"n_groups": len(groups), "nonzero_advantage_groups": nonzero_groups}
