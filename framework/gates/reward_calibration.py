"""Reward calibration (CPU) — sweep reward settings so that adaptive selective gathering is the true optimum
and cheap under-gathering / over-defensive shortcuts are not. Env-agnostic.

The reward shape is kept as R = U'[true_risk][action] - ask_cost * #asks, so the exact oracle/best-fixed
solvers apply; you supply candidate (U, ask_cost) pairs. For each candidate it re-solves the analytic policies
(oracle, best-fixed, ask-all, no-ask, 1-ask) by swapping env.U and reports the ordering + under-gathering /
over-escalation checks. Pre-registered lesson (docs/lessons_learned.md): lowering ask-cost fixes the
under-gathering exploit but can shrink the BC-gap (cheap asks -> thorough fixed checklist ~ oracle); resolve
that by widening the env, not by softening penalties.

Usage: import and call sweep(env, candidates); or adapt per task. See docs/framework.md.
"""
from __future__ import annotations
import copy
from framework.gates import bc_gap


def evaluate_candidate(env, U, cost):
    """Swap in candidate reward, run the Gate-1 analysis, restore. Returns (policy_table, deep_gap, checks)."""
    orig = copy.deepcopy(env.U)
    env.U.clear(); env.U.update(copy.deepcopy(U))
    if hasattr(env, "clear_caches"):
        env.clear_caches()
    try:
        R, gap, per_fam, BEST, checks = bc_gap.run(env, cost)
    finally:
        env.U.clear(); env.U.update(orig)
        if hasattr(env, "clear_caches"):
            env.clear_caches()
    return R, gap, per_fam, checks


def sweep(env, candidates):
    """candidates: list of (name, U_dict, cost). Prints a compact ordering + gate summary per candidate."""
    for name, U, cost in candidates:
        R, gap, per_fam, checks = evaluate_candidate(env, U, cost)
        print(f"=== {name} (cost={cost}) ===")
        for p in ("oracle", "best_fixed", "ask_all", "no_ask", "1ask", "blind_hold", "blind_escalate"):
            r = R[p]
            print(f"  {p:14s} all={r['all']:+.3f} high={r['high']:+.3f} benign={r['benign']:+.3f} asks={r['asks']:.2f}")
        print(f"  DEEP BC-gap={gap:+.3f}  passes_gate1={all(checks.values())}")
