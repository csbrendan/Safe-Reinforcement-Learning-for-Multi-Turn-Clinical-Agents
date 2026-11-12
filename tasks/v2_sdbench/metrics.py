"""Evaluation metrics over scored episodes (rows from env.Services.score / run_eval output).

Per-episode definitions (pre-registered in V2/sdbench_execution_plan.md; thresholds fixed before evaluation):
  correct          eval-judge pass (score >= 4)
  premature        wrong AND fewer than `premature_floor` record-grounded tests ordered
  early_commit     diagnosed with zero tests ordered (right or wrong)
  over_testing     total cost > over_ratio * cost of the case's documented workup
  invalid          no diagnosis, several diagnosis tags, hedged list, over-long answer, or context overflow
  frac_grounded    record-grounded tests / tests ordered
  *_corrected      the same with the corrected cost (question-channel test results charged) and, for
                   early_commit, counting test results obtained through questions as testing
  q_test_results   test results the agent asked for as patient questions (leak attempts; withheld if guarded)
  runon/truncated  replies cut at <end_of_turn> / replies that hit the per-turn token cap

Arms are compared paired on the same cases: bootstrap CI of the mean difference, McNemar for `correct`.
"""
from __future__ import annotations
import math
import random
import statistics as stx
from typing import Iterable

KEYS = ["correct", "judge_score", "total_cost", "n_tests", "n_questions", "n_grounded_tests", "frac_grounded",
        "n_repeat", "premature", "early_commit", "over_testing", "invalid", "n_rounds", "dx_leak_in_requests",
        "total_cost_corrected", "early_commit_corrected", "over_testing_corrected", "q_test_results",
        "guarded", "dx_leak_q", "runon", "truncated", "forced"]


def episode_metrics(row: dict, premature_floor: int = 1, over_ratio: float = 1.5) -> dict:
    n_tests = row.get("n_tests", 0)
    correct = bool(row.get("passed"))
    return {
        "case_id": str(row["case_id"]),
        "correct": float(correct),
        "judge_score": float(row.get("judge_score", 1)),
        "total_cost": float(row.get("total_cost", 0.0)),
        "n_tests": float(n_tests),
        "n_questions": float(row.get("n_questions", 0)),
        "n_grounded_tests": float(row.get("n_grounded_tests", 0)),
        "frac_grounded": (row.get("n_grounded_tests", 0) / n_tests) if n_tests else 0.0,
        "n_repeat": float(row.get("n_repeat", 0)),
        "premature": float((not correct) and row.get("n_grounded_tests", 0) < premature_floor),
        "early_commit": float(n_tests == 0 and bool(row.get("diagnosis"))),
        "over_testing": float(row.get("workup_cost", 0) > 0
                              and row.get("total_cost", 0) > over_ratio * row["workup_cost"]),
        "invalid": float(bool(row.get("invalid"))),
        "n_rounds": float(row.get("n_rounds", 0)),
        "dx_leak_in_requests": float(row.get("dx_leak_in_requests", 0.0)),
        "total_cost_corrected": float(row.get("total_cost_corrected", row.get("total_cost", 0.0))),
        "early_commit_corrected": float(n_tests == 0 and row.get("n_q_test_results", 0) == 0
                                        and bool(row.get("diagnosis"))),
        "over_testing_corrected": float(row.get("workup_cost", 0) > 0 and row.get(
            "total_cost_corrected", row.get("total_cost", 0)) > over_ratio * row["workup_cost"]),
        "q_test_results": float(row.get("n_q_test_results", 0)),
        "guarded": float(row.get("n_guarded", 0)),
        "dx_leak_q": float(row.get("n_dx_leak_q", 0)),
        "runon": float(row.get("n_runon", 0)),
        "truncated": float(row.get("n_truncated", 0)),
        "forced": float(bool(row.get("forced"))),
    }


def _boot_ci(vals: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(stx.mean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(rows: Iterable[dict], subset: set[str] | None = None, **kw) -> dict:
    ms = [episode_metrics(r, **kw) for r in rows]
    if subset is not None:
        ms = [m for m in ms if m["case_id"] in subset]
    out: dict = {"n": len(ms)}
    for k in KEYS:
        v = [m[k] for m in ms]
        out[k] = stx.mean(v) if v else float("nan")
        out[k + "_ci"] = _boot_ci(v)
    return out


def paired(rows_a: Iterable[dict], rows_b: Iterable[dict], key: str = "correct",
           subset: set[str] | None = None, **kw) -> dict:
    """Mean(b - a) over cases present in both arms (k=1 per case expected), with bootstrap CI."""
    a = {m["case_id"]: m for m in (episode_metrics(r, **kw) for r in rows_a)}
    b = {m["case_id"]: m for m in (episode_metrics(r, **kw) for r in rows_b)}
    ids = sorted(set(a) & set(b) & (subset if subset is not None else set(a)))
    d = [b[i][key] - a[i][key] for i in ids]
    out = {"n": len(ids), "mean_diff": stx.mean(d) if d else float("nan"), "ci": _boot_ci(d)}
    if key == "correct":
        n01 = sum(1 for i in ids if a[i]["correct"] == 0 and b[i]["correct"] == 1)
        n10 = sum(1 for i in ids if a[i]["correct"] == 1 and b[i]["correct"] == 0)
        out.update({"b_only": n01, "a_only": n10, "mcnemar_p": mcnemar_exact(n01, n10)})
    return out


def mcnemar_exact(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
