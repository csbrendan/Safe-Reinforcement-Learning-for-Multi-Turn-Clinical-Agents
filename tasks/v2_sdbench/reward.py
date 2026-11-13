"""Outcome reward for arms B and C (identical in both), plus the deterministic format checks.

  R_out = (judge_score - 1) / 4  -  lam_cost * total_cost / cost_scale  -  fmt_penalty * [format violation]

* judge_score: SD-Bench's 1-5 rubric via sdbench.judge.check_diagnosis with OUR judge callable. Training default
  prompt is `v3_reasoning` (diagnosis vs reference only), so scores are cacheable by (case, diagnosis) and the
  reward can't be moved by which tests were listed. Evaluation uses the official prompt and a different judge.
* No diagnosis -> score 1 without a judge call.
* lam_cost, cost_scale, fmt_penalty are set once before training from arm-A dev statistics (no sweep).
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable

from tasks.v2_sdbench import sdb
from tasks.v2_sdbench.cache import JsonlCache, make_key

HEDGE_RE = re.compile(r"(\bor\b|\bvs\.?\b|\bversus\b|\bdifferential\b|\brule out\b|\br/o\b|"
                      r"(^|\n)\s*(\d+[.)]|[-*•])\s+|;)", re.I)
MAX_DX_WORDS = 40


def format_flags(record: dict) -> dict:
    dx = record.get("diagnosis")
    flags = {
        "no_diagnosis": not dx,
        "multi_diagnosis_tag": record.get("n_diagnosis_tags", 0) > 1,
        "hedged": bool(dx and HEDGE_RE.search(dx)),
        "too_long": bool(dx and len(dx.split()) > MAX_DX_WORDS),
        "overflow": bool(record.get("overflow")),
    }
    flags["invalid"] = any(flags.values())
    return flags


class Judge:
    def __init__(self, judge_llm: Callable, cache: JsonlCache, dataset: str, prompt_version: str = "v3_reasoning",
                 max_turns: int = 12, repeats: int = 1):
        from tasks.v2_sdbench.runner import make_config
        self.llm = judge_llm
        self.cache = cache
        self.dataset = dataset
        self.prompt_version = prompt_version
        self.cfg = make_config(dataset, max_turns=max_turns, judge_prompt_version=prompt_version)
        self.model_id = getattr(judge_llm, "model_id", "unknown")
        self.uses_tests = "with_tests" in prompt_version
        # repeats > 1: score = median of that many independent judge calls (separate cache entries; repeat 0 keeps
        # the original key, so existing caches stay valid). Reduces judge-noise variance roughly by the count.
        self.repeats = repeats

    def score(self, case: dict, diagnosis: str | None, test_logs: list | None = None) -> dict:
        if self.repeats <= 1 or not diagnosis:
            return self._score_once(case, diagnosis, test_logs, 0)
        import statistics
        outs = [self._score_once(case, diagnosis, test_logs, r) for r in range(self.repeats)]
        med = int(statistics.median_low([o["score"] for o in outs]))
        pick = next(o for o in outs if o["score"] == med)
        return {**pick, "scores": [o["score"] for o in outs]}

    def _score_once(self, case: dict, diagnosis: str | None, test_logs: list | None, rep: int) -> dict:
        if not diagnosis:
            return {"score": 1, "passed": False, "reasoning": "no diagnosis", "judged": False}
        logs = [{k: l[k] for k in ("test_name", "result") if k in l} for l in (test_logs or [])] \
            if self.uses_tests else None
        key = make_key("judge", self.dataset, str(case["case_id"]), diagnosis.strip(), self.prompt_version,
                       self.model_id, logs, *([rep] if rep else []))

        def compute():
            passed, resp = sdb.judge.check_diagnosis(self.cfg, diagnosis, case, test_logs=logs, llm=self.llm)
            return {"score": int(resp["diagnosis_score"]), "passed": bool(passed),
                    "reasoning": resp.get("diagnosis_reasoning", "")[:2000], "judged": True}

        return self.cache.get_or_compute(key, compute, meta={"case": str(case["case_id"])})


@dataclass
class OutcomeWeights:
    lam_cost: float = 0.0
    cost_scale: float = 1000.0
    fmt_penalty: float = 0.25


def outcome_reward(judge_score: int, total_cost: float, flags: dict, w: OutcomeWeights) -> dict:
    s01 = (judge_score - 1) / 4.0
    cost_term = w.lam_cost * total_cost / w.cost_scale
    fmt = w.fmt_penalty if flags.get("invalid") else 0.0
    return {"r_out": s01 - cost_term - fmt, "score01": s01, "cost_term": cost_term, "fmt_term": fmt}
