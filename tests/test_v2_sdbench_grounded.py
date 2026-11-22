"""Grounded re-pricing adapter (tasks/v2_sdbench/grounded_costs.py). Offline: a stub replaces SD-Bench's CPT/LLM
estimator; SD-Bench's own harness.compute_agent_costs does the accounting.

Run:  ~/spark1/.venv/bin/python -m pytest -q tests/test_v2_sdbench_grounded.py
"""
from __future__ import annotations

import pytest

from tasks.v2_sdbench import grounded_costs as G
from tasks.v2_sdbench.cache import JsonlCache
from tests.test_v2_sdbench import CASE, _runner
from tasks.v2_sdbench.episode import Episode

PRICES = {"cbc": (25.0, "cpt_lookup"), "plasma metanephrines": (180.0, "cpt_lookup"),
          "mystery panel": (90.0, "llm_fallback")}


class StubEstimator:
    def __init__(self):
        self.calls = []

    def __call__(self, name, cfg):
        self.calls.append(name)
        cost, method = PRICES.get(name.lower().strip(), (None, "failure"))
        return {"test_name": name, "cost": cost, "method": method}


def _episode(tmp_path):
    runner, _, _ = _runner(tmp_path)
    ep = Episode(CASE, runner)
    ep.step("<test>CBC</test>\n<test>Plasma metanephrines</test>")
    ep.step("<question>Any palpitations?</question>")
    ep.step("<test>CBC</test>\n<test>mystery panel</test>")   # CBC repeated: priced once
    ep.step("<diagnosis>Pheochromocytoma</diagnosis>")
    rec = ep.record()
    row = {"case_id": rec["case_id"], "test_logs": rec["test_logs"], "n_test_rounds": rec["num_test_rounds"]}
    return rec, row


def _model(tmp_path, est):
    cfg = G.costing_config("synthetic")
    return G.GroundedCostModel(G.GroundedPricer(JsonlCache(str(tmp_path / "gp.jsonl")), cfg, estimator=est))


def test_rebuilt_result_matches_episode_record(tmp_path):
    rec, row = _episode(tmp_path)
    res = G.sdbench_result(row)
    assert res["requested_tests_only"] == rec["requested_tests_only"]
    assert res["num_test_rounds"] == rec["num_test_rounds"] == 2


def test_official_grounded_cost_uses_sdbench_formula(tmp_path):
    rec, row = _episode(tmp_path)
    est = StubEstimator()
    out = G.recost_row(row, _model(tmp_path, est), CASE)
    # SD-Bench: clinician_cost * (num_test_rounds + 1) + unique non-rejected test prices
    assert out["grounded_total_cost"] == pytest.approx(300 * 3 + 25 + 180 + 90)
    assert out["grounded_tests_cost"] == pytest.approx(295)
    assert (out["n_cpt"], out["n_llm_fallback"], out["n_price_failed"]) == (2, 1, 0)
    assert out["grounded_total_cost_corrected"] >= out["grounded_total_cost"] - 1e-9
    assert out["grounded_workup_cost"] == pytest.approx(180 + 300 * CASE["estimated_clinician_visits"])


def test_prices_are_cached_across_rows(tmp_path):
    _, row = _episode(tmp_path)
    est = StubEstimator()
    m = _model(tmp_path, est)
    G.recost_row(row, m)
    n = len(est.calls)
    G.recost_row(row, m)
    assert len(est.calls) == n  # second pass: all from cache
    # a fresh model on the same cache file also makes no calls
    est2 = StubEstimator()
    G.recost_row(row, _model(tmp_path, est2))
    assert est2.calls == []


def test_failed_prices_are_counted_and_not_cached(tmp_path):
    runner, _, _ = _runner(tmp_path)
    ep = Episode(CASE, runner)
    ep.step("<test>Unpriceable assay</test>")
    ep.step("<diagnosis>X</diagnosis>")
    rec = ep.record()
    row = {"case_id": "c1", "test_logs": rec["test_logs"], "n_test_rounds": rec["num_test_rounds"]}
    est = StubEstimator()
    m = _model(tmp_path, est)
    out = G.recost_row(row, m)
    assert out["n_price_failed"] == 1 and out["grounded_tests_cost"] == 0
    G.recost_row(row, m)
    assert est.calls.count("Unpriceable assay") >= 2  # retried, since failures may be transient


def test_summary_hit_rate():
    rows = [{"n_cpt": 3, "n_llm_fallback": 1, "n_price_failed": 0, "grounded_total_cost": 1000.0,
             "grounded_total_cost_corrected": 1100.0},
            {"n_cpt": 1, "n_llm_fallback": 0, "n_price_failed": 1, "grounded_total_cost": 500.0,
             "grounded_total_cost_corrected": 500.0}]
    s = G.summarize(rows)
    assert s["cpt_hit_rate"] == pytest.approx(4 / 6) and s["failed"] == 1
    assert s["mean_grounded_total_cost"] == pytest.approx(750.0)


def test_harness_estimator_is_restored(tmp_path):
    from sdbench.costing import costing as C
    orig = C.estimate_unified_test_costs
    _, row = _episode(tmp_path)
    G.recost_row(row, _model(tmp_path, StubEstimator()))
    assert C.estimate_unified_test_costs is orig


def test_test_rounds_rebuilt_from_turns_for_old_rows(tmp_path):
    rec, row = _episode(tmp_path)
    old = {k: v for k, v in row.items() if k != "n_test_rounds"}
    assert G.sdbench_result(old)["num_test_rounds"] == rec["num_test_rounds"] == 2
