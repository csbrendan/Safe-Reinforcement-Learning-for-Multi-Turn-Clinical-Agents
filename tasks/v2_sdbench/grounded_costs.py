"""Grounded (CPT / chargemaster) re-pricing of evaluation output, for costs comparable to the published SD-Bench numbers.

Why: SD-Bench's published costs match each test to CPT codes in the public Kaiser Permanente Washington chargemaster
(`unified_cpt_llm`), with an LLM estimate only as a fallback. Its README notes gpt-5-mini LLM-only estimates run
40-50% above grounded costs, so our training-time LLM prices (costs.py) aren't comparable to the paper. Training
keeps the cached LLM price; reported evaluation costs come from this module.

What it computes per evaluation row (tasks/v2_sdbench/run_eval.py JSONL):
  grounded_total_cost            SD-Bench's own harness.compute_agent_costs on the episode, method unified_cpt_llm
                                 (official accounting, unchanged SD-Bench code).
  grounded_total_cost_corrected  our corrected accounting (costs.CostModel.corrected_cost) with grounded prices.
  grounded_workup_cost           cost of the case record's real workup, same pricer.
  n_cpt / n_llm_fallback / n_price_failed   how each unique test was priced; the hit rate is the comparability check
                                 (README: the LLM fallback should fill gaps, not carry most tests).

Prices are cached per normalized test name (JSONL, first write wins), so every arm and condition shares the same
price for the same test.

Requirements for a real run (not needed by the CPU tests):
  * the chargemaster xlsx in <clone>/data/costing_data/ (see that folder's README);
  * SD-Bench's `cpt_costing` extra (pandas, openpyxl, faiss-cpu, fuzzywuzzy, python-Levenshtein);
  * OPENAI_API_KEY or OPENROUTER_API_KEY for the matcher's text-embedding-3-large embeddings (SD-Bench has no Azure
    path for embeddings). The matcher embeds the whole chargemaster once and caches it next to the xlsx;
  * the LLM fallback goes through SD-Bench's own client: OpenRouter by default, or our Azure gpt-5-mini with
    SDBENCH_FORCE_AZURE=1 and the AZURE_OPENAI_* variables in the environment. These fallback calls don't go
    through llm.SpendMeter; they're tiny (one short gpt-5-mini call per unmatched test, cached).

  python -m tasks.v2_sdbench.grounded_costs --in results/.../eval.jsonl --out results/.../eval.grounded.jsonl
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import os
from typing import Any, Callable

from tasks.v2_sdbench import sdb  # noqa: F401  (puts the SD-Bench clone on sys.path)
from tasks.v2_sdbench.cache import JsonlCache, make_key
from tasks.v2_sdbench.costs import CostModel, strip_prefix
from tasks.v2_sdbench.channel import is_question
from tasks.v2_sdbench.runner import make_config, normalize_request

METHOD = "unified_cpt_llm"
GROUNDED = "cpt_lookup"
FALLBACK = ("llm_fallback", "llm_only", "batch_llm_only")


def costing_config(dataset: str = "mimic", costing_model: str = "gpt-5-mini"):
    """SD-Bench CostingConfig for grounded pricing (sequential: our own pool and cache handle concurrency)."""
    from sdbench.config import CostingConfig
    cfg = make_config(dataset)
    cfg = dataclasses.replace(cfg, costing_method=METHOD, costing_model=costing_model, costing_parallelism=1,
                              costing_multiprocess=False)
    return CostingConfig.from_sdbench_config(cfg)


def _default_estimator(name: str, cfg) -> dict:
    from sdbench.costing.costing import estimate_unified_test_costs
    return estimate_unified_test_costs(name, cfg)


class GroundedPricer:
    """price(name) -> USD or None, via SD-Bench's unified CPT-then-LLM estimator, cached by normalized name."""

    def __init__(self, cache: JsonlCache, cfg, estimator: Callable[[str, Any], dict] | None = None):
        self.cache, self.cfg = cache, cfg
        self.estimate = estimator or _default_estimator
        self.tag = f"{METHOD}:{os.path.basename(cfg.costing_source_file_path)}:{cfg.costing_model}"

    def lookup(self, name: str) -> dict:
        name = strip_prefix(name)
        key = make_key("grounded_price", normalize_request(name), self.tag)
        v = self.cache.get(key)
        if v is not None:
            return v
        r = self.estimate(name, self.cfg)
        v = {"cost": r.get("cost"), "method": r.get("method")}
        if v["cost"] is not None:  # failures aren't cached (may be transient)
            self.cache.put(key, v, meta={"test": name})
        return v


class GroundedCostModel(CostModel):
    """Our cost formulas (official / corrected / workup) with grounded prices instead of LLM prices."""

    def __init__(self, pricer: GroundedPricer, clinician_cost: int = 300):
        super().__init__(None, pricer.cache, fallback_price=0.0, clinician_cost=clinician_cost)
        self.pricer = pricer
        self.methods: dict[str, str] = {}

    def price(self, test_name: str) -> float | None:
        v = self.pricer.lookup(test_name)
        self.methods[normalize_request(strip_prefix(test_name))] = v["method"] or "failure"
        return None if v["cost"] is None else float(v["cost"])

    def prefetch(self, names: list[str]) -> None:  # sequential; the cache makes repeats free
        for n in dict.fromkeys(names):
            self.price(n)


def sdbench_result(row: dict) -> dict:
    """The fields SD-Bench's harness costing reads, rebuilt from an evaluation row. Every request gets exactly one
    test log (runner.CachedTestRunner), so requested_tests_only is the non-question logs in order."""
    logs = row["test_logs"]
    tests = [l for l in logs if not is_question(l["test_name"])]
    rounds = row.get("n_test_rounds")
    if rounds is None:  # older eval files: a test round is a turn with at least one test (runner's has_tests rule)
        rounds = len({l["turn"] for l in tests})
    return {"requested_tests_only": [l["test_name"] for l in tests], "test_logs": logs, "num_test_rounds": rounds}


def _harness_cost(result: dict, cfg, estimator) -> dict:
    """SD-Bench's compute_agent_costs, with its per-test estimator swapped for our cached pricer."""
    from sdbench.costing import costing as C
    from sdbench.harness.harness import compute_agent_costs
    orig = C.estimate_unified_test_costs
    C.estimate_unified_test_costs = estimator
    try:
        compute_agent_costs(result, cfg)
    finally:
        C.estimate_unified_test_costs = orig
    return result[f"cost_analysis_{METHOD}"]


def recost_row(row: dict, model: GroundedCostModel, case: dict | None = None) -> dict:
    res = sdbench_result(row)
    pr = model.pricer

    def cached_estimator(name, cfg):
        v = pr.lookup(name)
        return {"test_name": name, "cost": v["cost"], "method": v["method"] or "failure", "cpt_estimate": None,
                "llm": None, "explanation": "cached grounded price"}

    official = _harness_cost(res, pr.cfg, cached_estimator)
    corr = model.corrected_cost({"requested_tests_only": res["requested_tests_only"], "test_logs": res["test_logs"],
                                 "num_test_rounds": res["num_test_rounds"]})
    methods = [d["method"] for d in official.get("cost_details", [])]
    out = {"grounded_total_cost": official["totals"]["total_with_visits"],
           "grounded_tests_cost": official["totals"]["tests_only"],
           "grounded_total_cost_corrected": corr["total_cost_corrected"],
           "n_cpt": sum(m == GROUNDED for m in methods),
           "n_llm_fallback": sum(m in FALLBACK for m in methods),
           "n_price_failed": sum(m == "failure" for m in methods),
           "grounded_method": METHOD, "grounded_tag": pr.tag}
    if case is not None:
        out["grounded_workup_cost"] = model.workup_cost(case)["total_cost"]
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tests = sum(r["n_cpt"] + r["n_llm_fallback"] + r["n_price_failed"] for r in rows)
    mean = lambda k: sum(r[k] for r in rows) / n if n else 0.0  # noqa: E731
    return {"episodes": n, "unique_tests": tests,
            "cpt_hit_rate": (sum(r["n_cpt"] for r in rows) / tests) if tests else None,
            "llm_fallback_rate": (sum(r["n_llm_fallback"] for r in rows) / tests) if tests else None,
            "failed": sum(r["n_price_failed"] for r in rows),
            "mean_grounded_total_cost": mean("grounded_total_cost"),
            "mean_grounded_total_cost_corrected": mean("grounded_total_cost_corrected")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="run_eval JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="mimic")
    ap.add_argument("--cache", default="results/v2_sdbench/grounded_prices.jsonl")
    ap.add_argument("--costing-model", default="gpt-5-mini")
    a = ap.parse_args()
    cfg = costing_config(a.dataset, a.costing_model)
    model = GroundedCostModel(GroundedPricer(JsonlCache(a.cache), cfg), clinician_cost=cfg.clinician_cost)
    from tasks.v2_sdbench.cases import get_case
    rows_out = []
    with open(a.inp) as f, open(a.out, "w") as g:
        for line in f:
            row = json.loads(line)
            row.update(recost_row(row, model, get_case(a.dataset, row["case_id"])))
            rows_out.append(row)
            g.write(json.dumps(row) + "\n")
    print(json.dumps(summarize(rows_out), indent=1))


if __name__ == "__main__":
    main()
