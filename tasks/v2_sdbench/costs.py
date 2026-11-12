"""Investigation cost, mirroring SD-Bench's harness formula:

  cost = clinician_cost * (num_test_rounds + 1) + sum(price(t) for t in unique non-rejected tests)

Prices come from SD-Bench's own LLM batch estimator prompt (one test per call), cached by normalized test name,
so a test always gets the same price. Grounded CPT costing (official charge master) can replace `price_fn` later
for evaluation. A test whose price can't be parsed gets `fallback_price` and is counted in `n_price_fallback`.

Two cost views per episode (both always reported):
  episode_cost    OFFICIAL SD-Bench accounting, unchanged. Benchmark comparisons use this.
  corrected_cost  TRAINING accounting. Same formula, but it closes two undercharges:
                  * a test result that reached the agent through a patient question (channel_kind test_result,
                    not guarded) is priced as the test it is, and that round counts as a test round;
                  * a compound order ("CBC, CMP; troponin") is priced per component, not as one item.
"""
from __future__ import annotations
from typing import Callable

from tasks.v2_sdbench import sdb  # noqa: F401  (puts the SD-Bench clone on sys.path)
from sdbench.costing.llm_batch_estimator import cost_tests_batch
from tasks.v2_sdbench.cache import JsonlCache, make_key
from tasks.v2_sdbench.channel import canonical_request, question_subject, split_compound
from tasks.v2_sdbench.runner import is_rejected, normalize_request

CLINICIAN_COST = 300
TEST_PREFIX = "Test to order: "


def strip_prefix(name: str) -> str:
    n = name.strip()
    return n[len(TEST_PREFIX):].strip() if n.lower().startswith(TEST_PREFIX.lower()) else n


class CostModel:
    def __init__(self, costing_llm: Callable | None, cache: JsonlCache, fallback_price: float = 100.0,
                 clinician_cost: int = CLINICIAN_COST):
        self.llm = costing_llm
        self.cache = cache
        self.fallback_price = fallback_price
        self.clinician_cost = clinician_cost
        self.model_id = getattr(costing_llm, "model_id", "none")

    def price(self, test_name: str) -> float | None:
        name = strip_prefix(test_name)
        key = make_key("price", normalize_request(name), self.model_id)
        v = self.cache.get(key)
        if v is not None:
            self.cache.hits += 1
            return v
        if self.llm is None:
            return None
        r = cost_tests_batch([name], None, llm=self.llm).get(name.lower().strip(), {})
        if "cost" in r and r["cost"] is not None:
            self.cache.misses += 1
            self.cache.put(key, float(r["cost"]), meta={"test": name})
            return float(r["cost"])
        return None  # failures are not cached (may be transient)

    def episode_cost(self, record: dict) -> dict:
        """Costs for one episode record (from Episode.record())."""
        rejected = {l["test_name"] for l in record["test_logs"] if is_rejected(l.get("result", ""))}
        tests = [strip_prefix(t) for t in record["requested_tests_only"] if t not in rejected]
        unique = list(dict.fromkeys(tests))
        prices, n_fb = [], 0
        for t in unique:
            p = self.price(t)
            if p is None:
                p, n_fb = self.fallback_price, n_fb + 1
            prices.append(p)
        visits = self.clinician_cost * (record["num_test_rounds"] + 1)
        return {"tests_cost": sum(prices), "visits_cost": visits, "total_cost": sum(prices) + visits,
                "n_priced_tests": len(unique), "n_price_fallback": n_fb}

    def corrected_cost(self, record: dict) -> dict:
        logs = record["test_logs"]
        rejected = {l["test_name"] for l in logs if is_rejected(l.get("result", ""))}
        items: list[str] = []
        for t in record["requested_tests_only"]:
            if t not in rejected:
                items.extend(split_compound(strip_prefix(t)))
        q_turns: set[int] = set()
        n_q_charged = 0
        seen_q: set[str] = set()
        for l in logs:
            if not (l.get("is_question") and l.get("channel_kind") == "test_result" and not l.get("guarded")):
                continue
            canon = normalize_request(canonical_request(l["test_name"]))
            if canon in seen_q:
                continue
            seen_q.add(canon)
            items.append(question_subject(l["test_name"]))
            q_turns.add(l.get("turn", -1))
            n_q_charged += 1
        unique = list(dict.fromkeys(normalize_request(i) for i in items))
        prices, n_fb = [], 0
        for t in unique:
            p = self.price(t)
            if p is None:
                p, n_fb = self.fallback_price, n_fb + 1
            prices.append(p)
        rounds = record["num_test_rounds"] + len(q_turns)  # question-only turns: they can't contain a <test>
        visits = self.clinician_cost * (rounds + 1)
        return {"tests_cost_corrected": sum(prices), "visits_cost_corrected": visits,
                "total_cost_corrected": sum(prices) + visits, "n_priced_corrected": len(unique),
                "n_question_tests_charged": n_q_charged, "n_price_fallback_corrected": n_fb}

    def workup_cost(self, case: dict) -> dict:
        """Cost of the real workup in the case record (SD-Bench's compute_actual_workup_costs formula)."""
        tests = [t["name"] for t in case.get("tests", []) if not t.get("in_initial_presentation", False)]
        prices, n_fb = [], 0
        for t in dict.fromkeys(tests):
            p = self.price(t)
            if p is None:
                p, n_fb = self.fallback_price, n_fb + 1
            prices.append(p)
        visits = self.clinician_cost * case.get("estimated_clinician_visits", 1)
        return {"total_cost": sum(prices) + visits, "n_price_fallback": n_fb}
