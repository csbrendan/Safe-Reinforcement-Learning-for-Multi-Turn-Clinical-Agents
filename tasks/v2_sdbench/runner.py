"""Deterministic, cached SD-Bench test runner.

Subclasses SD-Bench's own BatchTestRunner, so turn accounting, per-turn limits and the single-turn-type rule are
the official code. Only the gatekeeper call differs:

  * Each request is answered by its own gatekeeper call, with an empty `previous_tests` list, at temperature 0.
  * The answer is cached on (dataset, case, normalized request, gatekeeper prompt version, gatekeeper model).

So the same request on the same case always returns the same text, across GRPO rollouts, arms and evaluation.
Deviation from official SD-Bench (report it): official batches up to 10 requests per call and shows the gatekeeper
the previously completed tests. We trade that context for determinism and reuse.

Each log entry adds:
  record_grounded = gatekeeper says the result is in the case record (is_synthetic False) AND it actually answered
                    (not the "Results not available" placeholder a parse failure produces, not a refusal such as
                    "not available in the patient's current record", and not withheld by the guard).
  repeat          = same canonical request was already made in this episode.
  channel_kind    = channel.classify(request, result): rejected | test_result | historic_test | info.
  guarded         = (guard on) a question whose answer was a test result or named the reference diagnosis; the
                    agent got GUARD_TEXT / DX_GUARD_TEXT instead and the real answer is kept in `withheld_result`
                    for audit only.

Options (training environment; official SD-Bench behaviour is all-off except parallelism):
  question_guard  block test results returned through the question channel (information-leakage fix).
  canonicalize    cut a pasted answer off a question ("X?: 38.6") before calling the gatekeeper.
  parallel        answer the requests of one turn concurrently. Each request is already an independent call
                  with empty previous_tests, so this changes wall time only, not results.
"""
from __future__ import annotations
import datetime
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from tasks.v2_sdbench import sdb
from tasks.v2_sdbench.cache import JsonlCache, make_key
from tasks.v2_sdbench.channel import DX_GUARD_TEXT, GUARD_TEXT, canonical_request, classify, dx_key_terms, is_question

UNAVAILABLE_RE = re.compile(r"^\s*results not available for", re.I)
_REJECTION = ("i can only provide specific test results or answer specific patient questions",
              "not available", "please request a specific diagnostic test")


def normalize_request(name: str) -> str:
    s = re.sub(r"\s+", " ", name.strip().lower())
    return s.strip(" .;:,")


def is_rejected(result_text: str) -> bool:
    """Same rejection phrases SD-Bench's harness uses to drop tests from costing."""
    t = result_text.lower()
    return any(p in t for p in _REJECTION)


def make_config(dataset: str, max_turns: int = 12, config_name: str = "default", **overrides: Any):
    """SD-Bench config for `dataset` with our turn cap. Model names are placeholders: we always pass `llm=`."""
    from dataclasses import replace
    cfg = sdb.sdb_config.get_sdbench_config(dataset, config_name)
    return replace(cfg, max_turns=max_turns, **overrides)


_GK_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="gk")  # separate from callers' pools: no deadlock


class Timer:
    """Thread-safe wall-time accumulator per component (seconds and call counts)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.secs: dict[str, float] = {}
        self.calls: dict[str, int] = {}

    def add(self, name: str, secs: float, n: int = 1) -> None:
        with self._lock:
            self.secs[name] = self.secs.get(name, 0.0) + secs
            self.calls[name] = self.calls.get(name, 0) + n

    def snapshot(self) -> dict:
        with self._lock:
            return {"secs": dict(self.secs), "calls": dict(self.calls)}


class CachedTestRunner(sdb.gatekeeper.BatchTestRunner):
    def __init__(self, config, case: dict, collector: dict, gatekeeper_llm: Callable, cache: JsonlCache,
                 dataset: str, question_guard: bool = False, canonicalize: bool = False, parallel: bool = True,
                 timer: Timer | None = None):
        super().__init__(config, case, collector, llm=gatekeeper_llm)
        self._cache = cache
        self._dataset = dataset
        self._gk_id = getattr(gatekeeper_llm, "model_id", "unknown")
        self._seen: set[str] = set()
        self.question_guard = question_guard
        self.canonicalize = canonicalize
        self.parallel = parallel
        self.timer = timer
        self.dx_terms = dx_key_terms(case.get("final_diagnosis", ""))

    def _order_one(self, name: str) -> dict:
        sent = canonical_request(name) if self.canonicalize else name
        key = make_key("gatekeeper", self._dataset, str(self._case["case_id"]), normalize_request(sent),
                       self._config.gatekeeper_prompt_version, self._gk_id)

        def compute():
            t0 = time.time()
            r = sdb.gatekeeper.order_tests(self._config, self._case, [sent], previous_tests=[],
                                           collector={}, llm=self._llm)[0]
            if self.timer:
                self.timer.add("gatekeeper_llm", time.time() - t0)
            return {"results": r["results"], "is_synthetic": bool(r["is_synthetic"])}

        v = self._cache.get_or_compute(key, compute, meta={"case": str(self._case["case_id"]), "req": sent})
        return {"test_name": name, "sent": sent, "results": v["results"], "is_synthetic": v["is_synthetic"]}

    def __call__(self, test_names: list[str]):
        if self._turns_left <= 0:
            raise ValueError("No turns left")
        test_names, limit_messages = self._parse_and_limit_requests(test_names)
        has_tests = any(not t.startswith("Question for the patient:") for t in test_names)
        self._turns_left -= 1
        self._requested_tests.extend(test_names)
        self._num_rounds += 1
        if has_tests:
            self._num_test_rounds += 1
        t0 = time.time()
        if self.parallel and len(test_names) > 1:
            results = list(_GK_POOL.map(self._order_one, test_names))
        else:
            results = [self._order_one(t) for t in test_names]
        if self.timer:
            self.timer.add("gatekeeper_turn", time.time() - t0)
        for r in results:
            if r["test_name"] not in self._completed_tests:
                self._completed_tests.append(r["test_name"])
            norm = normalize_request(canonical_request(r["test_name"]))
            unavailable = bool(UNAVAILABLE_RE.match(r["results"]))
            q = is_question(r["test_name"])
            kind = "rejected" if unavailable else classify(r["test_name"], r["results"], self.dx_terms)
            guarded = self.question_guard and q and kind in ("test_result", "dx_leak")
            log = {
                "test_name": r["test_name"], "reason": "", "result": r["results"],
                "is_synthetic": r["is_synthetic"], "timestamp": datetime.datetime.now().isoformat(),
                "batch_mode": True,
                "record_grounded": (not r["is_synthetic"]) and kind != "rejected" and not guarded,
                "unavailable": unavailable,
                "repeat": norm in self._seen,
                "is_question": q,
                "turn": self._num_rounds,
                "channel_kind": kind,
                "guarded": guarded,
            }
            if guarded:
                text = DX_GUARD_TEXT if kind == "dx_leak" else GUARD_TEXT
                log["withheld_result"], log["result"] = r["results"], text
                r["results"] = text
            if r["sent"] != r["test_name"]:
                log["sent"] = r["sent"]
            self._collector["test_logs"].append(log)
            self._seen.add(norm)
        results = [{k: v for k, v in r.items() if k != "sent"} for r in results]
        return sdb.gatekeeper.TestRunnerResponse(test_names=test_names, test_results=results,
                                                 limit_messages=limit_messages)
