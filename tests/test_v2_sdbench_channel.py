"""Regression tests for the question-channel fixes (tasks/v2_sdbench/channel.py, runner guard, corrected cost)
and for arm C's record-grounded credit. CPU only, fake gatekeeper, no servers.

  ~/spark1/.venv/bin/python -m pytest -q tests/test_v2_sdbench_channel.py
"""
from __future__ import annotations
import re
import time
from types import SimpleNamespace

import pytest

from tasks.v2_sdbench import channel as ch
from tasks.v2_sdbench.cache import JsonlCache
from tasks.v2_sdbench.costs import CostModel
from tasks.v2_sdbench.env import grounded_test_credit
from tasks.v2_sdbench.episode import Episode
from tasks.v2_sdbench.runner import CachedTestRunner, make_config

CASE = {"case_id": "c1", "case_title": "t", "case_summary": "A 39-year-old man with episodic headache.",
        "final_diagnosis": "Pheochromocytoma", "estimated_clinician_visits": 2, "initial_presentation": "ip",
        "full_text": "x", "tests": []}

# Answers in the shapes Qwen3-8B actually produced on SD-Bench synthetic cases (smoke run, 2026-11-12).
ANSWERS = {
    "what is your white blood cell count?": ("White-cell count: 3.6 × 10⁹/L.", False),
    "what did your plasma metanephrine levels show?":
        ("The patient's plasma free metanephrine level was 2.8 nmol/L, above the reference limit.", False),
    "what did your bone marrow biopsy show?": ("The bone marrow biopsy showed hemophagocytosis.", False),
    "what is your temperature?": ("36.8°C", False),
    "do you smoke?": ("The patient does not smoke.", False),
    "what is your pulse rate?": ("The patient's pulse rate was 96 beats per minute.", False),
    "what is your diet like?": ("Regular diet; he has not restricted his sodium intake.", False),
    "what is your ldh?": ("These test results are not available in the patient's current record. "
                          "Would you like to order this test?", False),
    "plasma metanephrines": ("Plasma free metanephrine: 2.8 nmol/L (reference <0.50).", False),
    "cbc": ("Hemoglobin 13.1 g/dL; white-cell count 7.2 × 10⁹/L.", False),
    "mystery assay": ("Synthetic normal result.", True),
}


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, role="assistant",
                                                                            tool_calls=None,
                                                                            model_dump=lambda **k: {}))])


class FakeGK:
    model_id = "fake-gk-channel"

    def __init__(self, delay=0.0):
        self.calls, self.delay, self.sent = 0, delay, []

    def __call__(self, messages, **kw):
        self.calls += 1
        time.sleep(self.delay)
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        out = []
        for i, name in re.findall(r"ID (\d+): (.+)", user):
            self.sent.append(name.strip())
            b = ch.body(name.strip()).lower()
            res, syn = ANSWERS.get(b, (f"Answer to {b}.", False))
            out.append(f'<test_result id="{i}"><result>{res}</result>'
                       f'<is_synthetic>{"true" if syn else "false"}</is_synthetic></test_result>')
        return _resp("\n".join(out))


def _runner(tmp_path, guard=False, canon=False, parallel=True, gk=None, name="gk"):
    gk = gk or FakeGK()
    r = CachedTestRunner(make_config("synthetic", max_turns=6), CASE, {}, gk,
                         JsonlCache(str(tmp_path / f"{name}.jsonl")), "synthetic",
                         question_guard=guard, canonicalize=canon, parallel=parallel)
    return r, gk


Q = ch.QUESTION_PREFIX + " "
T = ch.TEST_PREFIX + " "


def _cost_model():
    cm = CostModel(None, JsonlCache(None))
    prices = {"white blood cell count": 10.0, "plasma metanephrine": 200.0, "bone marrow biopsy": 1500.0,
              "plasma metanephrines": 200.0, "cbc": 30.0, "cmp": 40.0, "troponin": 25.0,
              "csf analysis (cell count, protein)": 120.0, "mystery assay": 60.0}
    cm.price = lambda name: prices.get(name.lower().strip(), 50.0)
    return cm


def _record(runner, extra=None):
    return {"test_logs": runner._collector["test_logs"], "requested_tests_only": runner.requested_tests_only,
            "num_test_rounds": runner.num_test_rounds, **(extra or {})}


# ---------------------------------------------------------------- classifier
@pytest.mark.parametrize("q,res", [
    ("What is your white blood cell count?", "White-cell count: 3.6 × 10⁹/L."),
    ("What is your kidney function?", "The patient's kidney function is normal."),
    ("What is your ECG?", "The electrocardiogram showed sinus rhythm at 92 beats per minute."),
    ("What did your adrenal CT scan show?", "The computed tomography scan revealed a 4.8-cm right adrenal mass."),
    ("What did your fat pad biopsy show?", "The fat-pad biopsy showed amyloid by Congo red staining."),
    ("What did your genetic testing show?", "A pathogenic variant in RET at codon 634."),
    ("What were your free light chains?", "Lambda light chain 240 mg/L."),
])
def test_classifier_flags_test_results_in_questions(q, res):
    assert ch.classify(Q + q, res) == "test_result"


@pytest.mark.parametrize("q,res", [
    ("What is your temperature?", "36.8°C"),
    ("What is your pulse rate?", "The patient's pulse rate was 96 beats per minute."),
    ("What is your typical blood pressure?", "150/95 mm Hg when not having an episode."),
    ("What is your weight and height?", "Weight 68 kg, height 175 cm."),
    ("What is your neurological exam?", "Awake but inattentive, disoriented to the date."),
    ("Are you taking any medications?", "Amlodipine 5 mg once daily and lisinopril 10 mg once daily."),
    ("What is your diet like?", "Regular diet; he has not restricted his sodium intake."),
    ("Family history?", "His father took thyroid hormone afterward and was monitored with blood tests."),
])
def test_classifier_leaves_history_vitals_and_exam_free(q, res):
    assert ch.classify(Q + q, res) == "info"


def test_classifier_rejections_and_history():
    assert ch.classify(Q + "What is your LDH?", ANSWERS["what is your ldh?"][0]) == "rejected"
    assert ch.classify(Q + "Have you had a prior colonoscopy?",
                       "A colonoscopy five years ago was normal.") == "historic_test"
    # current-admission result asked as 'prior' is still a test result
    assert ch.classify(Q + "Have you had a prior CT?", "The CT scan of the abdomen showed a 4.8-cm mass.") \
        == "test_result"
    assert ch.classify(T + "CBC", "Hemoglobin 13 g/dL") == "test_result"


def test_canonical_request_cuts_pasted_answer_only():
    assert ch.canonical_request(Q + "What is your temperature?: 38.6°C") == Q + "What is your temperature?"
    multi = Q + "What is your pulse rate? What is your resting heart rate?"
    assert ch.canonical_request(multi) == multi
    assert ch.canonical_request(T + "CBC: done") == T + "CBC: done"


def test_split_compound_and_subject():
    assert ch.split_compound("CBC, CMP; troponin") == ["CBC", "CMP", "troponin"]
    assert ch.split_compound("CSF analysis (cell count, protein)") == ["CSF analysis (cell count, protein)"]
    assert ch.question_subject(Q + "What did your bone marrow biopsy show?") == "bone marrow biopsy"
    assert ch.question_subject(Q + "What is the patient's white blood cell count?: 3.6") == "white blood cell count"


# ---------------------------------------------------------------- guard (information leakage)
def test_guard_withholds_test_results_from_questions(tmp_path):
    r, _ = _runner(tmp_path, guard=True)
    out = r([Q + "What did your bone marrow biopsy show?", Q + "Do you smoke?"])
    assert out.test_results[0]["results"] == ch.GUARD_TEXT          # agent never sees the biopsy
    assert out.test_results[1]["results"] == "The patient does not smoke."
    log = r._collector["test_logs"][0]
    assert log["guarded"] and log["result"] == ch.GUARD_TEXT and "hemophagocytosis" in log["withheld_result"]
    assert not log["record_grounded"]
    ep = Episode(CASE, _runner(tmp_path, guard=True, name="gk2")[0])
    ep.step("<question>What did your bone marrow biopsy show?</question>")
    assert "hemophagocytosis" not in ep.messages[-1]["content"] and ch.GUARD_TEXT in ep.messages[-1]["content"]


def test_guard_off_is_official_behaviour(tmp_path):
    r, _ = _runner(tmp_path, guard=False)
    out = r([Q + "What did your bone marrow biopsy show?"])
    assert "hemophagocytosis" in out.test_results[0]["results"]
    log = r._collector["test_logs"][0]
    assert log["channel_kind"] == "test_result" and not log["guarded"]


def test_guard_never_touches_ordered_tests(tmp_path):
    r, _ = _runner(tmp_path, guard=True)
    out = r([T + "Plasma metanephrines"])
    assert "2.8 nmol/L" in out.test_results[0]["results"] and not r._collector["test_logs"][0]["guarded"]


# ---------------------------------------------------------------- corrected cost
def test_question_channel_abuse_is_charged_but_official_is_unchanged(tmp_path):
    r, _ = _runner(tmp_path, guard=False)
    r([Q + "What is your white blood cell count?", Q + "What did your plasma metanephrine levels show?",
       Q + "What did your bone marrow biopsy show?"])
    cm = _cost_model()
    rec = _record(r)
    off, corr = cm.episode_cost(rec), cm.corrected_cost(rec)
    assert off["total_cost"] == 300 and off["n_priced_tests"] == 0       # official: free
    assert corr["n_question_tests_charged"] == 3
    assert corr["tests_cost_corrected"] == 10 + 200 + 1500
    assert corr["visits_cost_corrected"] == 600                           # that round becomes a test round


def test_guarded_questions_are_not_charged(tmp_path):
    r, _ = _runner(tmp_path, guard=True)
    r([Q + "What did your bone marrow biopsy show?"])
    corr = _cost_model().corrected_cost(_record(r))
    assert corr["n_question_tests_charged"] == 0 and corr["total_cost_corrected"] == 300


def test_duplicates_and_parroted_questions_charged_once(tmp_path):
    r, gk = _runner(tmp_path, guard=False, canon=True)
    r([Q + "What is your white blood cell count?"] * 3)
    r([Q + "What is your white blood cell count?: 3.6 × 10⁹/L"])
    logs = r._collector["test_logs"]
    assert [l["repeat"] for l in logs] == [False, True, True, True]
    assert gk.calls == 1                                                  # cache + canonicalization
    assert logs[-1]["sent"] == Q + "What is your white blood cell count?"
    corr = _cost_model().corrected_cost(_record(r))
    assert corr["n_question_tests_charged"] == 1 and corr["tests_cost_corrected"] == 10
    # both turns obtained a charged result only in the first; the repeat turn adds no visit
    assert corr["visits_cost_corrected"] == 600


def test_legitimate_history_questions_cost_nothing(tmp_path):
    r, _ = _runner(tmp_path, guard=True)
    r([Q + "Do you smoke?", Q + "What is your temperature?", Q + "What is your pulse rate?",
       Q + "What is your diet like?"])
    rec = _record(r)
    assert all(l["channel_kind"] == "info" and not l["guarded"] for l in rec["test_logs"])
    corr = _cost_model().corrected_cost(rec)
    assert corr["total_cost_corrected"] == _cost_model().episode_cost(rec)["total_cost"] == 300


def test_unavailable_and_refused_results_cost_nothing(tmp_path):
    def broken(messages, **kw):
        return _resp("garbage")
    broken.model_id = "broken"
    r, _ = _runner(tmp_path, gk=broken)
    r([Q + "What is your LDH?"])
    r2, _ = _runner(tmp_path, name="gk_refuse")
    r2([Q + "What is your LDH?"])
    for rr in (r, r2):
        log = rr._collector["test_logs"][0]
        assert log["channel_kind"] == "rejected" and not log["record_grounded"]
        assert _cost_model().corrected_cost(_record(rr))["n_question_tests_charged"] == 0


def test_compound_order_priced_per_component(tmp_path):
    r, _ = _runner(tmp_path)
    r([T + "CBC, CMP; troponin"])
    cm = _cost_model()
    rec = _record(r)
    assert cm.episode_cost(rec)["tests_cost"] == 50.0                     # official: one unknown item
    assert cm.corrected_cost(rec)["tests_cost_corrected"] == 30 + 40 + 25


def test_corrected_never_below_official_on_same_record(tmp_path):
    r, _ = _runner(tmp_path)
    r([T + "CBC"])
    r([Q + "What is your white blood cell count?", Q + "Do you smoke?"])
    r([T + "CBC"])
    cm = _cost_model()
    rec = _record(r)
    assert cm.corrected_cost(rec)["total_cost_corrected"] >= cm.episode_cost(rec)["total_cost"]


# ---------------------------------------------------------------- throughput change is semantics-free
def test_parallel_equals_serial(tmp_path):
    reqs = [Q + "Do you smoke?", Q + "What is your white blood cell count?", Q + "What is your temperature?",
            Q + "Do you smoke?"]
    rp, gkp = _runner(tmp_path, parallel=True, gk=FakeGK(delay=0.2), name="p")
    rs, _ = _runner(tmp_path, parallel=False, gk=FakeGK(), name="s")
    t0 = time.time()
    op = rp(reqs)
    tp = time.time() - t0
    os_ = rs(reqs)
    assert op.test_results == os_.test_results
    strip = lambda logs: [{k: v for k, v in l.items() if k != "timestamp"} for l in logs]  # noqa: E731
    assert strip(rp._collector["test_logs"]) == strip(rs._collector["test_logs"])
    assert gkp.calls == 3 and tp < 0.5                                    # 3 unique requests concurrently


# ---------------------------------------------------------------- arm C relevance proxy
def test_grounded_test_credit_rules(tmp_path):
    r, _ = _runner(tmp_path, guard=True)
    r([T + "Plasma metanephrines", T + "mystery assay", T + "Plasma metanephrines"])  # grounded, synthetic, repeat
    r([Q + "Do you smoke?", Q + "What did your bone marrow biopsy show?"])          # questions earn nothing
    g = grounded_test_credit(r._collector["test_logs"], cap_n=10)
    assert g["n_new_grounded_tests"] == 1 and g["r_gather"] == pytest.approx(0.1)
    logs = [{"is_question": False, "record_grounded": True, "repeat": False, "channel_kind": "test_result"}] * 30
    assert grounded_test_credit(logs, cap_n=10)["r_gather"] == 1.0
