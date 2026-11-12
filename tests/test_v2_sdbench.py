"""CPU tests for the V2 SD-Bench scaffolding (tasks/v2_sdbench). No servers, GPUs or restricted data.

Run with the shared venv:  ~/spark1/.venv/bin/python -m pytest -q tests/test_v2_sdbench.py
"""
from __future__ import annotations
import inspect
import json
import os
import re
from types import SimpleNamespace

import pytest

from tasks.v2_sdbench import sdb
from tasks.v2_sdbench.cache import JsonlCache, make_key
from tasks.v2_sdbench.episode import Episode, SYSTEM_PROMPT, FIRST_USER, NAG, FORCE, first_user_message
from tasks.v2_sdbench.runner import CachedTestRunner, make_config, normalize_request
from tasks.v2_sdbench.reward import format_flags, outcome_reward, OutcomeWeights, Judge
from tasks.v2_sdbench.costs import CostModel
from tasks.v2_sdbench import metrics, judge_kit
from tasks.v2_sdbench.gather_credit import Reader, findings_from_logs, dx_leak_in_requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(REPO, "checkpoints", "medgemma4b_text")


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, role="assistant",
                                                                            tool_calls=None,
                                                                            model_dump=lambda **k: {}))])


class FakeGatekeeper:
    """Answers each 'ID n: request' with a deterministic result; 'mystery' tests are synthetic."""
    model_id = "fake-gk"

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, **kw):
        self.calls += 1
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        blocks = []
        for i, name in re.findall(r"ID (\d+): (.+)", user):
            syn = "true" if "mystery" in name.lower() else "false"
            blocks.append(f'<test_result id="{i}"><result>RESULT[{name.strip()}]</result>'
                          f'<is_synthetic>{syn}</is_synthetic></test_result>')
        return _resp("\n".join(blocks))


CASE = {"case_id": "c1", "case_title": "t", "case_summary": "A 39-year-old man with episodic headache.",
        "final_diagnosis": "Pheochromocytoma of the right adrenal gland", "estimated_clinician_visits": 2,
        "initial_presentation": "ip", "full_text": "full text of the case",
        "tests": [{"name": "Plasma metanephrines", "in_initial_presentation": False},
                  {"name": "CBC", "in_initial_presentation": True}]}


def _runner(tmp_path, gk=None, max_turns=4):
    gk = gk or FakeGatekeeper()
    cfg = make_config("synthetic", max_turns=max_turns)
    cache = JsonlCache(str(tmp_path / "gk.jsonl"))
    return CachedTestRunner(cfg, CASE, {}, gk, cache, "synthetic"), gk, cache


# ---------------------------------------------------------------- protocol fidelity
def test_prompts_match_sdbench_baseline_source():
    src = inspect.getsource(sdb.baseline.run)
    assert SYSTEM_PROMPT in src
    assert FIRST_USER.replace("\n", "\\n") in src
    assert NAG in src and FORCE in src


def test_first_user_message_uses_case_summary():
    assert CASE["case_summary"] in first_user_message(CASE)


# ---------------------------------------------------------------- runner
def test_runner_is_cached_and_deterministic(tmp_path):
    r1, gk, cache = _runner(tmp_path)
    out = r1(["Test to order: Plasma metanephrines", "Test to order: mystery panel"])
    assert [x["results"] for x in out.test_results] == ["RESULT[Test to order: Plasma metanephrines]",
                                                        "RESULT[Test to order: mystery panel]"]
    assert gk.calls == 2
    logs = r1._collector["test_logs"]
    assert logs[0]["record_grounded"] and not logs[1]["record_grounded"]
    # same request again (different case/spacing) -> cache hit and flagged as repeat
    r1(["Test to order:   PLASMA metanephrines "])
    assert gk.calls == 2 and r1._collector["test_logs"][-1]["repeat"]
    # a fresh runner reloading the cache file makes no calls
    r2, gk2, _ = _runner(tmp_path)
    r2(["Test to order: plasma metanephrines"])
    assert gk2.calls == 0
    assert r1.num_test_rounds == 2 and r1.turns_left == 2


def test_runner_unavailable_is_not_grounded(tmp_path):
    def broken(messages, **kw):
        return _resp("I cannot answer that")
    broken.model_id = "broken"
    r, _, _ = _runner(tmp_path, gk=broken)
    r(["Test to order: CBC"])
    log = r._collector["test_logs"][0]
    assert log["unavailable"] and not log["record_grounded"]


def test_runner_mixed_turn_is_rejected_but_consumes_turn(tmp_path):
    r, gk, _ = _runner(tmp_path)
    out = r(["Question for the patient: pain?", "Test to order: CBC"])
    assert out.test_results == [] and out.limit_messages and r.turns_left == 3 and gk.calls == 0


# ---------------------------------------------------------------- episode
def test_episode_flow_and_forced_merge(tmp_path):
    r, _, _ = _runner(tmp_path, max_turns=2)
    ep = Episode(CASE, r)
    assert not ep.step("<test>Plasma metanephrines</test>")
    assert ep.messages[-1]["role"] == "user" and ep.messages[-1]["content"].startswith("Results:\n- ")
    assert not ep.step("I am thinking")                      # no tags -> nag, no turn used
    assert ep.messages[-1]["content"] == NAG and r.turns_left == 1
    assert not ep.step("<question>Any flushing?</question>")  # last turn -> forced request merged in
    assert ep.forced and ep.messages[-1]["content"].endswith("\n\n" + FORCE)
    roles = [m["role"] for m in ep.messages]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))  # strict alternation after system
    assert ep.step("<diagnosis>Pheochromocytoma</diagnosis>")
    rec = ep.record()
    assert rec["diagnosis"] == "Pheochromocytoma" and rec["num_test_rounds"] == 1 and rec["n_nags"] == 1


def test_episode_nag_cap_forces_end(tmp_path):
    r, _, _ = _runner(tmp_path, max_turns=2)
    ep = Episode(CASE, r, max_agent_msgs=3)
    for _ in range(3):
        ep.step("nothing")
    assert ep.forced
    assert ep.step("still nothing") and ep.diagnosis is None


# ---------------------------------------------------------------- reward and costs
def test_format_flags():
    ok = format_flags({"diagnosis": "Pheochromocytoma", "n_diagnosis_tags": 1})
    assert not ok["invalid"]
    assert format_flags({"diagnosis": "Pheo or paraganglioma", "n_diagnosis_tags": 1})["hedged"]
    assert format_flags({"diagnosis": "1. Pheo\n2. Thyroid storm", "n_diagnosis_tags": 1})["hedged"]
    assert format_flags({"diagnosis": None})["no_diagnosis"]
    assert format_flags({"diagnosis": "x", "n_diagnosis_tags": 2})["multi_diagnosis_tag"]


def test_outcome_reward_terms():
    w = OutcomeWeights(lam_cost=0.1, cost_scale=1000, fmt_penalty=0.25)
    r = outcome_reward(5, 2000.0, {"invalid": False}, w)
    assert r["r_out"] == pytest.approx(1.0 - 0.2)
    r = outcome_reward(1, 0.0, {"invalid": True}, w)
    assert r["r_out"] == pytest.approx(-0.25)


def test_judge_wrapper_and_cache(tmp_path):
    calls = []

    def judge_llm(messages, **kw):
        calls.append(1)
        return _resp("**REASONING**: matches. **SCORE**: 5")
    judge_llm.model_id = "fake-judge"
    j = Judge(judge_llm, JsonlCache(str(tmp_path / "j.jsonl")), "synthetic", "v3_reasoning")
    assert j.score(CASE, "Pheochromocytoma")["score"] == 5
    assert j.score(CASE, "Pheochromocytoma ")["score"] == 5 and len(calls) == 1
    assert j.score(CASE, None)["score"] == 1 and len(calls) == 1


def test_cost_model(tmp_path):
    def price_llm(messages, **kw):
        return _resp("Test: x\nCost: $250\nExplanation: y")
    price_llm.model_id = "fake-price"
    cm = CostModel(price_llm, JsonlCache(str(tmp_path / "p.jsonl")))
    rec = {"test_logs": [{"test_name": "Test to order: CBC", "result": "ok"},
                         {"test_name": "Test to order: junk", "result": "Results not available for junk"}],
           "requested_tests_only": ["Test to order: CBC", "Test to order: CBC", "Test to order: junk"],
           "num_test_rounds": 2}
    c = cm.episode_cost(rec)
    assert c["tests_cost"] == 250 and c["visits_cost"] == 900 and c["n_priced_tests"] == 1
    assert cm.workup_cost(CASE)["total_cost"] == 250 + 600


# ---------------------------------------------------------------- gathering credit
def test_findings_exclude_request_wording_and_junk():
    logs = [{"test_name": "Test to order: biopsy for pheochromocytoma", "result": "Adrenal mass 4 cm"},
            {"test_name": "Test to order: Biopsy for pheochromocytoma", "result": "Adrenal mass 4 cm"},
            {"test_name": "Test to order: x", "result": "Results not available for x", "unavailable": True}]
    f = findings_from_logs(logs)
    assert f == ["Adrenal mass 4 cm"]
    assert dx_leak_in_requests(logs, CASE["final_diagnosis"]) > 0


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer not present")
def test_reader_credit_alignment(tmp_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    seen = []

    def score_fn(prefix, target):
        seen.append((len(prefix), list(target)))
        has_findings = "Findings" in tok.decode(prefix)
        return [-0.5 if has_findings else -2.0] * len(target)
    rd = Reader("unused", "base", tok, JsonlCache(str(tmp_path / "phi.jsonl")), score_fn=score_fn)
    logs = [{"test_name": "Test to order: metanephrines", "result": "Plasma normetanephrine 12x ULN"}]
    c = rd.credit(CASE, logs, cap=1.0)
    assert c["phi0"] == pytest.approx(-2.0) and c["phiT"] == pytest.approx(-0.5)
    assert c["r_gather"] == pytest.approx(1.0)  # 1.5 clipped to cap
    assert seen[0][1] == tok(CASE["final_diagnosis"], add_special_tokens=False)["input_ids"]
    rd.credit(CASE, logs)  # cached: no new reader calls
    assert len(seen) == 2


# ---------------------------------------------------------------- token masking for GRPO
@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="tokenizer not present")
def test_env_mask_covers_only_agent_tokens(tmp_path):
    from transformers import AutoTokenizer
    from framework.rl.grpo_disagg import _ids, _env_mask
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    r, _, _ = _runner(tmp_path, max_turns=2)
    ep = Episode(CASE, r)
    agent = ["<test>Plasma metanephrines</test>", "<question>Any flushing?</question>",
             "<diagnosis>Pheochromocytoma</diagnosis>"]
    for a in agent:
        ep.step(a)
    pid = _ids(tok, ep.messages[:2], True)
    full = _ids(tok, ep.messages, False)
    comp = full[len(pid):]
    mask = _env_mask(tok, ep.messages, len(pid), len(comp))
    trained = tok.decode([t for t, m in zip(comp, mask) if m])
    for a in agent:
        assert a in trained
    assert "RESULT[" not in trained and FORCE not in trained


# ---------------------------------------------------------------- metrics and judge kit
def test_metrics_paired_and_mcnemar():
    a = [{"case_id": str(i), "passed": i < 5, "judge_score": 4, "n_tests": 1, "n_grounded_tests": 1,
          "total_cost": 100, "workup_cost": 100} for i in range(20)]
    b = [{**r, "passed": int(r["case_id"]) < 12} for r in a]
    p = metrics.paired(a, b, "correct")
    assert p["n"] == 20 and p["mean_diff"] == pytest.approx(7 / 20) and p["b_only"] == 7 and p["a_only"] == 0
    assert p["mcnemar_p"] < 0.05
    m = metrics.episode_metrics({"case_id": "x", "passed": False, "n_tests": 0, "diagnosis": "d",
                                 "total_cost": 900, "workup_cost": 500})
    assert m["premature"] == 1 and m["early_commit"] == 1 and m["over_testing"] == 1


def test_judge_kit(tmp_path):
    rows = [{"case_id": "c1", "diagnosis": "Pheo", "judge_score": s} for s in (1, 2, 3, 4, 5)]
    key = judge_kit.export_pairs(rows, {"c1": CASE}, str(tmp_path / "pairs.csv"), n=5)
    txt = open(tmp_path / "pairs.csv").read()
    assert len(key) == 5 and "judge_score" not in txt.splitlines()[0]
    probes = judge_kit.adversarial_probes(CASE)
    assert {p["must"] for p in probes} >= {"pass", "fail"}
    ag = judge_kit.agreement([True, True, False, False], [True, False, False, False])
    assert ag["false_pass_rate"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- splits and GRPO prompt building
def test_synthetic_train_prompts_map_back_to_cases():
    from tasks.v2_sdbench import env
    rows = env.build_train_prompts(SimpleNamespace(env_config=None))
    assert len(rows) == 5
    for row in rows:
        assert row["prompt"][0]["content"] == SYSTEM_PROMPT
        assert row["prompt"][1]["content"] in env._CASE_BY_USER


def test_split_guards(tmp_path, monkeypatch):
    from tasks.v2_sdbench import cases as C
    fake = {"train": [{"case_id": f"t{i}", "x": i} for i in range(5)],
            "validation": [{"case_id": f"v{i}", "x": i} for i in range(10)],
            "test": [{"case_id": f"s{i}", "x": i} for i in range(3)]}
    monkeypatch.setattr(C, "official_sets", lambda d: fake)
    lookup = {c["case_id"]: c for v in fake.values() for c in v}
    monkeypatch.setattr(C, "get_case", lambda d, i: lookup[i])
    s = C.freeze_splits("fakeds", n_dev=4, out_dir=str(tmp_path))
    assert len(s["splits"]["dev"]) == 4 and len(s["splits"]["eval_val"]) == 6
    with pytest.raises(FileExistsError):
        C.freeze_splits("fakeds", n_dev=4, out_dir=str(tmp_path))
    with pytest.raises(PermissionError):
        C.assert_not_test("fakeds", ["t0", "s1"], out_dir=str(tmp_path))
    assert len(C.load_split("fakeds", "dev", out_dir=str(tmp_path))) == 4
    lookup["t0"]["x"] = 99  # silently changed case -> detected
    with pytest.raises(ValueError):
        C.load_split("fakeds", "train", out_dir=str(tmp_path))


def test_cache_tolerates_torn_line(tmp_path):
    p = tmp_path / "c.jsonl"
    c = JsonlCache(str(p))
    c.put(make_key("a"), 1)
    with open(p, "a") as f:
        f.write('{"k": "broken')
    assert JsonlCache(str(p)).get(make_key("a")) == 1
