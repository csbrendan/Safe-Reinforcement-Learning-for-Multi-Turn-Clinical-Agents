"""CPU tests for the SkyRL wrapper (tasks/v2_sdbench/skyrl_env.py). No SkyRL install, servers, GPUs or paid calls:
real Services/Episode with the fake gatekeeper; scoring is stubbed (Services.score is tested elsewhere).

Run:  ~/spark1/.venv/bin/python -m pytest -q tests/test_v2_sdbench_skyrl.py
"""
from __future__ import annotations
import json
import os

import pytest

from tasks.v2_sdbench import skyrl_env as S
from tasks.v2_sdbench.cases import load_split
from tasks.v2_sdbench.env import Services, load_config
from tasks.v2_sdbench.episode import Episode, FORCE, NAG
from tests.test_v2_sdbench import FakeGatekeeper, TOKENIZER_DIR

REPLIES = ["<test>CBC</test>\n<test>Basic metabolic panel</test>",
           "<question>Any recent travel?</question>",
           "<diagnosis>Community-acquired pneumonia</diagnosis>"]


def _svc(tmp_path, **over):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"cache_dir": str(tmp_path / "cache"), "max_turns": 4, **over}))
    svc = Services(load_config(str(cfg_path)))
    svc.gk_llm = FakeGatekeeper()
    scored = []

    def score(ep):
        scored.append(ep)
        return {"reward": 0.5, "r_out": 0.5, "judge_score": 3, "passed": True, "overflow": ep.overflow,
                "forced": ep.forced, "n_agent_msgs": ep.n_agent_msgs}
    svc.score = score
    return svc, scored


def _row(svc, i=0):
    return S.build_rows(svc.cfg)[i]


def _run(env, prompt, replies):
    """Mimic SkyRL's agent loop: chat = init(prompt) + [assistant, *observations] per step."""
    chat, _ = env.init(prompt)
    chat = list(chat)
    for r in replies:
        out = env.step(r)
        chat.append({"role": "assistant", "content": r})
        chat.extend(out["observations"])
        if out["done"]:
            return chat, out
    raise AssertionError("episode did not finish")


@pytest.fixture(autouse=True)
def _reset():
    yield
    S.set_services(None)


def test_build_rows_synthetic(tmp_path):
    svc, _ = _svc(tmp_path)
    rows = S.build_rows(svc.cfg)
    assert len(rows) == len(load_split("synthetic", "train")) and rows
    assert all(r["env_class"] == "sdbench" and r["dataset"] == "synthetic" for r in rows)
    assert len({r["case_id"] for r in rows}) == len(rows)


def test_transcript_and_reward_match_direct_episode(tmp_path):
    svc, scored = _svc(tmp_path)
    S.set_services(svc)
    row = _row(svc)
    env = S.SDBenchEnv({}, {"case_id": row["case_id"], "dataset": "synthetic"})
    chat, out = _run(env, row["prompt"], REPLIES)
    assert out["reward"] == 0.5 and out["observations"] == [] and len(scored) == 1
    assert chat == env.ep.messages  # what SkyRL trains on == what the Episode (and the scorer) saw
    # The same replies through a bare Episode (the TRL path) give the same transcript.
    ep = svc.new_episode(load_split("synthetic", "train")[0])
    for r in REPLIES:
        ep.step(r)
    assert ep.messages == chat
    assert env.get_metrics()["reward"] == 0.5


def test_forced_diagnosis_path(tmp_path):
    svc, scored = _svc(tmp_path, max_turns=1)
    S.set_services(svc)
    row = _row(svc)
    env = S.SDBenchEnv({}, {"case_id": row["case_id"]})
    chat, out = _run(env, row["prompt"], ["no tags", "still none", "<diagnosis>X</diagnosis>"])
    assert out["done"] and scored[0].forced
    assert NAG in chat[3]["content"] and FORCE in chat[-2]["content"]


@pytest.mark.skipif(not os.path.isdir(TOKENIZER_DIR), reason="MedGemma tokenizer not present")
def test_overflow_is_scored_like_trl_loop(tmp_path):
    from transformers import AutoTokenizer
    svc, scored = _svc(tmp_path, max_context_tokens=500, turn_max_tokens=400)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    S.set_services(svc, tokenizer=tok)
    row = _row(svc)
    env = S.SDBenchEnv({}, {"case_id": row["case_id"]})
    env.init(row["prompt"])
    out = env.step(REPLIES[0])
    assert out["done"] and scored and scored[0].overflow


def test_prompt_mismatch_and_bad_rows_refused(tmp_path):
    svc, _ = _svc(tmp_path)
    S.set_services(svc)
    row = _row(svc)
    with pytest.raises(ValueError):
        S.SDBenchEnv({}, {})
    with pytest.raises(ValueError):
        S.SDBenchEnv({}, {"case_id": row["case_id"], "dataset": "mimic"})
    env = S.SDBenchEnv({}, {"case_id": row["case_id"]})
    bad = [dict(m) for m in row["prompt"]]
    bad[1]["content"] += " (edited)"
    with pytest.raises(ValueError):
        env.init(bad)


def test_build_data_cli(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"cache_dir": str(tmp_path / "cache")}))
    out = tmp_path / "rows.jsonl"
    monkeypatch.setattr("sys.argv", ["x", "build-data", "--env-config", str(cfg), "--out", str(out)])
    S.main()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert rows and rows[0]["env_class"] == "sdbench"
