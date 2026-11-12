"""Azure auxiliary roles: official call shape, retries, credential handling and the MIMIC data-policy guard.
Offline: a fake client stands in for Azure."""
from __future__ import annotations
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from tasks.v2_sdbench import llm
from tasks.v2_sdbench.env import check_data_policy, load_config


class FakeClient:
    def __init__(self, fail_first: int = 0, exc=None):
        self.calls, self.fail_first, self.exc = [], fail_first, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_first:
            raise self.exc
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=30))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=usage)


def _rate_limit():
    req = httpx.Request("POST", "https://x.openai.azure.com")
    return openai.RateLimitError("429", response=httpx.Response(429, request=req, headers={"retry-after-ms": "1"}),
                                 body=None)


def test_azure_call_is_official_shape():
    fc = FakeClient()
    call = llm.azure_chat("gpt-5", {}, client=fc)
    call(messages=[{"role": "user", "content": "hi"}])
    kw = fc.calls[0]
    assert kw["model"] == "gpt-5" and kw["stream"] is False
    assert not ({"temperature", "max_tokens", "seed", "extra_body"} & kw.keys())
    assert call.model_id == "azure:gpt-5"
    assert call.usage == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 40, "reasoning_tokens": 30,
                          "retries": 0}


@pytest.mark.parametrize("bad", ["temperature", "max_tokens", "seed"])
def test_azure_rejects_non_official_args(bad):
    with pytest.raises(ValueError):
        llm.azure_chat("gpt-5", {bad: 0}, client=FakeClient())


def test_azure_retries_rate_limits_then_succeeds():
    fc = FakeClient(fail_first=2, exc=_rate_limit())
    call = llm.azure_chat("gpt-5", {}, client=fc, max_retries=3)
    call(messages=[])
    assert len(fc.calls) == 3 and call.usage["retries"] == 2


def test_azure_gives_up_after_max_retries():
    fc = FakeClient(fail_first=10, exc=_rate_limit())
    with pytest.raises(openai.RateLimitError):
        llm.azure_chat("gpt-5", {}, client=fc, max_retries=1)(messages=[])
    assert len(fc.calls) == 2


def test_azure_does_not_retry_bad_requests():
    req = httpx.Request("POST", "https://x")
    exc = openai.BadRequestError("400", response=httpx.Response(400, request=req), body=None)
    fc = FakeClient(fail_first=10, exc=exc)
    with pytest.raises(openai.BadRequestError):
        llm.azure_chat("gpt-5", {}, client=fc)(messages=[])
    assert len(fc.calls) == 1


def test_env_file_read_not_exported(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    f = tmp_path / "azure.env"
    f.write_text("# c\nexport AZURE_OPENAI_API_KEY='k1'\nAZURE_OPENAI_ENDPOINT=https://r.openai.azure.com/\n")
    s = llm.azure_settings(str(f))
    assert s == {"api_key": "k1", "endpoint": "https://r.openai.azure.com/", "api_version": llm.AZURE_API_VERSION}
    import os
    assert "AZURE_OPENAI_API_KEY" not in os.environ


def test_missing_credentials_message(tmp_path, monkeypatch):
    for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        llm.azure_settings(str(tmp_path / "none.env"))


def _cfg(tmp_path, **over):
    base = json.load(open("V2/sdbench_configs/azure_synthetic.json"))
    base.update(over)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(base))
    return str(p)


def test_azure_config_loads_for_synthetic(tmp_path):
    cfg = load_config(_cfg(tmp_path))
    assert cfg["judge"]["prompt"] == "v4_reasoning_with_tests"
    assert cfg["judge"]["deployment"] == "gpt-5" and cfg["costing"]["deployment"] == "gpt-5-mini"
    assert cfg["question_guard"] and cfg["cost_mode"] == "corrected"


def test_mimic_to_azure_refused_without_opt_out(tmp_path):
    with pytest.raises(PermissionError, match="human review"):
        load_config(_cfg(tmp_path, dataset="mimic"))


def test_mimic_to_azure_allowed_with_opt_out(tmp_path):
    cfg = load_config(_cfg(tmp_path, dataset="mimic", azure_mimic_human_review_opt_out=True))
    check_data_policy(cfg)


def test_mimic_pricer_only_on_azure_is_allowed(tmp_path):
    local = {"provider": "local", "url": "http://x/v1", "model": "m"}
    load_config(_cfg(tmp_path, dataset="mimic", gatekeeper=local, judge={**local, "prompt": "v4_reasoning_with_tests"}))


def test_make_aux_llm_dispatch():
    assert llm.make_aux_llm({"provider": "local", "url": "http://x/v1", "model": "m"}).model_id == "m@http://x/v1"
    with pytest.raises(ValueError):
        llm.make_aux_llm({"provider": "openrouter"})
