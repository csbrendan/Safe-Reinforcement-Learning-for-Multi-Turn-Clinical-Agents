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
                          "retries": 0, "by_deployment": {"gpt-5": 1}}


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


def test_overflow_moves_to_second_pool_after_429(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    fc = FakeClient(fail_first=1, exc=_rate_limit())
    call = llm.azure_chat("gpt-5", {}, client=fc, max_retries=3, overflow=["gpt-5-datazone"])
    call(messages=[])
    assert [c["model"] for c in fc.calls] == ["gpt-5", "gpt-5-datazone"]
    assert call.model_id == "azure:gpt-5"  # caches unchanged
    assert call.usage["by_deployment"] == {"gpt-5": 0, "gpt-5-datazone": 1}


def test_overflow_first_attempt_always_primary_and_rotates_back(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    fc = FakeClient(fail_first=2, exc=_rate_limit())
    call = llm.azure_chat("gpt-5", {}, client=fc, max_retries=3, overflow=["gpt-5-datazone"])
    call(messages=[])
    assert [c["model"] for c in fc.calls] == ["gpt-5", "gpt-5-datazone", "gpt-5"]
    call(messages=[])
    assert fc.calls[-1]["model"] == "gpt-5"


def test_overflow_deployment_charged_at_its_own_price(tmp_path):
    m = llm.SpendMeter(100, 100, None)
    fc = FakeClient(fail_first=1, exc=_rate_limit())
    llm.azure_chat("gpt-5", {}, client=fc, max_retries=1, meter=m, overflow=["gpt-5-datazone"])(messages=[])
    assert m.spent == pytest.approx((100 * 1.375 + 40 * 11.0) / 1e6)


def test_unpriced_overflow_refused_at_build():
    with pytest.raises(llm.SpendCapExceeded):
        llm.azure_chat("gpt-5", {}, client=FakeClient(), overflow=["gpt-5-alias"])


def test_make_aux_llm_passes_overflow():
    import inspect
    assert "overflow" in inspect.signature(llm.azure_chat).parameters
    cfg = load_config("V2/sdbench_configs/azure_synthetic.json")
    assert cfg["judge"]["overflow_deployments"] == ["gpt-5-datazone"]
    assert cfg["gatekeeper"]["overflow_deployments"] == ["gpt-5-datazone"]


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


# ---------------------------------------------------------------- spend cap
@pytest.fixture(autouse=True)
def _isolated_meter(tmp_path, monkeypatch):
    """Every test gets its own ledger, so no test touches the real ~/.cache ledger."""
    monkeypatch.setenv("V2_AZURE_SPEND_LEDGER", str(tmp_path / "ledger.jsonl"))
    llm.reset_spend_meter()
    yield
    llm.reset_spend_meter()


def _usage(prompt=1_000_000, cached=0, out=0):
    return SimpleNamespace(prompt_tokens=prompt, completion_tokens=out,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


def test_call_cost_uses_list_prices_and_cached_rate():
    assert llm.call_cost("gpt-5", _usage(prompt=1_000_000)) == pytest.approx(1.25)
    assert llm.call_cost("gpt-5", _usage(prompt=1_000_000, cached=1_000_000)) == pytest.approx(0.125)
    assert llm.call_cost("gpt-5", _usage(prompt=0, out=1_000_000)) == pytest.approx(10.0)
    assert llm.call_cost("gpt-5-mini", _usage(prompt=1_000_000, out=1_000_000)) == pytest.approx(2.25)


def test_unpriced_deployment_is_refused_before_any_call():
    fc = FakeClient()
    with pytest.raises(llm.SpendCapExceeded):
        llm.azure_chat("gpt-6-unknown", {}, client=fc)
    assert fc.calls == []


def test_process_cap_stops_calls(tmp_path):
    class Big(FakeClient):
        def create(self, **kw):
            self.calls.append(kw)
            return SimpleNamespace(choices=[], usage=_usage(prompt=1_000_000))  # $1.25 per call on gpt-5
    meter = llm.SpendMeter(cap_usd=2.0, daily_cap_usd=100.0, ledger=str(tmp_path / "l.jsonl"))
    fc = Big()
    call = llm.azure_chat("gpt-5", {}, client=fc, meter=meter)
    call(messages=[]); call(messages=[])          # $2.50: the second call was allowed at $1.25 < $2
    with pytest.raises(llm.SpendCapExceeded):
        call(messages=[])
    assert len(fc.calls) == 2 and meter.summary()["spent_usd"] == pytest.approx(2.5)


def test_daily_cap_counts_other_processes_via_ledger(tmp_path):
    led = tmp_path / "l.jsonl"
    day = llm.SpendMeter._day()
    led.write_text(json.dumps({"day": day, "usd": 9.5}) + "\n" + json.dumps({"day": "2000-01-01", "usd": 99}) + "\n")
    meter = llm.SpendMeter(cap_usd=100.0, daily_cap_usd=10.0, ledger=str(led))
    assert meter.others_today == pytest.approx(9.5)
    class Big(FakeClient):
        def create(self, **kw):
            self.calls.append(kw)
            return SimpleNamespace(choices=[], usage=_usage(prompt=1_000_000))
    call = llm.azure_chat("gpt-5", {}, client=Big(), meter=meter)
    call(messages=[])
    with pytest.raises(llm.SpendCapExceeded):
        call(messages=[])
    rows = [json.loads(l) for l in led.read_text().splitlines()]
    assert rows[-1]["deployment"] == "gpt-5" and rows[-1]["usd"] == pytest.approx(1.25)


def test_default_meter_reads_env_caps(monkeypatch):
    monkeypatch.setenv("V2_AZURE_SPEND_CAP_USD", "0")
    llm.reset_spend_meter()
    with pytest.raises(llm.SpendCapExceeded):
        llm.azure_chat("gpt-5", {}, client=FakeClient())
