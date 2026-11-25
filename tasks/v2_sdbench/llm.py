"""Callables for the auxiliary roles (gatekeeper, judge, pricer), in the shape SD-Bench expects:
`llm(messages=[...]) -> ChatCompletion`.

Two providers:
  local   OpenAI-compatible vLLM server (Qwen3-8B in the overnight work). Temperature 0, fixed seed.
  azure   Azure AI Foundry deployment (GPT-5 / GPT-5-mini). Called the way SD-Bench calls it: only the deployment
          name plus SD-Bench's own per-model defaults, no temperature or token cap (GPT-5 accepts neither
          temperature 0 nor max_tokens). Output is therefore not deterministic; the JSONL caches (first write wins)
          make a run reproducible, and arms must share them.

Spend cap (azure only): every Azure call is priced from its returned token usage at list prices (PRICES_PER_M) and
charged to one process-wide SpendMeter shared by all roles. Before each call the meter refuses (SpendCapExceeded)
if this process has reached its cap or today's ledger total has reached the daily cap. Settings, from the
environment:
  V2_AZURE_SPEND_CAP_USD    per-process cap, default 5. Raise it deliberately, per approved run.
  V2_AZURE_DAILY_CAP_USD    cap on today's total (UTC) across processes sharing the ledger, default 25.
  V2_AZURE_SPEND_LEDGER     JSONL ledger, one line per call, default ~/.cache/v2_sdbench/azure_spend.jsonl.
A deployment with no price entry is refused (fail closed). Calls already in flight when a cap is hit still finish,
so the overshoot is at most max_concurrency calls per role. Other processes' spend is read from the ledger once,
when the meter is created, so concurrent processes can each overshoot the daily cap by their own spend; the
Azure budget alerts are the backstop.

Azure credentials are read from a file (default ~/.secrets/azure.env, KEY=VALUE lines) into this process only,
never exported to the shell: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION (optional).
"""
from __future__ import annotations
import os
import random
import threading
import time
from typing import Any

AZURE_ENV_FILE = "~/.secrets/azure.env"
AZURE_API_VERSION = "2025-01-01-preview"  # SD-Bench's default

# USD per 1M tokens (Azure Global Standard list prices, checked 2026-11-19): input, cached input, output.
# Keyed by deployment name. Deployment names are the real model name, plus a suffix for a second deployment type of
# the same model and version (e.g. "gpt-5-datazone"); never an alias for a different model.
PRICES_PER_M: dict[str, tuple[float, float, float]] = {
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.1": (1.25, 0.125, 10.00),
    # Data Zone Standard: same gpt-5 2025-08-07; assumed +10% over Global (check the Azure rate card).
    "gpt-5-datazone": (1.375, 0.1375, 11.00),
}
DEFAULT_LEDGER = "~/.cache/v2_sdbench/azure_spend.jsonl"


class SpendCapExceeded(RuntimeError):
    pass


def call_cost(deployment: str, usage) -> float:
    """USD for one response's usage object. Reasoning tokens are part of completion_tokens."""
    if deployment not in PRICES_PER_M:
        raise SpendCapExceeded(f"no price for Azure deployment {deployment!r}; add it to PRICES_PER_M first")
    p_in, p_cached, p_out = PRICES_PER_M[deployment]
    if usage is None:
        return 0.0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    d = getattr(usage, "prompt_tokens_details", None)
    cached = (getattr(d, "cached_tokens", 0) or 0) if d else 0
    out = getattr(usage, "completion_tokens", 0) or 0
    return ((prompt - cached) * p_in + cached * p_cached + out * p_out) / 1e6


class SpendMeter:
    def __init__(self, cap_usd: float, daily_cap_usd: float, ledger: str | None):
        self.cap, self.daily_cap = float(cap_usd), float(daily_cap_usd)
        self.ledger = os.path.expanduser(ledger) if ledger else None
        self.spent = 0.0        # this process, all days (per-process cap)
        self.spent_today = 0.0  # this process, current UTC day (daily cap)
        self.calls = 0
        self.lock = threading.Lock()
        self.today = self._day()
        self.others_today = self._ledger_total(self.today)

    @staticmethod
    def _day() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _roll(self) -> None:  # caller holds the lock
        d = self._day()
        if d != self.today:
            self.today, self.spent_today = d, 0.0
            self.others_today = self._ledger_total(d)

    def _ledger_total(self, day: str) -> float:
        if not self.ledger or not os.path.exists(self.ledger):
            return 0.0
        import json
        tot = 0.0
        with open(self.ledger) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("day") == day:
                    tot += float(r.get("usd", 0.0))
        return tot

    def check(self, deployment: str) -> None:
        if deployment not in PRICES_PER_M:
            raise SpendCapExceeded(f"no price for Azure deployment {deployment!r}; add it to PRICES_PER_M first")
        with self.lock:
            self._roll()
            if self.spent >= self.cap:
                raise SpendCapExceeded(f"Azure spend cap reached: ${self.spent:.2f} of ${self.cap:.2f} this process "
                                       "(V2_AZURE_SPEND_CAP_USD)")
            if self.others_today + self.spent_today >= self.daily_cap:
                raise SpendCapExceeded(f"Azure daily cap reached: ${self.others_today + self.spent_today:.2f} of "
                                       f"${self.daily_cap:.2f} today (V2_AZURE_DAILY_CAP_USD)")

    def charge(self, deployment: str, usage) -> float:
        usd = call_cost(deployment, usage)
        with self.lock:
            self._roll()
            self.spent += usd
            self.spent_today += usd
            self.calls += 1
            if self.ledger:
                import json
                os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
                with open(self.ledger, "a") as f:
                    f.write(json.dumps({"day": self.today, "t": round(time.time(), 1), "pid": os.getpid(),
                                        "deployment": deployment, "usd": round(usd, 6)}) + "\n")
        return usd

    def summary(self) -> dict:
        return {"spent_usd": round(self.spent, 4), "calls": self.calls, "cap_usd": self.cap,
                "daily_cap_usd": self.daily_cap, "others_today_usd": round(self.others_today, 4)}


_METER: SpendMeter | None = None
_METER_LOCK = threading.Lock()


def spend_meter() -> SpendMeter:
    """The process-wide meter, built from the environment on first use."""
    global _METER
    with _METER_LOCK:
        if _METER is None:
            _METER = SpendMeter(float(os.environ.get("V2_AZURE_SPEND_CAP_USD", "5")),
                                float(os.environ.get("V2_AZURE_DAILY_CAP_USD", "25")),
                                os.environ.get("V2_AZURE_SPEND_LEDGER", DEFAULT_LEDGER))
        return _METER


def reset_spend_meter(meter: SpendMeter | None = None) -> None:
    global _METER
    with _METER_LOCK:
        _METER = meter


def local_chat(base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 800,
               seed: int | None = 0, extra_body: dict | None = None, timeout: float = 600.0):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="sk-local", timeout=timeout)

    def call(messages, **kw: Any):
        args: dict[str, Any] = dict(model=model, messages=list(messages), temperature=temperature,
                                    max_tokens=max_tokens)
        if seed is not None:
            args["seed"] = seed
        if extra_body:
            args["extra_body"] = extra_body
        args.update(kw)
        return client.chat.completions.create(**args)

    call.model_id = f"{model}@{base_url}"  # type: ignore[attr-defined]
    return call


def qwen_no_think() -> dict:
    """Qwen3 chat-template switch to disable thinking (as in the smoke tests)."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def read_env_file(path: str = AZURE_ENV_FILE) -> dict[str, str]:
    """KEY=VALUE lines (optional `export `, quotes stripped). Returned, not exported."""
    out: dict[str, str] = {}
    p = os.path.expanduser(path)
    if not os.path.exists(p):
        return out
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def azure_settings(env_file: str = AZURE_ENV_FILE) -> dict[str, str]:
    """Credentials from the env file, falling back to the process environment."""
    f = read_env_file(env_file)
    get = lambda k, d=None: f.get(k) or os.environ.get(k) or d  # noqa: E731
    s = {"api_key": get("AZURE_OPENAI_API_KEY"), "endpoint": get("AZURE_OPENAI_ENDPOINT"),
         "api_version": get("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION)}
    missing = [k for k in ("api_key", "endpoint") if not s[k]]
    if missing:
        raise RuntimeError(f"Azure credentials missing ({', '.join(missing)}): put AZURE_OPENAI_API_KEY and "
                           f"AZURE_OPENAI_ENDPOINT in {env_file}")
    return s


def _is_retryable(e: Exception) -> bool:
    import openai
    return isinstance(e, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                          openai.InternalServerError))


def _retry_after(e: Exception) -> float | None:
    try:
        h = e.response.headers  # type: ignore[attr-defined]
        v = h.get("retry-after-ms")
        if v:
            return float(v) / 1000
        v = h.get("retry-after")
        return float(v) if v else None
    except Exception:
        return None


def azure_chat(deployment: str, model_args: dict | None = None, env_file: str = AZURE_ENV_FILE,
               max_concurrency: int = 16, max_retries: int = 8, timeout: float = 600.0, client=None,
               meter: SpendMeter | None = None, overflow: list[str] | tuple[str, ...] = ()):
    """Azure deployment as an SD-Bench llm. `model_args` are forwarded verbatim (default {} = SD-Bench official).

    A semaphore caps in-flight requests per role; 429/5xx/timeouts back off (honouring Retry-After) up to
    `max_retries`, then raise. Every call is checked against and charged to the spend meter (module docstring).
    `overflow`: extra deployments of the SAME model and version (e.g. "gpt-5-datazone"). The first attempt always goes
    to `deployment`; after a retryable error the next attempt moves to the next deployment in the rotation, so a 429 on
    one quota pool spills into the other. model_id stays f"azure:{deployment}", so caches are shared.
    `client` and `meter` are for tests."""
    meter = meter or spend_meter()
    rotation = [deployment, *[d for d in overflow if d != deployment]]
    for d in rotation:
        meter.check(d)  # fail closed at build time for unpriced deployments
    if client is None:
        from openai import AzureOpenAI
        s = azure_settings(env_file)
        client = AzureOpenAI(azure_endpoint=s["endpoint"], api_key=s["api_key"], api_version=s["api_version"],
                             timeout=timeout, max_retries=0)
    margs = dict(model_args or {})
    for bad in ("temperature", "max_tokens", "seed"):
        if bad in margs:
            raise ValueError(f"{bad} is not an official SD-Bench argument for Azure GPT-5 roles")
    sem = threading.BoundedSemaphore(max_concurrency)
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "retries": 0,
             "by_deployment": {d: 0 for d in rotation}}
    lock = threading.Lock()

    def call(messages, **kw: Any):
        args: dict[str, Any] = {**margs, **kw}
        for attempt in range(max_retries + 1):
            dep = rotation[attempt % len(rotation)]
            meter.check(dep)
            try:
                with sem:
                    r = client.chat.completions.create(model=dep, messages=list(messages), stream=False, **args)
                u = getattr(r, "usage", None)
                meter.charge(dep, u)
                with lock:
                    usage["calls"] += 1
                    usage["by_deployment"][dep] += 1
                    if u is not None:
                        usage["prompt_tokens"] += u.prompt_tokens or 0
                        usage["completion_tokens"] += u.completion_tokens or 0
                        d = getattr(u, "completion_tokens_details", None)
                        usage["reasoning_tokens"] += (getattr(d, "reasoning_tokens", 0) or 0) if d else 0
                return r
            except Exception as e:  # noqa: BLE001
                if attempt == max_retries or not _is_retryable(e):
                    raise
                with lock:
                    usage["retries"] += 1
                # Moving to another pool: short pause; same pool again (or one deployment): honour Retry-After.
                nxt = rotation[(attempt + 1) % len(rotation)]
                wait = 0.5 if nxt != dep else (_retry_after(e) or min(60.0, 2.0 ** attempt))
                time.sleep(wait + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    call.model_id = f"azure:{deployment}"  # type: ignore[attr-defined]
    call.usage = usage  # type: ignore[attr-defined]
    call.meter = meter  # type: ignore[attr-defined]
    return call


def make_aux_llm(role: dict, no_think: bool = False):
    """Build the llm for one auxiliary role from its config block.
      {"provider": "local", "url": ..., "model": ..., "max_tokens": ...}
      {"provider": "azure", "deployment": "gpt-5", "model_args": {}, "max_concurrency": 16}"""
    p = role.get("provider", "local")
    if p == "azure":
        return azure_chat(role["deployment"], role.get("model_args") or {},
                          env_file=role.get("env_file", AZURE_ENV_FILE),
                          max_concurrency=role.get("max_concurrency", 16),
                          overflow=role.get("overflow_deployments", ()))
    if p == "local":
        return local_chat(role["url"], role["model"], max_tokens=role.get("max_tokens", 800),
                          extra_body=qwen_no_think() if no_think else None)
    raise ValueError(f"unknown provider {p!r}")
