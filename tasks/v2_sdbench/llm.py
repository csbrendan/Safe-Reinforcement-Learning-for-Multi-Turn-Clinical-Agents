"""Callables for the auxiliary roles (gatekeeper, judge, pricer), in the shape SD-Bench expects:
`llm(messages=[...]) -> ChatCompletion`.

Two providers:
  local   OpenAI-compatible vLLM server (Qwen3-8B in the overnight work). Temperature 0, fixed seed.
  azure   Azure AI Foundry deployment (GPT-5 / GPT-5-mini). Called the way SD-Bench calls it: only the deployment
          name plus SD-Bench's own per-model defaults, no temperature or token cap (GPT-5 accepts neither
          temperature 0 nor max_tokens). Output is therefore not deterministic; the JSONL caches (first write wins)
          make a run reproducible, and arms must share them.

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
               max_concurrency: int = 16, max_retries: int = 8, timeout: float = 600.0, client=None):
    """Azure deployment as an SD-Bench llm. `model_args` are forwarded verbatim (default {} = SD-Bench official).

    A semaphore caps in-flight requests per role; 429/5xx/timeouts back off (honouring Retry-After) up to
    `max_retries`, then raise. `client` is for tests."""
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
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "retries": 0}
    lock = threading.Lock()

    def call(messages, **kw: Any):
        args: dict[str, Any] = {**margs, **kw}
        for attempt in range(max_retries + 1):
            try:
                with sem:
                    r = client.chat.completions.create(model=deployment, messages=list(messages), stream=False,
                                                       **args)
                u = getattr(r, "usage", None)
                with lock:
                    usage["calls"] += 1
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
                wait = _retry_after(e) or min(60.0, 2.0 ** attempt)
                time.sleep(wait + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    call.model_id = f"azure:{deployment}"  # type: ignore[attr-defined]
    call.usage = usage  # type: ignore[attr-defined]
    return call


def make_aux_llm(role: dict, no_think: bool = False):
    """Build the llm for one auxiliary role from its config block.
      {"provider": "local", "url": ..., "model": ..., "max_tokens": ...}
      {"provider": "azure", "deployment": "gpt-5", "model_args": {}, "max_concurrency": 16}"""
    p = role.get("provider", "local")
    if p == "azure":
        return azure_chat(role["deployment"], role.get("model_args") or {},
                          env_file=role.get("env_file", AZURE_ENV_FILE),
                          max_concurrency=role.get("max_concurrency", 16))
    if p == "local":
        return local_chat(role["url"], role["model"], max_tokens=role.get("max_tokens", 800),
                          extra_body=qwen_no_think() if no_think else None)
    raise ValueError(f"unknown provider {p!r}")
