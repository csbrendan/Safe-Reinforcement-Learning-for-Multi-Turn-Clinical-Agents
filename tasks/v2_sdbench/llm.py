"""Callables for local OpenAI-compatible (vLLM) servers, in the shape SD-Bench expects:
`llm(messages=[...]) -> ChatCompletion`. All roles default to temperature 0 for determinism.
"""
from __future__ import annotations
from typing import Any


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
