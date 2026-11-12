"""Gathering credit for arm C: does the evidence the agent revealed make the correct diagnosis more evident?

A frozen reader (base MedGemma 4B, no LoRA) scores

    phi(S) = mean log P(reference diagnosis tokens | case summary + findings S)

where S holds only what the environment returned, never the agent's own text. Then

    R_gather = clip(phi(S_final) - phi(S_0), 0, cap)

with S_0 = the case summary the agent starts from (no findings).

Anti-gaming choices:
* `findings_mode="results_only"` (default) passes result texts without the agent's request wording, so naming a
  guessed diagnosis inside a test request can't raise phi. "name_and_result" is available for the pre-check
  comparison only.
* Unavailable and rejected results are dropped. Repeats are deduplicated.
* `dx_leak_in_requests` audits whether request wording contains reference-diagnosis content words.

This is a training signal only. Evaluation never uses it.
"""
from __future__ import annotations
import re
from typing import Callable

from tasks.v2_sdbench.cache import JsonlCache, make_key
from tasks.v2_sdbench.runner import is_rejected, normalize_request

READER_INSTR = ("You are reading a clinical case. Based only on the information below, state the single most "
                "likely final diagnosis.")
_STOP = set("a an the of with and or in on to for by from due secondary primary acute chronic left right "
            "syndrome disease disorder type".split())


def findings_from_logs(test_logs: list, mode: str = "results_only") -> list[str]:
    out, seen = [], set()
    for l in test_logs:
        if l.get("unavailable") or is_rejected(l.get("result", "")):
            continue
        k = normalize_request(l["test_name"])
        if k in seen:
            continue
        seen.add(k)
        res = l.get("result", "").strip()
        if mode == "results_only":
            out.append(res)
        else:
            out.append(f"{l['test_name']}: {res}")
    return out


def reader_prompt(case_summary: str, findings: list[str]) -> str:
    body = f"{READER_INSTR}\n\nCase presentation:\n{case_summary}"
    if findings:
        body += "\n\nFindings:\n" + "\n".join(f"- {f}" for f in findings)
    return f"<start_of_turn>user\n{body}<end_of_turn>\n<start_of_turn>model\nFinal diagnosis: "


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w not in _STOP}


def dx_leak_in_requests(test_logs: list, reference: str) -> float:
    """Fraction of requests whose wording contains a content word of the reference diagnosis."""
    ref = content_words(reference)
    if not test_logs or not ref:
        return 0.0
    hits = sum(1 for l in test_logs if content_words(l["test_name"]) & ref)
    return hits / len(test_logs)


class Reader:
    """phi via vLLM /v1/completions with prompt_logprobs on the base model (served alongside the LoRA)."""

    def __init__(self, base_url: str, model: str, tokenizer, cache: JsonlCache, max_prompt_tokens: int = 7000,
                 findings_mode: str = "results_only", score_fn: Callable | None = None):
        self.base_url, self.model, self.tok, self.cache = base_url, model, tokenizer, cache
        self.max_prompt_tokens = max_prompt_tokens
        self.findings_mode = findings_mode
        self._score_fn = score_fn  # test hook: (prefix_ids, target_ids) -> list[float]
        self._client = None

    def _token_logprobs(self, prefix_ids: list[int], target_ids: list[int]) -> list[float]:
        if self._score_fn is not None:
            return self._score_fn(prefix_ids, target_ids)
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="sk-local", timeout=600)
        ids = prefix_ids + target_ids
        r = self._client.completions.create(model=self.model, prompt=ids, max_tokens=1, temperature=0.0,
                                            extra_body={"prompt_logprobs": 0})
        plp = r.choices[0].prompt_logprobs  # list aligned with prompt tokens; entry 0 is None
        out = []
        for pos in range(len(prefix_ids), len(ids)):
            entry = plp[pos]
            tid = str(ids[pos])
            out.append(float(entry[tid]["logprob"] if isinstance(entry[tid], dict) else entry[tid]))
        return out

    def phi(self, case: dict, findings: list[str]) -> float:
        reference = case["final_diagnosis"].strip()
        target = self.tok(reference, add_special_tokens=False)["input_ids"]
        kept = list(findings)
        while True:
            prefix = self.tok(reader_prompt(case.get("case_summary", ""), kept), add_special_tokens=False)["input_ids"]
            if len(prefix) + len(target) <= self.max_prompt_tokens or not kept:
                break
            kept = kept[:-1]  # drop the latest findings until the prompt fits the reader's context
        prefix = [self.tok.bos_token_id] + prefix if self.tok.bos_token_id is not None else prefix
        key = make_key("phi", str(case["case_id"]), self.model, prefix, target)
        lps = self.cache.get_or_compute(key, lambda: self._token_logprobs(prefix, target))
        return sum(lps) / max(len(lps), 1)

    def credit(self, case: dict, test_logs: list, cap: float = 2.0) -> dict:
        f = findings_from_logs(test_logs, self.findings_mode)
        p0 = self.phi(case, [])
        pT = self.phi(case, f) if f else p0
        return {"phi0": p0, "phiT": pT, "delta_phi": pT - p0, "r_gather": min(max(pT - p0, 0.0), cap),
                "n_findings": len(f), "dx_leak_in_requests": dx_leak_in_requests(test_logs, case["final_diagnosis"])}
