"""Cached, versioned Claude Opus teacher client for black-box rationale distillation.

Design constraints (see docs / CLAUDE.md):
- `claude-opus-4-8` DEPRECATES the `temperature` parameter -> never pass it.
- Strict-JSON output; one retry with a "JSON only" reminder; then a logged parse-failure.
- Every response is cached to JSONL keyed on (prompt_version, case_id, state_hash[, variant]) so re-runs and
  independent arms sharing the same states don't re-call the API. The model id + prompt version are stamped on
  every record.
- The client is a pure SCORER: it never sees privileged simulator state unless the caller puts it in the
  prompt. The Phase-0 pilot deliberately passes only student-visible text.

Key is read from the environment (source ~/.secrets/anthropic.env first; see memory).
"""
from __future__ import annotations
import json, os, re, time
from pathlib import Path

MODEL = "claude-opus-4-8"

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(raw):
    """Best-effort strict-JSON extraction from a model reply. Returns dict or None."""
    if not raw:
        return None
    for cand in _FENCE.findall(raw):
        try:
            return json.loads(cand)
        except Exception:
            pass
    i, j = raw.find("{"), raw.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(raw[i:j + 1])
        except Exception:
            pass
    try:
        return json.loads(raw.strip())
    except Exception:
        return None


class TeacherClient:
    def __init__(self, cache_path, model=MODEL, prompt_version="dss_pilot_v1", max_tokens=1024):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY not set — run: source ~/.secrets/anthropic.env")
        import anthropic
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=key)
        self.model, self.prompt_version, self.max_tokens = model, prompt_version, max_tokens
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            for line in open(self.cache_path):
                r = json.loads(line)
                self.cache[r["cache_key"]] = r

    def key(self, case_id, state_hash, variant=""):
        return f"{self.prompt_version}|{case_id}|{state_hash}|{variant}"

    def label(self, system, user, case_id, state_hash, variant="", force=False):
        """Return (record, from_cache). record carries parsed JSON, raw text, and parse_status."""
        ck = self.key(case_id, state_hash, variant)
        if not force and ck in self.cache:
            return self.cache[ck], True
        obj, raw, status = self._call(system, user)
        rec = {"cache_key": ck, "prompt_version": self.prompt_version, "teacher_model": self.model,
               "case_id": case_id, "state_hash": state_hash, "variant": variant,
               "parse_status": status, "parsed": obj, "raw": raw}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.cache[ck] = rec
        return rec, False

    def _call(self, system, user, max_attempts=6):
        # Resilient: a transient API/network error (429/529/connection/VPN blip) must NEVER crash a training run.
        # Retry the API call with exponential backoff; retry JSON-parse failures with a "JSON only" reminder;
        # if everything is exhausted return None so the caller treats this rollout as unscored (within-group
        # neutral in reward.assemble) rather than propagating an exception into GRPO.
        raw, last_exc = "", None
        for attempt in range(max_attempts):
            u = user if attempt == 0 else user + "\n\nReturn STRICT JSON only — no prose, no code fence."
            try:
                msg = self.client.messages.create(          # NOTE: no temperature (deprecated for opus-4-8)
                    model=self.model, max_tokens=self.max_tokens, system=system,
                    messages=[{"role": "user", "content": u}])
                raw = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
                obj = extract_json(raw)
                if obj is not None:
                    return obj, raw, ("ok" if attempt == 0 else "ok_retry")
                # parsed None -> loop retries immediately with the reminder (no backoff)
            except Exception as e:                          # anthropic/network error -> back off and retry
                last_exc = e
                time.sleep(min(2 ** attempt, 30))
        return None, (raw or f"<api_error:{type(last_exc).__name__}>" if last_exc else raw), \
            ("api_fail" if last_exc else "parse_fail")
