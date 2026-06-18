"""Hardened parser for one structured action per assistant turn.

Returns ("ask", intent_str) | ("commit", (decision_str, action_or_None)) | ("invalid", None).
Guards against list/non-string fields a sampled policy can emit (which would otherwise crash set()/hashing
downstream). Tasks resolve the intent/decision against their own vocabularies.
"""
from __future__ import annotations
import json, re

_JSON = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(text):
    for m in _JSON.finditer(text):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if o.get("action_type") == "ask" and isinstance(o.get("intent"), str) and o.get("intent"):
            return ("ask", o["intent"])                      # intent must be a hashable string
        if o.get("action_type") == "commit" and isinstance(o.get("decision"), str):
            act = o.get("action")
            act = act if isinstance(act, str) else (None if act is None else json.dumps(act))
            return ("commit", (o.get("decision"), act))
    # <answer>{...}</answer> fallback (older single-turn format)
    m = re.search(r"<answer>\s*(\{.*\})\s*</answer>", text, re.DOTALL)
    if m:
        try:
            j = json.loads(m.group(1)) or {}
            dec = j.get("decision"); act = j.get("action")
            return ("commit", (dec if isinstance(dec, str) else None,
                               act if isinstance(act, (str, type(None))) else json.dumps(act)))
        except Exception:
            return ("commit", (None, None))
    return ("invalid", None)
