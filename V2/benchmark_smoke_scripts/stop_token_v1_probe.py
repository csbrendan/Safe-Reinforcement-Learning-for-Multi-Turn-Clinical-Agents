#!/usr/bin/env python3
"""Does MedGemma (base / V1 adapters) keep generating after <end_of_turn> on the V1 diagnostics prompt when the
server's generation_config stops only at token 1? Read-only probe; V1 code, results and checkpoints are untouched."""
import json, sys, urllib.request, os
sys.path.insert(0, os.path.expanduser("~/rl-gains-in-clinical-tasks")); os.environ.setdefault("DX_VARIANT", "deephedge")
import tasks.diagnostics.env as env
URL = "http://127.0.0.1:8100/v1/chat/completions"
cases = env.gen_cases(5, 0.0, 100)
def go(model, msgs, stop):
    b = {"model": model, "messages": msgs, "temperature": 1.0, "max_tokens": 128, "seed": 1, "skip_special_tokens": False}
    if stop: b["stop_token_ids"] = [1, 106]
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=120))
    c = d["choices"][0]; return c["finish_reason"], d["usage"]["completion_tokens"], c["message"]["content"]
print(json.dumps({"note": "5 cases, first turn, V1 max_tokens=128, temp=1.0"}))
for model in ["medgemma4b", "plain_s1", "credit_s1"]:
    for stop in [False, True]:
        rows = [go(model, [{"role": "system", "content": env.SYS}, {"role": "user", "content": env.user_msg(c)}], stop) for c in cases]
        print(json.dumps({"model": model, "stop_ids": stop, "finish": [r[0] for r in rows], "tokens": [r[1] for r in rows], "example": rows[0][2][:260]}))
