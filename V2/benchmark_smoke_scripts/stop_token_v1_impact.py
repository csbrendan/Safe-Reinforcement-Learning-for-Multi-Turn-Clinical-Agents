#!/usr/bin/env python3
"""Bounded, read-only check: do V1 diagnostics behaviour metrics change if turns stop at <end_of_turn>?
Mirrors framework/gates/support_probe.py rollouts through the vLLM chat API. NOT a re-evaluation of V1: small n, one
adapter seed. KNOWN FLAW: with k=1 every case shares a seed at each turn, so samples are correlated; superseded by stop_token_v1_support_probe.sh. Writes only to the path passed as an argument."""
import json, os, sys, statistics as st, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.expanduser("~/rl-gains-in-clinical-tasks"))
os.environ.setdefault("DX_VARIANT", "deephedge")
import tasks.diagnostics.env_rv as env  # V1 main diagnostics environment (DX_VARIANT=deephedge)
from framework.parsing import parse_action
URL = "http://127.0.0.1:8100/v1/chat/completions"; MAX_TURNS = 12
def call(model, msgs, stop, seed):
    b = {"model": model, "messages": msgs, "temperature": 1.0, "top_p": 0.95, "max_tokens": 200, "seed": seed}
    if stop: b["stop_token_ids"] = [1, 106]
    d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json"}), timeout=300))
    return d["choices"][0]["message"]["content"] or "", d["usage"]["completion_tokens"]
def episode(arg):
    model, stop, c, seed = arg
    msgs = [{"role": "system", "content": env.SYS}, {"role": "user", "content": env.user_msg(c)}]; asks = []; toks = []; dec = None
    for t in range(MAX_TURNS):
        txt, n = call(model, msgs, stop, seed * 100 + t); toks.append(n); msgs.append({"role": "assistant", "content": txt})
        kind, val = parse_action(txt); canon = env.resolve_ask(val) if kind == "ask" else None
        if kind == "commit": dec = val[0] if isinstance(val, tuple) else val; break
        if kind == "ask" and canon: asks.append(canon); msgs.append({"role": "user", "content": env.answer(c, canon)})
        else: break  # support_probe ends the rollout on an invalid action
    r, oracle_grade, valid, _ = env.score(c, asks, dec)
    return {"model": model, "stop": stop, "case": c["case_id"], "risk": c["true_risk"], "asks": len(asks), "commit": dec is not None, "reward": r, "oracle_grade": oracle_grade, "valid_commit": valid, "tokens_per_turn": sum(toks) / len(toks)}
if __name__ == "__main__":
    cases = env.gen_cases(int(sys.argv[2]), 0.0, 100); k = int(sys.argv[3]); out = open(sys.argv[1], "w")
    for model in (sys.argv[4].split(",") if len(sys.argv) > 4 else ["medgemma4b", "plain_s1", "credit_s1"]):
        for stop in [False, True]:
            with ThreadPoolExecutor(8) as ex:
                rows = list(ex.map(episode, [(model, stop, c, s) for c in cases for s in range(k)]))
            for x in rows: out.write(json.dumps(x, default=str) + "\n")
            out.flush()
            high = [x for x in rows if x["risk"] == "high"]
            print(json.dumps({"model": model, "stop_ids": stop, "episodes": len(rows), "commit_rate": round(sum(x["commit"] for x in rows) / len(rows), 3), "mean_asks": round(st.mean(x["asks"] for x in rows), 2), "one_or_fewer_asks": round(sum(x["asks"] <= 1 for x in rows) / len(rows), 3), "mean_reward": round(st.mean(x["reward"] for x in rows), 3), "valid_commit_rate": round(sum(x["valid_commit"] for x in rows) / len(rows), 3), "high_risk_episodes": len(high), "high_risk_reward": round(st.mean(x["reward"] for x in high), 3) if high else None, "tokens_per_turn": round(st.mean(x["tokens_per_turn"] for x in rows), 1)}), flush=True)
