"""Summarize benchmark runs: wall time, component times, requests and validity counts per episode."""
import json
import sys

COLS = ["case_id", "secs", "policy", "gk_turn", "gk_llm_calls", "judge", "pricing", "policy_out_tokens", "n_rounds",
        "n_requests", "n_repeat", "n_runon", "n_truncated", "judge_score", "invalid", "forced",
        "total_cost", "total_cost_corrected", "n_q_test_results", "n_guarded", "n_dx_leak_q"]


def rows(path):
    for line in open(path):
        r = json.loads(line)
        t = r.get("timing", {"secs": {}, "calls": {}})
        yield {"case_id": r["case_id"], "secs": r["secs"], "policy": t["secs"].get("policy", 0),
               "gk_turn": t["secs"].get("gatekeeper_turn", 0), "gk_llm_calls": t["calls"].get("gatekeeper_llm", 0),
               "judge": t["secs"].get("judge", 0), "pricing": t["secs"].get("pricing", 0),
               "policy_out_tokens": r.get("policy_out_tokens"), "n_rounds": r["n_rounds"],
               "n_requests": len(r["test_logs"]), "n_repeat": r["n_repeat"], "n_runon": r.get("n_runon"),
               "n_truncated": r.get("n_truncated"), "judge_score": r["judge_score"], "invalid": r["invalid"],
               "forced": r["forced"], "total_cost": r["total_cost"],
               "total_cost_corrected": r.get("total_cost_corrected"), "n_q_test_results": r.get("n_q_test_results"),
               "n_guarded": r.get("n_guarded"), "n_dx_leak_q": r.get("n_dx_leak_q")}


for p in sys.argv[1:]:
    print(f"== {p}")
    print("\t".join(COLS))
    for r in sorted(rows(p), key=lambda r: r["case_id"]):
        print("\t".join(f"{r[c]:.0f}" if isinstance(r[c], float) else str(r[c]) for c in COLS))
