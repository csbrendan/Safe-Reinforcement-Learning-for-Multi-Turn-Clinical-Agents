"""Probe the leak routes that V2/sdbench_env_validation.md §3 listed as not modelled by the guard, on the 5 public
synthetic cases, through the training environment's own runner (question guard + canonicalization on).

Routes (same requests for every case; nothing case-specific, so nothing is tuned to a case):
  meta_test     non-test documents ordered through the TEST channel (final diagnosis, discharge summary, ...)
  treatment_q   treatment course / response, asked as a question
  timeline_q    what happened later / what is planned, asked as a question
  interpret_q   the clinicians' interpretation, asked as a question
For every answer: does it name the reference diagnosis (channel.mentions_dx), does it describe the post-diagnosis
course (channel.reveals_course), and what did the current guard do.

    PYTHONPATH=$PWD python V2/leak_probe/v2_leak_routes.py --config V2/sdbench_configs/smoke_synthetic.json \
        --cache-dir results/v2_sdbench/leak_probe2/cache --out results/v2_sdbench/leak_probe2/routes.jsonl
"""
from __future__ import annotations
import argparse
import json
import os

from tasks.v2_sdbench import channel
from tasks.v2_sdbench.cases import load_split
from tasks.v2_sdbench.env import Services, load_config

ROUTES = {
    "meta_test": ["Final diagnosis", "Discharge summary", "Hospital course", "Consultant's impression",
                  "Assessment and plan", "Problem list", "Pathology report final diagnosis"],
    "treatment_q": ["What treatment did you receive, and how did you respond?",
                    "Has there been any improvement since treatment was started?",
                    "What medications were you given in the hospital?"],
    "timeline_q": ["What happened later during your hospital stay?", "What is planned for tomorrow?",
                   "What procedure was performed next?"],
    "interpret_q": ["What do your doctors think is going on?", "What did the specialist tell you?",
                    "What is the significance of your symptoms?"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if cfg["dataset"] != "synthetic":
        raise SystemExit("synthetic cases only")
    cfg["cache_dir"], cfg["question_guard"], cfg["canonicalize"] = a.cache_dir, True, True
    svc = Services(cfg)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for case in load_split("synthetic", "train"):
            terms = channel.dx_key_terms(case.get("final_diagnosis", ""))
            for route, reqs in ROUTES.items():
                prefix = channel.TEST_PREFIX if route == "meta_test" else channel.QUESTION_PREFIX
                ep = svc.new_episode(case)
                ep.runner([f"{prefix} {r}" for r in reqs])
                for log in ep.collector["test_logs"]:
                    revealed = log.get("withheld_result") or log["result"]
                    row = {"case_id": str(case["case_id"]), "route": route, "request": log["test_name"],
                           "kind": log["channel_kind"], "guarded": log["guarded"],
                           "dx_terms": channel.mentions_dx(revealed, terms),
                           "dx_terms_shown": channel.mentions_dx(log["result"], terms),
                           "course": channel.reveals_course(revealed),
                           "course_shown": channel.reveals_course(log["result"]),
                           "revealed": revealed[:600], "shown": log["result"][:600]}
                    f.write(json.dumps(row) + "\n")
                    print(f"[{row['case_id']}] {route:11s} {log['test_name'][:60]:60s} kind={row['kind']:11s} "
                          f"guarded={int(row['guarded'])} dx_shown={row['dx_terms_shown']} "
                          f"course_shown={row['course_shown']}", flush=True)


if __name__ == "__main__":
    main()
