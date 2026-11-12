"""Evaluate one arm on one split. Same Episode protocol and cached gatekeeper as training; judge and prompt are
the evaluation's own (official `v4_reasoning_with_tests` by default, a different model from the training judge).

  python -m tasks.v2_sdbench.run_eval --env-config V2/sdbench_configs/eval_local.json \
      --split dev --policy-url http://127.0.0.1:8100/v1 --policy-model base --arm-name A --out results/...jsonl

--condition official (default; SD-Bench behaviour: no question guard, request text sent as written) or guarded
(the training environment). Both official and corrected costs are written either way.

Resumable (skips (case_id, k) already in --out). Writes one JSON line per episode, including the transcript.
The sealed test split needs --unseal-test, which should be passed exactly once, after all arms are frozen.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tasks.v2_sdbench.cases import assert_not_test, load_split
from tasks.v2_sdbench.env import Services, load_config
from tasks.v2_sdbench.llm import local_chat
from tasks.v2_sdbench.runner import Timer


def run_episode(svc: Services, case: dict, policy, k: int) -> dict:
    tm = Timer()
    ep = svc.new_episode(case, timer=tm)
    ep.overflow = False
    t0 = time.time()
    n_out_tokens = 0
    while not ep.done:
        tp = time.time()
        try:
            r = policy(messages=ep.messages)
        except Exception as e:  # context overflow or server error: end the episode, record why
            ep.overflow, ep.done, ep.error = True, True, f"{type(e).__name__}: {str(e)[:300]}"
            break
        tm.add("policy", time.time() - tp)
        n_out_tokens += getattr(getattr(r, "usage", None), "completion_tokens", 0) or 0
        ep.n_truncated += r.choices[0].finish_reason == "length"
        ts = time.time()
        ep.step(r.choices[0].message.content or "")
        tm.add("env_step", time.time() - ts)
    row = svc.score(ep)
    row.update({"k": k, "secs": round(time.time() - t0, 1), "error": getattr(ep, "error", None),
                "timing": tm.snapshot(), "policy_out_tokens": n_out_tokens,
                "transcript": ep.messages, "test_logs": ep.collector.get("test_logs", [])})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "eval_val", "test"])
    ap.add_argument("--policy-url", required=True)
    ap.add_argument("--policy-model", required=True, help="served model or LoRA name")
    ap.add_argument("--arm-name", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (plumbing runs)")
    ap.add_argument("--cases", default="", help="comma-separated case ids (benchmarks)")
    ap.add_argument("--cache-dir", default="", help="override the config's cache_dir (cold-cache benchmarks)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--condition", choices=["official", "guarded"], default="official")
    ap.add_argument("--serial-gatekeeper", action="store_true", help="benchmark baseline: one request at a time")
    ap.add_argument("--no-stop-fix", action="store_true",
                    help="benchmark/reproduction only: don't send stop_token_ids (pre-fix behaviour)")
    ap.add_argument("--timeout", type=float, default=0, help="seconds; stop submitting and exit after this")
    ap.add_argument("--unseal-test", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.env_config)
    cfg["arm"], cfg["lam_gather"] = "B", 0.0  # evaluation: no gathering credit, reward fields unused
    cfg["question_guard"] = cfg["canonicalize"] = a.condition == "guarded"
    cfg["gk_parallel"] = not a.serial_gatekeeper
    if a.cache_dir:
        cfg["cache_dir"] = a.cache_dir
    cases = load_split(cfg["dataset"], a.split)
    if a.split == "test" and not a.unseal_test:
        raise PermissionError("test split is sealed; pass --unseal-test only for the final frozen evaluation")
    if a.split != "test":
        assert_not_test(cfg["dataset"], [c["case_id"] for c in cases])
    if a.cases:
        want = a.cases.split(",")
        cases = [c for c in cases if str(c["case_id"]) in want]
    if a.limit:
        cases = cases[: a.limit]
    tok = None
    if cfg.get("log_phi"):  # audit only: phi is logged, never used as an eval metric
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["reader"]["tokenizer"])
    svc = Services(cfg, tokenizer=tok)
    stop = {} if a.no_stop_fix else {"stop_token_ids": cfg["policy_stop_token_ids"], "skip_special_tokens": False}
    policy = local_chat(a.policy_url, a.policy_model, temperature=a.temperature, max_tokens=a.max_tokens,
                        seed=None if a.temperature > 0 else 0, extra_body=stop or None)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["case_id"], r["k"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    todo = [(c, k) for c in cases for k in range(a.k) if (str(c["case_id"]), k) not in done]
    print(f"[eval] arm={a.arm_name} split={a.split} condition={a.condition} cases={len(cases)} k={a.k} "
          f"todo={len(todo)} judge={cfg['judge']['model']}/{cfg['judge']['prompt']} "
          f"parallel_gk={cfg['gk_parallel']}", flush=True)
    deadline = time.time() + a.timeout if a.timeout else None
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(a.out, "a") as f:
        futs = {ex.submit(run_episode, svc, c, policy, k): (c["case_id"], k) for c, k in todo}
        try:
            it = as_completed(futs, timeout=(deadline - time.time()) if deadline else None)
            for fut in it:
                row = fut.result()
                _write(f, row, a)
        except TimeoutError:
            print(f"[eval] TIMEOUT after {a.timeout:.0f}s; unfinished episodes are dropped", flush=True)
            for fu in futs:
                fu.cancel()
            f.flush()
            os._exit(3)  # worker threads may be blocked in HTTP calls; don't wait for them


def _write(f, row, a):
    row.update({"arm": a.arm_name, "split": a.split, "policy_model": a.policy_model,
                "temperature": a.temperature, "condition": a.condition})
    f.write(json.dumps(row) + "\n")
    f.flush()
    print(f"[eval] case={row['case_id']} k={row['k']} score={row['judge_score']} "
          f"tests={row['n_tests']} q={row['n_questions']} grounded={row['n_grounded_tests']} "
          f"cost={row['total_cost']:.0f} cost_corr={row['total_cost_corrected']:.0f} "
          f"guarded={row['n_guarded']} qtest={row['n_q_test_results']} invalid={row['invalid']} "
          f"secs={row['secs']}", flush=True)


if __name__ == "__main__":
    main()
