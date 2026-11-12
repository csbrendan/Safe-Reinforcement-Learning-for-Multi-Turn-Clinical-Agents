"""GRPO hooks for SD-Bench, used by framework/rl/grpo_disagg.py when `--env tasks.v2_sdbench.env`.

grpo_disagg calls:
    build_train_prompts(args) -> list of {"prompt": [system, user]}   (train split only; test refused)
    make_rollout_func(tok, args, helpers) -> rollout_func(prompts, trainer)

Settings come from a JSON file passed as `--env-config` (see V2/sdbench_configs/). Arms B and C must use the same
file except `arm` and `lam_gather` (C only). The reward module and caches are identical across arms.

Training-environment deviations from official SD-Bench (all reported; see V2/sdbench_env_validation.md):
  question_guard  test results asked for as patient questions are withheld (information-leakage fix)
  cost_mode       "corrected" charges what the agent actually obtained (channel.py, CostModel.corrected_cost)
Evaluation runs the official environment (guard off) and reports official and corrected cost side by side.

Arm C's gathering credit is "record_grounded": per new test whose result exists in the case record, capped.
It is a RELEVANCE PROXY (the record's clinicians ordered something like it), not information gain. The Φ reader
failed its pre-registered check and is not used for reward ("phi" remains selectable for audit only).
"""
from __future__ import annotations
import json
import os
import statistics as stx
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tasks.v2_sdbench.cache import JsonlCache
from tasks.v2_sdbench.cases import assert_not_test, load_split
from tasks.v2_sdbench.costs import CostModel
from tasks.v2_sdbench.episode import Episode, SYSTEM_PROMPT, first_user_message
from tasks.v2_sdbench.llm import local_chat, qwen_no_think
from tasks.v2_sdbench.reward import Judge, OutcomeWeights, format_flags, outcome_reward
from tasks.v2_sdbench.runner import CachedTestRunner, Timer, make_config

DEFAULTS: dict[str, Any] = {
    "dataset": "synthetic", "split": "train", "arm": "B", "lam_gather": 0.0, "gather_cap": 2.0,
    "max_turns": 12, "turn_max_tokens": 400, "max_context_tokens": 14000,
    "outcome": {"lam_cost": 0.0, "cost_scale": 1000.0, "fmt_penalty": 0.25},
    "gatekeeper": {"url": "http://192.168.100.2:8101/v1", "model": "local-aux", "max_tokens": 800},
    "judge": {"url": "http://192.168.100.2:8101/v1", "model": "local-aux", "prompt": "v3_reasoning",
              "max_tokens": 800},
    "costing": {"url": "http://192.168.100.2:8101/v1", "model": "local-aux", "max_tokens": 300},
    "reader": {"url": "http://192.168.100.2:8000/v1", "model": None, "findings_mode": "results_only"},
    "aux_no_think": True, "cache_dir": "results/v2_sdbench/cache", "rollout_log": "", "n_threads": 16,
    "question_guard": True, "canonicalize": True, "gk_parallel": True, "cost_mode": "corrected",
    "gather": "record_grounded", "gather_cap_n": 10,
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    if cfg["arm"] not in ("B", "C"):
        raise ValueError("arm must be B or C")
    if cfg["arm"] == "B" and cfg["lam_gather"] != 0.0:
        raise ValueError("arm B must have lam_gather 0")
    if cfg["cost_mode"] not in ("official", "corrected"):
        raise ValueError("cost_mode must be official or corrected")
    if cfg["gather"] not in ("record_grounded", "phi"):
        raise ValueError("gather must be record_grounded or phi")
    return cfg


def grounded_test_credit(test_logs: list, cap_n: int) -> dict:
    """Arm C relevance proxy: new <test> requests whose result is in the case record, capped at cap_n.
    Questions earn nothing (they're free and unlimited), nor do repeats, rejections, unavailable results or
    results the gatekeeper synthesized."""
    n = sum(1 for l in test_logs if not l.get("is_question") and l.get("record_grounded") and not l.get("repeat")
            and l.get("channel_kind") != "rejected")
    return {"n_new_grounded_tests": n, "r_gather": min(n, cap_n) / cap_n}


class Services:
    """Shared clients and caches. One instance per process."""

    def __init__(self, cfg: dict, tokenizer=None):
        cd = cfg["cache_dir"]
        os.makedirs(cd, exist_ok=True)
        eb = qwen_no_think() if cfg.get("aux_no_think") else None
        g, j, c = cfg["gatekeeper"], cfg["judge"], cfg["costing"]
        self.cfg = cfg
        self.sdb_cfg = make_config(cfg["dataset"], max_turns=cfg["max_turns"])
        self.gk_llm = local_chat(g["url"], g["model"], max_tokens=g["max_tokens"], extra_body=eb)
        self.gk_cache = JsonlCache(os.path.join(cd, "gatekeeper.jsonl"))
        self.judge = Judge(local_chat(j["url"], j["model"], max_tokens=j["max_tokens"], extra_body=eb),
                           JsonlCache(os.path.join(cd, "judge.jsonl")), cfg["dataset"], j["prompt"],
                           cfg["max_turns"])
        self.costs = CostModel(local_chat(c["url"], c["model"], max_tokens=c["max_tokens"], extra_body=eb),
                               JsonlCache(os.path.join(cd, "prices.jsonl")))
        self.timer = Timer()
        self.reader = None
        if cfg["gather"] == "phi" or cfg.get("log_phi"):
            from tasks.v2_sdbench.gather_credit import Reader
            r = cfg["reader"]
            self.reader = Reader(r["url"], r["model"], tokenizer, JsonlCache(os.path.join(cd, "phi.jsonl")),
                                 findings_mode=r.get("findings_mode", "results_only"))
        self.weights = OutcomeWeights(**cfg["outcome"])

    def new_episode(self, case: dict, timer: Timer | None = None) -> Episode:
        runner = CachedTestRunner(self.sdb_cfg, case, {}, self.gk_llm, self.gk_cache, self.cfg["dataset"],
                                  question_guard=self.cfg["question_guard"], canonicalize=self.cfg["canonicalize"],
                                  parallel=self.cfg["gk_parallel"], timer=timer or self.timer)
        return Episode(case, runner)

    def score(self, ep: Episode) -> dict:
        rec = ep.record()
        rec["overflow"] = getattr(ep, "overflow", False)
        flags = format_flags(rec)
        t0 = time.time()
        j = self.judge.score(ep.case, rec["diagnosis"], rec["test_logs"])
        t1 = time.time()
        cost = self.costs.episode_cost(rec)
        corr = self.costs.corrected_cost(rec)
        t2 = time.time()
        tm = ep.runner.timer or self.timer
        tm.add("judge", t1 - t0)
        tm.add("pricing", t2 - t1)
        used_cost = corr["total_cost_corrected"] if self.cfg["cost_mode"] == "corrected" else cost["total_cost"]
        out = outcome_reward(j["score"], used_cost, flags, self.weights)
        from tasks.v2_sdbench.gather_credit import dx_leak_in_requests
        logs = rec["test_logs"]
        g = grounded_test_credit(logs, self.cfg["gather_cap_n"])
        g["dx_leak_in_requests"] = dx_leak_in_requests(logs, ep.case["final_diagnosis"])
        if self.reader is not None:
            ph = self.reader.credit(ep.case, logs, cap=self.cfg["gather_cap"])
            tm.add("reader", time.time() - t2)
            if self.cfg["gather"] == "phi":
                g.update(ph)
            else:  # audit only
                g.update({"phi_" + k: v for k, v in ph.items() if k != "dx_leak_in_requests"})
        lam = self.cfg["lam_gather"] if self.cfg["arm"] == "C" else 0.0
        total = out["r_out"] + lam * g["r_gather"]
        q = [l for l in logs if l.get("is_question")]
        return {"case_id": rec["case_id"], "arm": self.cfg["arm"], "reward": total, **out, **g,
                "cost_mode": self.cfg["cost_mode"], "question_guard": self.cfg["question_guard"],
                "judge_score": j["score"], "passed": j["passed"], **cost, **corr, **flags,
                "n_tests": sum(1 for l in logs if not l.get("is_question")),
                "n_questions": len(q),
                "n_q_test_results": sum(1 for l in q if l.get("channel_kind") == "test_result"),
                "n_guarded": sum(1 for l in q if l.get("guarded")),
                "n_historic_tests": sum(1 for l in q if l.get("channel_kind") == "historic_test"),
                "n_canonicalized": sum(1 for l in logs if "sent" in l),
                "n_grounded": sum(1 for l in logs if l.get("record_grounded")),
                "n_grounded_tests": sum(1 for l in logs if l.get("record_grounded") and not l.get("is_question")),
                "n_unavailable": sum(1 for l in logs if l.get("unavailable")),
                "workup_cost": self.costs.workup_cost(ep.case)["total_cost"],
                "n_repeat": sum(1 for l in logs if l.get("repeat")),
                "n_rounds": rec["num_rounds"], "n_test_rounds": rec["num_test_rounds"],
                "n_agent_msgs": rec["n_agent_msgs"], "n_nags": rec["n_nags"],
                "forced": rec["forced"], "diagnosis": rec["diagnosis"]}


def build_train_prompts(args) -> list[dict]:
    cfg = load_config(getattr(args, "env_config", None))
    cases = load_split(cfg["dataset"], cfg["split"])
    assert_not_test(cfg["dataset"], [c["case_id"] for c in cases])
    if cfg["split"] not in ("train",) and cfg["dataset"] != "synthetic":
        raise PermissionError("GRPO may only train on the train split")
    global _CASE_BY_USER
    _CASE_BY_USER = {}
    rows = []
    for c in cases:
        u = first_user_message(c)
        if u in _CASE_BY_USER:
            raise ValueError(f"two cases share the same opening message: {c['case_id']}")
        _CASE_BY_USER[u] = c
        rows.append({"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": u}]})
    return rows


_CASE_BY_USER: dict[str, dict] = {}


def make_rollout_func(tok, args, helpers: dict):
    cfg = load_config(getattr(args, "env_config", None))
    svc = Services(cfg, tokenizer=tok)
    ids, env_mask = helpers["ids"], helpers["env_mask"]
    temp = args.rollout_temp
    pool = ThreadPoolExecutor(max_workers=cfg["n_threads"])

    def rollout_func(prompts, trainer):
        t0 = time.time()
        client = trainer.vllm_generation.vllm_client
        eps = [svc.new_episode(_CASE_BY_USER[p[-1]["content"]]) for p in prompts]
        for ep in eps:
            ep.overflow = False
        for _ in range(2 * cfg["max_turns"] + 1):
            active = [e for e in eps if not e.done]
            if not active:
                break
            for e in active:  # context guard: end the episode rather than overflow the server
                if len(ids(tok, e.messages, True)) + cfg["turn_max_tokens"] > cfg["max_context_tokens"]:
                    e.overflow, e.done = True, True
            active = [e for e in active if not e.done]
            if not active:
                break
            out = client.chat([e.messages for e in active], n=1, temperature=temp,
                              max_tokens=cfg["turn_max_tokens"], logprobs=0)
            texts = tok.batch_decode(out["completion_ids"], skip_special_tokens=True)
            list(pool.map(lambda et: et[0].step(et[1]), zip(active, texts)))  # gatekeeper calls in parallel
        scored = list(pool.map(svc.score, eps))
        prompt_ids, completion_ids, logprobs, masks, rewards = [], [], [], [], []
        for ep, s in zip(eps, scored):
            pid = ids(tok, ep.messages[:2], True)
            full = ids(tok, ep.messages, False)
            comp = full[len(pid):]
            prompt_ids.append(pid); completion_ids.append(comp); logprobs.append([0.0] * len(comp))
            masks.append(env_mask(tok, ep.messages, len(pid), len(comp)))
            rewards.append(s["reward"])
        m = lambda k: stx.mean(float(s[k]) for s in scored)  # noqa: E731
        print(f"[sdb-rollout] arm={cfg['arm']} n={len(eps)} reward={m('reward'):.3f} r_out={m('r_out'):.3f} "
              f"judge={m('judge_score'):.2f} pass={m('passed'):.2f} r_gather={m('r_gather'):.3f} "
              f"tests={m('n_tests'):.1f} q={m('n_questions'):.1f} grounded={m('n_grounded'):.1f} "
              f"repeat={m('n_repeat'):.1f} guarded={m('n_guarded'):.1f} cost={m('total_cost'):.0f} "
              f"cost_corr={m('total_cost_corrected'):.0f} invalid={m('invalid'):.2f} "
              f"overflow={m('overflow'):.2f} gk_cache={svc.gk_cache.hits}/{svc.gk_cache.hits + svc.gk_cache.misses} "
              f"wall={time.time() - t0:.0f}s", flush=True)
        if cfg.get("rollout_log"):
            with open(cfg["rollout_log"], "a") as f:
                for s in scored:
                    f.write(json.dumps(s) + "\n")
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": logprobs,
                "env_mask": masks, "traj_reward": rewards}

    return rollout_func
