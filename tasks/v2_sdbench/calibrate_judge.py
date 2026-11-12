"""Judge calibration: score the SAME stored episodes with a reference judge (Qwen3-8B, local) and a candidate judge
(Azure GPT-5), both with SD-Bench's official prompt, and report how the reward signal shifts.

Nothing is re-rolled: each episode's final diagnosis and the test log the agent saw are read from eval JSONL files
and sent to both judges. r_out is recomputed by swapping only the judge term:
    r_out' = r_out - score01 + (score' - 1) / 4
so cost and format terms are held fixed.

    PYTHONPATH=$PWD python -m tasks.v2_sdbench.calibrate_judge \
        --episodes results/v2_sdbench/bench/conc16_guarded.jsonl ... \
        --ref V2/sdbench_configs/smoke_synthetic.json --cand V2/sdbench_configs/azure_synthetic.json \
        --repeats 1 --max-calls 120 --out results/v2_sdbench/calibration/judge_calibration.json

--repeats N scores every episode N extra times with the candidate (separate caches) to measure its test-retest
agreement: GPT-5 on Azure takes no temperature, so it isn't deterministic.
--max-calls caps candidate (paid) calls; the run stops before exceeding it.
Synthetic cases only: the script refuses dataset "mimic" (see V2/sdbench_azure_switch.md).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from tasks.v2_sdbench.cache import JsonlCache
from tasks.v2_sdbench.cases import get_case
from tasks.v2_sdbench.env import load_config
from tasks.v2_sdbench.llm import make_aux_llm
from tasks.v2_sdbench.reward import Judge


def load_episodes(paths: list[str]) -> list[dict]:
    eps, seen = [], set()
    for p in paths:
        run = os.path.basename(p).removesuffix(".jsonl")
        for i, line in enumerate(open(p)):
            r = json.loads(line)
            if "test_logs" not in r:
                continue
            key = (str(r["case_id"]), (r.get("diagnosis") or "").strip(),
                   json.dumps([(l.get("test_name"), l.get("result")) for l in r["test_logs"]]))
            if key in seen:
                continue
            seen.add(key)
            eps.append({"run": run, "idx": i, "case_id": str(r["case_id"]), "diagnosis": r.get("diagnosis"),
                        "test_logs": r["test_logs"], "judge_score_logged": r.get("judge_score"),
                        "r_out": r.get("r_out"), "score01": r.get("score01"), "invalid": r.get("invalid"),
                        "group": f"{run}:{r['case_id']}"})
    return eps


def weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa on 1..k scores."""
    n = len(a)
    if n == 0:
        return float("nan")
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x - 1][y - 1] += 1
    ra = [sum(r) for r in O]
    cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j]
            den += w * ra[i] * cb[j] / n
    return 1 - num / den if den else float("nan")


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    if len(a) < 3 or st.pstdev(ra) == 0 or st.pstdev(rb) == 0:
        return float("nan")
    ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (len(a) * st.pstdev(ra) * st.pstdev(rb))


def agreement(a: list[int], b: list[int]) -> dict:
    n = len(a)
    return {"n": n, "exact": sum(x == y for x, y in zip(a, b)) / n if n else None,
            "within_1": sum(abs(x - y) <= 1 for x, y in zip(a, b)) / n if n else None,
            "pass_agree": sum((x >= 4) == (y >= 4) for x, y in zip(a, b)) / n if n else None,
            "kappa_quadratic": weighted_kappa(a, b), "spearman": spearman(a, b),
            "mean_delta": st.mean(y - x for x, y in zip(a, b)) if n else None}


def group_stats(eps: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for e in eps:
        groups[e["group"]].append(e[key])
    multi = {g: v for g, v in groups.items() if len(v) >= 2}
    stds = {g: st.pstdev(v) for g, v in multi.items()}
    return {"n_groups": len(multi), "mean_within_group_std": st.mean(stds.values()) if stds else None,
            "frac_zero_std": sum(s == 0 for s in stds.values()) / len(stds) if stds else None,
            "per_group_std": {g: round(s, 4) for g, s in stds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--ref", required=True, help="env config whose judge is the reference (Qwen)")
    ap.add_argument("--cand", required=True, help="env config whose judge is the candidate (Azure GPT-5)")
    ap.add_argument("--repeats", type=int, default=1, help="extra candidate scorings per episode (test-retest)")
    ap.add_argument("--max-calls", type=int, default=120, help="hard cap on candidate judge calls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--price-in", type=float, default=1.25, help="USD per 1M input tokens (assumed GPT-5 list)")
    ap.add_argument("--price-out", type=float, default=10.0, help="USD per 1M output tokens (assumed GPT-5 list)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref_cfg, cand_cfg = load_config(a.ref), load_config(a.cand)
    for c in (ref_cfg, cand_cfg):
        if c["dataset"] == "mimic":
            raise SystemExit("calibration is synthetic-only until the MIMIC data-policy blocker is cleared")
    if ref_cfg["judge"]["prompt"] != cand_cfg["judge"]["prompt"]:
        raise SystemExit("reference and candidate must use the same judge prompt")
    eps = load_episodes(a.episodes)
    need = len([e for e in eps if e["diagnosis"]]) * (1 + a.repeats)
    if need > a.max_calls:
        raise SystemExit(f"{need} candidate calls needed > --max-calls {a.max_calls}")
    print(f"[calib] {len(eps)} unique episodes, {need} candidate calls max", flush=True)

    ds = ref_cfg["dataset"]
    ref_j = Judge(make_aux_llm(ref_cfg["judge"], bool(ref_cfg.get("aux_no_think"))),
                  JsonlCache(os.path.join(ref_cfg["cache_dir"], "judge.jsonl")), ds, ref_cfg["judge"]["prompt"],
                  ref_cfg["max_turns"])
    cand_llm = make_aux_llm(cand_cfg["judge"], bool(cand_cfg.get("aux_no_think")))
    cd = os.path.join(cand_cfg["cache_dir"], "calibration")
    os.makedirs(cd, exist_ok=True)
    cand_js = [Judge(cand_llm, JsonlCache(os.path.join(cd, f"judge_rep{r}.jsonl")), ds, cand_cfg["judge"]["prompt"],
                     cand_cfg["max_turns"]) for r in range(1 + a.repeats)]
    cases = {cid: get_case(ds, cid) for cid in {e["case_id"] for e in eps}}

    def score(e):
        c = cases[e["case_id"]]
        e["ref"] = ref_j.score(c, e["diagnosis"], e["test_logs"])["score"]
        e["cand"] = [j.score(c, e["diagnosis"], e["test_logs"])["score"] for j in cand_js]
        return e

    with ThreadPoolExecutor(a.workers) as pool:
        eps = list(pool.map(score, eps))

    for e in eps:
        base = (e["r_out"] - e["score01"]) if e["r_out"] is not None else 0.0
        e["r_ref"] = base + (e["ref"] - 1) / 4
        e["r_cand"] = base + (e["cand"][0] - 1) / 4
    judged = [e for e in eps if e["diagnosis"]]
    ref_s = [e["ref"] for e in judged]
    cand_s = [e["cand"][0] for e in judged]
    usage = getattr(cand_llm, "usage", {})
    cost = (usage.get("prompt_tokens", 0) * a.price_in + usage.get("completion_tokens", 0) * a.price_out) / 1e6
    rep = {}
    if a.repeats:
        rep = agreement([e["cand"][0] for e in judged], [e["cand"][1] for e in judged])
    summary = {
        "n_episodes": len(eps), "n_judged": len(judged),
        "ref_judge": ref_j.model_id, "cand_judge": cand_js[0].model_id, "prompt": ref_cfg["judge"]["prompt"],
        "score_hist_ref": {s: ref_s.count(s) for s in range(1, 6)},
        "score_hist_cand": {s: cand_s.count(s) for s in range(1, 6)},
        "agreement_ref_vs_cand": agreement(ref_s, cand_s),
        "cand_test_retest": rep,
        "ref_matches_logged": sum(e["ref"] == e["judge_score_logged"] for e in judged) / max(1, len(judged)),
        "reward": {
            "mean_ref": st.mean(e["r_ref"] for e in eps), "mean_cand": st.mean(e["r_cand"] for e in eps),
            "mean_abs_shift": st.mean(abs(e["r_cand"] - e["r_ref"]) for e in eps),
            "max_abs_shift": max(abs(e["r_cand"] - e["r_ref"]) for e in eps),
            "std_ref": st.pstdev([e["r_ref"] for e in eps]), "std_cand": st.pstdev([e["r_cand"] for e in eps]),
            "spearman": spearman([e["r_ref"] for e in eps], [e["r_cand"] for e in eps]),
            "groups_ref": group_stats(eps, "r_ref"), "groups_cand": group_stats(eps, "r_cand"),
        },
        "cand_usage": usage,
        "cand_cost_usd_estimate": round(cost, 4),
        "cand_cost_note": f"assumed prices ${a.price_in}/1M in, ${a.price_out}/1M out; check your Azure rate card",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "episodes": [{k: e[k] for k in ("run", "idx", "case_id", "group", "invalid",
                                                                   "judge_score_logged", "ref", "cand", "r_ref",
                                                                   "r_cand")} for e in eps]}, f, indent=1)
    print(json.dumps(summary, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))


if __name__ == "__main__":
    main()
