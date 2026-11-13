"""Does the (local Qwen) judge's scoring noise change what GRPO would learn from a group?

Re-scores saved episodes R times with fresh judge caches (same model, same official prompt), keeps every term of
the reward except the judge term fixed, and compares the GRPO groups across scorings:

    reward' = reward - score01 + (score' - 1) / 4

Groups are what GRPO normalises over: all episodes of one case in one run (duplicates kept, no dedupe).

Per pair of scorings, and per scoring vs the consensus (mean score over all R scorings + the logged one):
  rank_tau        within-group Kendall tau-b of rewards, averaged over groups with any spread
  sign_agree      fraction of episodes whose GRPO advantage (reward - group mean) has the same sign, counting
                  "zero" as its own sign (|A| < 1e-9)
  sign_flip       fraction with strictly opposite non-zero signs (the harmful case: the update pushes the wrong way)
  adv_cosine      cosine between the two advantage vectors of a group (the update direction if per-sample
                  gradients were orthogonal), averaged over groups where both are non-zero
  zero_std_change groups that have spread under one scoring and none under the other
Plus the noise-to-signal ratio: mean per-episode variance of reward across scorings / mean within-group variance of
the consensus reward.

    PYTHONPATH=$PWD python -m tasks.v2_sdbench.judge_stability --episodes a.jsonl b.jsonl \
        --config V2/sdbench_configs/calib_qwen_ref.json --repeats 3 --out results/v2_sdbench/judge_stability.json

Local judge only (refuses an Azure judge: no paid calls). Synthetic cases only.
"""
from __future__ import annotations
import argparse
import itertools
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

EPS = 1e-9


def load_groups(paths: list[str]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        run = os.path.basename(p).removesuffix(".jsonl")
        for i, line in enumerate(open(p)):
            r = json.loads(line)
            if "test_logs" not in r or r.get("reward") is None:
                continue
            groups[f"{run}:{r['case_id']}"].append(
                {"run": run, "idx": i, "case_id": str(r["case_id"]), "diagnosis": r.get("diagnosis"),
                 "test_logs": r["test_logs"], "logged": r.get("judge_score"), "reward": r["reward"],
                 "score01": r["score01"]})
    return {g: v for g, v in groups.items() if len(v) >= 2}


def sign(x: float) -> int:
    return 0 if abs(x) < EPS else (1 if x > 0 else -1)


def adv(v: list[float]) -> list[float]:
    m = st.mean(v)
    return [x - m for x in v]


def kendall_tau_b(a: list[float], b: list[float]) -> float | None:
    conc = disc = ta = tb = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        da, db = sign(a[i] - a[j]), sign(b[i] - b[j])
        if da == 0 and db == 0:
            continue
        if da == 0:
            ta += 1
        elif db == 0:
            tb += 1
        elif da == db:
            conc += 1
        else:
            disc += 1
    n0 = conc + disc
    den = math.sqrt((n0 + ta) * (n0 + tb))
    return (conc - disc) / den if den else None


def cosine(a: list[float], b: list[float]) -> float | None:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na > EPS and nb > EPS else None


def compare(groups: dict, ka: str, kb: str) -> dict:
    taus, coss, n, agree, flip, zchange = [], [], 0, 0, 0, 0
    for eps in groups.values():
        ra, rb = [e["r"][ka] for e in eps], [e["r"][kb] for e in eps]
        aa, ab = adv(ra), adv(rb)
        for x, y in zip(aa, ab):
            n += 1
            agree += sign(x) == sign(y)
            flip += sign(x) * sign(y) == -1
        sa, sb = st.pstdev(ra) > EPS, st.pstdev(rb) > EPS
        zchange += sa != sb
        if sa or sb:
            t = kendall_tau_b(ra, rb)
            if t is not None:
                taus.append(t)
        c = cosine(aa, ab)
        if c is not None:
            coss.append(c)
    return {"n_episodes": n, "sign_agree": agree / n, "sign_flip": flip / n,
            "rank_tau": st.mean(taus) if taus else None, "adv_cosine": st.mean(coss) if coss else None,
            "adv_cosine_min": min(coss) if coss else None, "zero_std_change": zchange / len(groups)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--config", required=True, help="env config with a LOCAL judge")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-root", default="results/v2_sdbench/judge_stability_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if cfg["judge"].get("provider", "local") != "local":
        raise SystemExit("judge_stability makes many judge calls; local judge only (no paid API)")
    if cfg["dataset"] != "synthetic":
        raise SystemExit("synthetic cases only")
    groups = load_groups(a.episodes)
    eps = [e for g in groups.values() for e in g]
    print(f"[stab] {len(groups)} groups, {len(eps)} episodes, {sum(bool(e['diagnosis']) for e in eps)} judged",
          flush=True)
    llm = make_aux_llm(cfg["judge"], bool(cfg.get("aux_no_think")))
    judges = [Judge(llm, JsonlCache(os.path.join(a.cache_root, f"rep{r}.jsonl")), cfg["dataset"],
                    cfg["judge"]["prompt"], cfg["max_turns"]) for r in range(a.repeats)]
    cases = {cid: get_case(cfg["dataset"], cid) for cid in {e["case_id"] for e in eps}}

    def score(e):
        e["scores"] = {"logged": e["logged"]}
        for r, j in enumerate(judges):
            e["scores"][f"rep{r}"] = j.score(cases[e["case_id"]], e["diagnosis"], e["test_logs"])["score"]
        return e

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(score, eps))
    keys = ["logged"] + [f"rep{r}" for r in range(a.repeats)]
    for e in eps:
        base = e["reward"] - e["score01"]
        e["r"] = {k: base + (e["scores"][k] - 1) / 4 for k in keys}
        e["r"]["consensus"] = base + (st.mean(e["scores"][k] for k in keys) - 1) / 4
    pairs = {f"{x}~{y}": compare(groups, x, y) for x, y in itertools.combinations(keys, 2)}
    vs_cons = {k: compare(groups, k, "consensus") for k in keys}
    judged = [e for e in eps if e["diagnosis"]]
    noise = st.mean(st.pvariance([e["r"][k] for k in keys]) for e in eps)
    signal = st.mean(st.pvariance([e["r"]["consensus"] for e in g]) for g in groups.values())
    exact = st.mean(len({e["scores"][k] for k in keys}) == 1 for e in judged) if judged else None
    mean_of = lambda d, f: st.mean(v[f] for v in d.values() if v[f] is not None)  # noqa: E731
    summary = {
        "n_groups": len(groups), "n_episodes": len(eps), "n_judged": len(judged), "repeats": a.repeats,
        "judge": judges[0].model_id, "prompt": cfg["judge"]["prompt"],
        "frac_judged_all_scorings_identical": exact,
        "noise_var": noise, "signal_var_within_group": signal,
        "noise_to_signal": noise / signal if signal > EPS else None,
        "pairwise_mean": {f: mean_of(pairs, f) for f in ("sign_agree", "sign_flip", "rank_tau", "adv_cosine",
                                                         "zero_std_change")},
        "pairwise": pairs, "vs_consensus": vs_cons,
        "groups_with_spread_consensus": sum(st.pstdev([e["r"]["consensus"] for e in g]) > EPS
                                            for g in groups.values()),
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summary,
               "episodes": [{k: e[k] for k in ("run", "idx", "case_id", "scores", "r")} for e in eps]},
              open(a.out, "w"), indent=1)
    print(json.dumps({k: summary[k] for k in summary if k not in ("pairwise",)}, indent=1))


if __name__ == "__main__":
    main()
