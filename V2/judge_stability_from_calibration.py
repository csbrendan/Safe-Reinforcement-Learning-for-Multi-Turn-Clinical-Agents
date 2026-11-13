"""Priority 3 (no new judge calls): GRPO-group stability of the Qwen judge from the 2026-11-12 calibration, which
scored every unique episode four times: logged (during the run), reference re-score, and two candidate re-scores
with fresh caches (same Qwen3-8B, official v4_reasoning_with_tests prompt).

Groups are rebuilt with duplicates (an episode identical to another in case, diagnosis and test log gets that
episode's scores), because GRPO normalises over all rollouts of a prompt. Metrics: tasks.v2_sdbench.judge_stability.

    python V2/judge_stability_from_calibration.py
"""
import itertools
import json
import statistics as st
import sys

sys.path.insert(0, ".")
from tasks.v2_sdbench.judge_stability import compare, EPS  # noqa: E402

CAL = "results/v2_sdbench/calibration/qwen_vs_qwen.json"
B = "results/v2_sdbench/bench"
RUNS = {"conc16_guarded": f"{B}/conc16_guarded.jsonl", "t07_guarded": f"{B}/t07_guarded.jsonl"}


def key(r):
    return (str(r["case_id"]), (r.get("diagnosis") or "").strip(),
            json.dumps([(l.get("test_name"), l.get("result")) for l in r["test_logs"]]))


def main():
    cal = json.load(open(CAL))["episodes"]
    by_run_idx = {(e["run"], e["idx"]): e for e in cal}
    scores_by_key = {}
    for run, path in RUNS.items():
        rows = [json.loads(l) for l in open(path)]
        for i, r in enumerate(rows):
            if (run, i) in by_run_idx:
                e = by_run_idx[(run, i)]
                scores_by_key[key(r)] = {"logged": e["judge_score_logged"], "ref": e["ref"], "c0": e["cand"][0],
                                         "c1": e["cand"][1]}
    groups = {}
    for run, path in RUNS.items():
        for r in map(json.loads, open(path)):
            sc = scores_by_key[key(r)]
            base = r["reward"] - r["score01"]
            rs = {k: base + (v - 1) / 4 for k, v in sc.items()}
            rs["consensus"] = base + (st.mean(sc.values()) - 1) / 4
            groups.setdefault(f"{run}:{r['case_id']}", []).append({"r": rs, "s": sc, "dx": bool(r.get("diagnosis")),
                                                                    "invalid": r["invalid"]})
    keys = ["logged", "ref", "c0", "c1"]
    out = {"groups": {g: {"n": len(v), "n_judged": sum(e["dx"] for e in v), "n_invalid": sum(e["invalid"] for e in v),
                          "reward_std": {k: round(st.pstdev([e["r"][k] for e in v]), 4) for k in keys + ["consensus"]},
                          "scores": [[e["s"][k] for k in keys] for e in v]}
                      for g, v in groups.items()}}
    pairs = {f"{a}~{b}": compare(groups, a, b) for a, b in itertools.combinations(keys, 2)}
    out["pairwise"] = pairs
    out["pairwise_mean"] = {f: st.mean(p[f] for p in pairs.values() if p[f] is not None)
                            for f in ("sign_agree", "sign_flip", "rank_tau", "adv_cosine", "zero_std_change")}
    out["vs_consensus"] = {k: compare(groups, k, "consensus") for k in keys}
    eps = [e for v in groups.values() for e in v]
    noise = st.mean(st.pvariance([e["r"][k] for k in keys]) for e in eps)
    signal = st.mean(st.pvariance([e["r"]["consensus"] for e in v]) for v in groups.values())
    out["noise_var"], out["signal_var"], out["noise_to_signal"] = noise, signal, noise / signal
    judged = [e for e in eps if e["dx"]]
    out["n_episodes"], out["n_judged"] = len(eps), len(judged)
    out["frac_judged_identical_all4"] = st.mean(len(set(e["s"].values())) == 1 for e in judged)
    json.dump(out, open("results/v2_sdbench/judge_stability_from_calibration.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "pairwise"}, indent=1))


if __name__ == "__main__":
    main()
