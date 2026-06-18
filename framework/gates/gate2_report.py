"""Gate 2 verdict + full breakdown from support-probe outputs (CPU, no GPU).

Reads one or more probe runs (the .json summary for parse_validity + the .records.jsonl for everything else)
and prints, per condition (hidden prioritized), per risk subset {overall, high, moderate, benign, relevant}:
support@k, mean reward, reward std, mean asks, ask-hist, oracle-grade ask-hist; plus the blind/shortcut
discriminators (no-ask-commit, 1-ask, blind-escalate, blind-hold, benign over-escalation, high-risk support).
Then it compares each run against the analytic baselines and applies the Gate-2 pass criteria.

Usage:
  python -m framework.gates.gate2_report --run base=results/probe_base.json --run sft=results/probe_sft.json \
    [--cond hidden] [--k 12]
"""
from __future__ import annotations
import argparse, json, statistics as stx
from collections import Counter
from pathlib import Path

# Analytic baselines for rx_triage (Gate 1, widened env, cost 0.03) — overall hidden-condition values.
ANALYTIC = {"oracle": 0.854, "best_fixed": 0.749, "ask_all": 0.747, "no_ask": 0.601,
            "1ask": 0.608, "blind_escalate": 0.376}
SUBSETS = [("overall", None), ("high", "high"), ("moderate", "moderate"),
           ("benign", "none"), ("relevant", "relevant")]


def _sub(recs, key):
    if key is None:
        return recs
    if key == "relevant":
        return [r for r in recs if r["true_risk"] != "none"]
    return [r for r in recs if r["true_risk"] == key]


def _hist(xs):
    h = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for x in xs:
        h["0" if x == 0 else "1" if x == 1 else "2" if x == 2 else "3+"] += 1
    n = max(len(xs), 1)
    return {k: round(v / n, 3) for k, v in h.items()}


def metrics(recs):
    if not recs:
        return None
    byc = {}
    for r in recs:
        byc.setdefault(r["case_id"], []).append(r)
    supp = sum(1 for v in byc.values() if any(x["oracle_grade"] for x in v)) / len(byc)
    rew = [r["reward"] for r in recs]
    asks = [r["n_asks"] for r in recs]
    og_asks = [r["n_asks"] for r in recs if r["oracle_grade"]]
    committed = [r for r in recs if r["decision"] is not None]
    return dict(
        support=round(supp, 3), n_cases=len(byc), n_roll=len(recs),
        mean_reward=round(stx.mean(rew), 3), reward_std=round(stx.pstdev(rew), 3) if len(rew) > 1 else 0.0,
        mean_asks=round(stx.mean(asks), 2), ask_hist=_hist(asks), og_ask_hist=_hist(og_asks),
        og_blind_rate=round(sum(1 for x in og_asks if x == 0) / len(og_asks), 3) if og_asks else 0.0,
        no_ask_commit=round(sum(1 for r in committed if r["n_asks"] == 0) / max(len(recs), 1), 3),
        one_ask=round(sum(1 for r in recs if r["n_asks"] == 1) / max(len(recs), 1), 3),
        blind_escalate=round(sum(1 for r in recs if r["n_asks"] == 0 and r["decision"] == "escalate") / max(len(recs), 1), 3),
        blind_hold=round(sum(1 for r in recs if r["n_asks"] == 0 and r["decision"] == "hold_for_labs") / max(len(recs), 1), 3),
    )


def benign_overesc(recs):
    b = _sub(recs, "none")
    if not b:
        return float("nan")
    return round(sum(1 for r in b if r["decision"] in ("escalate", "hold_for_labs")) / len(b), 3)


def load_run(spec, cond):
    tag, path = spec.split("=", 1)
    summ = json.load(open(path))
    rp = path.replace(".json", ".records.jsonl")
    recs = [json.loads(l) for l in open(rp) if json.loads(l).get("cond") == cond]
    pv = summ.get(cond, {}).get("parse_validity")
    return tag, recs, pv


def report_run(tag, recs, pv, k):
    print(f"\n{'='*92}\n[{tag}] cond=hidden  parse_validity={pv}  (k={k}, rollouts={len(recs)})\n{'='*92}")
    print(f"  {'subset':9s} {'supp@k':>7} {'reward':>8} {'rstd':>6} {'asks':>5}  {'ask_hist(0/1/2/3+)':>26}  {'og_ask_hist':>26}")
    M = {}
    for name, key in SUBSETS:
        m = metrics(_sub(recs, key)); M[name] = m
        if not m:
            continue
        ah = "/".join(f"{m['ask_hist'][x]:.2f}" for x in ("0", "1", "2", "3+"))
        oh = "/".join(f"{m['og_ask_hist'][x]:.2f}" for x in ("0", "1", "2", "3+"))
        print(f"  {name:9s} {m['support']:>7.3f} {m['mean_reward']:>+8.3f} {m['reward_std']:>6.3f} "
              f"{m['mean_asks']:>5.2f}  {ah:>26}  {oh:>26}")
    o = M["overall"]
    print(f"\n  shortcut/blind:  no_ask_commit={o['no_ask_commit']}  1ask={o['one_ask']}  "
          f"blind_escalate={o['blind_escalate']}  blind_hold={o['blind_hold']}  "
          f"benign_overesc={benign_overesc(recs)}")
    print(f"  high-risk support={M['high']['support'] if M['high'] else '-'}  "
          f"og_blind_rate(overall)={o['og_blind_rate']}")
    return M


def gate2_checks(M, pv):
    o, hi = M["overall"], M.get("high")
    og0 = o["og_ask_hist"]["0"]
    selective_og = (og0 < 0.85)  # oracle-grade successes are not dominated by 0-ask blind commits
    return {
        "parse_validity near 1.0 (>=0.95)": pv is not None and pv >= 0.95,
        "non-trivial support@12 (>=0.15)": o["support"] >= 0.15,
        "reward diversity for RL (overall std >=0.08)": o["reward_std"] >= 0.08,
        "no collapse to no-ask-commit (<0.6)": o["no_ask_commit"] < 0.6,
        "no collapse to 1-ask shortcut (<0.6)": o["one_ask"] < 0.6,
        "no collapse to ask-all (mean_asks < rel-union ~7)": o["mean_asks"] < 6.5,
        "no collapse to blind-escalate (<0.3)": o["blind_escalate"] < 0.3,
        "oracle-grade incl. selective asks (og 0-ask frac <0.85)": selective_og,
        "high-risk support nonzero": bool(hi) and hi["support"] > 0.0,
        "benign over-escalation controlled (<0.4)": (M.get("_benign_oe", 0.0) < 0.4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="tag=path/to/probe.json (repeatable)")
    ap.add_argument("--cond", default="hidden")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    print("ANALYTIC BASELINES (rx_triage widened, hidden, cost 0.03):")
    print("  " + "  ".join(f"{k}={v:+.3f}" for k, v in ANALYTIC.items()))

    runs = {}
    for spec in args.run:
        tag, recs, pv = load_run(spec, args.cond)
        M = report_run(tag, recs, pv, args.k)
        M["_benign_oe"] = benign_overesc(recs)
        runs[tag] = (M, pv)

    # Gate-2 verdict on the SFT run if present, else the last run.
    target = "sft" if "sft" in runs else list(runs)[-1]
    M, pv = runs[target]
    print(f"\n{'='*92}\nGATE 2 CHECKS — target run: [{target}]\n{'='*92}")
    checks = gate2_checks(M, pv)
    for kk, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {kk}")
    print(f"  => GATE 2 {'PASSES' if all(checks.values()) else 'FAILS'} for [{target}]")
    if "base" in runs and target != "base":
        bM = runs["base"][0]
        print(f"\n  base->sft deltas (hidden, overall): "
              f"support {bM['overall']['support']:.3f}->{M['overall']['support']:.3f}  "
              f"reward {bM['overall']['mean_reward']:+.3f}->{M['overall']['mean_reward']:+.3f}  "
              f"parse_validity {runs['base'][1]}->{pv}")


if __name__ == "__main__":
    main()
