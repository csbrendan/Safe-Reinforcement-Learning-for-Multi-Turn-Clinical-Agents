"""Diagnostic report + four-way verdict for an SFT-vs-RL comparison on a task (env-agnostic).

Reads two probe record files (probe_*.records.jsonl from framework.gates.support_probe) for the SFT init and
the RL'd model, and judges WHETHER any reward gain comes from adaptive gathering vs ask-cost exploitation /
blind commits / over-escalation -- not from mean reward alone. Four verdicts:
  THESIS-POSITIVE | ASK-COST EXPLOITATION (under-gather, high-risk regress) | OPTIMIZATION-ONLY | INCONCLUSIVE.

Usage: python -m framework.report --sft results/probe_sft.records.jsonl --rl results/probe_rl.records.jsonl
"""
from __future__ import annotations
import argparse, json, statistics as stx
from pathlib import Path


def load(path, cond="hidden"):
    return [json.loads(l) for l in open(path) if json.loads(l).get("cond") == cond]


def agg(recs, sub=None):
    rs = [r for r in recs if sub is None or (sub == "relevant" and r["true_risk"] != "none")
          or r["true_risk"] == sub]
    if not rs:
        return None
    byc = {}
    for r in rs:
        byc.setdefault(r["case_id"], []).append(r)
    supp = sum(1 for v in byc.values() if any(x["oracle_grade"] for x in v)) / len(byc)
    og = [r["n_asks"] for r in rs if r["oracle_grade"]]
    return dict(value=round(stx.mean(r["reward"] for r in rs), 3), support=round(supp, 3),
                asks=round(stx.mean(r["n_asks"] for r in rs), 2),
                blind_og=round(sum(1 for x in og if x == 0) / len(og), 3) if og else 0.0,
                n=len(byc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True); ap.add_argument("--rl", required=True)
    ap.add_argument("--label", default="RL"); args = ap.parse_args()
    S, R = load(args.sft), load(args.rl)
    print("=" * 84)
    print(f"DIAGNOSTIC REPORT — SFT vs {args.label} (hidden)")
    print("=" * 84)
    print("\n## 1. overall + by subset  [value / support / asks]  (blind_og = oracle-grade with 0 asks)")
    for sub, name in [(None, "overall"), ("high", "high"), ("moderate", "moderate"),
                      ("none", "benign"), ("relevant", "relevant")]:
        s, r = agg(S, sub), agg(R, sub)
        if not s or not r: continue
        print(f"  {name:9s} SFT={s['value']:+.3f}/{s['support']:.2f}/{s['asks']:.1f}  "
              f"{args.label}={r['value']:+.3f}/{r['support']:.2f}/{r['asks']:.1f}  blind_og:{s['blind_og']}->{r['blind_og']}")

    so, ro = agg(S), agg(R)
    sh, rh = agg(S, "high"), agg(R, "high")
    d_rew, d_supp = ro["value"] - so["value"], ro["support"] - so["support"]
    d_asks = ro["asks"] - so["asks"]
    hr_reg = (sh and rh and rh["support"] < sh["support"] - 1e-9)
    reward_up, support_up = d_rew > 0.02, d_supp > 0.03
    print("\n## 2. VERDICT")
    print(f"  d_value={d_rew:+.3f}  d_support={d_supp:+.3f} (support ~±0.05 noise)  d_asks={d_asks:+.2f}  "
          f"high-risk support {sh['support'] if sh else '-'}->{rh['support'] if rh else '-'} (regressed={hr_reg})")
    if reward_up and support_up and not hr_reg:
        v = "THESIS-POSITIVE: reward AND oracle-grade support improved, no high-risk regression."
    elif (reward_up and not support_up and d_asks < -0.2) or hr_reg:
        v = ("ASK-COST EXPLOITATION / under-gathering: reward rose via FEWER asks, support flat/down and/or "
             "high-risk safety subset regressed. Not the intended gain -> revisit reward (ask-cost vs "
             "wrong-high-risk penalty) before more RL.")
    elif abs(d_rew) > 0.02 or abs(d_supp) > 0.05:
        v = "OPTIMIZATION-ONLY / task failure: policy moved, reward changed, but not in the intended direction."
    else:
        v = "INCONCLUSIVE: policy did not improve held-out reward/support (check KL / rollout diversity)."
    print("  =>", v)


if __name__ == "__main__":
    main()
