"""Titration-aware 3-way comparison (base / GRPO-control / GRPO-info-gain), held-out hidden, pure env.score.
framework.report's generic high/moderate/none subsets do not map to titration's JOINT clot_bleed states, so we
compute titration-specific subsets here: high-clot (state startswith 'high'), high-bleed (endswith 'high'), and the
safety-critical high_high. Reports value (mean reward), support@k (per-case any oracle-grade), mean asks, 1-ask rate.
Usage: python -m tasks.titration.compare <base.records> <ctrl.records> <ig.records>
"""
import json, sys, statistics as stx

def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]

def subset(recs, pred):
    rs = [r for r in recs if pred(r["true_risk"])]
    if not rs: return None
    by = {}
    for r in rs: by.setdefault(r["case_id"], []).append(r)
    supp = stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0.0 for v in by.values())
    one = stx.mean(1.0 if r["n_asks"] == 1 else 0.0 for r in rs)
    return {"n": len(by), "value": stx.mean(r["reward"] for r in rs), "supp": supp,
            "asks": stx.mean(r["n_asks"] for r in rs), "one_ask": one}

SUBSETS = [
    ("overall",      lambda s: True),
    ("high-clot",    lambda s: s.startswith("high")),
    ("high-bleed",   lambda s: s.endswith("high")),
    ("high_high",    lambda s: s == "high_high"),
    ("low-acuity",   lambda s: s in ("low_low", "low_mod", "mod_low")),
]

arms = sys.argv[1:]
labels = ["base", "control", "info-gain"][:len(arms)]
data = {lab: load(p) for lab, p in zip(labels, arms)}

for name, pred in SUBSETS:
    print(f"\n=== {name} ===")
    print(f"{'arm':10} {'n':>3} {'value':>7} {'supp@12':>8} {'asks':>6} {'1ask':>6}")
    for lab in labels:
        a = subset(data[lab], pred)
        if a:
            print(f"{lab:10} {a['n']:>3} {a['value']:>+7.3f} {a['supp']:>8.2f} {a['asks']:>6.2f} {a['one_ask']:>6.2f}")
