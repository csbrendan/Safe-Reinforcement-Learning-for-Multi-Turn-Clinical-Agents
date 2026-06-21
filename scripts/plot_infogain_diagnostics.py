"""Diagnostics deephedge: base vs GRPO-control(coef0) vs info-gain(coef0.2), held-out `hidden`, by risk subset.
Reads support_probe records (pure env.score). Bucketing matches framework.report (high/moderate/none=benign).
"""
import json, statistics as stx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("base",      "results/probe_dh_base_t12.records.jsonl",      "#888888"),
    ("GRPO ctrl", "results/probe_dh_grpo_final_t12.records.jsonl", "#d1495b"),
    ("info-gain", "results/probe_dh_ig_final_t12.records.jsonl",   "#2e7d32"),
]
SUBSETS = [("high", "high"), ("moderate", "moderate"), ("none", "benign"), ("relevant", "relevant")]

def load(path):
    return [json.loads(l) for l in open(path) if json.loads(l)["cond"] == "hidden"]

def agg(recs, sub):
    rs = [r for r in recs if (sub == "relevant" and r["true_risk"] != "none") or r["true_risk"] == sub]
    if not rs: return None
    # support@k: per CASE, did ANY of its k rollouts land oracle-grade (matches framework.report)
    by_case = {}
    for r in rs:
        by_case.setdefault(r["case_id"], []).append(r)
    supp = stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0.0 for v in by_case.values())
    return {"value": stx.mean(r["reward"] for r in rs),
            "support": supp,
            "asks": stx.mean(r["n_asks"] for r in rs)}

data = {name: load(p) for name, p, _ in ARMS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.arange(len(SUBSETS)); w = 0.26
for panel, (metric, title, ylab) in enumerate(
        [("value", "Clinical value (mean reward)", "reward"),
         ("support", "Oracle-grade support@12 (per case)", "support@12"),
         ("asks", "Mean asks (gathering)", "asks")]):
    ax = axes[panel]
    for k, (name, _p, col) in enumerate(ARMS):
        vals = [(agg(data[name], s) or {}).get(metric, 0.0) for s, _ in SUBSETS]
        ax.bar(x + (k-1)*w, vals, w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in SUBSETS])
    ax.set_title(title); ax.set_ylabel(ylab); ax.axhline(0, color="k", lw=0.6)
    if panel == 0: ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
# overall reward annotation
ov = {name: stx.mean(r["reward"] for r in data[name]) for name, _, _ in ARMS}
fig.suptitle("Diagnostics deephedge (held-out hidden, pure env.score): base vs GRPO-control vs info-gain   "
             f"|  overall reward  base {ov['base']:.3f} -> ctrl {ov['GRPO ctrl']:.3f} -> info-gain {ov['info-gain']:.3f}",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "results/figs/infogain_diagnostics_base_vs_ctrl_vs_ig.png"
plt.savefig(out, dpi=130)
print("saved", out)
# also dump the numbers
for name, _, _ in ARMS:
    print(f"\n{name}: overall={ov[name]:.3f}")
    for s, lbl in SUBSETS:
        a = agg(data[name], s)
        if a: print(f"  {lbl:9} value={a['value']:+.3f} support={a['support']:.2f} asks={a['asks']:.2f}")
