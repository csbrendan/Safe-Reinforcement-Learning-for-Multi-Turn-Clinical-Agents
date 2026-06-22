"""Diagnostics deephedge 4-arm: base / GRPO-control(coef0) / GRPO-info-gain(coef0.2) / DREAM-decoupled.
Held-out hidden, pure env.score, support@12 per-case (matches framework.report). GRPO arms = FINAL ckpt.
NOTE: DREAM's high-risk peak is checkpoint-40 (+0.248 / support 1.00), not final -- see the trajectory plot /
docs/eval_runs/dream_deephedge.md. Final-vs-final is shown here for parity with the other arms.
"""
import json, statistics as stx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("base",      "results/probe_dh_base_t12.records.jsonl",       "#888888"),
    ("GRPO ctrl", "results/probe_dh_grpo_final_t12.records.jsonl",  "#d1495b"),
    ("info-gain", "results/probe_dh_ig_final_t12.records.jsonl",    "#2e7d32"),
    ("DREAM",     "results/probe_dream_final_t12.records.jsonl",    "#6a4c93"),
]
SUBSETS = [("high", "high"), ("moderate", "moderate"), ("none", "benign"), ("relevant", "relevant")]

def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def agg(recs, sub):
    rs = [r for r in recs if (sub == "relevant" and r["true_risk"] != "none") or r["true_risk"] == sub]
    if not rs: return None
    by = {}
    for r in rs: by.setdefault(r["case_id"], []).append(r)
    supp = stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0.0 for v in by.values())
    return {"value": stx.mean(r["reward"] for r in rs), "support": supp, "asks": stx.mean(r["n_asks"] for r in rs)}

data = {n: load(p) for n, p, _ in ARMS}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(SUBSETS)); w = 0.2
for panel, (metric, title, ylab) in enumerate(
        [("value", "Clinical value (mean reward)", "reward"),
         ("support", "Oracle-grade support@12 (per case)", "support@12"),
         ("asks", "Mean asks (gathering)", "asks")]):
    ax = axes[panel]
    for k, (n, _p, col) in enumerate(ARMS):
        ax.bar(x + (k - 1.5) * w, [(agg(data[n], s) or {}).get(metric, 0.0) for s, _ in SUBSETS], w, label=n, color=col)
    if metric == "asks":
        ax.axhline(3.9, ls="--", color="k", lw=1, label="oracle VOI ~3.9")
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in SUBSETS]); ax.set_title(title); ax.set_ylabel(ylab)
    ax.axhline(0, color="k", lw=0.6); ax.grid(axis="y", alpha=0.3)
    if panel == 0: ax.legend(loc="upper left", fontsize=8)
ov = {n: stx.mean(r["reward"] for r in data[n]) for n, _, _ in ARMS}
fig.suptitle("Diagnostics deephedge (held-out hidden, pure env.score): base / control / info-gain / DREAM, by risk subset   "
             "|  GRPO arms = FINAL; DREAM high-risk peak is ckpt-40 (see trajectory)", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/figs/dream_diagnostics_4arm.png"
plt.savefig(out, dpi=130); print("saved", out)
for n, _, _ in ARMS:
    print(f"\n{n}: overall={ov[n]:.3f}")
    for s, lbl in SUBSETS:
        a = agg(data[n], s)
        if a: print(f"  {lbl:9} value={a['value']:+.3f} support@12={a['support']:.2f} asks={a['asks']:.2f}")
