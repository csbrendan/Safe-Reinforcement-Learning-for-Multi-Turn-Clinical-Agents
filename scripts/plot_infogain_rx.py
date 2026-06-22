"""rx_triage: base vs SFT-v3 vs GRPO-control vs GRPO-info-gain, held-out `hidden`, by risk subset.
Reads support_probe records (pure env.score). support@12 per-case (matches framework.report).
GRPO arms use the FINAL checkpoint. NOTE: info-gain's high-risk-preserving high-water mark is checkpoint-20,
not final (see docs/eval_runs/rx_triage_infogain.md) -- final-vs-final is shown here for parity with the control.
"""
import json, statistics as stx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [
    ("base",      "results/probe_base_t10.records.jsonl",       "#888888"),
    ("SFT-v3",    "results/probe_sft_v3_t10.records.jsonl",     "#1f77b4"),
    ("GRPO ctrl", "results/probe_grpo2_t10.records.jsonl",      "#d1495b"),
    ("info-gain", "results/probe_rx_ig_final_t10.records.jsonl", "#2e7d32"),
]
SUBSETS = [("high", "high"), ("moderate", "moderate"), ("none", "benign"), ("relevant", "relevant")]

def load(path):
    return [json.loads(l) for l in open(path) if json.loads(l)["cond"] == "hidden"]

def agg(recs, sub):
    rs = [r for r in recs if (sub == "relevant" and r["true_risk"] != "none") or r["true_risk"] == sub]
    if not rs: return None
    by_case = {}
    for r in rs:
        by_case.setdefault(r["case_id"], []).append(r)
    supp = stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0.0 for v in by_case.values())
    return {"value": stx.mean(r["reward"] for r in rs), "support": supp,
            "asks": stx.mean(r["n_asks"] for r in rs)}

data = {name: load(p) for name, p, _ in ARMS}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(SUBSETS)); w = 0.2
for panel, (metric, title, ylab) in enumerate(
        [("value", "Clinical value (mean reward)", "reward"),
         ("support", "Oracle-grade support@12 (per case)", "support@12"),
         ("asks", "Mean asks (gathering)", "asks")]):
    ax = axes[panel]
    for k, (name, _p, col) in enumerate(ARMS):
        vals = [(agg(data[name], s) or {}).get(metric, 0.0) for s, _ in SUBSETS]
        ax.bar(x + (k-1.5)*w, vals, w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in SUBSETS])
    ax.set_title(title); ax.set_ylabel(ylab); ax.axhline(0, color="k", lw=0.6)
    if panel == 0: ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
ov = {name: stx.mean(r["reward"] for r in data[name]) for name, _, _ in ARMS}
fig.suptitle("rx_triage (held-out hidden, pure env.score): base / SFT-v3 / GRPO-control / GRPO-info-gain, by risk subset   "
             "|  GRPO arms = FINAL ckpt; info-gain best-preserving ckpt is step-20 (see writeup)", fontsize=10.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/figs/infogain_rx_base_sft_ctrl_ig.png"
plt.savefig(out, dpi=130); print("saved", out)
for name, _, _ in ARMS:
    print(f"\n{name}: overall={ov[name]:.3f}")
    for s, lbl in SUBSETS:
        a = agg(data[name], s)
        if a: print(f"  {lbl:9} value={a['value']:+.3f} support@12={a['support']:.2f} asks={a['asks']:.2f}")
