"""Titration 'gathering dial': mean asks (and overall value) vs the info-gain coefficient lambda.
Shows the narrow safe optimum -- under-gather at lambda<=0.03, oracle-level at lambda=0.1, over-gather at
lambda=0.2 -- against the oracle ask-level band. Held-out hidden, pure env.score.
"""
import json, statistics as stx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [  # (lambda, label, records)
    (0.0,  "0 (plain GRPO)", "results/probe_ti_ctl_final_t10.records.jsonl"),
    (0.03, "0.03",          "results/probe_ti_ig003_final_t10.records.jsonl"),
    (0.10, "0.1",           "results/probe_ti_ig01_final_t10.records.jsonl"),
    (0.20, "0.2",           "results/probe_ti_ig_final_t10.records.jsonl"),
]
def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def asks(recs): return stx.mean(r["n_asks"] for r in recs)
def val(recs):  return stx.mean(r["reward"] for r in recs)

base = load("results/probe_ti_base.records.jsonl")
ORACLE_ASKS = 2.77
lams = [a[0] for a in ARMS]
A = [asks(load(a[2])) for a in ARMS]
V = [val(load(a[2])) for a in ARMS]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.axhspan(ORACLE_ASKS-0.1, ORACLE_ASKS+0.1, color="#9ecae1", alpha=0.5, zorder=0)
ax.axhline(ORACLE_ASKS, color="#3182bd", ls="--", lw=1, label=f"oracle ask-level ({ORACLE_ASKS:.1f})")
ax.plot(lams, A, "o-", color="#d1495b", lw=2, ms=8, label="mean asks (GRPO)")
for x, y, lab in zip(lams, A, ["under","under","oracle-level","over"]):
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="#7a2230")
ax.axhline(asks(base), color="#888", ls=":", lw=1, label=f"base/init ({asks(base):.2f})")
ax.set_xlabel(r"info-gain coefficient $\lambda$"); ax.set_ylabel("mean asks", color="#d1495b")
ax.set_xticks(lams); ax.set_xticklabels([a[1] for a in ARMS]); ax.set_ylim(0, 5)
ax.tick_params(axis="y", labelcolor="#d1495b")

ax2 = ax.twinx()
ax2.plot(lams, V, "s--", color="#2e7d32", lw=1.6, ms=6, label="overall value")
ax2.set_ylabel("overall value (mean reward)", color="#2e7d32"); ax2.tick_params(axis="y", labelcolor="#2e7d32")

ax.set_title("Titration: the credit coefficient is a gathering dial\n"
             "(under-gather $\\to$ oracle-level $\\to$ over-gather; held-out hidden, pure env.score)", fontsize=10)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
plt.tight_layout()
out = "results/figs/titration_lambda_dial.png"; plt.savefig(out, dpi=130)
print("saved", out)
for (lam, lab, _), a, v in zip(ARMS, A, V):
    print(f"  lambda={lab:14} asks={a:.2f} value={v:+.3f}")
print(f"  base asks={asks(base):.2f}  oracle~{ORACLE_ASKS}")
