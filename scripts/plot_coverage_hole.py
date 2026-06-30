"""Coverage-hole + Gate-2-predicts-repairability figure for the SFT-coverage cell.
Left: a coverage-biased SFT looks great on AGGREGATE but is unsafe on the rare HIGH-RISK cases
      (overall vs high-risk value), for the severe (keep-high=0) and milder (keep-high=0.30) biases.
Right: bias severity sets the high-risk action's latent support (emergent_tx sample rate) -> Gate 2 PASS/FAIL,
      which predicts whether RL can repair the hole. Held-out hidden, pure env.score.
"""
import json, statistics as stx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SFTS = [  # (label, records, gate2)
    ("keep-high=0\n(severe)",  "results/probe_dx_sftbiased_t12.records.jsonl",      "Gate 2 FAIL"),
    ("keep-high=0.30\n(mild)", "results/probe_dx_biased30_n200_t12.records.jsonl",  "Gate 2 PASS"),
]
def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def val(recs, pred): rs=[r for r in recs if pred(r["true_risk"])]; return stx.mean(r["reward"] for r in rs) if rs else 0
def emerg(recs):     rs=[r for r in recs if r["true_risk"]=="high"]; return stx.mean(1.0 if r["decision"]=="emergent_tx" else 0 for r in rs) if rs else 0

data = [load(s[1]) for s in SFTS]
overall = [val(d, lambda s: True) for d in data]
high    = [val(d, lambda s: s == "high") for d in data]
emr     = [emerg(d) for d in data]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
x = np.arange(len(SFTS)); w = 0.36
axL.bar(x-w/2, overall, w, color="#4c78a8", label="overall (aggregate)")
axL.bar(x+w/2, high,    w, color="#e45756", label="high-risk subset")
axL.axhline(0, color="k", lw=0.6)
axL.set_xticks(x); axL.set_xticklabels([s[0] for s in SFTS]); axL.set_ylabel("value (mean reward)")
axL.set_title("A coverage-biased SFT hides a high-risk hole\nbehind a strong aggregate score")
axL.legend(fontsize=9)
for xi, (o, h) in enumerate(zip(overall, high)):
    axL.annotate(f"{o:+.2f}", (xi-w/2, o), textcoords="offset points", xytext=(0,3), ha="center", fontsize=8)
    axL.annotate(f"{h:+.2f}", (xi+w/2, h), textcoords="offset points", xytext=(0,3 if h>=0 else -12), ha="center", fontsize=8)

axR.bar(x, emr, 0.5, color=["#b0b0b0", "#2e7d32"])
axR.set_xticks(x); axR.set_xticklabels([s[0] for s in SFTS]); axR.set_ylabel("emergent_tx sample rate on high-risk")
axR.set_title("Bias severity sets latent support (Gate 2),\nwhich predicts RL repairability")
axR.set_ylim(0, max(emr)*1.4 + 0.05)
for xi, (e, g) in enumerate(zip(emr, [s[2] for s in SFTS])):
    axR.annotate(f"{e:.2f}\n{g}", (xi, e), textcoords="offset points", xytext=(0,4), ha="center", fontsize=9,
                 color=("#7a2230" if "FAIL" in g else "#1b5e20"), fontweight="bold")
fig.suptitle("SFT coverage gap on diagnostics (high-risk under-sampled): the hole, and whether RL can repair it",
             fontsize=11)
plt.tight_layout(rect=[0,0,1,0.94])
out = "results/figs/coverage_hole_gate2.png"; plt.savefig(out, dpi=130)
print("saved", out)
for (lab,_,g), o, h, e in zip(SFTS, overall, high, emr):
    print(f"  {lab.split(chr(10))[0]:16} overall={o:+.3f} high={h:+.3f} emergent_rate={e:.3f} {g}")
