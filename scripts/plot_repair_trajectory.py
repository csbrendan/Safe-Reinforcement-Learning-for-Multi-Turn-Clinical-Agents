"""SFT-coverage repair trajectory: from the biased SFT (step 0), neither info-gain nor DREAM RL repairs the
high-risk hole -- they fail in OPPOSITE directions (info-gain under-gathers, DREAM over-gathers), and both
erode the SFT's strong overall support. Held-out hidden (n=200), pure env.score. (cf. the rx trajectory chart.)
"""
import json, statistics as stx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 20, 40, 60]
SFT = "results/probe_dx_biased30_n200_t12.records.jsonl"
IG  = {0: SFT, 20: "results/probe_dx_repair_s20_t12.records.jsonl", 40: "results/probe_dx_repair_s40_t12.records.jsonl", 60: "results/probe_dx_repair_final_t12.records.jsonl"}
DR  = {0: SFT, 20: "results/probe_dx_repdream_s20_t12.records.jsonl", 40: "results/probe_dx_repdream_s40_t12.records.jsonl", 60: "results/probe_dx_repdream_final_t12.records.jsonl"}

def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def osupp(recs):
    by={}
    for r in recs: by.setdefault(r["case_id"], []).append(r)
    return stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0 for v in by.values())
def masks(recs): return stx.mean(r["n_asks"] for r in recs)
def hival(recs): h=[r["reward"] for r in recs if r["true_risk"]=="high"]; return stx.mean(h) if h else 0
ig = {s: load(IG[s]) for s in STEPS}; dr = {s: load(DR[s]) for s in STEPS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (fn, title, ylab, oracle) in zip(axes, [
        (masks,  "Mean asks (opposite failures)", "asks", 3.9),
        (osupp,  "Overall oracle-grade support@12", "support@12", None),
        (hival,  "High-risk value", "mean reward (high-risk)", None)]):
    ax.plot(STEPS, [fn(ig[s]) for s in STEPS], "o-", color="#d1495b", label="info-gain repair")
    ax.plot(STEPS, [fn(dr[s]) for s in STEPS], "s-", color="#2e7d32", label="DREAM repair")
    ax.scatter([0], [fn(ig[0])], color="k", zorder=5, label="biased SFT (step 0)")
    if oracle is not None: ax.axhline(oracle, ls="--", color="#3182bd", lw=1, label="oracle (~3.9)")
    ax.set_xlabel("GRPO step"); ax.set_ylabel(ylab); ax.set_title(title); ax.set_xticks(STEPS)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Repairing a biased SFT's high-risk hole fails for BOTH RL variants "
             "(info-gain under-gathers, DREAM over-gathers; held-out hidden)", fontsize=11)
plt.tight_layout(rect=[0,0,1,0.95])
out = "results/figs/repair_trajectory.png"; plt.savefig(out, dpi=130); print("saved", out)
print(f"{'step':>4} | {'IG asks':>8}{'DR asks':>8} | {'IG osupp':>9}{'DR osupp':>9} | {'IG hival':>9}{'DR hival':>9}")
for s in STEPS:
    print(f"{s:>4} | {masks(ig[s]):>8.2f}{masks(dr[s]):>8.2f} | {osupp(ig[s]):>9.2f}{osupp(dr[s]):>9.2f} | {hival(ig[s]):>+9.2f}{hival(dr[s]):>+9.2f}")
