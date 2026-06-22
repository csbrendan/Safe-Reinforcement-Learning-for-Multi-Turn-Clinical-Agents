"""DREAM diagnostics checkpoint trajectory (base=step0 ref): high-risk reward, high-risk support@12, mean asks.
Shows the high-risk peak at ckpt-40 and the persistent over-gathering (asks > oracle VOI 3.9). Held-out hidden,
pure env.score. Info-gain final shown as a horizontal reference (the bar to beat).
"""
import json, statistics as stx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 20, 40, 60]
REC = {0: "results/probe_dh_base_t12.records.jsonl", 20: "results/probe_dream_s20_t12.records.jsonl",
       40: "results/probe_dream_s40_t12.records.jsonl", 60: "results/probe_dream_final_t12.records.jsonl"}
IG = "results/probe_dh_ig_final_t12.records.jsonl"

def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def hr_reward(r): h = [x["reward"] for x in r if x["true_risk"] == "high"]; return stx.mean(h) if h else 0.0
def hr_supp(r):
    h = [x for x in r if x["true_risk"] == "high"]
    by = {}
    for x in h: by.setdefault(x["case_id"], []).append(x)
    return stx.mean(1.0 if any(y["oracle_grade"] for y in v) else 0.0 for v in by.values()) if by else 0.0
def asks(r): return stx.mean(x["n_asks"] for x in r)

D = {s: load(REC[s]) for s in STEPS}; ig = load(IG)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (fn, title, ylab, igref, hl) in zip(axes, [
        (hr_reward, "High-risk reward", "mean reward", hr_reward(ig), None),
        (hr_supp,   "High-risk support@12", "support@12", hr_supp(ig), None),
        (asks,      "Mean asks (gathering)", "asks", asks(ig), 3.9)]):
    ax.plot(STEPS, [fn(D[s]) for s in STEPS], "o-", color="#6a4c93", label="DREAM")
    ax.axhline(igref, ls="--", color="#2e7d32", lw=1.2, label="info-gain final")
    if hl is not None: ax.axhline(hl, ls=":", color="k", lw=1, label="oracle VOI ~3.9")
    ax.set_xlabel("GRPO step (0 = base)"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks(STEPS); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("DREAM diagnostics deephedge: high-risk peaks at ckpt-40 (reward +0.248 / support 1.00), "
             "explorer over-gathers (asks > oracle VOI). Held-out hidden, pure env.score.", fontsize=10.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/figs/dream_diagnostics_trajectory.png"
plt.savefig(out, dpi=130); print("saved", out)
print(f"{'step':>4} | {'HRrew':>7} {'HRsupp':>7} {'asks':>6}")
for s in STEPS:
    print(f"{s:>4} | {hr_reward(D[s]):>7.3f} {hr_supp(D[s]):>7.2f} {asks(D[s]):>6.2f}")
print(f"  ig | {hr_reward(ig):>7.3f} {hr_supp(ig):>7.2f} {asks(ig):>6.2f}  (info-gain final ref)")
