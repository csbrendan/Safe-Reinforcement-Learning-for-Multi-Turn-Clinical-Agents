"""rx_triage checkpoint trajectory: GRPO-control vs GRPO-info-gain across training (SFT-v3 = step 0).
Shows the honest finding: info-gain DELAYS high-risk erosion + ask-trimming but does not prevent it.
Held-out hidden, pure env.score. high-risk reward = mean reward on true_risk=='high'; asks = mean n_asks.
"""
import json, statistics as stx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEPS = [0, 20, 40, 60]
CTRL = {0: "results/probe_sft_v3_t10.records.jsonl", 20: "results/probe_grpo2_s20_t10.records.jsonl",
        40: "results/probe_grpo2_s40_t10.records.jsonl", 60: "results/probe_grpo2_t10.records.jsonl"}
IG   = {0: "results/probe_sft_v3_t10.records.jsonl", 20: "results/probe_rx_ig_s20_t10.records.jsonl",
        40: "results/probe_rx_ig_s40_t10.records.jsonl", 60: "results/probe_rx_ig_final_t10.records.jsonl"}

def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def hr_reward(recs):
    h = [r["reward"] for r in recs if r["true_risk"] == "high"]; return stx.mean(h) if h else 0.0
def mean_asks(recs): return stx.mean(r["n_asks"] for r in recs)
def one_ask(recs):
    by = {}
    for r in recs: by.setdefault(r["case_id"], []).append(r)
    return stx.mean(stx.mean(1.0 if x["n_asks"] == 1 else 0.0 for x in v) for v in by.values())

ctrl = {s: load(CTRL[s]) for s in STEPS}; ig = {s: load(IG[s]) for s in STEPS}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (fn, title, ylab, hl) in zip(axes, [
        (hr_reward, "High-risk reward (true_risk=high)", "mean reward", 0.195),
        (mean_asks, "Mean asks (gathering)", "asks", None),
        (one_ask,   "1-ask rate (shortcut)", "P(exactly 1 ask)", None)]):
    ax.plot(STEPS, [fn(ctrl[s]) for s in STEPS], "o-", color="#d1495b", label="GRPO control")
    ax.plot(STEPS, [fn(ig[s]) for s in STEPS], "s-", color="#2e7d32", label="GRPO info-gain")
    if hl is not None: ax.axhline(hl, ls="--", color="#1f77b4", lw=1, label="SFT-v3 (step 0)")
    ax.set_xlabel("GRPO step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks(STEPS); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("rx_triage: info-gain DELAYS but does not prevent high-risk erosion / ask-trimming "
             "(held-out hidden, pure env.score)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "results/figs/infogain_rx_checkpoint_trajectory.png"
plt.savefig(out, dpi=130); print("saved", out)
print(f"{'step':>4} | {'ctrl HRrew':>10} {'ig HRrew':>9} | {'ctrl asks':>9} {'ig asks':>8} | {'ctrl 1ask':>9} {'ig 1ask':>8}")
for s in STEPS:
    print(f"{s:>4} | {hr_reward(ctrl[s]):>10.3f} {hr_reward(ig[s]):>9.3f} | "
          f"{mean_asks(ctrl[s]):>9.2f} {mean_asks(ig[s]):>8.2f} | {one_ask(ctrl[s]):>9.2f} {one_ask(ig[s]):>8.2f}")
