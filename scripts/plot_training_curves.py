"""Training diagnostics from the actual run logs (no invented data).
(A) Diagnostics RL: trl reward, KL, entropy, and rollout mean_asks over GRPO steps for control / info-gain / DREAM.
(B) Biased-SFT loss curve over steps.
Parses logs_*.log lines emitted by trl + the rollout/dream-diag prints.
"""
import re, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def trl_series(path, key):
    vals = []
    for l in open(path, errors="ignore"):
        m = re.search(r"'%s': '([0-9.eE+-]+)'" % re.escape(key), l)
        if m:
            try: vals.append(float(m.group(1)))
            except ValueError: pass
    return vals

def asks_series(path):
    out = []
    for l in open(path, errors="ignore"):
        m = re.search(r"mean_asks=([0-9.]+)", l)
        if m: out.append(float(m.group(1)))
    return out

RUNS = [  # label, log, color
    ("plain GRPO",  "logs_dhgrpo.log",    "#d1495b"),
    ("info-gain",   "logs_dhgrpo_ig.log", "#2e7d32"),
    ("DREAM",       "logs_dream_dh.log",  "#3b6fb6"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [("reward", "trl reward (optimized objective)"), ("kl", "KL to init"),
          ("entropy", "policy entropy"), (None, "rollout mean asks")]
for ax, (key, title) in zip(axes.flat, panels):
    for lab, log, c in RUNS:
        y = asks_series(log) if key is None else trl_series(log, key)
        if y: ax.plot(range(1, len(y)+1), y, lw=1.4, color=c, label=lab, alpha=0.9)
    ax.set_title(title); ax.set_xlabel("GRPO step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes.flat[3].axhline(3.9, ls="--", color="#888", lw=1, label="oracle asks")
fig.suptitle("Diagnostics deephedge: RL training diagnostics (from run logs)", fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
out = "results/figs/training_curves_diagnostics.png"; plt.savefig(out, dpi=130); print("saved", out)

# (B) SFT loss
fig2, ax = plt.subplots(figsize=(6.4, 4.2))
steps, loss = [], []
for l in open("logs_dx_sft_biased30.log", errors="ignore"):
    m = re.search(r"step (\d+)/\d+\] loss=([0-9.]+)", l)
    if m: steps.append(int(m.group(1))); loss.append(float(m.group(2)))
ax.plot(steps, loss, "-o", ms=3, color="#444")
ax.set_xlabel("SFT step"); ax.set_ylabel("training loss")
ax.set_title("Biased-SFT (diagnostics) training loss"); ax.grid(alpha=0.3)
plt.tight_layout(); out2 = "results/figs/sft_loss_curve.png"; plt.savefig(out2, dpi=130); print("saved", out2)
print(f"  diagnostics reward steps: ctrl={len(trl_series('logs_dhgrpo.log','reward'))} ig={len(trl_series('logs_dhgrpo_ig.log','reward'))} dream={len(trl_series('logs_dream_dh.log','reward'))}")
print(f"  sft loss points: {len(loss)} (final loss {loss[-1] if loss else 'NA'})")
