"""Multi-seed, moving-average-smoothed RL training curves for diagnostics deephedge.
For each variant (plain GRPO / info-gain / DREAM) we average per-step metrics across available seeds and shade
+/- SEM; a centered 10-step moving average reduces step noise. Metrics are read from the run logs (trl per-step
loss/reward/kl/entropy + rollout/dream-diag mean_asks). Support is an EVAL metric (per checkpoint), not per-step,
so it is not plotted here. x = training step; each panel's y is labeled; legend annotates variant and #seeds.
"""
import re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# variant -> list of seed logs (only existing ones are used)
import os
RUNS = {
    "plain GRPO": ["logs_dhgrpo.log", "logs_dh_ctl_s2.log", "logs_dh_ctl_s3.log"],
    "info-gain":  ["logs_dhgrpo_ig.log", "logs_dh_ig_s2.log", "logs_dh_ig_s3.log"],
    "DREAM":      ["logs_dream_dh.log"],
}
COLORS = {"plain GRPO": "#d1495b", "info-gain": "#2e7d32", "DREAM": "#3b6fb6"}

def trl(path, key):
    return [float(m.group(1)) for l in open(path, errors="ignore")
            for m in [re.search(r"'%s': '([0-9.eE+-]+)'" % key, l)] if m]
def asks(path):
    return [float(m.group(1)) for l in open(path, errors="ignore")
            for m in [re.search(r"mean_asks=([0-9.]+)", l)] if m]

def parse(path, metric):
    return asks(path) if metric == "mean_asks" else trl(path, metric)

def smooth(y, w=10):
    y = np.asarray(y, float)
    if len(y) == 0: return y
    out = np.empty_like(y)
    h = w // 2
    for i in range(len(y)):
        lo, hi = max(0, i - h), min(len(y), i + h + 1)
        out[i] = y[lo:hi].mean()
    return out

def stack(variant, metric):
    series = []
    for log in RUNS[variant]:
        if os.path.exists(log):
            y = parse(log, metric)
            if len(y) >= 30: series.append(smooth(y))
    if not series: return None, None, 0
    n = min(len(s) for s in series)
    M = np.vstack([s[:n] for s in series])
    mean = M.mean(0)
    sem = (M.std(0, ddof=1) / np.sqrt(M.shape[0])) if M.shape[0] > 1 else np.zeros(n)
    return mean, sem, M.shape[0]

PANELS = [("mean_asks", "rollout mean asks", 3.9), ("reward", "trl reward (objective)", None),
          ("kl", "KL to init", None), ("entropy", "policy entropy", None)]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (metric, ylab, oracle) in zip(axes.flat, PANELS):
    for v in RUNS:
        mean, sem, n = stack(v, metric)
        if mean is None: continue
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, color=COLORS[v], lw=2, label=f"{v} ({n} seed{'s' if n>1 else ''})")
        if n > 1: ax.fill_between(x, mean - sem, mean + sem, color=COLORS[v], alpha=0.2, linewidth=0)
    if oracle is not None: ax.axhline(oracle, ls="--", color="#888", lw=1, label="oracle asks")
    ax.set_xlabel("training step"); ax.set_ylabel(ylab); ax.set_xlim(1, 60)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
fig.suptitle("Diagnostics deephedge RL training curves (10-step moving average; mean $\\pm$ SEM across seeds)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = "results/figs/rl_curves_seeds.png"; plt.savefig(out, dpi=130); print("saved", out)
for v in RUNS:
    m, s, n = stack(v, "mean_asks")
    print(f"  {v:11} seeds={n}  final mean_asks={m[-1]:.2f}" if m is not None else f"  {v}: no data")
