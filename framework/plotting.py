"""Fresh plotting for NEW experiments only. Reads gate/probe outputs; writes PNGs to results/figs/.

Plots the thesis's headline relationships:
  * rl_gain_vs_demonstrability — RL-SFT advantage vs the BC-gap (across tasks / reward settings)
  * subset_bars               — value or oracle-grade support by risk subset (high / moderate / benign)
  * safety_overescalation     — high-risk support and benign over-escalation, SFT vs RL
  * ask_distribution          — ask-count histogram (0/1/2/3+) per policy

All functions take plain Python data (dicts/lists) so they don't depend on any specific run format; the thin
runners in scripts/ load JSON and call these. No prior figures are carried over.
"""
from __future__ import annotations
from pathlib import Path

FIGS = Path(__file__).resolve().parent.parent / "results" / "figs"


def _ax(title, xl, yl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel(yl)
    return plt, fig, ax


def rl_gain_vs_demonstrability(points, out="rl_gain_vs_bcgap.png"):
    """points: list of (bc_gap, rl_minus_sft, label). The core thesis figure."""
    plt, fig, ax = _ax("RL-SFT advantage vs demonstrability (BC-gap)", "BC-gap (oracle - best fixed)", "RL - SFT (held-out value)")
    if points:
        xs = [p[0] for p in points]; ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=40)
        for x, y, lbl in points:
            ax.annotate(lbl, (x, y), fontsize=7)
        ax.axhline(0, color="k", lw=0.7, ls="--")
    FIGS.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(FIGS / out, dpi=130); plt.close(fig)
    return FIGS / out


def subset_bars(sft, rl, metric="support", out="subset_bars.png"):
    """sft/rl: dict subset->value. Grouped bars SFT vs RL by risk subset."""
    plt, fig, ax = _ax(f"{metric} by risk subset (SFT vs RL)", "risk subset", metric)
    subs = [s for s in ("high", "moderate", "none", "relevant") if s in sft]
    x = range(len(subs))
    ax.bar([i - 0.2 for i in x], [sft[s] for s in subs], 0.4, label="SFT")
    ax.bar([i + 0.2 for i in x], [rl[s] for s in subs], 0.4, label="RL")
    ax.set_xticks(list(x)); ax.set_xticklabels(subs); ax.legend(fontsize=8)
    FIGS.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(FIGS / out, dpi=130); plt.close(fig)
    return FIGS / out


def safety_overescalation(rows, out="safety_overescalation.png"):
    """rows: list of (policy_label, high_support, benign_overescalation_rate)."""
    plt, fig, ax = _ax("Safety: high-risk support vs benign over-escalation", "policy", "rate")
    labels = [r[0] for r in rows]; x = range(len(labels))
    ax.bar([i - 0.2 for i in x], [r[1] for r in rows], 0.4, label="high-risk support")
    ax.bar([i + 0.2 for i in x], [r[2] for r in rows], 0.4, label="benign over-escalation")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7); ax.legend(fontsize=8)
    FIGS.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(FIGS / out, dpi=130); plt.close(fig)
    return FIGS / out


def ask_distribution(hists, out="ask_distribution.png"):
    """hists: dict policy_label -> {'0':p,'1':p,'2':p,'3+':p}. Stacked ask-count distribution."""
    plt, fig, ax = _ax("Ask-count distribution by policy", "policy", "fraction")
    buckets = ["0", "1", "2", "3+"]; labels = list(hists)
    bottom = [0.0] * len(labels)
    for b in buckets:
        vals = [hists[p].get(b, 0.0) for p in labels]
        ax.bar(labels, vals, bottom=bottom, label=f"{b} asks")
        bottom = [bottom[i] + vals[i] for i in range(len(labels))]
    ax.legend(fontsize=8); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    FIGS.mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(FIGS / out, dpi=130); plt.close(fig)
    return FIGS / out
