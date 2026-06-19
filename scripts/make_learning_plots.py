"""First-pass learning/debug dashboard for the rx_triage gated-RL experiments (provisional, NOT paper figures).

Plots are built from committed probe records (results/probe_*_t10.records.jsonl) and known analytic Gate-1
baselines / committed summary numbers. No training, no env/reward changes. Saves PNGs under
results/figs/learning_dashboard/ and is summarized by docs/learning_dashboard.md.

Run: python scripts/make_learning_plots.py
"""
from __future__ import annotations
import json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figs" / "learning_dashboard"
OUT.mkdir(parents=True, exist_ok=True)

# ---- analytic Gate-1 policies (widened env, cost 0.03) ----------------------------------------------
G1 = {  # policy: (value, mean_asks)
    "oracle": (0.854, 2.94), "best-fixed": (0.749, 5.0), "ask-all": (0.747, 6.5),
    "1-ask": (0.608, 1.0), "no-ask": (0.601, 0.0), "blind-escalate": (0.376, 0.0),
}
INTENDED = {"oracle", "best-fixed"}   # the policies we WANT; the rest are priced-out shortcuts

# ---- learned/base policies (committed Gate-2 @ MAX_TURNS=10) ----------------------------------------
MODELS = ["base", "smoke-SFT", "GRPO #1", "improved-SFT", "v3-SFT", "v4-SFT", "v5-SFT"]
M = {
    "base":         dict(reward=0.386, support=0.58, asks=2.19, one_ask=0.73, no_commit=0.085, hi_reward=0.151, hi_support=0.00),
    "smoke-SFT":    dict(reward=0.168, support=0.60, asks=4.48, one_ask=0.44, no_commit=0.304, hi_reward=-0.428, hi_support=0.14),
    "GRPO #1":      dict(reward=0.261, support=0.85, asks=1.10, one_ask=0.98, no_commit=0.004, hi_reward=-0.770, hi_support=0.43),
    "improved-SFT": dict(reward=0.409, support=0.93, asks=1.33, one_ask=0.90, no_commit=0.010, hi_reward=-0.080, hi_support=1.00),
    "v3-SFT":       dict(reward=0.318, support=0.967, asks=2.78, one_ask=0.539, no_commit=0.196, hi_reward=0.195, hi_support=1.00),
    "v4-SFT":       dict(reward=0.372, support=0.883, asks=1.53, one_ask=0.796, no_commit=0.029, hi_reward=0.197, hi_support=1.00),
    "v5-SFT":       dict(reward=0.411, support=0.95, asks=1.20, one_ask=0.918, no_commit=0.011, hi_reward=0.069, hi_support=1.00),
}
BEST_FIXED, ORACLE = 0.749, 0.854
RECS = {"base": "probe_base_t10", "smoke-SFT": "probe_sft_t10", "GRPO #1": "probe_grpo_t10",
        "improved-SFT": "probe_sft_v2_t10", "v3-SFT": "probe_sft_v3_t10", "v4-SFT": "probe_sft_v4_t10",
        "v5-SFT": "probe_sft_v5_t10"}
COL = {"base": "#7f7f7f", "smoke-SFT": "#1f77b4", "GRPO #1": "#d62728", "improved-SFT": "#2ca02c",
       "v3-SFT": "#9467bd", "v4-SFT": "#8c564b", "v5-SFT": "#e377c2"}
saved = []


def _save(fig, name, caption):
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=7.5, style="italic", wrap=True)
    fig.subplots_adjust(bottom=0.22)
    p = OUT / name
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(p))


def ask_composition(tag):
    """From records (hidden): fraction of rollouts in {no-commit, 0,1,2,3+ asks}."""
    f = ROOT / "results" / f"{RECS[tag]}.records.jsonl"
    recs = [json.loads(l) for l in open(f) if json.loads(l).get("cond") == "hidden"]
    n = len(recs); b = {"no-commit": 0, "0 asks": 0, "1 ask": 0, "2 asks": 0, "3+ asks": 0}
    for r in recs:
        if r["decision"] is None:
            b["no-commit"] += 1
        else:
            k = r["n_asks"]
            b["0 asks" if k == 0 else "1 ask" if k == 1 else "2 asks" if k == 2 else "3+ asks"] += 1
    return {k: v / max(n, 1) for k, v in b.items()}


# ============================================================ 1. Gate-1 policy value bar
def plot1():
    pols = list(G1)
    vals = [G1[p][0] for p in pols]
    colors = ["#2ca02c" if p in INTENDED else "#bbbbbb" for p in pols]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(pols, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(ORACLE, ls="--", c="#2ca02c", lw=1, alpha=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("analytic policy value"); ax.set_ylim(0, 0.95)
    ax.set_title("Gate 1 — policy values (widened RX triage, ask-cost 0.03)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "01_gate1_policy_values.png",
          "Gate 1 PASS: the oracle adaptive policy (0.854) beats the best demonstrable fixed checklist (0.749, "
          "the BC-gap), and every shortcut — no-ask, 1-ask, ask-all, blind-escalate — is priced strictly below it. "
          "The env genuinely rewards selective adaptive questioning, not shortcuts.")


# ============================================================ 2. Gate-1 asks vs reward scatter
def plot2():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for p, (v, a) in G1.items():
        c = "#2ca02c" if p in INTENDED else "#888888"
        ax.scatter(a, v, s=90, color=c, edgecolor="black", zorder=3)
        ax.annotate(p, (a, v), textcoords="offset points", xytext=(6, 6), fontsize=8.5)
    ax.set_xlabel("mean asks (info gathered)"); ax.set_ylabel("policy value")
    ax.set_title("Gate 1 — efficient frontier: value vs asks")
    ax.grid(alpha=0.3)
    _save(fig, "02_gate1_asks_vs_value.png",
          "The value-vs-cost frontier. no-ask/blind (0 asks) are cheap but unsafe; ask-all (~6.5) pays too much "
          "ask-cost; the oracle sits at the knee (~2.9 asks) by gathering SELECTIVELY. Adaptive selection — not "
          "more or fewer asks per se — is what wins.")


# ============================================================ 3. Model comparison (small multiples)
def plot3():
    specs = [("reward", "mean reward", (-0.85, 0.9), [BEST_FIXED, ORACLE]),
             ("support", "support@12", (0, 1.05), None),
             ("asks", "mean asks", (0, 7), [2.94]),
             ("one_ask", "1-ask rate", (0, 1.05), [0.6]),
             ("no_commit", "no-commit rate", (0, 0.4), None),
             ("hi_reward", "high-risk reward", (-0.85, 0.4), [0.0]),
             ("hi_support", "high-risk support@12", (0, 1.05), None)]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    axes = axes.flatten()
    for ax, (key, title, ylim, refs) in zip(axes, specs):
        vals = [M[m][key] for m in MODELS]
        ax.bar(MODELS, vals, color=[COL[m] for m in MODELS], edgecolor="black", linewidth=0.4)
        ax.axhline(0, c="black", lw=0.6)
        if refs:
            for ref in refs:
                ax.axhline(ref, ls="--", c="purple", lw=1, alpha=0.6)
        ax.set_title(title, fontsize=10); ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=35, labelsize=7.5)
    axes[-1].axis("off")
    fig.suptitle("Model comparison — aggregate reward/support hide the behavior (dashed = analytic refs)", fontsize=12)
    _save(fig, "03_model_comparison.png",
          "Aggregate reward/support are misleading: GRPO #1 and improved-SFT post high support@12 yet mean asks "
          "collapse toward 1 and high-risk reward craters (GRPO -0.77). Behavior metrics (asks, 1-ask rate, "
          "no-commit, high-risk reward) tell the real story; purple dashes mark best-fixed/oracle/oracle-asks/guards.")


# ============================================================ 4. Ask-behavior composition (stacked)
def plot4():
    order = ["no-commit", "0 asks", "1 ask", "2 asks", "3+ asks"]
    cols = {"no-commit": "#000000", "0 asks": "#d62728", "1 ask": "#ff7f0e",
            "2 asks": "#1f77b4", "3+ asks": "#2ca02c"}
    comp = {m: ask_composition(m) for m in MODELS}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bottom = np.zeros(len(MODELS))
    for seg in order:
        vals = np.array([comp[m][seg] for m in MODELS])
        ax.bar(MODELS, vals, bottom=bottom, label=seg, color=cols[seg], edgecolor="white", linewidth=0.5)
        bottom += vals
    ax.set_ylabel("fraction of hidden rollouts"); ax.set_ylim(0, 1.0)
    ax.set_title("Ask-behavior composition (hidden, from probe records)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, "04_ask_behavior_composition.png",
          "Each pathology is visible: smoke-SFT has a large black no-commit band + heavy 3+ asks (over-asks, "
          "fails to commit); GRPO #1 is almost entirely orange 1-ask (shortcut collapse); improved-SFT commits "
          "(no black) but is still mostly 1-ask (under-gathers). A clean baseline should be mostly green 3+ asks.")


# ============================================================ 5. Failure-mode heatmap (severity 0/1/2)
def plot5():
    cols = ["non-termination", "1-ask shortcut", "high-risk\nregression", "benign\nover-escalation",
            "over-gathering", "under-gathering"]
    # severity 0=controlled, 1=mild, 2=severe; derived from committed metrics (thresholds in docs)
    sev = {
        "base":          [0, 1, 2, 1, 0, 1],   # mild 1-ask, severe high-risk fail (supp 0), benign shortcut, mild under
        "smoke-SFT":     [2, 0, 2, 0, 2, 0],   # severe non-termination + over-gathering, poor high-risk reward
        "GRPO #1":       [0, 2, 2, 0, 0, 2],   # severe 1-ask collapse + high-risk reward regress + under-gather
        "improved-SFT":  [0, 2, 1, 1, 0, 2],   # termination fixed; 1-ask severe; under-gather severe; high-risk reward mild-neg
        "v3-SFT":        [1, 0, 0, 0, 0, 0],   # gathering+safety fixed; only MILD non-termination (no-commit 0.196)
        "v4-SFT":        [0, 2, 0, 0, 0, 2],   # validity+termination+safety fixed; reverted to 1-ask under-gathering
        "v5-SFT":        [0, 2, 0, 0, 0, 2],   # best reward/validity but deepest 1-ask (greedy 1.00) under-gather
        "target":        [0, 0, 0, 0, 0, 0],
    }
    rows = ["base", "smoke-SFT", "GRPO #1", "improved-SFT", "v3-SFT", "v4-SFT", "v5-SFT", "target"]
    data = np.array([sev[r] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, ("·", "mild", "SEV")[int(data[i, j])], ha="center", va="center", fontsize=8,
                    color="black" if data[i, j] < 1.5 else "white")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.7); cb.ax.set_yticklabels(["controlled", "mild", "severe"])
    ax.set_title("Failure-mode severity by stage")
    _save(fig, "05_failure_mode_heatmap.png",
          "Each stage fixes one pathology and exposes another: smoke-SFT = non-termination + over-gathering; "
          "GRPO #1 = 1-ask collapse + high-risk regression; improved-SFT fixes termination but is severe on "
          "1-ask/under-gathering. v3 targets all-controlled (bottom row = goal, not measured yet).")


# ============================================================ 6. Curriculum progression (small multiples + v3 band)
def plot6():
    stages = ["smoke-SFT", "improved-SFT", "v3-SFT", "v4-SFT", "v5-SFT"]
    specs = [("no_commit", "no-commit rate", (0, 0.35), (0.0, 0.05), "near 0"),
             ("one_ask", "1-ask rate", (0, 1.0), (0.0, 0.6), "< 0.6 guard"),
             ("asks", "mean asks", (0, 6.5), (2.8, 4.0), "~2.8-4 (risk-graded)"),
             ("hi_support", "high-risk support", (0, 1.1), (0.8, 1.0), "preserved"),
             ("reward", "reward", (0, 0.9), (0.65, 0.749), "-> best-fixed")]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))
    xs = [0, 1, 2, 3, 4]
    for ax, (key, title, ylim, band, lbl) in zip(axes, specs):
        ys = [M[s][key] for s in stages]
        ax.plot(xs, ys, "-o", color="#333", zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
        ax.axhspan(band[0], band[1], color="#2ca02c", alpha=0.18, label=f"target: {lbl}")
        ax.set_xticks([0, 1, 2, 3, 4, 5]); ax.set_xticklabels(["smoke", "v2", "v3", "v4", "v5", "tgt"], fontsize=7.5)
        ax.set_xlim(-0.3, 5.3); ax.set_ylim(*ylim); ax.set_title(title, fontsize=10)
        ax.scatter([5], [sum(band) / 2], marker="*", s=150, color="#2ca02c", zorder=4)
        ax.grid(alpha=0.25)
    fig.suptitle("Curriculum oscillation: only v3 gathers (2.78); v2/v4/v5 collapse to ~1-ask despite 68-70% full-checklist demos — gathering is unstable under SFT", fontsize=9.6)
    _save(fig, "06_curriculum_progression.png",
          "No SFT point holds BOTH gather and commit. Only v3 gathered (2.78 asks); v2/v4/v5 — different "
          "mix/epochs — all collapsed to ~1.2-1.5 asks (v5, the 'interpolation', gathered LEAST). The gather mode "
          "is bistable and not reliably controllable by curriculum; green band/star = the still-unmet target.")


# ============================================================ 7. Gate / safety status matrix (heatmap)
def plot7():
    cols = ["BC-gap\npasses", "latent\nsupport", "parse\nvalid", "no-commit\ncontrolled", "1-ask\ncontrolled",
            "high-risk\nsupport", "high-risk\nreward ok", "benign\nover-esc ok", "no exploit\nverdict"]
    N = np.nan
    grid = {  # parse>=0.95; no-commit<0.05; 1-ask<0.6; high-supp>0; high-rew>=-0.1; benign-oe<0.4
        "Gate 1 (widened env)": [1, N, N, N, N, N, N, N, N],
        "base":                 [1, 1, 1, 1, 0, 0, 1, 1, 1],
        "smoke-SFT":            [1, 1, 1, 0, 1, 1, 0, 1, 1],
        "GRPO #1":              [1, 1, 1, 1, 0, 1, 0, 1, 0],
        "improved-SFT":         [1, 1, 1, 1, 0, 1, 1, 1, 0],
        "v3-SFT":               [1, 1, 0, 0, 1, 1, 1, 1, 1],
        "v4-SFT":               [1, 1, 1, 1, 0, 1, 1, 1, 1],
        "v5-SFT":               [1, 1, 1, 1, 0, 1, 1, 1, 1],
        "clean target":         [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    rows = ["Gate 1 (widened env)", "base", "smoke-SFT", "GRPO #1", "improved-SFT", "v3-SFT", "v4-SFT", "v5-SFT", "clean target"]
    data = np.array([grid[r] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#2ca02c"])
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    im.cmap.set_bad("#dddddd")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = data[i, j]
            txt = "—" if np.isnan(v) else ("✓" if v == 1 else "✗")
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if (not np.isnan(v)) else "#888")
    ax.set_title("Gate / safety status by stage  (✓ pass·controlled   ✗ fail   — n/a)")
    _save(fig, "07_gate_status_matrix.png",
          "One-glance status. Gate 1 (BC-gap) and Gate 2 (latent support) both pass. Every learned stage trips ≥1 "
          "check: v3-SFT alone passes '1-ask controlled' but fails parse + no-commit; v4-SFT fixes those but "
          "reverts to failing '1-ask controlled'. The validity↔gathering tradeoff means no SFT stage is yet "
          "all-green ('clean target', bottom) — the bar to clear before any GRPO #2.")


# ============================================================ 8. Temperature sweep (learned vs sampling)
def plot8():
    sweeps = {}
    for v in ("v3", "v4", "v5"):
        p = ROOT / "results" / f"temp_sweep_{v}.json"
        if p.exists():
            sweeps[v] = json.load(open(p))["by_temp"]
    if not sweeps:
        return
    col = {"v3": "#9467bd", "v4": "#8c564b", "v5": "#e377c2"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, by in sweeps.items():
        ts = sorted(float(t) for t in by)
        axes[0].plot(ts, [by[str(t)]["mean_asks"] for t in ts], "-o", color=col[v], label=v)
        axes[1].plot(ts, [by[str(t)]["high_reward"] for t in ts], "-o", color=col[v], label=v)
    axes[0].axhline(ORACLE_ASKS := 2.94, ls="--", c="green", alpha=0.6, label="oracle asks 2.94")
    axes[0].axhline(1.0, ls=":", c="red", alpha=0.6, label="1-ask collapse")
    axes[0].set_xlabel("decoding temperature"); axes[0].set_ylabel("mean asks"); axes[0].set_ylim(0, 3.6)
    axes[0].set_title("mean asks vs temperature"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, c="black", lw=0.6)
    axes[1].set_xlabel("decoding temperature"); axes[1].set_ylabel("high-risk reward")
    axes[1].set_title("high-risk reward vs temperature"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Temperature diagnostic: under-gathering is a LEARNED policy, not a sampling artifact", fontsize=12)
    _save(fig, "08_temperature_diagnostic.png",
          "At greedy (temp 0) v4/v5 commit after exactly 1 ask (mean 1.00) — under-gathering is the learned policy, "
          "not stochastic eval; higher temp ADDS a little gathering (opposite of the confound hypothesis). v3 "
          "gathers at every temp (greedy 2.23). So eval temperature is neither the cause nor the fix.")


if __name__ == "__main__":
    plot1(); plot2(); plot3(); plot4(); plot5(); plot6(); plot7(); plot8()
    print("Saved learning-dashboard figures:")
    for p in saved:
        print("  " + p)
