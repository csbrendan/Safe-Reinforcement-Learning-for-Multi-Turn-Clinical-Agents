"""Aggregate the validation alpha sweep -> comparison table, risk-subgroup plots, reward-component plots,
reward diagnostics, training-stability, and a reward-hacking scan + alpha recommendation.

Reads results/procreward_pilot/sweep/: a{A}_{ckpt}_metrics.json (per-checkpoint eval), a{A}_rewardlog.jsonl
(reward components), a{A}_train.log (per-step [rollout-diag]/[claude-judge] + TRL kl/loss). Robust to missing
files (writes what it can). VALIDATION seed 202 only.

  DX_VARIANT=deephedge PYTHONPATH=$PWD python scripts/analyze_alpha_sweep.py
"""
from __future__ import annotations
import json, re, glob, statistics as stx
from pathlib import Path

SW = Path("results/procreward_pilot/sweep")
ALPHAS = ["0.05", "0.10", "0.20"]
CKPTS = ["checkpoint-10", "checkpoint-20", "final"]


def load_metrics(a, ck):
    p = SW / f"a{a}_{ck}_metrics.json"
    return json.load(open(p)) if p.exists() else None


def load_greedy(a):
    p = SW / f"a{a}_greedy_metrics.json"
    return json.load(open(p)) if p.exists() else None


def reward_diag(a):
    p = SW / f"a{a}_rewardlog.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p)]
    if not rows:
        return None
    def d(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return dict(mean=round(stx.mean(xs), 3), sd=round(stx.pstdev(xs), 3),
                    min=round(min(xs), 2), max=round(max(xs), 2)) if xs else None
    # group by (step-block, case): rewardlog rows are appended per step in order, 32/step
    n = len(rows)
    # ordering-change + nonzero-adv per case group within each 32-row step block
    flips = groups = nz = 0
    for s in range(0, n, 32):
        blk = rows[s:s + 32]
        by = {}
        for i, r in enumerate(blk):
            by.setdefault(r["case_id"], []).append(r)
        for g in by.values():
            if len(g) < 2:
                continue
            groups += 1
            if sorted(range(len(g)), key=lambda i: g[i]["analytic"]) != \
               sorted(range(len(g)), key=lambda i: g[i]["total"]):
                flips += 1
            if any(abs(x["claude_norm"]) > 1e-9 for x in g):
                nz += 1
    raws = [(r["claude_raw"], r.get("n_asks")) for r in rows if r.get("claude_raw") is not None and r.get("n_asks") is not None]
    corr = None
    if len(raws) > 2:
        cs, ls = [x[0] for x in raws], [x[1] for x in raws]
        mc, ml = stx.mean(cs), stx.mean(ls)
        den = (sum((c - mc) ** 2 for c in cs) * sum((l - ml) ** 2 for l in ls)) ** 0.5
        corr = round(sum((c - mc) * (l - ml) for c, l in raws) / den, 3) if den else None
    return dict(n=n, analytic=d("analytic"), claude_raw=d("claude_raw"), claude_norm=d("claude_norm"),
                alpha_x_norm=d("alpha_x_norm"), total=d("total"),
                pct_groups_reordered=round(100 * flips / max(groups, 1), 1),
                pct_groups_nonzero_adv=round(100 * nz / max(groups, 1), 1),
                corr_score_len=corr)


def train_stability(a):
    p = SW / f"a{a}_train.log"
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    diag = re.findall(r"mean_asks=([\-\d.]+) parse_validity=([\-\d.]+) committed=([\-\d.]+) mean_reward=([\-\d.]+)", txt)
    jud = re.findall(r"api_calls=(\d+) .*?cache_hits=(\d+)/(\d+) .*?retries=(\d+) parse_fail=(\d+) score_wall=([\d.]+)s api_latency/call=([\d.]+)s", txt)
    # TRL logs (best-effort): kl / loss / reward
    kl = [float(x) for x in re.findall(r"'kl':\s*([\-\d.eE]+)", txt)]
    loss = [float(x) for x in re.findall(r"'loss':\s*([\-\d.eE]+)", txt)]
    ent = [float(x) for x in re.findall(r"'entropy':\s*([\-\d.eE]+)", txt)]
    wall = re.search(r"DONE steps=\d+ wall=(\d+)s", txt)
    api_total = sum(int(j[0]) for j in jud)
    cache_hits = sum(int(j[1]) for j in jud)
    calls_total = sum(int(j[2]) for j in jud)
    retries = sum(int(j[3]) for j in jud)
    pf = sum(int(j[4]) for j in jud)
    swall = sum(float(j[5]) for j in jud)
    return dict(steps=len(diag),
                asks_traj=[float(d[0]) for d in diag],
                reward_traj=[float(d[3]) for d in diag],
                parse_traj=[float(d[1]) for d in diag],
                committed_traj=[float(d[2]) for d in diag],
                kl=kl, loss=loss, entropy=ent,
                api_calls=api_total, cache_hits=cache_hits, calls=calls_total, retries=retries, parse_fail=pf,
                claude_wall_s=round(swall, 0), wall_s=int(wall.group(1)) if wall else None)


PRIMARY = ["value", "unsafe_high_rate", "premature_rate", "mean_regret", "oracle_grade_rate"]
BURDEN = ["benign_overtriage_rate", "mean_asks", "redundant_rate", "one_ask_rate", "over_gather_rate"]


def fmt(m, k):
    return "-" if not m or k not in m else m[k]


def main():
    have = {a: load_metrics(a, "final") for a in ALPHAS}
    diag = {a: reward_diag(a) for a in ALPHAS}
    stab = {a: train_stability(a) for a in ALPHAS}
    L = ["# Validation alpha sweep — Claude process-reward GRPO (deephedge, seed 202)\n",
         "3 GRPO runs, identical config (300 cases, 30 steps, group 8, temp 1.0, seed 1), alpha in "
         "{0.05,0.10,0.20}. Validation seed 202 only; test seed 100 untouched. support@k is a diagnostic.\n"]

    # ---- final-checkpoint comparison, by subgroup ----
    for grp in ("high", "moderate", "none", "overall"):
        L.append(f"## Final validation metrics — {grp} subgroup")
        L.append("| metric | α=0.05 | α=0.10 | α=0.20 |")
        L.append("|---|---|---|---|")
        keys = PRIMARY + BURDEN if grp in ("high", "overall") else ["value", "mean_asks", "benign_overtriage_rate",
                                                                     "premature_rate", "over_gather_rate", "one_ask_rate"]
        for k in keys:
            L.append(f"| {k} | " + " | ".join(str(fmt(have[a].get(grp) if have[a] else None, k)) for a in ALPHAS) + " |")
        L.append("")

    # ---- greedy pass@1 (high) ----
    L.append("## Greedy pass@1 (k=1) high-risk")
    L.append("| metric | α=0.05 | α=0.10 | α=0.20 |")
    L.append("|---|---|---|---|")
    gr = {a: load_greedy(a) for a in ALPHAS}
    for k in ("value", "oracle_grade_rate", "unsafe_high_rate", "mean_asks"):
        L.append(f"| high {k} | " + " | ".join(str(fmt(gr[a].get("high") if gr[a] else None, k)) for a in ALPHAS) + " |")
    L.append("")

    # ---- checkpoint trajectories (high value / unsafe-high / mean asks) ----
    L.append("## Checkpoint trajectories (val seed 202)")
    for k, lbl in [("value", "high value"), ("unsafe_high_rate", "high unsafe rate"), ("mean_asks", "overall mean asks")]:
        L.append(f"- **{lbl}** by checkpoint (10/20/final):")
        for a in ALPHAS:
            row = []
            for ck in CKPTS:
                m = load_metrics(a, ck)
                g = (m.get("high") if k != "mean_asks" else m.get("overall")) if m else None
                row.append(str(fmt(g, k)))
            L.append(f"  - α={a}: " + " -> ".join(row))
    L.append("")

    # ---- reward diagnostics ----
    L.append("## Reward-component diagnostics (training rollouts)")
    L.append("| quantity | α=0.05 | α=0.10 | α=0.20 |")
    L.append("|---|---|---|---|")
    for comp in ("analytic", "claude_raw", "claude_norm", "alpha_x_norm", "total"):
        L.append(f"| {comp} mean±sd | " + " | ".join(
            (f"{diag[a][comp]['mean']}±{diag[a][comp]['sd']}" if diag[a] and diag[a][comp] else "-") for a in ALPHAS) + " |")
    for k, lbl in [("pct_groups_reordered", "% groups reordered"), ("pct_groups_nonzero_adv", "% groups nonzero-adv"),
                   ("corr_score_len", "corr(score,len)")]:
        L.append(f"| {lbl} | " + " | ".join((str(diag[a][k]) if diag[a] else "-") for a in ALPHAS) + " |")
    L.append("")

    # ---- API + stability ----
    L.append("## Claude API + training stability")
    L.append("| quantity | α=0.05 | α=0.10 | α=0.20 |")
    L.append("|---|---|---|---|")
    def sv(a, k):
        return stab[a][k] if stab[a] and k in stab[a] else "-"
    for k, lbl in [("api_calls", "API calls"), ("cache_hits", "cache hits"), ("retries", "retries"),
                   ("parse_fail", "parse fails"), ("claude_wall_s", "Claude wall (s)"), ("wall_s", "train wall (s)")]:
        L.append(f"| {lbl} | " + " | ".join(str(sv(a, k)) for a in ALPHAS) + " |")
    for a in ALPHAS:
        s = stab[a]
        if s:
            L.append(f"\n- α={a}: asks {s['asks_traj']}, reward {s['reward_traj']}, "
                     f"parse {s['parse_traj']}, committed {s['committed_traj']}"
                     + (f", KL {[round(x,4) for x in s['kl'][-5:]]}" if s['kl'] else " (KL not logged by TRL to stdout)"))
    L.append("")

    Path(SW / "alpha_sweep_report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[analyze] wrote {SW/'alpha_sweep_report.md'}")
    make_plots(have, diag)


def make_plots(have, diag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analyze] matplotlib unavailable ({e}); skipping plots")
        return
    x = range(len(ALPHAS)); labels = [f"α={a}" for a in ALPHAS]
    # risk-subgroup plot: high/mod/benign value + high unsafe rate + mean asks
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for grp, c in [("high", "crimson"), ("moderate", "orange"), ("none", "green")]:
        ax[0].plot(x, [have[a].get(grp, {}).get("value") if have[a] else None for a in ALPHAS], "o-", label=grp, color=c)
    ax[0].set_title("analytic value by risk subgroup"); ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].legend()
    ax[1].plot(x, [have[a].get("high", {}).get("unsafe_high_rate") if have[a] else None for a in ALPHAS], "o-", color="crimson", label="unsafe-high")
    ax[1].plot(x, [have[a].get("high", {}).get("premature_rate") if have[a] else None for a in ALPHAS], "s--", color="darkred", label="premature-high")
    ax[1].plot(x, [have[a].get("none", {}).get("benign_overtriage_rate") if have[a] else None for a in ALPHAS], "^:", color="green", label="benign over-triage")
    ax[1].set_title("safety / burden rates"); ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].legend()
    for grp, c in [("high", "crimson"), ("moderate", "orange"), ("none", "green"), ("overall", "black")]:
        ax[2].plot(x, [have[a].get(grp, {}).get("mean_asks") if have[a] else None for a in ALPHAS], "o-", label=grp, color=c)
    ax[2].set_title("mean asks by subgroup"); ax[2].set_xticks(x); ax[2].set_xticklabels(labels); ax[2].legend()
    fig.tight_layout(); fig.savefig(SW / "plot_risk_subgroups.png", dpi=110); plt.close(fig)
    # reward-component plot: mean±sd of components by alpha
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    comps = ["analytic", "claude_norm", "alpha_x_norm", "total"]
    for comp in comps:
        ys = [diag[a][comp]["mean"] if diag[a] and diag[a].get(comp) else None for a in ALPHAS]
        es = [diag[a][comp]["sd"] if diag[a] and diag[a].get(comp) else 0 for a in ALPHAS]
        ax[0].errorbar(x, ys, yerr=es, marker="o", capsize=4, label=comp)
    ax[0].set_title("reward components (mean±sd)"); ax[0].set_xticks(x); ax[0].set_xticklabels(labels); ax[0].legend()
    ax[1].plot(x, [diag[a]["pct_groups_reordered"] if diag[a] else None for a in ALPHAS], "o-", label="% groups reordered")
    ax[1].plot(x, [diag[a]["pct_groups_nonzero_adv"] if diag[a] else None for a in ALPHAS], "s-", label="% nonzero-adv")
    ax[1].set_title("process-signal reach"); ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].legend()
    fig.tight_layout(); fig.savefig(SW / "plot_reward_components.png", dpi=110); plt.close(fig)
    print(f"[analyze] wrote plots -> {SW}/plot_risk_subgroups.png, plot_reward_components.png")


if __name__ == "__main__":
    main()
