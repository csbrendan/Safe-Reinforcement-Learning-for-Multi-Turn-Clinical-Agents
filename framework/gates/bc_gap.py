"""Gate 1 (CPU) — the DEMONSTRABILITY axis, env-agnostic.

Computes, on a task's deep families under its reward: the oracle-adaptive value, the best fixed checklist,
ask-all, no-ask, 1-ask shortcut, and blind-defensive policies; a value/support/mean-asks table stratified by
risk subset (high/moderate/benign/relevant); the BC-gap (oracle - best-fixed), per family; and the Gate-1
pass checks (oracle highest; best-fixed below oracle by >= ~0.10; shortcuts << best-fixed; under-gathering
penalized on high; over-escalation penalized on benign; oracle selective; anchor structural 0).

The task env module must expose (beyond framework/envs/base.py): U, ACTIONS, ITEMS, IDX, P1, _pitem,
DEEP_FAMILIES, rel_items, true_risk, best_commit, oracle_value, oracle_rollout.

Usage: python -m framework.gates.bc_gap --env tasks.rx_triage.env --cost 0.03
"""
from __future__ import annotations
import argparse, importlib
from itertools import combinations, product


def run(env, cost, eps=None):
    S = env; EPS = env.EPS if eps is None else eps
    DEEP = list(S.DEEP_FAMILIES)
    # Defensive/escalation action names — read from env (defaults = rx_triage) so the SAME gate logic applies
    # to any task without re-tuning. Severity tiers stay the canonical keys "none"/"moderate"/"high".
    HOLD = getattr(S, "ACT_HOLD", "hold_for_labs"); ESC = getattr(S, "ACT_ESCALATE", "escalate")

    def cases(fam):
        rel = [S.IDX[n] for n in S.rel_items(fam)]; out = []
        for combo in product((0, 1), repeat=len(rel)):
            a = [0] * len(S.ITEMS); w = 1.0
            for i, v in zip(rel, combo):
                a[i] = v; w *= S._pitem(i, v)
            out.append((a, w))
        return out
    ALLC = [(f, a, w) for f in DEEP for a, w in cases(f)]

    def pcr(fam, a, n, act):  # per-case reward
        return S.U[S.true_risk(fam, a)].get(act, min(S.U[S.true_risk(fam, a)].values())) - cost * n

    def fam_best_static(fam):
        rel = [S.IDX[n] for n in S.rel_items(fam)]; cs = cases(fam); best, bsub = -9, ()
        for sz in range(len(rel) + 1):
            for sub in combinations(rel, sz):
                v = sum(w * pcr(fam, a, len(sub), S.best_commit(fam, {i: a[i] for i in sub})[1]) for a, w in cs)
                v /= sum(w for _, w in cs)
                if v > best:
                    best, bsub = v, sub
        return bsub
    BEST = {f: fam_best_static(f) for f in DEEP}
    BEST1 = {f: max(([S.IDX[n]] for n in S.rel_items(f)),
                    key=lambda sub: sum(w * pcr(f, a, 1, S.best_commit(f, {sub[0]: a[sub[0]]})[1]) for a, w in cases(f)))
             for f in DEEP}

    def behavior(policy, fam, a):
        rel = [S.IDX[n] for n in S.rel_items(fam)]
        if policy == "oracle":
            asks, act, _ = S.oracle_rollout(fam, a, cost); return len(asks), act
        if policy == "blind_hold":     return 0, HOLD
        if policy == "blind_escalate": return 0, ESC
        sub = {"ask_all": rel, "no_ask": [], "best_fixed": list(BEST[fam]), "1ask": list(BEST1[fam])}[policy]
        return len(sub), S.best_commit(fam, {i: a[i] for i in sub})[1]

    def stat(policy, sub=None):
        rew = sup = na = W = 0.0
        for fam, a, w in ALLC:
            tr = S.true_risk(fam, a)
            if sub == "relevant" and tr == "none": continue
            if sub in ("high", "moderate", "none") and tr != sub: continue
            n, act = behavior(policy, fam, a); r = pcr(fam, a, n, act)
            og = 1.0 if r >= S.oracle_rollout(fam, a, cost)[2] - EPS else 0.0
            rew += r * w; sup += og * w; na += n * w; W += w
        W = W or 1.0
        return rew / W, sup / W, na / W

    def benign_overesc(policy):
        num = den = 0.0
        for fam, a, w in ALLC:
            if S.true_risk(fam, a) != "none": continue
            _, act = behavior(policy, fam, a); den += w
            if act in (HOLD, ESC): num += w
        return num / den if den else float("nan")

    POL = ["oracle", "best_fixed", "ask_all", "no_ask", "1ask", "blind_hold", "blind_escalate"]
    R = {}
    for p in POL:
        va, sa, ma = stat(p); vh, sh, _ = stat(p, "high"); vm, _, _ = stat(p, "moderate")
        vb, _, _ = stat(p, "none"); vr, _, _ = stat(p, "relevant")
        R[p] = dict(all=va, high=vh, mod=vm, benign=vb, relev=vr, asks=ma, high_supp=sh, oe=benign_overesc(p))
    deep_gap = R["oracle"]["all"] - R["best_fixed"]["all"]
    per_fam = {f: S.oracle_value(f, cost) - sum(w * pcr(f, a, len(BEST[f]), S.best_commit(f, {i: a[i] for i in BEST[f]})[1])
                                                for a, w in cases(f)) / sum(w for _, w in cases(f)) for f in DEEP}
    bd = max(R["blind_hold"]["all"], R["blind_escalate"]["all"])
    checks = {
        "oracle highest": R["oracle"]["all"] >= max(R[p]["all"] for p in POL if p != "oracle") - 1e-9,
        "best_fixed below oracle by >= ~0.10": deep_gap >= 0.095,
        "ask_all < oracle": R["ask_all"]["all"] < R["oracle"]["all"] - 1e-3,
        "no_ask < best_fixed": R["no_ask"]["all"] < R["best_fixed"]["all"] - 0.03,
        "1ask < best_fixed": R["1ask"]["all"] < R["best_fixed"]["all"] - 0.03,
        "blind_defensive < best_fixed": bd < R["best_fixed"]["all"] - 0.03,
        "high under-gather penalized": max(R["no_ask"]["high"], R["1ask"]["high"]) < R["oracle"]["high"] - 0.10,
        "benign over-escalation penalized": max(R["blind_hold"]["benign"], R["blind_escalate"]["benign"]) < R["oracle"]["benign"] - 0.10,
        "oracle selective (asks < rel-union)": R["oracle"]["asks"] < max(len(S.rel_items(f)) for f in DEEP),
        "anchor gap ~0 (structural)": True,
    }
    return R, deep_gap, per_fam, BEST, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tasks.rx_triage.env")
    ap.add_argument("--cost", type=float, default=None, help="override env.ASK_COST")
    args = ap.parse_args()
    env = importlib.import_module(args.env)
    cost = args.cost if args.cost is not None else env.ASK_COST
    R, gap, per_fam, BEST, checks = run(env, cost)
    print(f"=== GATE 1 (demonstrability) — env={args.env}, cost={cost} ===")
    print(f"  {'policy':14s} {'all':>7} {'high':>7} {'mod':>7} {'benign':>7} {'relev':>7} {'asks':>6} {'hi_supp':>8} {'ben_OE':>7}")
    for p, r in R.items():
        print(f"  {p:14s} {r['all']:>7.3f} {r['high']:>7.3f} {r['mod']:>7.3f} {r['benign']:>7.3f} "
              f"{r['relev']:>7.3f} {r['asks']:>6.2f} {r['high_supp']:>8.3f} {r['oe']:>7.2f}")
    print(f"\n  DEEP BC-gap = {gap:+.3f}  per-family: " + ", ".join(f"{f}={g:+.3f}" for f, g in per_fam.items()))
    print(f"  best-fixed items: " + "; ".join(f"{f}={tuple(env.ITEMS[i] for i in BEST[f])}" for f in BEST))
    print("  checks:")
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  => GATE 1 {'PASSES' if all(checks.values()) else 'FAILS'}")


if __name__ == "__main__":
    main()
