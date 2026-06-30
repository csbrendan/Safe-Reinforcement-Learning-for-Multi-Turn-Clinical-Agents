"""Generate ORACLE-adaptive SFT trajectories on diagnostics deephedge, with a deliberate COVERAGE BIAS:
high-risk (emergent) cases are under-sampled (default: dropped). This produces a competent adaptive gatherer
on benign/moderate cases that has a HIGH-RISK HOLE (it rarely/never demonstrates the deep gather + emergent_tx
commit). The biased-SFT->RL cell then tests whether verifiable RL on the FULL environment repairs that hole.

Trajectory = the exact oracle path: ask each oracle-selected finding (with the simulator's answer), then commit
the oracle disposition. A fraction of rho=1.0 commit-now cases is mixed in for reliable termination teaching.

Run (env_rv deephedge):
  DX_VARIANT=deephedge python -m tasks.diagnostics.gen_sft_biased --n 800 --keep-high 0.0 --out data/dx_sft_biased.jsonl
"""
from __future__ import annotations
import argparse, json, random
from collections import Counter
from pathlib import Path
import tasks.diagnostics.env_rv as E


def build_oracle_traj(c):
    """One adaptive oracle trajectory: ask the oracle's findings (with answers), then commit the oracle action."""
    fam, assign = c["family"], c["assign"]
    revealed0 = {E.IDX[n]: assign[E.IDX[n]] for n in c["observed"]}
    asks, action, _ = E.oracle_rollout(fam, assign, E.ASK_COST, revealed0)
    msgs = [{"role": "system", "content": E.SYS}, {"role": "user", "content": E.user_msg(c)}]
    for it in asks:
        msgs.append({"role": "assistant", "content": json.dumps({"action_type": "ask", "intent": it})})
        msgs.append({"role": "user", "content": E.answer(c, it)})
    rationale = ("The chart already shows the decision-relevant findings; deciding now." if not asks
                 else "Sufficient information gathered; deciding now.")
    msgs.append({"role": "assistant", "content": json.dumps(
        {"action_type": "commit", "decision": action, "rationale": rationale})})
    return msgs, len(asks), action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="target trajectory count after bias filtering")
    ap.add_argument("--keep-high", type=float, default=0.0, help="fraction of high-risk cases to keep (0=drop all)")
    ap.add_argument("--commit-now-frac", type=float, default=0.12, help="fraction of rho=1.0 commit-now cases")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/dx_sft_biased.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = []
    # ---- commit-now bucket (rho=1.0): chart already complete -> 0 asks -> commit. Keep its high-risk biased too.
    n_now = max(1, round(args.n * args.commit_now_frac))
    for c in E.gen_cases(n_now * 3, rho=1.0, seed=args.seed, families=E.DEEP_FAMILIES, prefix="now-"):
        if c["true_risk"] == "high" and rng.random() >= args.keep_high:
            continue
        msgs, na, dec = build_oracle_traj(c)
        rows.append({"case_id": c["case_id"], "messages": msgs,
                     "metadata": {"family": c["family"], "rho": 1.0, "n_asks": na,
                                  "true_risk": c["true_risk"], "decision": dec}})
        if sum(1 for r in rows if r["metadata"]["rho"] == 1.0) >= n_now:
            break
    # ---- main rho=0 gather bucket, biased: drop high-risk with prob (1 - keep_high) ----
    pool = E.gen_cases(args.n * 4, rho=0.0, seed=args.seed + 1, families=E.DEEP_FAMILIES, prefix="bias-")
    for c in pool:
        if len(rows) >= args.n:
            break
        if c["true_risk"] == "high" and rng.random() >= args.keep_high:
            continue
        msgs, na, dec = build_oracle_traj(c)
        rows.append({"case_id": c["case_id"], "messages": msgs,
                     "metadata": {"family": c["family"], "rho": 0.0, "n_asks": na,
                                  "true_risk": c["true_risk"], "decision": dec}})
    rng.shuffle(rows)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- audit ----
    md = [r["metadata"] for r in rows]
    n = len(rows)
    risk = Counter(m["true_risk"] for m in md)
    commit_dist = Counter(m["decision"] for m in md)
    commit_rate = sum(1 for r in rows if json.loads(r["messages"][-1]["content"]).get("action_type") == "commit") / n
    ask_hist = Counter(min(m["n_asks"], 6) for m in md)
    print(f"[gen_sft_biased] {n} trajectories (keep_high={args.keep_high}) -> {out}")
    print(f"  risk dist: {dict(risk)}  (high frac = {risk.get('high',0)/n:.3f})")
    print(f"  commit dist: {dict(commit_dist)}  commit-rate={commit_rate:.3f}")
    print(f"  ask-count dist (0..6+): " + ", ".join(f"{k}:{ask_hist.get(k,0)}" for k in range(7)))
    print(f"  mean asks={sum(m['n_asks'] for m in md)/n:.2f}  emergent_tx demos={commit_dist.get('emergent_tx',0)}")


if __name__ == "__main__":
    main()
