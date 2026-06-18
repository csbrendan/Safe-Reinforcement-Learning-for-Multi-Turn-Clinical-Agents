"""Generate SFT-init trajectories for rx_triage. The demonstrated gathering policy is the per-family best
*fixed* checklist (the demonstrable baseline RL is compared against — non-adaptive, so RL retains headroom).
No data is committed — regenerate on demand.

Two modes:
  --mode fixed       (legacy) rho=0 only; ask the FULL fixed checklist every case -> commit. Every trajectory
                     is long (4-6 asks) and commit is rare in the token mix -> the undertrained smoke SFT learned
                     to ask but often failed to COMMIT (~30% no-commit; see docs/eval_runs/gate2_audit_rx_triage.md).
  --mode curriculum  (default) SAME fixed-checklist gathering policy, but over a spread of pre-observation (rho):
                     the demo asks only the checklist items NOT already on the chart, then commits. This teaches
                     "ask relevant info -> stop when sufficient -> commit" with many SHORT trajectories (1-3 asks)
                     and explicit COMMIT-NOW examples (chart already sufficient -> 0 asks -> commit), rebalancing
                     the ask:commit ratio so commitment is learned reliably. The held-out eval stays rho=0 (the
                     model must still gather from scratch), and the policy stays non-adaptive (fixed checklist
                     minus what's already known) -> the demonstrability/BC-gap framing is preserved.

Usage: python -m tasks.rx_triage.gen_data --mode curriculum --n 480 --out data/rx_triage_sft_v2.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from tasks.rx_triage import env as E
from framework.gates import bc_gap

# curriculum pre-observation mix: (rho, fraction). rho=0 -> full gather; rho=1.0 -> commit-now (chart complete).
RHO_MIX = [(0.0, 0.40), (0.4, 0.25), (0.7, 0.20), (1.0, 0.15)]


def build_traj(c, sub, commit_now_min_obs=0):
    """One ask->(answer)->commit trajectory: ask checklist items not already on the chart, then commit.
    Returns (messages, n_asks, decision)."""
    fam, assign, observed = c["family"], c["assign"], c["observed"]
    to_ask = [i for i in sub if E.ITEMS[i] not in observed]      # checklist items still unknown
    revealed = {i: assign[i] for i in sub}                        # all checklist items (observed + asked)
    msgs = [{"role": "system", "content": E.SYS}, {"role": "user", "content": E.user_msg(c)}]
    for i in to_ask:
        msgs.append({"role": "assistant", "content": json.dumps({"action_type": "ask", "intent": E.ITEMS[i]})})
        msgs.append({"role": "user", "content": E.answer(c, E.ITEMS[i])})
    _, dec = E.best_commit(fam, revealed)
    if not to_ask:
        rationale = "The chart already shows the decision-relevant findings; deciding now."
    else:
        rationale = "Sufficient information gathered; deciding now."
    msgs.append({"role": "assistant", "content": json.dumps(
        {"action_type": "commit", "decision": dec, "rationale": rationale})})
    return msgs, len(to_ask), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "curriculum"], default="curriculum")
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--out", default="data/rx_triage_sft_v2.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    _, _, _, BEST, _ = bc_gap.run(E, E.ASK_COST)   # best fixed checklist per deep family (under the env reward)

    rows = []
    if args.mode == "fixed":
        cases = E.gen_cases(args.n, rho=0.0, seed=args.seed, families=E.DEEP_FAMILIES)
        for c in cases:
            msgs, na, dec = build_traj(c, list(BEST[c["family"]]))
            rows.append({"case_id": c["case_id"], "messages": msgs, "n_assistant_turns": na + 1,
                         "metadata": {"family": c["family"], "rho": 0.0, "n_asks": na}})
    else:
        idx = 0
        for rho, frac in RHO_MIX:
            m = max(1, round(args.n * frac))
            cases = E.gen_cases(m, rho=rho, seed=args.seed + idx, families=E.DEEP_FAMILIES, prefix=f"v2r{int(rho*100)}-")
            for c in cases:
                msgs, na, dec = build_traj(c, list(BEST[c["family"]]))
                rows.append({"case_id": c["case_id"], "messages": msgs, "n_assistant_turns": na + 1,
                             "metadata": {"family": c["family"], "rho": rho, "n_asks": na}})
            idx += 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # validation summary (CPU): commit rate must be 1.0; ask-length spread; family/rho mix
    ask_hist = Counter(min(r["metadata"]["n_asks"], 7) for r in rows)
    commit_now = sum(1 for r in rows if r["metadata"]["n_asks"] == 0)
    short = sum(1 for r in rows if 1 <= r["metadata"]["n_asks"] <= 3)
    print(f"[gen_data:{args.mode}] {len(rows)} trajectories -> {out}")
    print(f"  every trajectory ends in commit: {all(json.loads(r['messages'][-1]['content']).get('action_type')=='commit' for r in rows)}")
    print(f"  commit-now (0 asks): {commit_now} ({100*commit_now/len(rows):.0f}%)   short (1-3 asks): {short} ({100*short/len(rows):.0f}%)")
    print(f"  asks/traj histogram (0..7+): " + ", ".join(f"{k}:{ask_hist.get(k,0)}" for k in range(8)))
    print(f"  family mix: {dict(Counter(r['metadata']['family'] for r in rows))}")
    print(f"  checklists: " + "; ".join(f"{f}={tuple(E.ITEMS[i] for i in BEST[f])}" for f in BEST))


if __name__ == "__main__":
    main()
