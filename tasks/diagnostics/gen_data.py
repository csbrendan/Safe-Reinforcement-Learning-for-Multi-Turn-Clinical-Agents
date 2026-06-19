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
from tasks.diagnostics import env as E
from framework.gates import bc_gap

# curriculum pre-observation mix: (rho, fraction). rho=0 -> full gather; rho=1.0 -> commit-now (chart complete).
# v2 (40% rho=0) terminated but collapsed to ~1.3 asks: too much short/commit-now signal swamped gathering.
# v3 puts ~70% on rho=0 full-checklist (4-6 asks -> commit, incl. high/moderate multi-ask), trims termination
# teaching to the minimum (12% commit-now + 18% rho=0.4 mid-length 2-4 asks), and DROPS the 1-ask-prone rho=0.7
# bucket -> mean demo asks ~4, few 1-ask commits, while still teaching reliable commitment.
RHO_MIX = {
    "curriculum":    [(0.0, 0.40), (0.4, 0.25), (0.7, 0.20), (1.0, 0.15)],
    "curriculum_v3": [(0.0, 0.70), (0.4, 0.18), (1.0, 0.12)],
    # v3 learned risk-graded gathering but its longer trajectories compounded a ~5% per-turn invalid rate into
    # ~20% no-commit (Gate-2 parse FAIL). v4 keeps v3's gathering mix almost unchanged (deep gather stays
    # dominant; do NOT drift back to v2's 1-ask under-gathering) but nudges a bit more 2-4-ask termination
    # (rho=0.4) so commit is demonstrated at moderate lengths too. The main fix is MORE TRAINING on this clean,
    # 100%-valid-vocabulary data (n=800, epochs=3) to raise per-turn validity -> fewer invalid-turn no-commits.
    "curriculum_v4": [(0.0, 0.68), (0.4, 0.22), (1.0, 0.10)],
}


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
    ap.add_argument("--mode", choices=["fixed", "curriculum", "curriculum_v3", "curriculum_v4"], default="curriculum_v4")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="data/dx_sft.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    _, _, _, BEST, _ = bc_gap.run(E, E.ASK_COST)   # best fixed checklist per deep family (under the env reward)

    rows = []
    if args.mode == "fixed":
        cases = E.gen_cases(args.n, rho=0.0, seed=args.seed, families=E.DEEP_FAMILIES)
        for c in cases:
            msgs, na, dec = build_traj(c, list(BEST[c["family"]]))
            rows.append({"case_id": c["case_id"], "messages": msgs, "n_assistant_turns": na + 1,
                         "metadata": {"family": c["family"], "rho": 0.0, "n_asks": na,
                                      "true_risk": c["true_risk"], "decision": dec}})
    else:
        idx = 0
        for rho, frac in RHO_MIX[args.mode]:
            m = max(1, round(args.n * frac))
            cases = E.gen_cases(m, rho=rho, seed=args.seed + idx, families=E.DEEP_FAMILIES, prefix=f"v2r{int(rho*100)}-")
            for c in cases:
                msgs, na, dec = build_traj(c, list(BEST[c["family"]]))
                rows.append({"case_id": c["case_id"], "messages": msgs, "n_assistant_turns": na + 1,
                             "metadata": {"family": c["family"], "rho": rho, "n_asks": na,
                                          "true_risk": c["true_risk"], "decision": dec}})
            idx += 1

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- sanity audit (CPU) ----------------------------------------------------------------------
    n = len(rows)
    md = [r["metadata"] for r in rows]
    ask_hist = Counter(min(m["n_asks"], 7) for m in md)
    commit_rate = sum(1 for r in rows if json.loads(r["messages"][-1]["content"]).get("action_type") == "commit") / n
    one_ask = sum(1 for m in md if m["n_asks"] == 1)
    two_four = sum(1 for m in md if 2 <= m["n_asks"] <= 4)
    commit_now = sum(1 for m in md if m["n_asks"] == 0)
    risk = Counter(m.get("true_risk") for m in md)
    hi_mod = [m for m in md if m.get("true_risk") in ("high", "moderate")]
    himod_3plus = sum(1 for m in hi_mod if m["n_asks"] >= 3)
    # valid item-name rate: every assistant ASK turn must use a canonical item name (directive 4/5/10)
    ask_turns = bad_names = 0
    for r in rows:
        for mmsg in r["messages"]:
            if mmsg["role"] == "assistant":
                try:
                    o = json.loads(mmsg["content"])
                except Exception:
                    continue
                if o.get("action_type") == "ask":
                    ask_turns += 1
                    if o.get("intent") not in E.IDX:
                        bad_names += 1
    valid_name_rate = 1.0 - bad_names / max(ask_turns, 1)
    # hidden-state leakage: assistant turns must contain ONLY action JSON (no chart/assign values); the only
    # state in any user turn is the case's own observed chart (rho) or an answer to an item the model asked.
    leak = 0
    for r in rows:
        for mmsg in r["messages"]:
            if mmsg["role"] == "assistant":
                try:
                    o = json.loads(mmsg["content"])
                    if o.get("action_type") not in ("ask", "commit"):
                        leak += 1
                except Exception:
                    leak += 1
    lengths = Counter(r["n_assistant_turns"] for r in rows)
    print(f"[gen_data:{args.mode}] {n} trajectories -> {out}")
    print(f"  commit-rate: {commit_rate:.3f} (target 1.0)   no-hidden-leak (assistant=action-only): {leak==0}")
    print(f"  ask-count dist (0..7+): " + ", ".join(f"{k}:{ask_hist.get(k,0)}" for k in range(8)))
    print(f"  traj-length dist (assistant turns): " + ", ".join(f"{k}:{v}" for k, v in sorted(lengths.items())))
    print(f"  commit-position (asks before commit): 0-ask {100*commit_now/n:.0f}% | 1-ask {100*one_ask/n:.0f}% | "
          f"2-4 ask {100*two_four/n:.0f}% | 5+ ask {100*sum(1 for m in md if m['n_asks']>=5)/n:.0f}%")
    print(f"  valid item-name rate (ask turns): {valid_name_rate:.4f} ({ask_turns} ask turns, {bad_names} invalid)")
    print(f"  risk distribution: {dict(risk)}")
    print(f"  high/moderate cases with 3+ asks: {himod_3plus}/{len(hi_mod)} ({100*himod_3plus/max(len(hi_mod),1):.0f}% of hi+mod)")
    print(f"  family mix: {dict(Counter(m['family'] for m in md))}")
    print(f"  checklists: " + "; ".join(f"{f}={tuple(E.ITEMS[i] for i in BEST[f])}" for f in BEST))


if __name__ == "__main__":
    main()
