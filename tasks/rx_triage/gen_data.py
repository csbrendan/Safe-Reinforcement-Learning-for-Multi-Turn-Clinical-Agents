"""Generate SFT-init trajectories for rx_triage: clone the strongest *demonstrable* policy (the per-family
best fixed checklist) as multi-turn ask->answer->commit dialogues. This is the demonstrable baseline RL is
compared against; it asks a fixed set (retains gathering) but is non-adaptive, so RL can improve by selective
gathering. No data is committed — regenerate on demand.

Usage: python -m tasks.rx_triage.gen_data --n 120 --out data/rx_triage_sft.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from tasks.rx_triage import env as E
from framework.gates import bc_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="data/rx_triage_sft.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    # best fixed checklist per deep family, under the env's reward
    _, _, _, BEST, _ = bc_gap.run(E, E.ASK_COST)
    cases = E.gen_cases(args.n, rho=0.0, seed=args.seed, families=E.DEEP_FAMILIES)
    rows = []
    for c in cases:
        fam = c["family"]; assign = c["assign"]
        sub = list(BEST[fam])                     # item indices to ask (fixed checklist)
        msgs = [{"role": "system", "content": E.SYS}, {"role": "user", "content": E.user_msg(c)}]
        revealed = {}
        for i in sub:
            item = E.ITEMS[i]
            msgs.append({"role": "assistant", "content": json.dumps({"action_type": "ask", "intent": item})})
            msgs.append({"role": "user", "content": E.answer(c, item)})
            revealed[i] = assign[i]
        _, dec = E.best_commit(fam, revealed)
        msgs.append({"role": "assistant", "content": json.dumps(
            {"action_type": "commit", "decision": dec, "rationale": "Decided on the gathered findings."})})
        rows.append({"case_id": c["case_id"], "messages": msgs, "n_assistant_turns": len(sub) + 1,
                     "metadata": {"family": fam}})
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[rx_triage gen_data] {len(rows)} trajectories -> {out}")
    print(f"  checklists: " + "; ".join(f"{f}={tuple(E.ITEMS[i] for i in BEST[f])}" for f in BEST))


if __name__ == "__main__":
    main()
