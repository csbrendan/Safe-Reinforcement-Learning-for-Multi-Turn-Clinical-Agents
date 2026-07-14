"""Build the two Phase-0 SFT dataset variants from labeled pilot states, and audit the TRAINING TEXT itself.

Variants (Distilling Step-by-Step, word/reasoning supervision — no teacher logits):
  A. action-only        — assistant turns carry only the structured action (teacher-free neutral commit note).
  B. rationale+action   — assistant turns carry Claude's concise rationale, THEN the same structured action.

Both are emitted as full-trajectory `{"messages":[...]}` rows in the exact format `framework.rl.sft_warmup`
consumes: the student system prompt is the real `env.SYS` (NOT the teacher rubric), each target action is
embedded in its exact conversation prefix, and answers come from `env.answer`. Privileged fields from
pilot_states.jsonl are never read here, so nothing hidden can enter the training text.

Audits on the built data (student-visible surface only):
  - JSON-validity: every assistant action must parse via the real `framework.parsing.parse_action`.
  - leakage: no assistant turn may name an UNrevealed finding descriptor (mirrors gen_data.py's leak audit).

  PYTHONPATH=$PWD python -m framework.distill.build_sft
NO training is run.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

from tasks.diagnostics import env as E
from framework.parsing import parse_action

NEUTRAL_COMMIT_NOTE = "Deciding based on the findings gathered."


def _user_msg(fam):
    """Hidden-condition (rho=0, no chart) user message — matches env.user_msg for a no-observed case."""
    return "\n".join([f"Presentation: {E._FAM_DESC[fam]}.",
                      "Establish the work-up before deciding disposition.",
                      "Chart: no results on file yet."])


def action_json(claude_action, rationale_for_commit):
    kind, val = claude_action
    if kind == "ask" and val is not None:
        return json.dumps({"action_type": "ask", "intent": val})
    if kind == "commit" and val is not None:
        return json.dumps({"action_type": "commit", "decision": val, "rationale": rationale_for_commit})
    return None  # invalid actions are dropped (should be rare; reported)


def build_case_trajectory(case_states, with_rationale):
    """One full trajectory per case from its ordered visited states. Returns (messages, dropped)."""
    fam = case_states[0]["family"]
    # reconstruct the visible dialogue from revealed_order deltas; each state's claude_action is that turn's target
    msgs = [{"role": "system", "content": E.SYS},
            {"role": "user", "content": _user_msg(fam)}]
    dropped = 0
    prev_revealed = []
    for s in case_states:
        # emit the answer turns for any findings revealed since the previous state (the asks already emitted)
        for it in s["revealed_order"][len(prev_revealed):]:
            msgs.append({"role": "user", "content": _answer_str(fam, it, s)})
        prev_revealed = s["revealed_order"]
        aj = action_json(tuple(s["claude_action"]), NEUTRAL_COMMIT_NOTE)
        if aj is None:
            dropped += 1
            continue
        content = (f"{s['claude_rationale'].strip()}\n{aj}" if with_rationale and s["claude_rationale"]
                   else aj)
        msgs.append({"role": "assistant", "content": content})
    return msgs, dropped


# --- small helpers that avoid needing the concrete case object (student-visible only) ------------------
def _answer_str(fam, item, state):
    # value is encoded in the visible descriptor already chosen at label time; recover it from revealed order
    # by re-reading the descriptor the teacher saw is overkill — the answer text is env.answer-equivalent.
    # We stored only student-visible text, so reconstruct from the descriptor pair via the assign bit is NOT
    # available here; instead the pilot guarantees revealed findings were real env answers, so we re-derive
    # the value from the descriptor recorded in visible_user.
    v = _value_from_visible(state["visible_user"], item)
    return f"Result: {E._REVEAL[item][v]}."


def _value_from_visible(visible_user, item):
    neg, pos = E._REVEAL[item]
    if pos.lower() in visible_user.lower():
        return 1
    return 0


def _audit_messages(rows):
    bad_json = leaks = ask_turns = 0
    for r in rows:
        revealed = []
        for m in r["messages"]:
            if m["role"] == "user" and m["content"].startswith("Result:"):
                for it in E.ITEMS:
                    if E._REVEAL[it][1].lower() in m["content"].lower() or E._REVEAL[it][0].lower() in m["content"].lower():
                        revealed.append(it)
            if m["role"] != "assistant":
                continue
            kind, val = parse_action(m["content"])
            if kind == "invalid":
                bad_json += 1
            ask_target = E.resolve_ask(val) if kind == "ask" else None
            if kind == "ask":
                ask_turns += 1
            # leakage: the assistant ASSERTS the concerning RESULT of an UNrevealed finding it is NOT ordering
            # this turn (naming the test being ordered is not a leak).
            low = m["content"].lower()
            for it in E.ITEMS:
                if it in revealed or it == ask_target:
                    continue
                if E._REVEAL[it][1].lower() in low:
                    leaks += 1
                    break
    return bad_json, leaks, ask_turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="results/distill_pilot/pilot_states.jsonl")
    ap.add_argument("--out-dir", default="results/distill_pilot")
    args = ap.parse_args()
    out = Path(args.out_dir)
    states = [json.loads(l) for l in open(args.states)]
    by_case = defaultdict(list)
    for s in states:
        by_case[s["case_id"]].append(s)
    for v in by_case.values():
        v.sort(key=lambda s: s["turn"])

    for name, with_rat in [("action_only", False), ("rationale_action", True)]:
        rows, total_dropped = [], 0
        for cid, css in by_case.items():
            msgs, dropped = build_case_trajectory(css, with_rat)
            total_dropped += dropped
            rows.append({"case_id": cid, "messages": msgs,
                         "n_assistant_turns": sum(1 for m in msgs if m["role"] == "assistant")})
        path = out / f"dss_{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        bad, leaks, asks = _audit_messages(rows)
        print(f"[build:{name}] {len(rows)} trajectories -> {path}")
        print(f"    assistant turns: {sum(r['n_assistant_turns'] for r in rows)}  invalid-action turns: {bad}  "
              f"dropped invalid: {total_dropped}")
        print(f"    leakage (assistant names an UNrevealed finding): {leaks}  [MUST be 0]")


if __name__ == "__main__":
    main()
