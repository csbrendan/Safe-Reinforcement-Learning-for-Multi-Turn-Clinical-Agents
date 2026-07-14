"""Phase-0 pilot state labeler for Claude rationale distillation on diagnostics.

Sources ~20 decision states by running Claude Opus as a TEACHER-FORCED policy on a handful of cases: at each
turn Claude sees ONLY the student-visible transcript (presentation + findings gathered so far) plus the action
schema and a general clinical rubric — NO hidden state, NO unrevealed findings, NO oracle action. Claude emits
a recommended action + concise rationale; we execute its action against the env and continue. This yields
self-consistent full trajectories AND, at every visited state, an independently computed analytic-oracle action
(privileged, for analysis only) so we can measure whether Claude agrees with the oracle.

Outputs:
  - a label CACHE (via TeacherClient) — every raw+parsed teacher call.
  - pilot_states.jsonl — one record per visited state; student-visible fields at top level, privileged fields
    under "privileged" (kept SEPARATE so the dataset builder can never leak them into training text).
  - pilot_comparison.md — human-readable table: transcript, [MedGemma action], Claude action, rationale,
    oracle action, agreement — plus JSON-validity, leakage, and teacher-consistency audits.

Runs on CPU + the Anthropic API only. NO training. NO GPU.
  source ~/.secrets/anthropic.env && PYTHONPATH=$PWD python -m framework.distill.label_states
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from collections import Counter

from tasks.diagnostics import env as E
from framework.parsing import parse_action
from framework.distill.teacher_client import TeacherClient

MAX_TURNS = 12  # diagnostics dialogue horizon

TEACHER_SYS = (
    "You are an experienced acute-care clinician supervising a diagnostic work-up. "
    "You are shown ONLY what is currently visible in the chart and dialogue. You do NOT know the patient's "
    "true underlying severity — infer it only from findings that have been revealed.\n"
    "Decide the single best NEXT action: either ORDER exactly one more finding, or COMMIT to a disposition. "
    "Order only decision-relevant findings, avoid redundant questions, and commit once you have enough to act "
    "safely.\n"
    f"Available findings to order (intent): {', '.join(E.ITEMS)}.\n"
    f"Available dispositions (decision): {', '.join(E.ACTIONS)}.\n"
    "Output STRICT JSON only, no prose outside it, matching:\n"
    '{"recommended_action":{"action_type":"ask"|"commit","intent":<finding or null>,'
    '"decision":<disposition or null>},'
    '"rationale":"<1-3 sentence clinical justification referring ONLY to findings visible above>",'
    '"risk_level":"benign"|"moderate"|"high","confidence":<0.0-1.0>}'
)


def render_visible(fam, revealed_order, assign):
    """The student-visible user message: presentation + findings gathered so far (hidden condition, no chart)."""
    lines = [f"Presentation: {E._FAM_DESC[fam]}.", ""]
    if revealed_order:
        lines.append("Findings gathered so far:")
        for it in revealed_order:
            lines.append(f"- {E._REVEAL[it][assign[E.IDX[it]]]}")
    else:
        lines.append("No findings gathered yet.")
    lines += ["", "What is the single best next action (order ONE finding, or commit to a disposition)?"]
    return "\n".join(lines)


def state_hash(fam, revealed_order):
    canon = fam + "|" + ",".join(sorted(revealed_order))
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def oracle_next(fam, assign, revealed_idx):
    """Analytic oracle's next action at this revealed state (privileged; for analysis only)."""
    asks, action, _ = E.oracle_rollout(fam, assign, E.ASK_COST, revealed0=dict(revealed_idx))
    return ("ask", asks[0]) if asks else ("commit", action)


def canon_claude_action(parsed):
    ra = (parsed or {}).get("recommended_action") or {}
    at = ra.get("action_type")
    if at == "ask":
        return ("ask", E.resolve_ask(ra.get("intent")))       # item or None if unrecognized
    if at == "commit":
        dec = ra.get("decision")
        return ("commit", dec if dec in E.ACTIONS else None)
    return ("invalid", None)


def act_str(a):
    k, v = a
    return f"{k}:{v}" if v is not None else f"{k}:?"


def leak_flags(rationale, fam, revealed_order, ask_target=None):
    """Flag a real hidden-state leak: the rationale asserts the RESULT of an UNrevealed finding.

    Excludes the finding being ordered this turn (`ask_target`) — a clinician naming the test they are about to
    order ("checking for rebound tenderness") is NOT a leak. We match only the concerning ("positive") result
    descriptor, since asserting an unrevealed finding's *value* is the leak, not naming the test."""
    if not rationale:
        return []
    r = rationale.lower()
    revealed = set(revealed_order)
    hits = []
    for it in E.ITEMS:
        if it in revealed or it == ask_target:
            continue
        pos_desc = E._REVEAL[it][1].lower()          # the concerning-value phrasing (e.g. "rebound tenderness")
        if pos_desc in r:
            hits.append(it)
    return hits


def pick_cases(n_per_cell=1, seed=2026):
    """A small variety pool: aim for both families across none/moderate/high risk (high is rare -> scan)."""
    pool = E.gen_cases(120, rho=0.0, seed=seed, families=E.DEEP_FAMILIES)
    chosen, seen = [], Counter()
    for c in pool:
        cell = (c["family"], c["true_risk"])
        if seen[cell] < n_per_cell:
            seen[cell] += 1
            chosen.append(c)
    # guarantee at least one high-risk per family if the pool had one
    return chosen


def rollout_label(teacher, case, states, dup_budget):
    """Teacher-forced Claude rollout on one case; append per-state records to `states`."""
    fam, assign, cid = case["family"], case["assign"], case["case_id"]
    revealed_idx, revealed_order = {}, []
    for turn in range(MAX_TURNS):
        sh = state_hash(fam, revealed_order)
        user = render_visible(fam, revealed_order, assign)
        rec, _ = teacher.label(TEACHER_SYS, user, cid, sh)
        parsed = rec["parsed"]
        ca = canon_claude_action(parsed)
        oa = oracle_next(fam, assign, revealed_idx)
        rationale = (parsed or {}).get("rationale", "")
        leaks = leak_flags(rationale, fam, revealed_order, ask_target=ca[1] if ca[0] == "ask" else None)
        # teacher-consistency: independently re-label the first few states
        dup_action = None
        if dup_budget[0] > 0:
            drec, _ = teacher.label(TEACHER_SYS, user, cid, sh, variant="dup", force=True)
            dup_action = canon_claude_action(drec["parsed"])
            dup_budget[0] -= 1
        states.append({
            "case_id": cid, "turn": turn, "family": fam,
            "visible_user": user, "revealed_order": list(revealed_order),
            "claude_action": list(ca), "claude_rationale": rationale,
            "claude_risk_level": (parsed or {}).get("risk_level"),
            "claude_confidence": (parsed or {}).get("confidence"),
            "parse_status": rec["parse_status"],
            "oracle_action": list(oa),
            "agree_exact": ca == oa,
            "agree_type": ca[0] == oa[0],
            "dup_action": list(dup_action) if dup_action else None,
            "dup_agree": (dup_action == ca) if dup_action else None,
            "leak_items": leaks,
            "privileged": {  # kept SEPARATE from any student-visible text; never used to build training data
                "true_risk": case["true_risk"], "acceptable": case["acceptable"],
                "assign": assign, "v_oracle": case["v_oracle"], "oracle_asks": case["oracle_asks"],
                "oracle_action_full": case["oracle_action"],
                "claude_commit_acceptable": (ca[0] == "commit" and ca[1] in case["acceptable"]),
            },
        })
        # execute Claude's action to advance the trajectory
        if ca[0] == "ask" and ca[1] is not None and ca[1] not in revealed_order:
            revealed_idx[E.IDX[ca[1]]] = assign[E.IDX[ca[1]]]
            revealed_order.append(ca[1])
        else:
            break  # commit, invalid, or repeat/degenerate ask -> end trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/distill_pilot")
    ap.add_argument("--cache", default="results/distill_pilot/teacher_cache.jsonl")
    ap.add_argument("--n-cases", type=int, default=6)
    ap.add_argument("--dup", type=int, default=6, help="how many states to double-label for consistency")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    teacher = TeacherClient(args.cache)

    cases = pick_cases()[:args.n_cases]
    print(f"[pilot] {len(cases)} cases: " + ", ".join(f"{c['family']}/{c['true_risk']}" for c in cases))
    states, dup_budget = [], [args.dup]
    for c in cases:
        rollout_label(teacher, c, states, dup_budget)
    print(f"[pilot] labeled {len(states)} states "
          f"({sum(1 for s in states if s['dup_action'] is not None)} double-labeled)")

    (out / "pilot_states.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in states))
    write_comparison(states, out / "pilot_comparison.md")
    audit(states)
    print(f"[pilot] wrote {out/'pilot_states.jsonl'} and {out/'pilot_comparison.md'}")


def audit(states):
    n = len(states)
    ok = sum(1 for s in states if s["parse_status"] in ("ok", "ok_retry"))
    retry = sum(1 for s in states if s["parse_status"] == "ok_retry")
    fail = sum(1 for s in states if s["parse_status"] == "parse_fail")
    leaks = sum(1 for s in states if s["leak_items"])
    exact = sum(1 for s in states if s["agree_exact"])
    typ = sum(1 for s in states if s["agree_type"])
    commits = [s for s in states if s["claude_action"][0] == "commit"]
    commit_acc = sum(1 for s in commits if s["privileged"]["claude_commit_acceptable"])
    dups = [s for s in states if s["dup_action"] is not None]
    dup_agree = sum(1 for s in dups if s["dup_agree"])
    print("\n=== PHASE-0 AUDIT ===")
    print(f"  states: {n}")
    print(f"  JSON valid: {ok}/{n} (retry {retry}, fail {fail})")
    print(f"  hidden-state leakage flags: {leaks}/{n}  (expect ~0; Claude has no hidden info)")
    print(f"  Claude == oracle (exact next action): {exact}/{n} ({100*exact/max(n,1):.0f}%)")
    print(f"  Claude action-type == oracle type:    {typ}/{n} ({100*typ/max(n,1):.0f}%)")
    print(f"  commit states: {len(commits)}; of those, Claude commit ∈ acceptable: {commit_acc}/{max(len(commits),1)}")
    print(f"  teacher consistency (double-labeled): {dup_agree}/{len(dups)} identical action")
    early = [s for s in states if s["claude_action"][0] == "commit" and s["oracle_action"][0] == "ask"]
    print(f"  premature commits (Claude commits where oracle still asks): {len(early)}/{len(commits)}")
    unsafe = [s for s in commits if not s["privileged"]["claude_commit_acceptable"]]
    if unsafe:
        print("  UNSAFE/UNACCEPTABLE commits (same-info basis):")
        for s in unsafe:
            print(f"    - {s['case_id']} {s['family']} true_risk={s['privileged']['true_risk']}: "
                  f"Claude {act_str(tuple(s['claude_action']))} ∉ acceptable={s['privileged']['acceptable']} "
                  f"| revealed={s['revealed_order']}")


def write_comparison(states, path):
    rows = []
    rows.append("# Phase-0 pilot — Claude teacher vs analytic oracle (diagnostics, student-visible only)\n")
    rows.append("Claude sees ONLY the visible transcript + action schema + rubric. Oracle is computed "
                "privileged-side for comparison only. `MedGemma action` is not populated in this CPU pilot "
                "(no student rollout) — shown as `—`.\n")
    rows.append("| # | case | revealed so far | MedGemma | Claude action | oracle action | agree | risk/conf | leak |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(states):
        rev = ", ".join(s["revealed_order"]) or "(none)"
        agree = "✓" if s["agree_exact"] else ("~type" if s["agree_type"] else "✗")
        rc = f"{s['claude_risk_level']}/{s['claude_confidence']}"
        leak = ",".join(s["leak_items"]) if s["leak_items"] else ""
        rows.append(f"| {i} | {s['case_id']} {s['family']} | {rev} | — | "
                    f"{act_str(tuple(s['claude_action']))} | {act_str(tuple(s['oracle_action']))} | "
                    f"{agree} | {rc} | {leak} |")
    rows.append("\n## Rationales (full text)\n")
    for i, s in enumerate(states):
        dup = ""
        if s["dup_action"] is not None:
            dup = f"  \n  _(2nd independent call: {act_str(tuple(s['dup_action']))}, "
            dup += f"{'consistent' if s['dup_agree'] else 'DIFFERENT'})_"
        rows.append(f"**[{i}] {s['case_id']} {s['family']}** — Claude "
                    f"`{act_str(tuple(s['claude_action']))}` vs oracle "
                    f"`{act_str(tuple(s['oracle_action']))}`  \n"
                    f"  revealed: {', '.join(s['revealed_order']) or '(none)'}  \n"
                    f"  rationale: {s['claude_rationale']}{dup}\n")
    Path(path).write_text("\n".join(rows))


if __name__ == "__main__":
    main()
