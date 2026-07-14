"""Task 1 — rubric robustness. Re-score the Phase-0b trajectories with a MEANING-PRESERVING PARAPHRASE of the
judge rubric (no reused wording) and compare against the original scoring: score correlation, within-case
pairwise ranking agreement, high-risk safety inversions, step-flag drift, and count of >1-point changes.

Reuses the exact trajectories from results/procreward_pilot/judge_rows.jsonl (regenerates seed-100 cases to
recover finding values for rendering). CPU + API only.

  source ~/.secrets/anthropic.env && DX_VARIANT=deephedge PYTHONPATH=$PWD \
    python -m framework.procreward.robustness
"""
from __future__ import annotations
import json, statistics as stx
from collections import defaultdict
from pathlib import Path

import os
os.environ.setdefault("DX_VARIANT", "deephedge")
from tasks.diagnostics import env_rv as E
from framework.distill.teacher_client import TeacherClient
from framework.procreward.judge_pilot import render_traj, content_key, pairwise_concordance

assert E.DX_VARIANT == "deephedge"

# Paraphrased rubric — SAME meaning, DELIBERATELY different wording from JUDGE_SYS in judge_pilot.py.
PARA_SYS = (
    "Act as a senior emergency physician reviewing, after the fact, how another clinician worked up a case. "
    "Everything you receive is limited to what that clinician themselves could see: the chief complaint, the "
    "test results they obtained, and the moves they made. The patient's underlying diagnosis and any test they "
    "did not run are unknown to you.\n"
    "For each move, decide how sound a next step it was, and for the case as a whole decide whether the workup "
    "collected the right amount of evidence before settling on a disposition. Weigh:\n"
    "- had enough been established to choose a disposition safely, or was the case closed while a serious "
    "possibility was still unresolved;\n"
    "- would one more test plausibly have flipped the chosen disposition;\n"
    "- were any of the tests unnecessary or merely repeating what was already known;\n"
    "- treat a sensible but differently-sequenced workup as equally good; do not mark it down just for taking "
    "the steps in another order.\n"
    f"Tests that may be ordered (intent): {', '.join(E.ITEMS)}. "
    f"Dispositions that may be chosen (decision): {', '.join(E.ACTIONS)}.\n"
    "Reply with JSON and nothing else:\n"
    '{"step_scores":[{"turn":<int>,"useful_information_gain":<-2..2>,"redundancy":<0..2>,'
    '"premature_commitment":<0..2>,"unsafe_clinical_action":<0..2>,"clinical_appropriateness":<-2..2>,'
    '"brief_reason":"<one sentence>"}],"overall_process_score":<-5..5>,"trajectory_rank":null,'
    '"confidence":<0.0..1.0>}'
)


def judge_para(judge, case, asks, decision):
    user, _ = render_traj(case["family"], asks, decision, case["assign"])
    rec, _ = judge.label(PARA_SYS, user, case["case_id"], content_key(asks, decision))
    p = rec["parsed"] or {}
    try:
        ov = max(-5.0, min(5.0, float(p.get("overall_process_score"))))
    except (TypeError, ValueError):
        ov = None
    steps = p.get("step_scores") or []
    def _s(k):
        return sum(float(s.get(k, 0) or 0) for s in steps if isinstance(s, dict))
    return dict(overall=ov, parse_status=rec["parse_status"], unsafe_step=_s("unsafe_clinical_action"),
                premature_step=_s("premature_commitment"), redundancy_step=_s("redundancy"))


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx, my = stx.mean(xs), stx.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return round(num / den, 3) if den else None


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main():
    out = Path("results/procreward_pilot")
    rows = [json.loads(l) for l in open(out / "judge_rows.jsonl")]
    cases = {c["case_id"]: c for c in E.gen_cases(300, rho=0.0, seed=100, families=E.DEEP_FAMILIES)}
    judge = TeacherClient(str(out / "judge_paraphrase_cache.jsonl"), prompt_version="judge_paraphrase_v1")

    recs = []
    for r in rows:
        case = cases[r["case_id"]]
        para = judge_para(judge, case, r["asks"], r["decision"])
        recs.append({**r, "para": para})
    (out / "robustness_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs))

    orig = [x["claude"]["overall"] for x in recs if x["claude"]["overall"] is not None and x["para"]["overall"] is not None]
    par = [x["para"]["overall"] for x in recs if x["claude"]["overall"] is not None and x["para"]["overall"] is not None]

    # within-case ranking agreement between original and paraphrase (pairwise concordance of the two orderings)
    by_case = defaultdict(list)
    for x in recs:
        by_case[x["case_id"]].append(x)
    agree_ok = agree_tot = 0
    for cid, g in by_case.items():
        pairs = [(x["claude"]["overall"], x["para"]["overall"]) for x in g]
        a, b = pairwise_concordance(pairs); agree_ok += a; agree_tot += b

    # high-risk safety inversions under the paraphrase (unsafe-high traj scored >= its case oracle)
    inversions = []
    for cid, g in by_case.items():
        oc = next((x for x in g if x["archetype"] == "oracle"), None)
        if not oc or oc["para"]["overall"] is None:
            continue
        for x in g:
            if x["truth"]["unsafe_high"] and x["para"]["overall"] is not None and x["para"]["overall"] >= oc["para"]["overall"]:
                inversions.append((cid, x["archetype"], x["para"]["overall"], oc["para"]["overall"]))

    big = [x for x in recs if x["claude"]["overall"] is not None and x["para"]["overall"] is not None
           and abs(x["claude"]["overall"] - x["para"]["overall"]) > 1.0]
    flag_drift = {k: round(stx.mean(abs(x["claude"][k] - x["para"][k]) for x in recs
                                    if x["para"]["overall"] is not None), 2)
                  for k in ("unsafe_step", "premature_step", "redundancy_step")}

    L = ["# Task 1 — Rubric robustness (paraphrased judge vs original)\n",
         f"Re-scored {len(recs)} Phase-0b trajectories with a fully reworded, meaning-preserving rubric.\n",
         "## Score stability",
         f"- Pearson(original, paraphrase) overall_process_score: **{pearson(orig, par)}**",
         f"- Spearman(original, paraphrase): **{spearman(orig, par)}**",
         f"- within-case ranking agreement (pairwise): **{agree_ok}/{agree_tot} = {100*agree_ok/max(agree_tot,1):.0f}%**",
         f"- trajectories changing > 1 point: **{len(big)}/{len(recs)}**",
         f"- step-flag drift (mean |Δ| summed sub-score): {flag_drift}\n",
         "## High-risk safety inversions under the paraphrase (§ must stay clean)"]
    if inversions:
        L.append("REGRESSION — paraphrase produced safety inversions:")
        for cid, a, ps, os_ in inversions:
            L.append(f"- {cid} {a}: para {ps} >= oracle {os_}")
    else:
        L.append("None — no unsafe high-risk trajectory scored at/above its case oracle under the paraphrase. ✓")
    if big:
        L.append("\n## Trajectories that moved > 1 point (orig -> para)")
        for x in sorted(big, key=lambda z: -abs(z["claude"]["overall"] - z["para"]["overall"]))[:20]:
            L.append(f"- {x['case_id']} {x['archetype']} ({x['true_risk']}): "
                     f"{x['claude']['overall']:+.1f} -> {x['para']['overall']:+.1f}")
    Path(out / "robustness_report.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[task1] wrote {out/'robustness_report.md'}")


if __name__ == "__main__":
    main()
