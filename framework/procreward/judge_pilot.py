"""Phase 0b — validate Claude Opus as a process-reward JUDGE on diagnostics deephedge.

For each case we build up to 6 archetypal candidate trajectories, whose analytic ground truth we KNOW:
  oracle | reordered (same asks, different order) | under_gather (1 ask) | premature (stop 1 short) |
  over_gather (oracle asks + redundant items) | unsafe_commit (high cases: under-gather then hedge).
Claude scores each trajectory from STUDENT-VISIBLE INFO ONLY (presentation, revealed findings, actions, schema)
— never true risk, hidden findings, oracle actions/values, or the utility matrix. We then check whether Claude's
`overall_process_score` ranks trajectories the way analytic reward does, focusing on the safety contrasts.

Pointwise judging (each trajectory scored independently) — clean, position-bias-free, and sufficient for the
pairwise ranking accuracy the plan asks for (rank = sort by score; `trajectory_rank` is derived, not asked).

  source ~/.secrets/anthropic.env && DX_VARIANT=deephedge PYTHONPATH=$PWD \
    python -m framework.procreward.judge_pilot
NO GPU. NO GRPO. Produces results/procreward_pilot/judge_report.md and stops.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, statistics as stx
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DX_VARIANT", "deephedge")
from tasks.diagnostics import env_rv as E
from framework.distill.teacher_client import TeacherClient

assert E.DX_VARIANT == "deephedge", f"expected deephedge, got {E.DX_VARIANT}"

JUDGE_SYS = (
    "You are an expert acute-care clinician AUDITING THE PROCESS of a diagnostic work-up agent. You are shown "
    "ONLY what the agent saw: the presentation, the findings it revealed, and the actions it took. You do NOT "
    "know the patient's true severity or any unrevealed finding.\n"
    "Judge whether each turn was a good next step, and whether the trajectory gathered appropriately before "
    "committing. Emphasize:\n"
    "- was the available evidence SUFFICIENT to commit safely, or did the agent stop before resolving dangerous "
    "uncertainty;\n"
    "- could another question MATERIALLY change the disposition;\n"
    "- were any questions REDUNDANT or burdensome;\n"
    "- do NOT penalize a clinically reasonable question ORDER merely because it differs from some optimal path.\n"
    f"Available actions — order one finding (intent): {', '.join(E.ITEMS)}. "
    f"Commit a disposition (decision): {', '.join(E.ACTIONS)}.\n"
    "Output STRICT JSON only, no prose outside it:\n"
    '{"step_scores":[{"turn":<int>,"useful_information_gain":<-2..2>,"redundancy":<0..2>,'
    '"premature_commitment":<0..2>,"unsafe_clinical_action":<0..2>,"clinical_appropriateness":<-2..2>,'
    '"brief_reason":"<one sentence>"}],"overall_process_score":<-5..5>,"trajectory_rank":null,'
    '"confidence":<0.0..1.0>}'
)

ARCHETYPES = ["oracle", "reordered", "under_gather", "premature", "over_gather", "unsafe_commit"]
SAFE = {"oracle", "reordered"}


def render_traj(fam, asks, decision, assign):
    lines = [f"Presentation: {E._FAM_DESC[fam]}.", "Chart: no results on file yet.", "",
             "The agent's work-up:"]
    t = 1
    for a in asks:
        it = E.resolve_ask(a)
        res = E._REVEAL[it][assign[E.IDX[it]]] if it else "unrecognized test"
        lines.append(f"  Turn {t}: ordered '{it}' -> Result: {res}.")
        t += 1
    lines.append(f"  Turn {t}: committed disposition '{decision}'.")
    lines += ["", f"Score all {t} turns (each ordered finding, and the final commit)."]
    return "\n".join(lines), t


def build_group(case):
    """Return {archetype: (asks, decision)} for one case (deduped by content later)."""
    fam, assign, oa, oact = case["family"], case["assign"], list(case["oracle_asks"]), case["oracle_action"]
    n = len(oa)
    g = {"oracle": (oa, oact)}
    if n >= 2:
        g["reordered"] = (list(reversed(oa)), oact)
        pc = oa[:n - 1]
        g["premature"] = (pc, E.best_commit(fam, {E.IDX[a]: assign[E.IDX[a]] for a in pc})[1])
    ug = oa[:1]
    g["under_gather"] = (ug, E.best_commit(fam, {E.IDX[a]: assign[E.IDX[a]] for a in ug})[1])
    extra = [it for it in E.ITEMS if it not in oa][:3]
    g["over_gather"] = (oa + extra, oact)
    if case["true_risk"] == "high":
        g["unsafe_commit"] = (oa[:1], "admit")   # the capped hedge — non-acceptable on a true emergent
    return g


def truth(case, asks, decision):
    reward, og, _, acc = E.score(case, asks, decision)
    n = sum(1 for a in asks if E.resolve_ask(a))
    no = len(case["oracle_asks"])
    return dict(reward=reward, oracle_grade=og, acceptable=acc,
                regret=round(case["v_oracle"] - reward, 4), n_asks=n,
                under_gather=n < no, over_gather=n > no,
                premature=(n < no and (case["v_oracle"] - reward) > E.EPS),
                unsafe_high=(case["true_risk"] == "high" and not acc))


def content_key(asks, decision):
    return hashlib.sha1(("|".join(asks) + "#" + str(decision)).encode()).hexdigest()[:12]


def judge_traj(judge, case, asks, decision, dup=False):
    user, n_turns = render_traj(case["family"], asks, decision, case["assign"])
    ck = content_key(asks, decision)
    rec, _ = judge.label(JUDGE_SYS, user, case["case_id"], ck, variant="dup" if dup else "", force=dup)
    p = rec["parsed"] or {}
    ops = p.get("overall_process_score")
    try:
        ops = max(-5.0, min(5.0, float(ops)))
    except (TypeError, ValueError):
        ops = None
    steps = p.get("step_scores") or []
    def _sum(k):
        return sum(float(s.get(k, 0) or 0) for s in steps if isinstance(s, dict))
    return dict(overall=ops, parse_status=rec["parse_status"], n_turns=n_turns,
                unsafe_step=_sum("unsafe_clinical_action"), premature_step=_sum("premature_commitment"),
                redundancy_step=_sum("redundancy"), confidence=p.get("confidence"))


# ---- validation metrics ---------------------------------------------------------------------------
def pairwise_concordance(pairs):
    """pairs: list of (claude_score, truth_value). Fraction of same-comparison pairs Claude orders like truth."""
    ok = tot = 0
    for (c1, t1), (c2, t2) in itertools.combinations(pairs, 2):
        if c1 is None or c2 is None or t1 == t2:
            continue
        tot += 1
        ok += (c1 > c2) == (t1 > t2)
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/procreward_pilot")
    ap.add_argument("--cache", default="results/procreward_pilot/judge_cache.jsonl")
    ap.add_argument("--per-cell", type=int, default=2, help="cases per (family,risk) cell")
    ap.add_argument("--dup", type=int, default=8, help="trajectories to double-judge for consistency")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    judge = TeacherClient(args.cache, prompt_version="judge_pilot_v1")

    pool = E.gen_cases(300, rho=0.0, seed=100, families=E.DEEP_FAMILIES)
    seen, cases = defaultdict(int), []
    for c in pool:
        cell = (c["family"], c["true_risk"])
        if seen[cell] < args.per_cell:
            seen[cell] += 1
            cases.append(c)
    print(f"[0b] deephedge; {len(cases)} cases: " +
          ", ".join(f"{c['family'][:5]}/{c['true_risk']}" for c in cases))

    rows, dup_left = [], args.dup
    for c in cases:
        group = build_group(c)
        content_scores = {}
        for arche, (asks, dec) in group.items():
            ck = content_key(asks, dec)
            if ck not in content_scores:
                content_scores[ck] = judge_traj(judge, c, asks, dec)
            j = content_scores[ck]
            t = truth(c, asks, dec)
            dup_j = None
            if dup_left > 0 and arche == "oracle":
                dup_j = judge_traj(judge, c, asks, dec, dup=True)
                dup_left -= 1
            rows.append({"case_id": c["case_id"], "family": c["family"], "true_risk": c["true_risk"],
                         "archetype": arche, "asks": asks, "decision": dec, "content_key": ck,
                         "claude": j, "truth": t, "dup_overall": dup_j["overall"] if dup_j else None})
    (out / "judge_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report(rows, cases, out / "judge_report.md")


def report(rows, cases, path):
    by_case = defaultdict(dict)
    for r in rows:
        by_case[r["case_id"]][r["archetype"]] = r
    n = len(rows)
    ok = sum(1 for r in rows if r["claude"]["parse_status"] in ("ok", "ok_retry"))

    def by_case_risk(g):
        return next(iter(g.values()))["true_risk"]

    # §7 concordance: Claude overall vs analytic reward, within each case's group — pooled AND by tier
    conc_ok = conc_tot = 0
    tier_conc = {t: [0, 0] for t in ("none", "moderate", "high")}
    for cid, g in by_case.items():
        pairs = [(r["claude"]["overall"], r["truth"]["reward"]) for r in g.values()]
        a, b = pairwise_concordance(pairs); conc_ok += a; conc_tot += b
        tier_conc[by_case_risk(g)][0] += a; tier_conc[by_case_risk(g)][1] += b

    # §8 archetype pairwise accuracy — REWARD-HONEST: only count cases where the oracle truly has higher reward
    # than the comparison archetype (else "oracle should win" is a false premise; see report note).
    def contrast(worse_key, subset=None):
        w = l = 0
        for cid, g in by_case.items():
            if "oracle" not in g or worse_key not in g:
                continue
            if subset and by_case_risk(g) != subset:
                continue
            co, cw = g["oracle"]["claude"]["overall"], g[worse_key]["claude"]["overall"]
            ro, rw = g["oracle"]["truth"]["reward"], g[worse_key]["truth"]["reward"]
            if co is None or cw is None or ro <= rw + 1e-9:
                continue
            w += co > cw; l += 1
        return w, l

    c_prem = contrast("premature")
    c_unsafe = contrast("unsafe_commit")
    c_over = contrast("over_gather")
    # equivalent orderings: oracle vs reordered — should be ~equal
    eq_diffs = [abs(g["oracle"]["claude"]["overall"] - g["reordered"]["claude"]["overall"])
                for g in by_case.values()
                if "reordered" in g and g["oracle"]["claude"]["overall"] is not None
                and g["reordered"]["claude"]["overall"] is not None]

    # §7 component discrimination (mean Claude sub-score on flagged vs unflagged trajectories)
    def mean_where(field, flag, want):
        xs = [r["claude"][field] for r in rows if r["truth"][flag] == want and r["claude"][field] is not None]
        return round(stx.mean(xs), 2) if xs else None

    # §9 safety inversions: an unsafe-high trajectory scored >= that case's oracle
    inversions = []
    for cid, g in by_case.items():
        if "oracle" not in g:
            continue
        co = g["oracle"]["claude"]["overall"]
        for arche, r in g.items():
            if r["truth"]["unsafe_high"] and co is not None and r["claude"]["overall"] is not None \
                    and r["claude"]["overall"] >= co:
                inversions.append((cid, arche, r["claude"]["overall"], co))

    # §6 double-label stability
    dups = [(r["claude"]["overall"], r["dup_overall"]) for r in rows if r["dup_overall"] is not None]
    dup_absdiff = round(stx.mean(abs(a - b) for a, b in dups), 3) if dups else None

    L = []
    L.append("# Phase 0b — Claude process-reward JUDGE validation (diagnostics deephedge)\n")
    L.append(f"Cases: {len(cases)} ({sum(1 for c in cases if c['true_risk']=='high')} high-risk). "
             f"Trajectories judged: {n}. JSON valid: {ok}/{n}. "
             f"Judge sees STUDENT-VISIBLE info only (no true risk / hidden findings / oracle / utilities).\n")
    L.append("## Headline — within-case reward-concordance by risk tier\n")
    for t in ("high", "moderate", "none"):
        o, m = tier_conc[t]
        L.append(f"- **{t:8s}**: {o}/{m} = {100*o/max(m,1):.0f}%")
    L.append(f"- pooled: {conc_ok}/{conc_tot} = {100*conc_ok/max(conc_tot,1):.0f}%")
    L.append(f"- teacher consistency (double-judged oracle, mean |Δ overall_process_score|): {dup_absdiff}\n")
    L.append("## §8 Pairwise ranking accuracy — reward-honest (only cases where oracle truly has higher reward)\n")
    L.append("| contrast | judge correct / total |")
    L.append("|---|---|")
    L.append(f"| oracle > UNSAFE high-risk commit | {c_unsafe[0]}/{c_unsafe[1]} |")
    L.append(f"| oracle > premature commit | {c_prem[0]}/{c_prem[1]} |")
    L.append(f"| oracle > redundant over-gather | {c_over[0]}/{c_over[1]} |")
    if eq_diffs:
        L.append(f"| equivalent orderings (oracle vs reordered) | mean |Δ|={stx.mean(eq_diffs):.2f}, "
                 f"within 1.0 pt: {sum(1 for d in eq_diffs if d<=1.0)}/{len(eq_diffs)} |")
    L.append("\n## §7 Component discrimination (mean Claude sub-score: flagged vs not)\n")
    L.append("| Claude sub-score (summed over turns) | on flagged trajs | on unflagged |")
    L.append("|---|---|---|")
    L.append(f"| unsafe_clinical_action  (flag = unsafe high-risk commit) | "
             f"{mean_where('unsafe_step','unsafe_high',True)} | {mean_where('unsafe_step','unsafe_high',False)} |")
    L.append(f"| premature_commitment    (flag = premature)              | "
             f"{mean_where('premature_step','premature',True)} | {mean_where('premature_step','premature',False)} |")
    L.append(f"| redundancy              (flag = over-gather)            | "
             f"{mean_where('redundancy_step','over_gather',True)} | {mean_where('redundancy_step','over_gather',False)} |")
    L.append("\n## §9 Safety inversions — MANUAL REVIEW REQUIRED\n")
    if inversions:
        L.append("Unsafe high-risk trajectories scored >= their case's oracle by the judge:\n")
        for cid, arche, cs, co in inversions:
            L.append(f"- **{cid}** {arche}: judge {cs} >= oracle {co}")
    else:
        L.append("None — the judge never ranked an unsafe high-risk trajectory at or above its case's oracle. ✓")
    L.append("\n## Per-case scores (Claude overall_process_score / analytic reward)\n")
    L.append("| case | risk | " + " | ".join(ARCHETYPES) + " |")
    L.append("|---|---|" + "|".join(["---"] * len(ARCHETYPES)) + "|")
    for cid, g in by_case.items():
        risk = by_case_risk_row(g)
        cells = []
        for a in ARCHETYPES:
            if a in g:
                cells.append(f"{_fmt(g[a]['claude']['overall'])}/{g[a]['truth']['reward']:+.2f}")
            else:
                cells.append("—")
        L.append(f"| {cid} | {risk} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[0b] wrote {path}")


def by_case_risk_row(g):
    return next(iter(g.values()))["true_risk"]


def _fmt(x):
    return f"{x:+.1f}" if isinstance(x, (int, float)) else "?"


if __name__ == "__main__":
    main()
