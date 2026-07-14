"""Task 2 — real-rollout judge validation. Judge REAL MedGemma trajectories (base / plain-GRPO / info-gain-GRPO)
from existing deephedge probe records, BLIND to which model produced them, with the existing process rubric, and
compare Claude's scores/rankings against the analytic ground truth.

No GPU: the probe records (results/probe_dh_*_t12.records.jsonl) already contain each rollout's asks + decision;
finding values are recovered by regenerating the seed-100 eval cases. Identical (case,asks,decision) trajectories
are judged once (blind) and mapped back to every arm that produced them.

  source ~/.secrets/anthropic.env && DX_VARIANT=deephedge PYTHONPATH=$PWD \
    python -m framework.procreward.real_rollout_judge
"""
from __future__ import annotations
import json, os, random, statistics as stx
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DX_VARIANT", "deephedge")
from tasks.diagnostics import env_rv as E
from framework.distill.teacher_client import TeacherClient
from framework.procreward.judge_pilot import (JUDGE_SYS, render_traj, content_key, truth, pairwise_concordance)

assert E.DX_VARIANT == "deephedge"

ARMS = {"base": "results/probe_dh_base_t12.records.jsonl",
        "plain_grpo": "results/probe_dh_grpo_final_t12.records.jsonl",
        "infogain_grpo": "results/probe_dh_ig_final_t12.records.jsonl"}
TARGET_PER_TIER = {"high": 999, "moderate": 28, "none": 28}   # take ALL unique high; cap the common tiers


def judge_traj(judge, case, asks, decision):
    dec = decision if decision in E.ACTIONS else "(no disposition committed)"
    user, _ = render_traj(case["family"], asks, dec, case["assign"])
    rec, _ = judge.label(JUDGE_SYS, user, case["case_id"], content_key(asks, decision), variant="rr")
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


def benign_overtriage(case, decision):
    return case["true_risk"] == "none" and decision in E.ACTIONS and decision not in case["acceptable"]


def main():
    rng = random.Random(7)
    cases = {c["case_id"]: c for c in E.gen_cases(300, rho=0.0, seed=100, families=E.DEEP_FAMILIES)}
    out = Path("results/procreward_pilot"); out.mkdir(parents=True, exist_ok=True)
    judge = TeacherClient(str(out / "real_rollout_cache.jsonl"), prompt_version="judge_pilot_v1", max_tokens=3072)

    # collect hidden-condition rollouts per arm; index unique trajectories by content
    uniq = {}   # content_key -> {case_id, asks, decision, tier, arms:set}
    per_arm_all = defaultdict(list)
    for arm, path in ARMS.items():
        for r in (json.loads(l) for l in open(path)):
            if r.get("cond") != "hidden":
                continue
            cid = r["case_id"]
            if cid not in cases:
                continue
            ck = content_key(r["asks"], r["decision"])
            per_arm_all[arm].append((cid, ck))
            u = uniq.setdefault(ck, {"case_id": cid, "asks": r["asks"], "decision": r["decision"],
                                     "tier": cases[cid]["true_risk"], "arms": set()})
            u["arms"].add(arm)

    # stratified blind sample of UNIQUE trajectories (all high; capped moderate/benign)
    by_tier = defaultdict(list)
    for ck, u in uniq.items():
        by_tier[u["tier"]].append(ck)
    chosen = []
    for tier, cks in by_tier.items():
        rng.shuffle(cks)
        chosen += cks[:TARGET_PER_TIER[tier]]
    print(f"[task2] {sum(len(v) for v in per_arm_all.values())} rollouts -> {len(uniq)} unique; "
          f"judging {len(chosen)} (by tier: " +
          ", ".join(f"{t}:{min(len(by_tier[t]),TARGET_PER_TIER[t])}" for t in ('high','moderate','none')) + ")")

    judged = {}
    for ck in chosen:
        u = uniq[ck]
        case = cases[u["case_id"]]
        j = judge_traj(judge, case, u["asks"], u["decision"])
        t = truth(case, u["asks"], u["decision"])
        judged[ck] = {"case_id": u["case_id"], "tier": u["tier"], "arms": sorted(u["arms"]),
                      "asks": u["asks"], "decision": u["decision"], "claude": j, "truth": t,
                      "benign_overtriage": benign_overtriage(case, u["decision"])}
    (out / "real_rollout_rows.jsonl").write_text("".join(json.dumps(v) + "\n" for v in judged.values()))
    report(judged, per_arm_all, out / "real_rollout_report.md")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(stx.mean(xs), 2) if xs else None


def report(judged, per_arm_all, path):
    rows = list(judged.values())
    n = len(rows)
    ok = sum(1 for r in rows if r["claude"]["parse_status"] in ("ok", "ok_retry"))

    # within-case reward concordance by tier (pool arms' unique trajectories per case)
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r)
    tier_conc = {t: [0, 0] for t in ("none", "moderate", "high")}
    for cid, g in by_case.items():
        a, b = pairwise_concordance([(r["claude"]["overall"], r["truth"]["reward"]) for r in g])
        tier_conc[g[0]["tier"]][0] += a; tier_conc[g[0]["tier"]][1] += b

    def split(flag_fn):
        yes = [r["claude"]["overall"] for r in rows if flag_fn(r)]
        no = [r["claude"]["overall"] for r in rows if not flag_fn(r)]
        return _mean(yes), _mean(no), sum(1 for r in rows if flag_fn(r))

    unsafe = split(lambda r: r["truth"]["unsafe_high"])
    prem = split(lambda r: r["truth"]["premature"])
    over = split(lambda r: r["truth"]["over_gather"])
    otri = split(lambda r: r["benign_overtriage"])

    # arm-level: analytic value vs mean Claude score (does the judge rank the arms like the verifier?)
    arm_val, arm_claude = {}, {}
    ck_claude = {ck: judged[ck]["claude"]["overall"] for ck in judged}
    ck_reward = {ck: judged[ck]["truth"]["reward"] for ck in judged}
    for arm, lst in per_arm_all.items():
        rewards = [ck_reward[ck] for cid, ck in lst if ck in ck_reward]
        cs = [ck_claude[ck] for cid, ck in lst if ck in ck_claude and ck_claude[ck] is not None]
        arm_val[arm] = round(stx.mean(rewards), 3) if rewards else None
        arm_claude[arm] = round(stx.mean(cs), 2) if cs else None

    # §9 high-risk disagreements: high-risk trajectory where Claude score sign disagrees with safety
    hi_disagree = [r for r in rows if r["tier"] == "high"
                   and ((r["truth"]["acceptable"] and r["claude"]["overall"] is not None and r["claude"]["overall"] < 0)
                        or (not r["truth"]["acceptable"] and r["claude"]["overall"] is not None and r["claude"]["overall"] > 2))]
    bm_sample = [r for r in rows if r["tier"] in ("none", "moderate")
                 and r["claude"]["overall"] is not None
                 and ((r["truth"]["reward"] > 0.6 and r["claude"]["overall"] < -1)
                      or (r["truth"]["reward"] < 0.3 and r["claude"]["overall"] > 2))][:12]

    L = ["# Task 2 — Real-rollout judge validation (blind; base / plain-GRPO / info-gain-GRPO on deephedge)\n",
         f"Judged {n} unique real trajectories (blind to arm). JSON valid {ok}/{n}. "
         "Analytic labels from the pure verifier; Claude saw student-visible transcript only.\n",
         "## Within-case reward concordance by tier",
         f"- high: {tier_conc['high'][0]}/{tier_conc['high'][1]} = {100*tier_conc['high'][0]/max(tier_conc['high'][1],1):.0f}%",
         f"- moderate: {tier_conc['moderate'][0]}/{tier_conc['moderate'][1]} = {100*tier_conc['moderate'][0]/max(tier_conc['moderate'][1],1):.0f}%",
         f"- benign: {tier_conc['none'][0]}/{tier_conc['none'][1]} = {100*tier_conc['none'][0]/max(tier_conc['none'][1],1):.0f}%\n",
         "## Claude mean overall_process_score: flagged vs unflagged (n flagged)",
         "| analytic flag | flagged | unflagged | n |",
         "|---|---|---|---|",
         f"| unsafe high-risk commit | {unsafe[0]} | {unsafe[1]} | {unsafe[2]} |",
         f"| premature commit | {prem[0]} | {prem[1]} | {prem[2]} |",
         f"| redundant / over-gather | {over[0]} | {over[1]} | {over[2]} |",
         f"| benign over-triage | {otri[0]} | {otri[1]} | {otri[2]} |\n",
         "## Arm-level: analytic value vs mean Claude score (does the judge rank the arms like the verifier?)",
         "| arm | analytic mean reward | mean Claude score |",
         "|---|---|---|"]
    for arm in ("base", "plain_grpo", "infogain_grpo"):
        L.append(f"| {arm} | {arm_val[arm]} | {arm_claude[arm]} |")
    L.append("\n## §9 High-risk disagreements (Claude sign disagrees with safety) — MANUAL REVIEW")
    if hi_disagree:
        for r in hi_disagree:
            L.append(f"- {r['case_id']} arms={r['arms']} asks={r['asks']} -> {r['decision']} "
                     f"(reward {r['truth']['reward']:+.2f}, acceptable={r['truth']['acceptable']}): "
                     f"Claude {r['claude']['overall']:+.1f}")
    else:
        L.append("None — on high-risk trajectories, Claude's score sign never contradicted the safety label. ✓")
    L.append("\n## Benign/moderate disagreement sample")
    if bm_sample:
        for r in bm_sample:
            L.append(f"- {r['case_id']} ({r['tier']}) reward {r['truth']['reward']:+.2f}: Claude {r['claude']['overall']:+.1f}")
    else:
        L.append("None in the sampled subset.")
    Path(path).write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[task2] wrote {path}")


if __name__ == "__main__":
    main()
