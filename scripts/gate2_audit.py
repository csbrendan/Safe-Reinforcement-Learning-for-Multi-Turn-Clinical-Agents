"""Gate-2 anomaly audit: WHY is base mean reward (0.361) > smoke-SFT mean reward (0.180) on hidden, even though
SFT lifts high-risk support 0.00 -> 0.43? Determines: (A) legitimate (SFT pays ask-cost + mis-commits),
(B) implementation artifact (reward/env/parser/prompt bug), or (C) SFT-data issue (expert over-asks).

CPU-only, deterministic. Reads results/probe_{base,sft}.records.jsonl (hidden cond) + the live env +
data/rx_triage_sft_smoke.jsonl. Independently re-scores every record against the CURRENT env and cross-checks
case parameters between the two runs, then breaks behavior down by risk group. No GPU, no model.

Usage: python scripts/gate2_audit.py
"""
from __future__ import annotations
import os, sys, json, statistics as stx
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.rx_triage import env as E

HIDDEN_SEED = 100      # support_probe uses seed=100 for both runs
N_CASES = 60
REC = {"base": "results/probe_base.records.jsonl", "sft": "results/probe_sft.records.jsonl"}


def load(tag, cond="hidden"):
    return [json.loads(l) for l in open(REC[tag]) if json.loads(l).get("cond") == cond]


def rel_set(fam):
    return set(E.rel_items(fam))


# ============================================================= 1. independent re-score + consistency
def section1():
    print("=" * 96)
    print("1. INDEPENDENT RE-SCORE + CASE-PARAMETER CONSISTENCY")
    print("=" * 96)
    print(f"   env constants: ASK_COST={E.ASK_COST}  EPS={E.EPS}  ACTIONS={E.ACTIONS}")
    cases = {c["case_id"]: c for c in E.gen_cases(N_CASES, 0.0, seed=HIDDEN_SEED, families=E.DEEP_FAMILIES)}
    base, sft = load("base"), load("sft")

    # (a) per-case_id parameter agreement across base/SFT and vs regenerated env
    b_by = {r["case_id"]: r for r in base}
    s_by = {r["case_id"]: r for r in sft}
    mism_param = 0
    for cid, c in cases.items():
        for tag, by in (("base", b_by), ("sft", s_by)):
            r = by.get(cid)
            if r is None:
                continue
            if r["true_risk"] != c["true_risk"] or abs(r["v_oracle"] - c["v_oracle"]) > 1e-6 or r["family"] != c["family"]:
                mism_param += 1
                if mism_param <= 5:
                    print(f"   [param mismatch] {tag} {cid}: rec(risk={r['true_risk']},v={r['v_oracle']},{r['family']}) "
                          f"vs env(risk={c['true_risk']},v={c['v_oracle']},{c['family']})")
    print(f"   (a) case-parameter mismatches base/SFT vs regenerated env: {mism_param}  "
          f"(0 => identical hidden set, same true_risk/v_oracle/family, no old/widened env drift)")

    # (b) re-score every record against the CURRENT env.score
    for tag, recs in (("base", base), ("sft", sft)):
        dr = dog = 0
        for r in recs:
            c = cases[r["case_id"]]
            r2, og2, _, _ = E.score(c, r["asks"], r["decision"])
            if abs(r2 - r["reward"]) > 1e-4:
                dr += 1
            if bool(og2) != bool(r["oracle_grade"]):
                dog += 1
        print(f"   (b) [{tag}] re-score vs stored: reward mismatches={dr}/{len(recs)}  "
              f"oracle_grade mismatches={dog}/{len(recs)}  (0 => probe used the same reward fn as now)")

    # (c) n_asks in record == count of resolvable asks (cost base)
    bad_nask = 0
    for recs in (base, sft):
        for r in recs:
            if r["n_asks"] != sum(1 for a in r["asks"] if E.resolve_ask(a)):
                bad_nask += 1
    print(f"   (c) record n_asks vs recomputed resolvable-ask count: {bad_nask} mismatches (0 => consistent)")


# ============================================================= helpers for behavior
def by_group(recs):
    g = defaultdict(list)
    for r in recs:
        g[r["true_risk"]].append(r)
        if r["true_risk"] != "none":
            g["relevant"].append(r)
        g["overall"].append(r)
    return g


def percase_mean(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["case_id"]].append(r["reward"])
    return {k: stx.mean(v) for k, v in d.items()}


# ============================================================= 2. paired-case inspection
def section2():
    print("\n" + "=" * 96)
    print("2. PAIRED-CASE INSPECTION (per case_id mean reward over k=12, hidden)")
    print("=" * 96)
    base, sft = load("base"), load("sft")
    cases = {c["case_id"]: c for c in E.gen_cases(N_CASES, 0.0, seed=HIDDEN_SEED, families=E.DEEP_FAMILIES)}
    bm, sm = percase_mean(base), percase_mean(sft)
    b_supp = {cid: any(r["oracle_grade"] for r in base if r["case_id"] == cid) for cid in cases}
    s_supp = {cid: any(r["oracle_grade"] for r in sft if r["case_id"] == cid) for cid in cases}
    b_asks = defaultdict(list); s_asks = defaultdict(list)
    b_dec = defaultdict(Counter); s_dec = defaultdict(Counter)
    for r in base:
        b_asks[r["case_id"]].append(r["n_asks"]); b_dec[r["case_id"]][r["decision"]] += 1
    for r in sft:
        s_asks[r["case_id"]].append(r["n_asks"]); s_dec[r["case_id"]][r["decision"]] += 1

    rows = []
    for cid, c in cases.items():
        rows.append((cid, c["true_risk"], c["family"], len(c["oracle_asks"]), c["v_oracle"],
                     bm[cid], sm[cid], sm[cid] - bm[cid], b_supp[cid], s_supp[cid],
                     stx.mean(b_asks[cid]), stx.mean(s_asks[cid]),
                     b_dec[cid].most_common(1)[0][0], s_dec[cid].most_common(1)[0][0]))

    def show(title, subset):
        print(f"\n   -- {title} --")
        print(f"   {'case':6s} {'risk':8s} {'fam':9s} {'orA':>3} {'vorc':>6} {'baseR':>7} {'sftR':>7} "
              f"{'d':>6} {'bsup':>4} {'ssup':>4} {'bAsk':>5} {'sAsk':>5} {'baseDec':>10} {'sftDec':>10}")
        for x in subset:
            print(f"   {x[0]:6s} {x[1]:8s} {x[2]:9s} {x[3]:>3} {x[4]:>6.2f} {x[5]:>7.3f} {x[6]:>7.3f} "
                  f"{x[7]:>+6.3f} {str(x[8])[0]:>4} {str(x[9])[0]:>4} {x[10]:>5.1f} {x[11]:>5.1f} "
                  f"{str(x[12]):>10} {str(x[13]):>10}")

    base_wins = sorted([r for r in rows if r[7] > 0.03], key=lambda r: r[7])[:8]
    sft_wins = sorted([r for r in rows if r[7] < -0.03], key=lambda r: r[7])[:8]
    hi_sft = [r for r in rows if r[1] == "high" and r[9] and not r[8]][:8]
    benign_cheap = sorted([r for r in rows if r[1] == "none"], key=lambda r: -r[5])[:6]
    show("base WINS (baseR>sftR): where SFT loses reward", base_wins)
    show("SFT WINS (sftR>baseR)", sft_wins)
    show("HIGH-RISK: SFT oracle-grade succeeds & base fails", hi_sft)
    show("BENIGN: base's cheapest high-reward cases", benign_cheap)


# ============================================================= 3. SFT vs base behavior breakdown
def behavior_table(tag, recs):
    cases = {c["case_id"]: c for c in E.gen_cases(N_CASES, 0.0, seed=HIDDEN_SEED, families=E.DEEP_FAMILIES)}
    g = by_group(recs)
    print(f"\n   [{tag}] behavior by risk group:")
    print(f"   {'group':9s} {'n':>4} {'meanAsk':>8} {'noCommit':>9} {'repeatAsk':>10} {'irrelAsk':>9} "
          f"{'suffInfo':>9} {'insuffInfo':>11}")
    for grp in ("overall", "high", "moderate", "none", "relevant"):
        rs = g.get(grp, [])
        if not rs:
            continue
        n = len(rs)
        mean_ask = stx.mean(r["n_asks"] for r in rs)
        nocommit = sum(1 for r in rs if r["decision"] is None) / n
        repeat = sum(1 for r in rs if len(r["asks"]) != len(set(r["asks"]))) / n
        # irrelevant: asked an item not in the family's relevant set
        irrel = 0; suff = insuff = committed = 0
        for r in rs:
            rel = rel_set(r["family"])
            if any(a not in rel for a in r["asks"]):
                irrel += 1
            if r["decision"] is not None:
                committed += 1
                need = set(cases[r["case_id"]]["oracle_asks"])
                if need.issubset(set(r["asks"])):
                    suff += 1
                else:
                    insuff += 1
        irrel /= n
        suff_r = suff / committed if committed else 0.0
        insuff_r = insuff / committed if committed else 0.0
        print(f"   {grp:9s} {n:>4} {mean_ask:>8.2f} {nocommit:>9.3f} {repeat:>10.3f} {irrel:>9.3f} "
              f"{suff_r:>9.3f} {insuff_r:>11.3f}")
    # decision distribution by group
    print(f"   [{tag}] decision distribution by risk group:")
    for grp in ("high", "moderate", "none"):
        c = Counter(r["decision"] for r in g.get(grp, []))
        tot = sum(c.values()) or 1
        dist = ", ".join(f"{k}:{v/tot:.2f}" for k, v in c.most_common())
        print(f"     {grp:9s} {dist}")


def section3():
    print("\n" + "=" * 96)
    print("3. BEHAVIOR BREAKDOWN  (noCommit=decision None; suffInfo/insuffInfo among COMMITTED rollouts:")
    print("   suffInfo = model asked a superset of the case's oracle_asks before committing)")
    print("=" * 96)
    behavior_table("base", load("base"))
    behavior_table("sft", load("sft"))


# ============================================================= 4. SFT data-generation audit
def section4():
    print("\n" + "=" * 96)
    print("4. SFT DATA-GENERATION AUDIT  (data/rx_triage_sft_smoke.jsonl)")
    print("=" * 96)
    rows = [json.loads(l) for l in open("data/rx_triage_sft_smoke.jsonl")]
    # best-fixed checklist per family, as gen_data used
    from framework.gates import bc_gap
    _, _, _, BEST, _ = bc_gap.run(E, E.ASK_COST)
    best_names = {f: tuple(E.ITEMS[i] for i in BEST[f]) for f in BEST}
    print(f"   best-fixed checklist (the SFT expert): " + "; ".join(f"{f}={best_names[f]}" for f in best_names))
    oracle_mean_ask = stx.mean(len(c["oracle_asks"]) for c in
                               E.gen_cases(400, 0.0, seed=7, families=E.DEEP_FAMILIES))
    print(f"   oracle mean asks (adaptive) ~ {oracle_mean_ask:.2f}  vs  fixed-checklist sizes "
          + ", ".join(f"{f}={len(best_names[f])}" for f in best_names))

    bad_fmt = leak = nonchecklist = 0
    asks_per_fam = defaultdict(list)
    adaptive_stop = 0  # demos that stopped early (asked fewer than full checklist)
    for r in rows:
        fam = r["metadata"]["family"]; msgs = r["messages"]
        # roles must be sys,user,(assistant ask,user answer)*,assistant commit
        if msgs[0]["role"] != "system" or msgs[1]["role"] != "user":
            bad_fmt += 1; continue
        # hidden leak: initial user msg must not reveal labs (rho=0 => "no recent labs")
        if "no recent labs" not in msgs[1]["content"]:
            leak += 1
        asks = []
        for m in msgs[2:]:
            if m["role"] != "assistant":
                continue
            try:
                obj = json.loads(m["content"])
            except Exception:
                bad_fmt += 1; break
            if obj.get("action_type") == "ask":
                asks.append(obj["intent"])
            elif obj.get("action_type") == "commit":
                if obj.get("decision") not in E.ACTIONS:
                    bad_fmt += 1
            else:
                bad_fmt += 1
        asks_per_fam[fam].append(len(asks))
        if set(asks) != set(best_names[fam]):
            nonchecklist += 1
        if len(asks) < len(best_names[fam]):
            adaptive_stop += 1
    print(f"   trajectories={len(rows)}  malformed_turns={bad_fmt}  hidden_leaks_in_prompt={leak}  "
          f"demos_not_matching_checklist={nonchecklist}  demos_that_stopped_early={adaptive_stop}")
    for f in best_names:
        v = asks_per_fam[f]
        print(f"     {f}: demo asks/traj mean={stx.mean(v):.2f} (always = checklist size {len(best_names[f])} "
              f"=> NON-ADAPTIVE: asks full fixed checklist every case, never stops early)")


# ============================================================= 5. base distribution exploit
def section5():
    print("\n" + "=" * 96)
    print("5. BASE BEHAVIOR — benign-distribution exploit?")
    print("=" * 96)
    base = load("base")
    cases = list(E.gen_cases(N_CASES, 0.0, seed=HIDDEN_SEED, families=E.DEEP_FAMILIES))
    risk_n = Counter(c["true_risk"] for c in cases)
    print(f"   hidden eval-case risk mix (n={N_CASES}): " + ", ".join(f"{k}={v}" for k, v in risk_n.items()))
    g = by_group(base)
    contrib = {}
    for grp in ("high", "moderate", "none"):
        rs = g[grp]
        share = len(rs) / len(base)
        mr = stx.mean(r["reward"] for r in rs)
        contrib[grp] = (share, mr, share * mr)
        cont = Counter(r["decision"] for r in rs)
        tot = sum(cont.values()) or 1
        print(f"   base {grp:9s}: rollout-share={share:.2f}  meanR={mr:+.3f}  contribToOverall={share*mr:+.3f}  "
              f"mean_asks={stx.mean(r['n_asks'] for r in rs):.2f}  continue_rate={cont['continue']/tot:.2f}")
    overall = stx.mean(r["reward"] for r in base)
    print(f"   base overall meanR={overall:+.3f}; benign contributes {contrib['none'][2]:+.3f} "
          f"({100*contrib['none'][2]/overall:.0f}% of it). High+moderate contribute "
          f"{contrib['high'][2]+contrib['moderate'][2]:+.3f}.")


if __name__ == "__main__":
    section1(); section2(); section3(); section4(); section5()
    print("\n" + "=" * 96)
    print("Run `python scripts/gate2_audit.py | tee docs/eval_runs/gate2_audit_rx_triage.txt` to persist.")
    print("=" * 96)
