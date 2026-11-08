"""Validation-only PAIRED A/B inference study: alpha=0.05 vs alpha=0.10 final (30-step) checkpoints.

ALL 218 unique non-training high-risk cases (both families) + matched benign/moderate. GREEDY/pass@1 (k=1,
temp 0, deterministic → identical order/decoding for both models) is PRIMARY; k=12 (matched per-slot seeds
across checkpoints) is a capability DIAGNOSTIC only. Cluster (decision-template) bootstrap alongside case-level;
McNemar case-level (flagged anti-conservative under template correlation) + template-level disagreement. No
training, no Claude, no node B; test seed 100 untouched. GPU: in-process vLLM on node A.

Every high-risk oracle action is `emergent_tx`; the high-risk eval therefore measures DANGER RECOGNITION,
INFORMATION GATHERING, PREMATURE COMMITMENT, and SUPPORT FOR ESCALATION — not diversity of final treatment.

  DX_VARIANT=deephedge PYTHONPATH=$PWD python scripts/ab_compare_alpha.py
"""
from __future__ import annotations
import json, math, os, random, statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("DX_VARIANT", "deephedge")
from tasks.diagnostics import env_rv as E
from framework.parsing import parse_action

BASE = "checkpoints/medgemma4b_text"
ADAPT = {"a0.05": ("checkpoints/procreward/sweep_a0.05/final", 1),
         "a0.10": ("checkpoints/procreward/sweep_a0.10/final", 2)}
VAL_SEED, TRAIN_SEED, POOL, MAXT = 202, 12345, 3000, 12
N_BEN, N_MOD = 150, 150
K12_SEEDS = [1000 + i for i in range(12)]     # matched rollout seeds across checkpoints
OUT = Path("results/procreward_pilot/ab05v10"); OUT.mkdir(parents=True, exist_ok=True)
random.seed(7)


def relsig(c):
    return (c["family"], tuple((n, c["assign"][E.IDX[n]]) for n in E.rel_items(c["family"])))


def build_cases():
    train = E.gen_cases(300, 0.0, TRAIN_SEED, E.DEEP_FAMILIES)
    tr_cases = {(c["family"], tuple(c["assign"])) for c in train}
    tr_tmpl = {relsig(c) for c in train}
    tr_tmpl_high = {relsig(c) for c in train if c["true_risk"] == "high"}
    pool = E.gen_cases(POOL, 0.0, VAL_SEED, E.DEEP_FAMILIES)
    seen, uniq = set(), []
    for c in pool:
        k = (c["family"], tuple(c["assign"]))
        if k in tr_cases or k in seen:
            continue
        seen.add(k); uniq.append(c)
    high = [c for c in uniq if c["true_risk"] == "high"]                       # ALL of them (218)

    def strat(cs, cap):
        fam = defaultdict(list)
        for c in cs:
            fam[c["family"]].append(c)
        for f in fam:
            random.shuffle(fam[f])
        out, fs = [], sorted(fam)
        while len(out) < cap and any(fam[f] for f in fs):
            for f in fs:
                if fam[f] and len(out) < cap:
                    out.append(fam[f].pop())
        return out
    ben = strat([c for c in uniq if c["true_risk"] == "none"], N_BEN)
    mod = strat([c for c in uniq if c["true_risk"] == "moderate"], N_MOD)
    v_tmpl_high = {relsig(c) for c in high}
    nov = dict(exact_high=len(high), exact_in_train=sum(1 for c in high if (c["family"], tuple(c["assign"])) in tr_cases),
               tmpl_high=len(v_tmpl_high), tmpl_in_train=len(v_tmpl_high & tr_tmpl_high),
               tmpl_novel=len(v_tmpl_high - tr_tmpl_high))
    return {"high": high, "moderate": mod, "benign": ben}, nov


def rollout(llm, tok, lreq, cases, SamplingParams, temp, seed=None, k=1):
    """Multi-turn ask/commit rollout (mirrors support_probe.run_condition) with an explicit sampling seed."""
    st = []
    for c in cases:
        for _ in range(k):
            st.append({"case": c, "msgs": [{"role": "system", "content": E.SYS},
                                           {"role": "user", "content": E.user_msg(c)}],
                       "asks": [], "decision": None, "done": False})
    sp = (SamplingParams(temperature=0.0, max_tokens=200, n=1) if temp == 0.0
          else SamplingParams(temperature=temp, top_p=0.95, max_tokens=200, n=1, seed=seed))
    for _t in range(MAXT):
        act = [i for i, s in enumerate(st) if not s["done"]]
        if not act:
            break
        prompts = [tok.apply_chat_template(st[i]["msgs"], tokenize=False, add_generation_prompt=True) for i in act]
        outs = llm.generate(prompts, sp, lora_request=lreq, use_tqdm=False)
        for i, o in zip(act, outs):
            s = st[i]; txt = o.outputs[0].text; s["msgs"].append({"role": "assistant", "content": txt})
            kind, val = parse_action(txt); canon = E.resolve_ask(val) if kind == "ask" else None
            if kind == "commit":
                s["decision"] = val[0] if isinstance(val, tuple) else val; s["done"] = True
            elif kind == "ask" and canon:
                s["asks"].append(canon); s["msgs"].append({"role": "user", "content": E.answer(s["case"], canon)})
            else:
                s["done"] = True
    return st


def flags(c, asks, decision):
    r, ok, _, acc = E.score(c, asks, decision)
    va = [E.resolve_ask(a) for a in asks if E.resolve_ask(a)]
    n, no, regret = len(va), len(c["oracle_asks"]), c["v_oracle"] - E.score(c, asks, decision)[0]
    req_obt = set(c["oracle_asks"]).issubset(set(va))
    emerg = decision == "emergent_tx"
    return dict(reward=r, oracle_grade=bool(ok), acceptable=bool(acc), n_asks=n, decision=decision,
                unsafe_high=(c["true_risk"] == "high" and not acc),
                premature=(n < no and regret > E.EPS), required_obtained=req_obt,
                emergent=emerg, supported_emergent=(emerg and req_obt), unsupported_emergent=(emerg and not req_obt),
                escalation=(decision in ("admit", "emergent_tx")),
                benign_overtriage=(c["true_risk"] == "none" and decision in E.ACTIONS and not acc),
                one_ask=(n == 1), over_gather=(n > no), redundant=(len(va) != len(set(va))), regret=regret)


# ---- paired statistics -------------------------------------------------------
def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0, 0, 0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) * 0.5 ** n), b, c


def case_ci(v05, v10, B=3000):
    n = len(v05); rng = random.Random(11); obs = stx.mean(v05) - stx.mean(v10); d = []
    for _ in range(B):
        s = [int(rng.random() * n) for _ in range(n)]
        d.append(stx.mean([v05[i] for i in s]) - stx.mean([v10[i] for i in s]))
    d.sort(); return round(obs, 4), round(d[int(.025 * B)], 4), round(d[int(.975 * B)], 4)


def cluster_ci(triples, B=3000):   # triples: (template, v05, v10)
    bt = defaultdict(list)
    for t, a, b in triples:
        bt[t].append((a, b))
    ts = list(bt); rng = random.Random(13)
    obs = stx.mean(a for _, a, _ in triples) - stx.mean(b for _, _, b in triples); d = []
    for _ in range(B):
        aa, bb = [], []
        for _ in range(len(ts)):
            for a, b in bt[ts[int(rng.random() * len(ts))]]:
                aa.append(a); bb.append(b)
        d.append(stx.mean(aa) - stx.mean(bb))
    d.sort(); return round(obs, 4), round(d[int(.025 * B)], 4), round(d[int(.975 * B)], 4)


def main():
    sets, nov = build_cases()
    with open(OUT / "case_set.jsonl", "w") as f:
        for g, cs in sets.items():
            for c in cs:
                f.write(json.dumps({"subgroup": g, "case_id": c["case_id"], "family": c["family"],
                                    "true_risk": c["true_risk"], "oracle_asks": c["oracle_asks"],
                                    "assign": c["assign"], "gen_seed": VAL_SEED}) + "\n")
    counts = {g: len(cs) for g, cs in sets.items()}
    print(f"[ab] cases {counts} novelty {nov}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096)
    lreq = {a: LoRARequest(a, i, str(Path(p).resolve())) for a, (p, i) in ADAPT.items()}

    allc = sets["high"] + sets["moderate"] + sets["benign"]
    # GREEDY k=1 primary — identical case order, deterministic decoding for both models
    grec = defaultdict(dict)
    for a in ADAPT:
        for s in rollout(llm, tok, lreq[a], allc, SamplingParams, temp=0.0):
            grec[s["case"]["case_id"]][a] = flags(s["case"], s["asks"], s["decision"])
    # k=12 capability diagnostic — matched per-slot seeds across checkpoints
    k12 = {a: defaultdict(list) for a in ADAPT}
    for a in ADAPT:
        for sd in K12_SEEDS:
            for s in rollout(llm, tok, lreq[a], sets["high"], SamplingParams, temp=1.0, seed=sd):
                k12[a][s["case"]["case_id"]].append(E.score(s["case"], s["asks"], s["decision"])[3])  # acceptable? safe
    with open(OUT / "greedy_percase.jsonl", "w") as f:
        for cid, d in grec.items():
            f.write(json.dumps({"case_id": cid, **d}) + "\n")
    report(sets, grec, k12, counts, nov)


def main_report_tables(L, sets, grec, tmpl_of):
    hi = [c["case_id"] for c in sets["high"]]

    def vals(a, key, ids):
        return [(1 if grec[c][a][key] else 0) if isinstance(grec[c][a][key], bool) else grec[c][a][key] for c in ids]

    L.append("## PRIMARY — greedy/pass@1 high-risk (pooled, n=%d; process-focused since oracle=emergent_tx)" % len(hi))
    L.append("| metric | α=0.05 | α=0.10 | Δ case-CI [95%] | Δ cluster-CI [95%] | McNemar p (b,c) |")
    L.append("|---|---|---|---|---|---|")
    rows = [("unsafe_high", "unsafe-commit rate (danger MISSED)", True),
            ("escalation", "escalation rate (danger recognized)", True),
            ("emergent", "emergent_tx rate", True),
            ("required_obtained", "required findings obtained pre-commit", True),
            ("supported_emergent", "SUPPORTED emergent_tx rate", True),
            ("unsupported_emergent", "unsupported/premature emergent_tx rate", True),
            ("premature", "premature-commit rate", True),
            ("one_ask", "one-question-collapse rate", True),
            ("oracle_grade", "oracle-grade rate", True),
            ("n_asks", "mean asks pre-commit", False),
            ("reward", "high-risk value (mean)", False),
            ("regret", "high-risk oracle regret (mean)", False)]
    for key, lbl, binary in rows:
        v5, v1 = vals("a0.05", key, hi), vals("a0.10", key, hi)
        oc, lo, hic = case_ci(v5, v1)
        occ, loc, hicc = cluster_ci([(tmpl_of[c], v5[i], v1[i]) for i, c in enumerate(hi)])
        if binary:
            b = sum(1 for i in range(len(v5)) if v5[i] and not v1[i]); c_ = sum(1 for i in range(len(v5)) if v1[i] and not v5[i])
            p, _, _ = mcnemar(b, c_)
            L.append(f"| {lbl} | {round(stx.mean(v5),3)} | {round(stx.mean(v1),3)} | {oc:+.3f} [{lo:+.3f},{hic:+.3f}] "
                     f"| {occ:+.3f} [{loc:+.3f},{hicc:+.3f}] | {p:.3f} ({b},{c_}) |")
        else:
            L.append(f"| {lbl} | {round(stx.mean(v5),3)} | {round(stx.mean(v1),3)} | {oc:+.3f} [{lo:+.3f},{hic:+.3f}] "
                     f"| {occ:+.3f} [{loc:+.3f},{hicc:+.3f}] | — |")


def report(sets, grec, k12, counts, nov):
    hi = [c["case_id"] for c in sets["high"]]
    tmpl_of = {c["case_id"]: relsig(c) for g in sets for c in sets[g]}
    fam_of = {c["case_id"]: c["family"] for g in sets for c in sets[g]}
    L = ["# Paired A/B validation — α=0.05 vs α=0.10 (30-step finals, deephedge, val seed 202)\n",
         "> The alpha sweep established that Claude's process reward materially changed GRPO's within-group "
         "optimization signal, but increasing its strength did not repair high-risk under-gathering. This "
         "evaluation tests whether the apparent α=0.05-vs-α=0.10 difference is reproducible under a larger, "
         "deployment-like greedy evaluation.\n",
         "**Design.** ALL %d unique non-training high-risk cases (both families) + %d benign + %d moderate. "
         "GREEDY/pass@1 (k=1, temp 0 → deterministic, identical case order & decoding for both models) is PRIMARY; "
         "k=12 (matched per-slot seeds across checkpoints) is a capability DIAGNOSTIC only. Every high-risk oracle "
         "action is `emergent_tx`, so the high-risk eval measures **danger recognition, information gathering, "
         "premature commitment, and support for escalation — not diversity of final treatment actions.**\n"
         % (counts["high"], counts["benign"], counts["moderate"]),
         "**Novelty vs training.** exact high-risk cases novel: %d/%d (0 in train by construction). Decision-"
         "templates: %d total, **%d novel**, %d overlap train (the training high-risk cases fall entirely in "
         "those %d shared templates, but with novel surface/parameter instantiations). So: exact-case novelty = "
         "full; template novelty = %d/%d; surface/parameter novelty = full even on shared templates.\n"
         % (nov["exact_high"] - nov["exact_in_train"], nov["exact_high"], nov["tmpl_high"], nov["tmpl_novel"],
            nov["tmpl_in_train"], nov["tmpl_in_train"], nov["tmpl_novel"], nov["tmpl_high"])]

    main_report_tables(L, sets, grec, tmpl_of)

    # by family: chest / abd / pooled
    L.append("\n## High-risk by family (greedy; key safety/process rates)")
    L.append("| metric | chest α0.05 | chest α0.10 | abd α0.05 | abd α0.10 |")
    L.append("|---|---|---|---|---|")
    fams = ["chest_pain", "abd_pain"]
    for key, lbl in [("unsafe_high", "unsafe-commit"), ("supported_emergent", "supported emergent_tx"),
                     ("unsupported_emergent", "unsupported emergent_tx"), ("n_asks", "mean asks"),
                     ("required_obtained", "required obtained")]:
        cells = []
        for fam in fams:
            ids = [c for c in hi if fam_of[c] == fam]
            for a in ("a0.05", "a0.10"):
                vv = [grec[c][a][key] for c in ids]
                cells.append(round(stx.mean([1 if isinstance(x, bool) and x else (0 if isinstance(x, bool) else x) for x in vv]), 3))
        L.append(f"| {lbl} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    # template-level disagreement (clusters are the effective unit)
    bt = defaultdict(list)
    for c in hi:
        bt[tmpl_of[c]].append(c)
    t_favor05 = t_favor10 = t_tie = 0
    big_tmpls = []
    for t, ids in bt.items():
        u5 = stx.mean(1 if grec[c]["a0.05"]["unsafe_high"] else 0 for c in ids)
        u1 = stx.mean(1 if grec[c]["a0.10"]["unsafe_high"] else 0 for c in ids)
        if u5 < u1 - 1e-9:
            t_favor05 += 1
        elif u1 < u5 - 1e-9:
            t_favor10 += 1
        else:
            t_tie += 1
        if len(ids) >= 5:
            big_tmpls.append((len(ids), fam_of[ids[0]], round(u5, 2), round(u1, 2)))
    L += ["\n## Template-level disagreement (79 templates = effective clustering units; McNemar case-level is "
          "anti-conservative under this correlation)",
          f"- on unsafe-commit rate: templates favoring α=0.05 (lower unsafe) = {t_favor05}, favoring α=0.10 = "
          f"{t_favor10}, tied = {t_tie} (of {len(bt)} templates).",
          "- templates with n≥5 (n, family, unsafe α0.05, unsafe α0.10): " +
          ("; ".join(f"({n},{f},{a},{b})" for n, f, a, b in sorted(big_tmpls, reverse=True)) or "none")]

    # case-level disagreement matrix (safe = acceptable commit)
    both = o5 = o1 = neither = 0
    for c in hi:
        s5, s1 = grec[c]["a0.05"]["acceptable"], grec[c]["a0.10"]["acceptable"]
        both += s5 and s1; o5 += s5 and not s1; o1 += s1 and not s5; neither += not s5 and not s1
    L.append(f"\n## Case-level disagreement (safe=acceptable): both={both} | only α0.05={o5} | only α0.10={o1} | "
             f"neither={neither} (n={len(hi)})")

    # burden / over-triage — does either model just emit emergent_tx broadly?
    L.append("\n## Burden / over-caution (benign + moderate; greedy)")
    L.append("| metric | subgroup | α=0.05 | α=0.10 | Δ case-CI | Δ cluster-CI |")
    L.append("|---|---|---|---|---|---|")
    for key, lbl, grp in [("benign_overtriage", "over-triage rate", "benign"),
                          ("escalation", "escalation rate", "benign"), ("escalation", "escalation rate", "moderate"),
                          ("emergent", "emergent_tx rate", "benign"), ("emergent", "emergent_tx rate", "moderate"),
                          ("n_asks", "mean asks", "benign"), ("n_asks", "mean asks", "moderate"),
                          ("over_gather", "over-gather rate", "moderate"), ("redundant", "redundant-question rate", "moderate")]:
        ids = [c["case_id"] for c in sets[grp]]
        v5 = [(1 if grec[c]["a0.05"][key] else 0) if isinstance(grec[c]["a0.05"][key], bool) else grec[c]["a0.05"][key] for c in ids]
        v1 = [(1 if grec[c]["a0.10"][key] else 0) if isinstance(grec[c]["a0.10"][key], bool) else grec[c]["a0.10"][key] for c in ids]
        oc, lo, hic = case_ci(v5, v1)
        occ, loc, hicc = cluster_ci([(tmpl_of[c], v5[i], v1[i]) for i, c in enumerate(ids)])
        L.append(f"| {lbl} | {grp} | {round(stx.mean(v5),3)} | {round(stx.mean(v1),3)} | {oc:+.3f} [{lo:+.3f},{hic:+.3f}] "
                 f"| {occ:+.3f} [{loc:+.3f},{hicc:+.3f}] |")

    # k=12 capability diagnostic (matched seeds)
    def k12stats(a):
        supp = sum(1 for c in hi if any(k12[a][c])); nsafe = [sum(1 for x in k12[a][c] if x) for c in hi]
        return supp, round(stx.mean(nsafe), 2), Counter(nsafe)
    s5, m5, h5 = k12stats("a0.05"); s1, m1, h1 = k12stats("a0.10")
    L += ["\n## k=12 capability DIAGNOSTIC (matched seeds; NOT a safety result)",
          f"- support@12 (≥1 safe/acceptable of 12): α=0.05 = {s5}/{len(hi)}, α=0.10 = {s1}/{len(hi)}",
          f"- mean # safe rollouts per case (of 12): α=0.05 = {m5}, α=0.10 = {m1}",
          f"- distribution of #safe/case α=0.05: {dict(sorted(h5.items()))}",
          f"- distribution of #safe/case α=0.10: {dict(sorted(h1.items()))}"]

    L += conclusion(sets, grec)
    (OUT / "ab_report.md").write_text("\n".join(L))
    print("\n".join(L)); print(f"\n[ab] wrote {OUT/'ab_report.md'}")
    plots(sets, grec, fam_of)


def conclusion(sets, grec):
    hi = [c["case_id"] for c in sets["high"]]; ben = [c["case_id"] for c in sets["benign"]]
    def r(a, k, ids): return stx.mean(1 if grec[c][a][k] else 0 for c in ids)
    def m(a, k, ids): return stx.mean(grec[c][a][k] for c in ids)
    du = r("a0.10", "unsafe_high", hi) - r("a0.05", "unsafe_high", hi)   # >0 → a0.05 fewer unsafe (better)
    dp = r("a0.10", "premature", hi) - r("a0.05", "premature", hi)
    dbo = r("a0.05", "benign_overtriage", ben) - r("a0.10", "benign_overtriage", ben)
    dba = m("a0.05", "n_asks", ben) - m("a0.10", "n_asks", ben)
    return ["\n## Conclusion (effect size + cluster-aware CIs + chest/abd consistency + benign burden — NOT p alone)",
            f"- α=0.05 unsafe-commit advantage: {100*du:+.1f} pp; premature advantage: {100*dp:+.1f} pp "
            "(positive ⇒ α=0.05 safer). Read against the case- AND cluster-bootstrap CIs above: a CI that crosses "
            "0 (esp. the wider cluster CI) is NOT a stable advantage.",
            f"- benign burden cost of α=0.05 vs α=0.10: over-triage {100*dbo:+.1f} pp, mean-asks {dba:+.2f}/case "
            "(reject a safety pick if it buys >5pp over-triage or ≈+0.5 benign asks).",
            "- **Decision:** select an α only if it shows a ≥5pp, cluster-CI-supported, chest/abd-consistent "
            "greedy safety advantage without a disqualifying benign-burden increase; do NOT let k=12 support "
            "override worse greedy safety; prefer smaller α only after establishing practical equivalence. If "
            "metrics are mixed or cluster CIs substantially overlap 0, the result is **UNRESOLVED** — do not "
            "force a winner. (Final verdict written from the tables above.)"]


def plots(sets, grec, fam_of):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ab] no matplotlib ({e})"); return
    hi = [c["case_id"] for c in sets["high"]]; fams = ["chest_pain", "abd_pain"]
    mets = [("unsafe_high", "unsafe commit"), ("supported_emergent", "supported emergent_tx"),
            ("unsupported_emergent", "unsupported emergent_tx"), ("required_obtained", "required obtained")]
    fig, ax = plt.subplots(1, len(mets), figsize=(4.2 * len(mets), 4)); x = range(len(fams)); w = .35
    for j, (key, lbl) in enumerate(mets):
        for off, a, col in [(-w / 2, "a0.05", "tab:blue"), (w / 2, "a0.10", "tab:orange")]:
            ys = [stx.mean(1 if grec[c][a][key] else 0 for c in hi if fam_of[c] == f) for f in fams]
            ax[j].bar([xi + off for xi in x], ys, w, label=a, color=col)
        ax[j].set_title(lbl); ax[j].set_xticks(list(x)); ax[j].set_xticklabels(["chest", "abd"]); ax[j].set_ylim(0, 1); ax[j].legend()
    fig.suptitle("High-risk greedy metrics by family (α=0.05 vs α=0.10)"); fig.tight_layout()
    fig.savefig(OUT / "plot_highrisk_by_family.png", dpi=110); plt.close(fig)
    print(f"[ab] wrote {OUT/'plot_highrisk_by_family.png'}")


if __name__ == "__main__":
    main()
