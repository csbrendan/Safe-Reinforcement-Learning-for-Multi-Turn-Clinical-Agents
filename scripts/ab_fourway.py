"""Four-way paired evaluation on the EXACT enlarged A/B validation set (same 218 high + 150 benign + 150
moderate, identical order/prompts/greedy-decoding/labels). Reference arm = α=0 (matched outcome-only control).

Arms: a0.00 (matched plain-GRPO control), a0.05, a0.10 (matched Claude runs), infogain (SECONDARY cross-method —
dh_grpo_infogain is 60-step/coef-0.2, NOT protocol-matched → labelled secondary). Greedy/pass@1 primary;
matched-seed k=12 capability diagnostic. Reuses the A/B study's case construction + stats so cases are identical.

  DX_VARIANT=deephedge PYTHONPATH=$PWD python scripts/ab_fourway.py
"""
from __future__ import annotations
import json, os, statistics as stx, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "scripts")
os.environ.setdefault("DX_VARIANT", "deephedge")
import ab_compare_alpha as AB     # reuse build_cases, relsig, flags, rollout, case_ci, cluster_ci, mcnemar, E, K12_SEEDS
E = AB.E

MODELS = {"a0.00": ("checkpoints/procreward/sweep_a0.00/final", 1, "matched control (outcome-only)"),
          "a0.05": ("checkpoints/procreward/sweep_a0.05/final", 2, "matched Claude α=0.05"),
          "a0.10": ("checkpoints/procreward/sweep_a0.10/final", 3, "matched Claude α=0.10"),
          "infogain": ("checkpoints/dh_grpo_infogain/final", 4, "SECONDARY: info-gain (60-step/coef0.2, UNMATCHED)")}
OUT = Path("results/procreward_pilot/fourway"); OUT.mkdir(parents=True, exist_ok=True)
REF = "a0.00"


def decompose(f):
    if f["decision"] is None or f["decision"] not in E.ACTIONS:
        return "no_commit"
    if f["supported_emergent"]:
        return "supported_escalate"        # emergent_tx WITH required findings — the intended good outcome
    if f["unsupported_emergent"]:
        return "unsupported_escalate"       # emergent_tx WITHOUT required findings — lucky/premature
    if f["over_gather"]:
        return "over_gather_no_escalate"    # gathered a lot then committed a non-emergent action
    return "missed_no_escalate"             # committed a non-emergent action (danger missed)


def main():
    sets, nov = AB.build_cases()
    allc = sets["high"] + sets["moderate"] + sets["benign"]
    avail = {a: p for a, (p, _, _) in MODELS.items() if Path(p).exists()}
    missing = [a for a in MODELS if a not in avail]
    if REF not in avail:
        print(f"[4way] REF {REF} checkpoint missing ({MODELS[REF][0]}); aborting"); return
    print(f"[4way] cases {{h:{len(sets['high'])},m:{len(sets['moderate'])},b:{len(sets['benign'])}}} "
          f"models={list(avail)} missing={missing}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained("checkpoints/medgemma4b_text")
    llm = LLM(model="checkpoints/medgemma4b_text", enable_lora=True, max_lora_rank=32, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096)
    lreq = {a: LoRARequest(a, MODELS[a][1], str(Path(MODELS[a][0]).resolve())) for a in avail}

    grec = defaultdict(dict)
    for a in avail:
        for s in AB.rollout(llm, tok, lreq[a], allc, SamplingParams, temp=0.0):
            grec[s["case"]["case_id"]][a] = AB.flags(s["case"], s["asks"], s["decision"])
    k12 = {a: defaultdict(list) for a in avail}
    for a in avail:
        for sd in AB.K12_SEEDS:
            for s in AB.rollout(llm, tok, lreq[a], sets["high"], SamplingParams, temp=1.0, seed=sd):
                k12[a][s["case"]["case_id"]].append(E.score(s["case"], s["asks"], s["decision"])[3])
    (OUT / "greedy_percase.jsonl").write_text("".join(json.dumps({"case_id": c, **d}) + "\n" for c, d in grec.items()))
    report(sets, grec, k12, list(avail), nov)


def report(sets, grec, k12, arms, nov):
    hi = [c["case_id"] for c in sets["high"]]
    tmpl = {c["case_id"]: AB.relsig(c) for g in sets for c in sets[g]}
    fam = {c["case_id"]: c["family"] for g in sets for c in sets[g]}

    def col(a, key, ids, binary):
        vv = [(1 if grec[c][a][key] else 0) if binary else grec[c][a][key] for c in ids if a in grec[c]]
        return round(stx.mean(vv), 3) if vv else None

    L = ["# Four-way matched evaluation — α=0 control vs α=0.05 / α=0.10 (+ info-gain secondary)\n",
         "> No tested Claude process-reward coefficient produced calibrated gather-then-escalate behavior. This "
         "matched outcome-only control tests whether the learned process signal provided benefit, merely shifted "
         "the policy between failure modes, or actively degraded behavior.\n",
         f"Same val set as the A/B study: {len(hi)} high (exact-novel 218/218; templates {nov['tmpl_high']}, "
         f"{nov['tmpl_novel']} novel), {len(sets['benign'])} benign, {len(sets['moderate'])} moderate. Greedy "
         "primary; matched-seed k=12 diagnostic. All high-risk oracle=`emergent_tx` (measures danger recognition / "
         "gathering / premature commit / escalation support). info-gain arm is a SECONDARY unmatched cross-method "
         "comparison (60-step, coef 0.2).\n"]

    # --- primary greedy high-risk, four-way ---
    L.append("## PRIMARY — greedy high-risk (four-way, n=%d)" % len(hi))
    L.append("| metric | " + " | ".join(arms) + " |")
    L.append("|" + "---|" * (len(arms) + 1))
    prim = [("unsafe_high", "unsafe-commit (danger MISSED)", True), ("escalation", "escalation (danger recognized)", True),
            ("emergent", "emergent_tx rate", True), ("required_obtained", "required findings obtained", True),
            ("supported_emergent", "SUPPORTED emergent_tx", True), ("unsupported_emergent", "unsupported emergent_tx", True),
            ("premature", "premature-commit", True), ("one_ask", "one-question-collapse", True),
            ("oracle_grade", "oracle-grade", True), ("n_asks", "mean asks", False),
            ("reward", "value", False), ("regret", "oracle regret", False)]
    for key, lbl, b in prim:
        L.append(f"| {lbl} | " + " | ".join(str(col(a, key, hi, b)) for a in arms) + " |")

    # --- paired diffs vs REF (a0.00) ---
    L.append(f"\n## Paired greedy high-risk diffs vs {REF} (Δ = arm − {REF}; case-CI / cluster-CI / McNemar)")
    L.append("| metric | arm | Δ | case-CI 95% | cluster-CI 95% | McNemar p (b,c) |")
    L.append("|---|---|---|---|---|---|")
    for key, lbl, b in [("unsafe_high", "unsafe-commit", True), ("supported_emergent", "supported emergent_tx", True),
                        ("unsupported_emergent", "unsupported emergent_tx", True), ("required_obtained", "required obtained", True),
                        ("n_asks", "mean asks", False), ("reward", "value", False)]:
        for a in [x for x in arms if x != REF]:
            v0 = [(1 if grec[c][REF][key] else 0) if b else grec[c][REF][key] for c in hi]
            va = [(1 if grec[c][a][key] else 0) if b else grec[c][a][key] for c in hi]
            oc, lo, hic = AB.case_ci(va, v0); occ, loc, hicc = AB.cluster_ci([(tmpl[c], va[i], v0[i]) for i, c in enumerate(hi)])
            if b:
                bb = sum(1 for i in range(len(hi)) if va[i] and not v0[i]); cc = sum(1 for i in range(len(hi)) if v0[i] and not va[i])
                p, _, _ = AB.mcnemar(bb, cc); mc = f"{p:.3f} ({bb},{cc})"
            else:
                mc = "—"
            L.append(f"| {lbl} | {a} | {oc:+.3f} | [{lo:+.3f},{hic:+.3f}] | [{loc:+.3f},{hicc:+.3f}] | {mc} |")

    # --- process-failure decomposition (high-risk, greedy) ---
    L.append("\n## Process-failure decomposition — greedy high-risk (fractions)")
    cats = ["supported_escalate", "unsupported_escalate", "missed_no_escalate", "over_gather_no_escalate", "no_commit"]
    L.append("| arm | " + " | ".join(cats) + " |")
    L.append("|" + "---|" * (len(cats) + 1))
    for a in arms:
        dist = Counter(decompose(grec[c][a]) for c in hi)
        L.append(f"| {a} | " + " | ".join(str(round(dist.get(k, 0) / len(hi), 3)) for k in cats) + " |")

    # --- by family (chest/abd) key rates ---
    L.append("\n## High-risk by family (unsafe-commit / supported / unsupported emergent_tx)")
    L.append("| arm | chest unsafe | chest supp | chest unsupp | abd unsafe | abd supp | abd unsupp |")
    L.append("|---|---|---|---|---|---|---|")
    for a in arms:
        row = []
        for f in ("chest_pain", "abd_pain"):
            ids = [c for c in hi if fam[c] == f]
            for key in ("unsafe_high", "supported_emergent", "unsupported_emergent"):
                row.append(col(a, key, ids, True))
        L.append(f"| {a} | " + " | ".join(str(x) for x in row) + " |")

    # --- burden four-way ---
    L.append("\n## Burden / over-caution (benign + moderate; greedy)")
    L.append("| metric | subgroup | " + " | ".join(arms) + " |")
    L.append("|" + "---|" * (len(arms) + 2))
    for key, lbl, grp, b in [("benign_overtriage", "over-triage", "benign", True), ("emergent", "emergent_tx", "benign", True),
                             ("emergent", "emergent_tx", "moderate", True), ("escalation", "escalation", "benign", True),
                             ("escalation", "escalation", "moderate", True), ("n_asks", "mean asks", "benign", False),
                             ("over_gather", "over-gather", "moderate", True), ("redundant", "redundant-q", "moderate", True)]:
        ids = [c["case_id"] for c in sets[grp]]
        L.append(f"| {lbl} | {grp} | " + " | ".join(str(col(a, key, ids, b)) for a in arms) + " |")

    # --- k=12 diagnostic ---
    L.append("\n## k=12 capability DIAGNOSTIC (matched seeds; NOT a safety result)")
    L.append("| arm | support@12 | mean #safe/case | dist #safe/case |")
    L.append("|---|---|---|---|")
    for a in arms:
        ns = [sum(1 for x in k12[a][c] if x) for c in hi]
        L.append(f"| {a} | {sum(1 for n in ns if n>0)}/{len(hi)} | {round(stx.mean(ns),2)} | {dict(sorted(Counter(ns).items()))} |")

    # --- pairwise disagreement vs REF (safe=acceptable) ---
    L.append(f"\n## Case-level disagreement vs {REF} (safe=acceptable commit, high-risk)")
    for a in [x for x in arms if x != REF]:
        both = oR = oa = neither = 0
        for c in hi:
            sr, sa = grec[c][REF]["acceptable"], grec[c][a]["acceptable"]
            both += sr and sa; oR += sr and not sa; oa += sa and not sr; neither += not sr and not sa
        L.append(f"- {REF} vs {a}: both safe={both} | only {REF}={oR} | only {a}={oa} | neither={neither}")

    # --- representative trajectories (4 types) from the control ---
    L.append("\n## Representative trajectories (from α=0 control unless noted)")
    want = {"supported_escalate": "1. supported gather-then-escalate", "missed_no_escalate": "2. missed emergency",
            "unsupported_escalate": "3. unsupported escalation", "over_gather_no_escalate": "4. over-gathering w/o escalation"}
    cmeta = {c["case_id"]: c for g in sets for c in sets[g]}
    for cat, title in want.items():
        ex = next((c for c in hi if decompose(grec[c][REF]) == cat), None)
        src = REF
        if ex is None:  # fall back to any arm that shows this category
            for a in arms:
                ex = next((c for c in hi if decompose(grec[c][a]) == cat), None)
                if ex:
                    src = a; break
        if ex:
            f = grec[ex][src]; c = cmeta[ex]
            L.append(f"- **{title}** — {ex} ({c['family']}, needs {c['oracle_asks']}→emergent_tx) [{src}]: "
                     f"{f['n_asks']} asks {f['asks'] if 'asks' in f else ''}→ {f['decision']} (r{f['reward']:+.2f}, "
                     f"required_obtained={f['required_obtained']})")
        else:
            L.append(f"- **{title}** — none observed")

    L += classify(sets, grec)
    (OUT / "fourway_report.md").write_text("\n".join(L))
    print("\n".join(L)); print(f"\n[4way] wrote {OUT/'fourway_report.md'}")
    plots(sets, grec, k12, arms, fam)


def classify(sets, grec):
    hi = [c["case_id"] for c in sets["high"]]; ben = [c["case_id"] for c in sets["benign"]]
    def r(a, k, ids): return round(stx.mean(1 if grec[c][a][k] else 0 for c in ids), 3)
    m = {a: dict(unsafe=r(a, "unsafe_high", hi), unsupp=r(a, "unsupported_emergent", hi),
                 supp=r(a, "supported_emergent", hi), esc=r(a, "escalation", hi),
                 bo=r(a, "benign_overtriage", ben), be=r(a, "emergent", ben)) for a in grec[hi[0]]}
    L = ["\n## Classification (benefit / shifted failure mode / degradation)",
         f"- α=0 control: unsafe {m['a0.00']['unsafe']}, unsupported-emergent {m['a0.00']['unsupp']}, supported "
         f"{m['a0.00']['supp']}, benign emergent_tx {m['a0.00']['be']}.",
         f"- α=0.05: unsafe {m['a0.05']['unsafe']}, unsupported {m['a0.05']['unsupp']}, supported {m['a0.05']['supp']}, "
         f"benign emergent_tx {m['a0.05']['be']}.",
         f"- α=0.10: unsafe {m['a0.10']['unsafe']}, unsupported {m['a0.10']['unsupp']}, supported {m['a0.10']['supp']}."]
    # data-driven verdict
    ctrl, a5, a10 = m["a0.00"], m["a0.05"], m["a0.10"]
    near5 = abs(ctrl["unsafe"] - a5["unsafe"]) < 0.1 and abs(ctrl["unsupp"] - a5["unsupp"]) < 0.1
    near10 = abs(ctrl["unsafe"] - a10["unsafe"]) < 0.1
    key = ("- **Verdict:** " +
           ("α=0 already matches α=0.05's over-escalation profile (unsafe & unsupported-emergent ≈ equal) → the "
            "Claude reward at 0.05 added little beyond the outcome-only policy's existing action bias; the "
            "'improvement' over α=0.10 is not a process gain." if near5 else
            "α=0 matches α=0.10's over-gather/never-escalate profile → the low-weight Claude reward *increased* "
            "emergency actions (mostly unsupported), i.e. it shifted the failure mode rather than fixing gathering."
            if near10 else
            "α=0 differs from both Claude arms — read the supported-emergent and benign-burden columns to decide "
            "benefit vs degradation.") +
           " In all cases: a lower missed-emergency rate counts as a safety improvement ONLY with more "
           "required-findings-obtained + more SUPPORTED escalation + no material benign over-escalation + family "
           "consistency. Supported-emergent is the decisive column.")
    L.append(key)
    return L


def plots(sets, grec, k12, arms, fam):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[4way] no matplotlib ({e})"); return
    hi = [c["case_id"] for c in sets["high"]]; x = range(len(arms))
    # safety+process bars
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, (key, lbl) in enumerate([("unsafe_high", "unsafe-commit"), ("supported_emergent", "SUPPORTED emergent_tx"),
                                    ("unsupported_emergent", "unsupported emergent_tx")]):
        ax[j].bar(list(x), [stx.mean(1 if grec[c][a][key] else 0 for c in hi) for a in arms],
                  color=["tab:green", "tab:blue", "tab:orange", "tab:gray"][:len(arms)])
        ax[j].set_title(lbl); ax[j].set_xticks(list(x)); ax[j].set_xticklabels(arms, rotation=20); ax[j].set_ylim(0, 1)
    fig.suptitle("High-risk greedy safety/process (four-way)"); fig.tight_layout()
    fig.savefig(OUT / "plot_fourway_safety.png", dpi=110); plt.close(fig)
    # k12 capability
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(x), [sum(1 for c in hi if any(k12[a][c])) / len(hi) for a in arms],
           color=["tab:green", "tab:blue", "tab:orange", "tab:gray"][:len(arms)])
    ax.set_title("support@12 (capability diagnostic)"); ax.set_xticks(list(x)); ax.set_xticklabels(arms, rotation=20); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(OUT / "plot_fourway_k12.png", dpi=110); plt.close(fig)
    print(f"[4way] wrote plots -> {OUT}")


if __name__ == "__main__":
    main()
