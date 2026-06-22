"""Task 2 — Acute-presentation diagnostic work-up as an adaptive-VOI POMDP (Gate-1 target).

Structurally DIFFERENT from rx_triage (a triage-by-presentation diagnostic task: order tests/findings, then
commit a disposition) but with the SAME reward GEOMETRY (mirrored utility matrix, per-ask cost 0.03, a rare
high-stakes "emergent" subset, steep penalty for missing it). Purpose: test whether rx_triage's
"gates-necessary-but-not-sufficient / 1-ask shortcut collapse" pattern REPLICATES on a different task with
similar geometry. Reuses the gate framework unchanged (conforms to framework/envs/base.py).

Two faces, identical machinery to rx_triage:
  * analytic POMDP   — oracle_value / best_commit / oracle_rollout (exact) for the gates.
  * dialogue rollout — SYS / gen_cases / user_msg / answer / score for SFT, the support probe, and GRPO.
Verifiable reward (true severity known in-sim); no data/weights here, cases generated on demand.
"""
from __future__ import annotations
import random
from itertools import product, combinations

# ---- hidden findings/tests (binary: 0 = reassuring, 1 = concerning) ------------------------------
ITEMS = ["troponin_high", "ecg_ischemia", "hypotension", "dyspnea_severe", "radiation_arm", "pleuritic_pain",
         "prior_cad", "abd_rigidity", "rebound_tender", "lactate_high", "fever_high", "wbc_high",
         "guarding", "ct_free_air",
         # extra moderate/benign-gate findings (used by the deephedge trees to add disjoint decision points,
         # forcing a fixed checklist to over-order -> bigger BC-gap; harmless/unused in the other variants).
         "renal_impair", "age_high", "anticoag_use", "copd"]
import os as _os
DX_VARIANT = _os.environ.get("DX_VARIANT", "base")   # base | lowcost | antihedge | ah_lc | highprev
#     trop  ecg   hypo  dysp  radi  pleu  pcad  rigid rebnd lact  fever wbc   guard ctair  renal age   antic copd
P1 = [0.45, 0.45, 0.35, 0.40, 0.45, 0.40, 0.40, 0.50, 0.45, 0.35, 0.45, 0.45, 0.40, 0.30, 0.45, 0.45, 0.40, 0.40]
IDX = {n: i for i, n in enumerate(ITEMS)}
# highprev: raise prevalence of the emergent-gating findings so the rare high-stakes subset is DENSER ->
# gathering's benefit is less diffuse (pays off on more cases). Reward function itself is unchanged.
if DX_VARIANT == "highprev":
    for _n, _p in dict(troponin_high=0.62, ecg_ischemia=0.62, hypotension=0.55,
                       abd_rigidity=0.66, rebound_tender=0.62, lactate_high=0.55).items():
        P1[IDX[_n]] = _p
if DX_VARIANT == "deephedge":
    # emergent is a 4-finding conjunction (see trees below); raise those findings' prevalence so emergent stays
    # ~10% (not vanishing) and the high-subset BC-gap contribution is meaningful.
    for _n in ("troponin_high", "ecg_ischemia", "hypotension", "dyspnea_severe",
               "abd_rigidity", "rebound_tender", "lactate_high", "ct_free_air"):
        P1[IDX[_n]] = 0.52
DEEP_FAMILIES = ("chest_pain", "abd_pain")     # primary RL subset (deepest trees, largest adaptive gap)
SHALLOW_FAMILIES = ("syncope",)

ACTIONS = ("discharge", "symptomatic_tx", "observe", "admit", "emergent_tx")
# Severity tiers use the framework's canonical keys none/moderate/high (described as benign/urgent/emergent in
# prose). The framework's defensive/escalation actions are read from these two attrs (it defaults to rx_triage):
ACT_HOLD = "admit"            # the hedge (mirrors rx_triage hold_for_labs)
ACT_ESCALATE = "emergent_tx"  # the escalation (mirrors rx_triage escalate)
EPS_TIE = 0.05
# lowcost / ah_lc: halve the per-ask penalty so gathering is cheaper relative to its (diffuse) benefit.
ASK_COST = 0.015 if DX_VARIANT in ("lowcost", "ah_lc") else 0.03
EPS = 0.03                                 # oracle-grade tolerance: reward >= v_oracle - EPS
_AH = DX_VARIANT in ("antihedge", "ah_lc")  # antihedge: admit-on-emergent goes negative (breaks Gate 1; see screen)
# deephedge (the co-designed capped-hedge variant): keep admit POSITIVE & safe-ish everywhere (never catastrophic)
# but CAP it well below the correct disposition on both moderate and emergent, so blind-hedging is clearly
# suboptimal and gathering to the right action is rewarded on BOTH subsets -- yet 1-ask still hedges (scores low
# on true emergents), so Gate 1's under-gather penalty holds. emergent stays a multi-finding confirmation.
_CAP = DX_VARIANT == "deephedge"

# Reward geometry MIRRORS rx_triage (renamed actions/severities): missing an emergent OR urgent presentation is
# strongly penalized so cheap under-gathering -> wrong disposition is unprofitable; `admit` is the only hedge and
# it is NOT good on emergent (must disambiguate). NO disposition is good across severities -> must know the branch.
#   discharge<-continue  symptomatic_tx<-dose_adjust  observe<-bridge  admit<-hold_for_labs  emergent_tx<-escalate
#   benign<-none  urgent<-moderate  emergent<-high
#   severity keys: none=benign, moderate=urgent, high=emergent (canonical framework keys)
U = {
    "none":     {"discharge": 1.00, "symptomatic_tx": 0.40, "observe": 0.45, "admit": 0.30, "emergent_tx": 0.10},
    "moderate": {"discharge": -0.20, "symptomatic_tx": 0.90, "observe": 0.30,
                 "admit": (0.45 if _CAP else 0.86), "emergent_tx": 0.40},
    "high":     {"discharge": -1.00, "symptomatic_tx": -0.50, "observe": -0.60,
                 "admit": (-0.30 if _AH else 0.20 if _CAP else 0.40), "emergent_tx": 0.95},
}

# Branching relevance trees. node = ("commit", severity) | ("ask", item, {0: node, 1: node}).
# Fresh topology (NOT a relabel of rx_triage): disjoint per-branch finding-modules so a SAFE fixed test panel
# must over-order (cost-crushed) or stay short and miss a branch (steep penalty); the adaptive oracle walks one
# short path. emergent is engineered rare (deep concerning conjunction).
FAMILIES = {
    "chest_pain": ("ask", "troponin_high", {                        # rel-union 7
        1: ("ask", "ecg_ischemia", {
            1: ("ask", "hypotension", {
                1: ("commit", "high"),
                0: ("ask", "dyspnea_severe", {1: ("commit", "high"), 0: ("commit", "moderate")})}),
            0: ("ask", "radiation_arm", {
                1: ("commit", "moderate"),
                0: ("ask", "hypotension", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
        0: ("ask", "pleuritic_pain", {
            1: ("commit", "moderate"),
            0: ("ask", "prior_cad", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
    "abd_pain": ("ask", "abd_rigidity", {                           # rel-union 7 (disjoint per-branch modules)
        1: ("ask", "rebound_tender", {
            1: ("ask", "lactate_high", {
                1: ("commit", "high"),
                0: ("ask", "ct_free_air", {1: ("commit", "high"), 0: ("commit", "moderate")})}),
            0: ("ask", "guarding", {
                1: ("commit", "moderate"),
                0: ("ask", "lactate_high", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
        0: ("ask", "fever_high", {
            1: ("commit", "moderate"),
            0: ("ask", "wbc_high", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
    "syncope": ("ask", "ecg_ischemia", {                            # shallow (SFT-solvable)
        1: ("ask", "hypotension", {1: ("commit", "high"), 0: ("commit", "moderate")}),
        0: ("commit", "none")}),
}

if _CAP:
    # deephedge trees: emergent is a 3-finding CONJUNCTION, and each confirming finding is REUSED across
    # non-emergent branches so NO single finding confirms emergent (P(emergent | one finding) stays low). Then a
    # 1-ask "escalate on a positive finding" strategy is wrong (over-treats), so under-gathering is penalized on
    # the high subset (Gate 1 holds) — yet the full path yields the large emergent reward (gathering is required).
    # Emergent conjunction findings (troponin/ecg/hypotension, rigidity/rebound/lactate) are REUSED across branches
    # so none is individually diagnostic; the moderate/benign splits use DISJOINT per-branch findings (radiation,
    # dyspnea, pleuritic, prior_cad / fever, wbc, guarding) so a fixed checklist must over-order them (BC-gap)
    # while the oracle walks one short path.
    FAMILIES = {
        "chest_pain": ("ask", "troponin_high", {        # emergent = troponin & ecg & hypotension & dyspnea (4, reused)
            1: ("ask", "ecg_ischemia", {
                1: ("ask", "hypotension", {
                    1: ("ask", "dyspnea_severe", {1: ("commit", "high"),
                        0: ("ask", "radiation_arm", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
                    0: ("ask", "dyspnea_severe", {1: ("commit", "moderate"),
                        0: ("commit", "none")})}),
                0: ("ask", "hypotension", {
                    1: ("ask", "prior_cad", {1: ("commit", "moderate"), 0: ("commit", "none")}),
                    0: ("ask", "anticoag_use", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
            0: ("ask", "ecg_ischemia", {
                1: ("ask", "hypotension", {1: ("commit", "moderate"),
                    0: ("ask", "renal_impair", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
                0: ("ask", "pleuritic_pain", {1: ("commit", "moderate"),
                    0: ("ask", "copd", {1: ("commit", "moderate"), 0: ("commit", "none")})})})}),
        "abd_pain": ("ask", "abd_rigidity", {           # emergent = rigidity & rebound & lactate & ct_free_air (4, reused)
            1: ("ask", "rebound_tender", {
                1: ("ask", "lactate_high", {
                    1: ("ask", "ct_free_air", {1: ("commit", "high"),
                        0: ("ask", "fever_high", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
                    0: ("ask", "ct_free_air", {1: ("commit", "moderate"),
                        0: ("commit", "none")})}),
                0: ("ask", "lactate_high", {
                    1: ("ask", "wbc_high", {1: ("commit", "moderate"), 0: ("commit", "none")}),
                    0: ("ask", "renal_impair", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
            0: ("ask", "rebound_tender", {
                1: ("ask", "lactate_high", {1: ("commit", "moderate"),
                    0: ("ask", "guarding", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
                0: ("ask", "fever_high", {1: ("commit", "moderate"),
                    0: ("ask", "copd", {1: ("commit", "moderate"), 0: ("commit", "none")})})})}),
        "syncope": FAMILIES["syncope"],
    }

# ---------------------------------------------------------------------------------------------------
def acceptable(risk, eps=EPS_TIE):
    best = max(U[risk].values())
    return [a for a in ACTIONS if best - U[risk][a] <= eps + 1e-9]

def _walk_items(node, acc):
    if node[0] == "ask":
        acc.add(node[1])
        for child in node[2].values():
            _walk_items(child, acc)
    return acc

def rel_items(fam):
    # Deterministic order (by item index) so oracle_rollout's greedy tie-break is hash-seed-independent; only the
    # oracle-grade THRESHOLD depends on it, never model reward (see rx_triage env for the full rationale).
    return sorted(_walk_items(FAMILIES[fam], set()), key=lambda n: IDX[n])

def screen_item(fam):
    return FAMILIES[fam][1]

def true_risk(fam, assign):
    node = FAMILIES[fam]
    while node[0] == "ask":
        node = node[2][assign[IDX[node[1]]]]
    return node[1]

def _pitem(i, v):
    return P1[i] if v == 1 else 1 - P1[i]

# ---- belief / oracle (memoized; caches cleared if U or cost context changes) ----------------------
_RISK_CACHE, _ORACLE_CACHE = {}, {}
def clear_caches():
    _RISK_CACHE.clear(); _ORACLE_CACHE.clear()

def risk_dist(fam, revealed):
    key = (fam, frozenset(revealed.items()))
    if key in _RISK_CACHE:
        return _RISK_CACHE[key]
    rel = [IDX[n] for n in rel_items(fam)]
    unknown = [i for i in rel if i not in revealed]
    dist = {}
    for combo in product((0, 1), repeat=len(unknown)):
        a = dict(revealed); w = 1.0
        for i, v in zip(unknown, combo):
            a[i] = v; w *= _pitem(i, v)
        full = [a.get(i, 0) for i in range(len(ITEMS))]
        dist[true_risk(fam, full)] = dist.get(true_risk(fam, full), 0.0) + w
    _RISK_CACHE[key] = dist
    return dist

def best_commit(fam, revealed):
    d = risk_dist(fam, revealed)
    vals = {a: sum(p * U[r][a] for r, p in d.items()) for a in ACTIONS}
    a = max(ACTIONS, key=lambda a: vals[a])
    return vals[a], a

def oracle_value(fam, cost, revealed=None):
    revealed = dict(revealed or {})
    okey = (fam, cost, frozenset(revealed.items()))
    if okey in _ORACLE_CACHE:
        return _ORACLE_CACHE[okey]
    best, _ = best_commit(fam, revealed)
    for i in [IDX[n] for n in rel_items(fam)]:
        if i in revealed:
            continue
        ev = sum(_pitem(i, v) * oracle_value(fam, cost, {**revealed, i: v}) for v in (0, 1))
        best = max(best, -cost + ev)
    _ORACLE_CACHE[okey] = best
    return best

def oracle_rollout(fam, assign, cost, revealed0=None):
    """Adaptive trajectory on a concrete case -> (asks, action, reward). revealed0 items are free."""
    revealed, asks = dict(revealed0 or {}), []
    while True:
        commit_v, commit_a = best_commit(fam, revealed)
        best_ask, best_val = None, commit_v
        for i in [IDX[n] for n in rel_items(fam) if IDX[n] not in revealed]:
            ev = sum(_pitem(i, v) * oracle_value(fam, cost, {**revealed, i: v}) for v in (0, 1))
            if -cost + ev > best_val + 1e-12:
                best_val, best_ask = -cost + ev, i
        if best_ask is None:
            return asks, commit_a, U[true_risk(fam, assign)][commit_a] - cost * len(asks)
        revealed[best_ask] = assign[best_ask]; asks.append(ITEMS[best_ask])

# ---- dialogue rollout interface (SFT / probe / GRPO) ----------------------------------------------
ASK_INTENTS = list(ITEMS)
_INTENT2ITEM = {n: n for n in ITEMS}

def resolve_ask(raw):
    """Map a model-emitted intent to a canonical item (lenient: strips 'order_'/'ask_' prefix)."""
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    for pre in ("order_", "ask_", "test_"):
        if r.startswith(pre):
            r = r[len(pre):]
    return r if r in _INTENT2ITEM else None

_FAM_DESC = {
    "chest_pain": "adult with acute chest pain",
    "abd_pain": "adult with acute abdominal pain",
    "syncope": "adult with a syncopal episode",
}
_REVEAL = {
    "troponin_high": ("troponin normal", "troponin elevated"),
    "ecg_ischemia": ("ECG without ischemia", "ECG shows ischemia"),
    "hypotension": ("blood pressure stable", "hypotensive"),
    "dyspnea_severe": ("no significant dyspnea", "severe dyspnea"),
    "radiation_arm": ("no radiating pain", "pain radiating to arm/jaw"),
    "pleuritic_pain": ("not pleuritic", "pleuritic chest pain"),
    "prior_cad": ("no prior cardiac disease", "known coronary disease"),
    "abd_rigidity": ("abdomen soft", "abdominal rigidity"),
    "rebound_tender": ("no rebound", "rebound tenderness"),
    "lactate_high": ("lactate normal", "lactate elevated"),
    "fever_high": ("afebrile", "febrile"),
    "wbc_high": ("WBC normal", "WBC elevated"),
    "guarding": ("no guarding", "abdominal guarding"),
    "ct_free_air": ("CT: no free air", "CT: free air"),
    "renal_impair": ("renal function normal", "renal impairment"),
    "age_high": ("younger patient", "older patient"),
    "anticoag_use": ("no anticoagulation", "on anticoagulation"),
    "copd": ("no COPD", "COPD history"),
}

SYS = (
    "You are an acute-care triage assistant. The decision-relevant findings depend on what you order, so "
    "gather only what you need before deciding. Each turn output EXACTLY ONE action as JSON (one item per turn).\n"
    'To order a test/finding: {"action_type":"ask","intent":"<item>"} where <item> is one of: ' + ", ".join(ITEMS) + "\n"
    'To decide disposition: {"action_type":"commit","decision":"discharge|symptomatic_tx|observe|admit|emergent_tx",'
    '"rationale":"<short>"}\n'
    "Ordering has a small cost; order only decision-relevant tests (which depend on earlier results), then commit."
)

def gen_cases(n, rho, seed, families=DEEP_FAMILIES, prefix="dx-"):
    rng = random.Random(seed)
    fams = list(families)
    cases = []
    for i in range(n):
        fam = fams[i % len(fams)] if rho == 0.0 else rng.choice(fams)
        assign = [1 if rng.random() < P1[j] else 0 for j in range(len(ITEMS))]
        observed = {ITEMS[j]: assign[j] for j in range(len(ITEMS)) if rng.random() < rho}
        revealed0 = {IDX[n]: assign[IDX[n]] for n in observed}
        oasks, oaction, oreward = oracle_rollout(fam, assign, ASK_COST, revealed0)
        cases.append({"case_id": f"{prefix}{i}", "family": fam, "assign": assign, "observed": observed,
                      "rho": rho, "v_oracle": round(oreward, 4), "oracle_asks": oasks,
                      "oracle_action": oaction, "true_risk": true_risk(fam, assign),
                      "acceptable": acceptable(true_risk(fam, assign))})
    return cases

def user_msg(case):
    lines = [f"Presentation: {_FAM_DESC[case['family']]}.", "Establish the work-up before deciding disposition."]
    if case["observed"]:
        for n, v in case["observed"].items():
            lines.append(f"Chart: {_REVEAL[n][v]}.")
    else:
        lines.append("Chart: no results on file yet.")
    return "\n".join(lines)

def answer(case, intent):
    item = resolve_ask(intent)
    if item is None:
        return "That is not a recognized test."
    return f"Result: {_REVEAL[item][case['assign'][IDX[item]]]}."

def score(case, asks, decision):
    """Verifiable reward; oracle_grade = reward >= v_oracle - EPS. Returns (reward, oracle_grade, valid, acc)."""
    n_asks = sum(1 for a in asks if resolve_ask(a) is not None)
    if decision not in ACTIONS:
        return 0.0, False, False, False
    reward = U[case["true_risk"]][decision] - ASK_COST * n_asks
    return round(reward, 4), reward >= case["v_oracle"] - EPS, True, decision in case["acceptable"]


# ---- DREAM-inspired decoupled explore/exploit helpers (training + composed eval only) -----------
# These do NOT touch score()/U()/the analytic gates. They build the two mode-tokened prompts:
#   * explore mode: ask-only (no commit), used to gather findings;
#   * commit mode: present ONLY the explore-gathered findings, then commit in one turn.
# The commit-mode prompt is constructed from the supplied `asks` list ALONE -- it never reads
# true_risk / v_oracle / unrevealed items, so privileged severity (mu) cannot leak. assert_no_leak()
# is a hard guard the rollout/probe call before generating a commit.
MODE_EXPLORE = "[MODE: explore] Order tests to gather decision-relevant findings. Do NOT commit yet."
MODE_COMMIT = "[MODE: commit] Based ONLY on the findings below, output exactly one commit action now."


def explore_user_msg(case):
    """Explore-phase user turn: the normal presentation + the explore mode token (ask-only)."""
    return user_msg(case) + "\n" + MODE_EXPLORE


def commit_user_msg(case, asks):
    """Commit-phase user turn: re-present ONLY the explore-gathered findings, then the commit mode token.
    Reads case['assign'] for the gathered items only (the values the explore phase already saw)."""
    lines = [f"Presentation: {_FAM_DESC[case['family']]}."]
    seen = []
    for a in asks:
        it = resolve_ask(a)
        if it is None or it in seen:
            continue
        seen.append(it)
        lines.append(f"Finding: {_REVEAL[it][case['assign'][IDX[it]]]}.")
    if not seen:
        lines.append("Finding: none gathered.")
    lines.append(MODE_COMMIT)
    return "\n".join(lines)


def assert_no_leak(case, asks, commit_text):
    """Hard guard: the commit-phase context must expose ONLY explore-gathered findings -- never the hidden
    severity (mu), v_oracle, or any unrevealed item's result. Raises AssertionError on leak/tamper.
    Primary guarantee is structural: commit_text must equal the canonical gathered-only prompt (which is built
    from `asks` alone, so it cannot contain mu / v_oracle / unrevealed findings). A reveal-string scan over the
    UNREVEALED items is kept as a redundant catch."""
    canonical = commit_user_msg(case, asks)
    assert commit_text.strip() == canonical.strip(), "LEAK/tamper: commit prompt deviates from gathered-only canonical"
    gathered = {resolve_ask(a) for a in asks} - {None}
    for it in ITEMS:                       # redundant: no unrevealed item's result may appear
        if it in gathered:
            continue
        for val_str in _REVEAL[it]:
            assert val_str not in commit_text, f"LEAK: unrevealed item '{it}' result in commit prompt"
    return True
