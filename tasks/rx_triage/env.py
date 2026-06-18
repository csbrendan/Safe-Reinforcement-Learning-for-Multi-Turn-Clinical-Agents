"""Task 1 — RX refill triage as an adaptive-workup POMDP (Gate-1-passing).

Carried over (code only) from the prior project's validated environment. It is the single task that has
cleared Gate 1 under the recalibrated, safety-steep reward: optimal information-gathering is *sequentially
adaptive* (which item matters next depends on the previous answer's revealed value), so no fixed checklist
matches the oracle, while the reward prices out BOTH under-gathering (penalized on high-risk) and
over-escalation (penalized on benign).

Conforms to framework/envs/base.py. Two faces:
  * analytic POMDP   — oracle_value / best_commit / oracle_rollout (exact, enumerable) for the gates.
  * dialogue rollout — SYS / gen_cases / user_msg / answer / score for SFT, the support probe, and GRPO.

No data or weights live here; cases are generated on demand. Reward is verifiable (true state known in-sim).
"""
from __future__ import annotations
import random
from itertools import product, combinations

# ---- hidden items (binary: 0 = benign, 1 = concerning) -------------------------------------------
ITEMS = ["renal_decline", "potassium_high", "volume_dehydrated", "bp_low", "nsaid_use", "adherence_poor",
         "acute_symptoms", "egfr_severe", "sodium_low", "gi_bleed_risk", "lft_abnormal", "a1c_high"]
#     renal  K     vol   bp    nsaid adher acute egfrS naLow giBld lft   a1c
P1 = [0.45, 0.40, 0.45, 0.35, 0.50, 0.40, 0.40, 0.50, 0.40, 0.40, 0.20, 0.40]   # P(item == 1); screens ~0.5
IDX = {n: i for i, n in enumerate(ITEMS)}
DEEP_FAMILIES = ("ACEi_ARB", "NSAID")     # primary RL subset (deepest trees, largest adaptive gap)
SHALLOW_FAMILIES = ("metformin",)

ACTIONS = ("continue", "dose_adjust", "bridge", "hold_for_labs", "escalate")
EPS_TIE = 0.05
ASK_COST = 0.03                            # recalibrated (lower cost; see docs/lessons_learned.md)
EPS = 0.03                                 # oracle-grade tolerance: reward >= v_oracle - EPS

# Recalibrated reward "steepHigh+tightMod": mishandling a high OR moderate risk is strongly penalized so
# cheap under-gathering -> wrong commit is unprofitable; hold_for_labs is NOT good on high (no moderate/high
# hedge -> the agent must disambiguate). NO action is good across risks -> you must know the branch.
U = {
    "none":     {"continue": 1.00, "dose_adjust": 0.40, "bridge": 0.45, "hold_for_labs": 0.30, "escalate": 0.10},
    "moderate": {"continue": -0.20, "dose_adjust": 0.90, "bridge": 0.30, "hold_for_labs": 0.86, "escalate": 0.40},
    "high":     {"continue": -1.00, "dose_adjust": -0.50, "bridge": -0.60, "hold_for_labs": 0.40, "escalate": 0.95},
}

# Branching relevance trees. node = ("commit", risk) | ("ask", item, {0: node, 1: node}).
# Deep families use disjoint per-branch item-modules so a SAFE fixed checklist must over-ask (cost-crushed)
# or stay short and miss a branch (steep penalty); the adaptive oracle walks one short path.
FAMILIES = {
    "ACEi_ARB": ("ask", "renal_decline", {                      # rel-union 7
        1: ("ask", "egfr_severe", {
            1: ("ask", "potassium_high", {
                1: ("commit", "high"),
                0: ("ask", "acute_symptoms", {1: ("commit", "high"), 0: ("commit", "moderate")})}),
            0: ("ask", "volume_dehydrated", {
                1: ("commit", "moderate"),
                0: ("ask", "potassium_high", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
        0: ("ask", "adherence_poor", {
            1: ("commit", "moderate"),
            0: ("ask", "bp_low", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
    "NSAID": ("ask", "nsaid_use", {                             # rel-union 6
        1: ("ask", "renal_decline", {
            1: ("ask", "gi_bleed_risk", {
                1: ("commit", "high"),
                0: ("ask", "acute_symptoms", {1: ("commit", "high"), 0: ("commit", "moderate")})}),
            0: ("ask", "gi_bleed_risk", {
                1: ("commit", "moderate"),
                0: ("ask", "sodium_low", {1: ("commit", "moderate"), 0: ("commit", "none")})})}),
        0: ("ask", "adherence_poor", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
    "diuretic": ("ask", "volume_dehydrated", {                  # medium
        1: ("ask", "renal_decline", {
            1: ("commit", "high"),
            0: ("ask", "sodium_low", {1: ("commit", "high"), 0: ("commit", "moderate")})}),
        0: ("ask", "bp_low", {1: ("commit", "moderate"), 0: ("commit", "none")})}),
    "metformin": ("ask", "renal_decline", {                     # shallow (SFT-solvable)
        1: ("ask", "potassium_high", {1: ("commit", "high"), 0: ("commit", "moderate")}),
        0: ("commit", "none")}),
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
    # Deterministic order (by item index): oracle_rollout's greedy best-ask tie-break iterates this, so an
    # unordered set made the oracle path (hence v_oracle / the oracle-grade THRESHOLD) wobble ±ask_cost across
    # PYTHONHASHSEED on value-of-information ties. oracle_value/risk_dist are max/sum reductions (order-free) and
    # model reward (env.score) uses neither — so sorting fixes only support-threshold reproducibility.
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
    """Map a model-emitted intent to a canonical item (lenient: strips 'ask_' prefix)."""
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    if r.startswith("ask_"):
        r = r[4:]
    return r if r in _INTENT2ITEM else None

_FAM_DESC = {
    "ACEi_ARB": "ACE inhibitor / ARB (e.g., lisinopril) refill",
    "NSAID": "chronic NSAID (e.g., naproxen) refill",
    "diuretic": "diuretic (e.g., furosemide) refill",
    "metformin": "metformin refill",
}
_REVEAL = {
    "renal_decline": ("eGFR stable", "eGFR declining"),
    "potassium_high": ("potassium normal", "potassium elevated"),
    "volume_dehydrated": ("euvolemic", "dehydrated"),
    "bp_low": ("blood pressure ok", "blood pressure low"),
    "nsaid_use": ("no NSAID use", "ongoing NSAID use"),
    "adherence_poor": ("adherent", "poor adherence"),
    "acute_symptoms": ("no acute symptoms", "acute symptoms present"),
    "egfr_severe": ("decline mild", "decline severe"),
    "sodium_low": ("sodium normal", "sodium low"),
    "gi_bleed_risk": ("low GI-bleed risk", "high GI-bleed risk"),
    "lft_abnormal": ("LFTs normal", "LFTs abnormal"),
    "a1c_high": ("A1c at goal", "A1c above goal"),
}

SYS = (
    "You are a medication-refill triage assistant. The decision-relevant facts depend on what you find, so "
    "gather only what you need before deciding. Each turn output EXACTLY ONE action as JSON (one item per turn).\n"
    'To ask: {"action_type":"ask","intent":"<item>"} where <item> is one of: ' + ", ".join(ITEMS) + "\n"
    'To decide: {"action_type":"commit","decision":"continue|dose_adjust|bridge|hold_for_labs|escalate",'
    '"rationale":"<short>"}\n'
    "Asking has a small cost; ask only decision-relevant items (which depend on earlier answers), then commit."
)

def gen_cases(n, rho, seed, families=DEEP_FAMILIES, prefix="wk-"):
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
    lines = [f"Refill request: {_FAM_DESC[case['family']]}.", "Confirm current status before deciding."]
    if case["observed"]:
        for n, v in case["observed"].items():
            lines.append(f"Chart: {_REVEAL[n][v]}.")
    else:
        lines.append("Chart: no recent labs/notes on file.")
    return "\n".join(lines)

def answer(case, intent):
    item = resolve_ask(intent)
    if item is None:
        return "That is not a recognized query."
    return f"Result: {_REVEAL[item][case['assign'][IDX[item]]]}."

def score(case, asks, decision):
    """Verifiable reward; oracle_grade = reward >= v_oracle - EPS. Returns (reward, oracle_grade, valid, acc)."""
    n_asks = sum(1 for a in asks if resolve_ask(a) is not None)
    if decision not in ACTIONS:
        return 0.0, False, False, False
    reward = U[case["true_risk"]][decision] - ASK_COST * n_asks
    return round(reward, 4), reward >= case["v_oracle"] - EPS, True, decision in case["acceptable"]


if __name__ == "__main__":   # CPU self-check
    for rho, nm in ((0.0, "hidden"), (1.0, "anchor")):
        cs = gen_cases(200, rho, seed=1)
        mv = sum(c["v_oracle"] for c in cs) / len(cs)
        ma = sum(len(c["oracle_asks"]) for c in cs) / len(cs)
        print(f"{nm:7s} deep: mean V_oracle={mv:.3f} mean oracle_asks={ma:.2f}; "
              f"e.g. {cs[0]['family']} risk={cs[0]['true_risk']} asks={cs[0]['oracle_asks']} -> {cs[0]['oracle_action']}")
