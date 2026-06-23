"""Task 3 — risk-benefit titration as an adaptive-VOI POMDP (Gate-1 target).

Clinically DISTINCT from rx_triage (refill yes/no) and diagnostics (which disposition bucket): a TITRATION
decision — *where on an ordered intensity scale* (none < low < standard < high) to treat — under a stylized
clot-vs-bleed trade. Same project spine: ask hidden factors, then ONE ordered commit; verifiable analytic reward.

Latent state = a JOINT (clot, bleed) severity, each an additive risk SCORE over binary factors (stylized
CHA2DS2-VASc / HAS-BLED), bucketed low/mod/high. Utility U[joint_state][intensity] = benefit(clot,A) - harm(bleed,A):
benefit rises with intensity and matters more at high clot risk; harm rises with intensity and matters more at high
bleed risk. So the optimal intensity MOVES with the joint state -> adaptive gathering has value (Gate-1 BC-gap > 0).

Shared factors (age_high, htn) feed BOTH scores -> one ask can inform both axes (the source of interesting VOI).
Stylized, guideline-INSPIRED, NOT clinically authoritative (same toy-POMDP status as the other two tasks). Cases are
synthetic, generated on demand from per-factor prevalences; the env's U/risk_dist IS the ground truth.

Two faces, identical machinery to the other tasks:
  * analytic POMDP   — oracle_value / best_commit / oracle_rollout (exact) for the gates.
  * dialogue rollout — SYS / gen_cases / user_msg / answer / score for SFT, the support probe, and GRPO.
"""
from __future__ import annotations
import os as _os
import random
from itertools import product

# ---- hidden factors (binary: 0 = absent/reassuring, 1 = present/concerning) ----------------------
ITEMS = ["prior_clot", "age_high", "chf", "htn", "bleed_history", "fall_risk"]
IDX = {n: i for i, n in enumerate(ITEMS)}
#     pclot age   chf   htn   bleed fall
P1 = [0.35, 0.50, 0.40, 0.45, 0.35, 0.40]

# additive risk scores (stylized CHA2DS2-VASc / HAS-BLED) with a few PIVOTAL high-weight factors so a single ask
# can decisively move a bucket (crisp per-ask VOI). age_high + htn are SHARED across both axes (overlap = VOI).
CLOT_W = {"prior_clot": 3, "age_high": 2, "chf": 1, "htn": 1}        # max 7
BLEED_W = {"bleed_history": 3, "fall_risk": 2, "age_high": 1, "htn": 1}  # max 7

DEEP_FAMILIES = ("anticoag",)     # single presentation archetype (decision structure is presentation-invariant)
SHALLOW_FAMILIES = ()

# ---- ordered intensity scale (NEW vs the unordered commits of the other tasks) --------------------
ACTIONS = ("none", "low", "high")   # ordered: no therapy / reduced-intensity / full-intensity
ACT_HOLD = "low"        # framework hooks: the reduced-intensity "hold"-like middle and the full-intensity "escalate"
ACT_ESCALATE = "high"
EPS_TIE = 0.05
ASK_COST = 0.03
EPS = 0.03              # oracle-grade tolerance: reward >= v_oracle - EPS

# ---- multi-objective utility (single-objective by default; goal weights optional & cheap) ---------
# GOAL sets the weight on benefit (clot reduction) vs harm (bleed exposure). Default balanced = single-objective.
_GOAL_W = {"balanced": (1.0, 1.0), "clot_first": (1.5, 0.7), "bleed_first": (0.7, 1.5)}
GOAL = _os.environ.get("GOAL", "balanced")
WB, WH = _GOAL_W.get(GOAL, (1.0, 1.0))

EFF = {"none": 0.0, "low": 0.50, "high": 1.00}     # treatment efficacy (saturating)
EXP = {"none": 0.0, "low": 0.35, "high": 1.00}     # bleed exposure
CLOT_VAL = {"low": 0.10, "mod": 0.70, "high": 1.50}                  # value of preventing clots, by clot severity
BLEED_VAL = {"low": 0.10, "mod": 0.70, "high": 1.60}                 # cost of a bleed, by bleed severity
STATES = [f"{c}_{b}" for c in ("low", "mod", "high") for b in ("low", "mod", "high")]


def benefit(state, a):
    return CLOT_VAL[state.split("_")[0]] * EFF[a]


def harm(state, a):
    return BLEED_VAL[state.split("_")[1]] * EXP[a]


def utility(state, a, wb=WB, wh=WH):
    return wb * benefit(state, a) - wh * harm(state, a)


U = {s: {a: round(utility(s, a), 4) for a in ACTIONS} for s in STATES}


# ---------------------------------------------------------------------------------------------------
def acceptable(risk, eps=EPS_TIE):
    best = max(U[risk].values())
    return [a for a in ACTIONS if best - U[risk][a] <= eps + 1e-9]


def rel_items(fam):
    # all scoring factors are decision-relevant; deterministic order (by IDX) for hash-seed-independent tie-breaks.
    return list(ITEMS)


def screen_item(fam):
    return rel_items(fam)[0]


def _bucket(score):
    return "low" if score <= 1 else "mod" if score <= 4 else "high"


def true_risk(fam, assign):
    cs = sum(w * assign[IDX[n]] for n, w in CLOT_W.items())
    bs = sum(w * assign[IDX[n]] for n, w in BLEED_W.items())
    return f"{_bucket(cs)}_{_bucket(bs)}"


def _pitem(i, v):
    return P1[i] if v == 1 else 1 - P1[i]


# ---- belief / oracle (memoized; caches cleared if U or goal context changes) ----------------------
_RISK_CACHE, _ORACLE_CACHE = {}, {}


def clear_caches():
    _RISK_CACHE.clear(); _ORACLE_CACHE.clear()


def risk_dist(fam, revealed):
    """Posterior over the JOINT (clot,bleed) state given revealed factors. Entropy over this drives info-gain."""
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
    """Map a model-emitted intent to a canonical factor (lenient: strips 'ask_'/'check_'/'order_' prefixes)."""
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    for pre in ("ask_", "check_", "order_", "assess_"):
        if r.startswith(pre):
            r = r[len(pre):]
    return r if r in _INTENT2ITEM else None


_FAM_DESC = {
    "anticoag": "older adult with atrial fibrillation, deciding antithrombotic intensity",
}
_REVEAL = {
    "prior_clot": ("no prior thromboembolism", "prior stroke/VTE"),
    "chf": ("no heart failure", "heart failure"),
    "diabetes": ("no diabetes", "diabetes"),
    "vascular": ("no vascular disease", "vascular disease"),
    "age_high": ("younger patient", "elderly (>=75)"),
    "htn": ("normotensive", "hypertension"),
    "bleed_history": ("no prior bleeding", "prior major bleed"),
    "renal_impair": ("renal function normal", "renal impairment"),
    "fall_risk": ("low fall risk", "high fall risk"),
    "labile_control": ("stable INR/adherence", "labile control / poor adherence"),
}

SYS = (
    "You are a clinical decision-support assistant titrating antithrombotic therapy. The right intensity depends "
    "on the patient's clot vs bleeding risk, which depend on factors you must check. Gather only what you need, "
    "then choose an intensity. Each turn output EXACTLY ONE action as JSON (one factor per turn).\n"
    'To check a factor: {"action_type":"ask","intent":"<factor>"} where <factor> is one of: ' + ", ".join(ITEMS) + "\n"
    'To decide intensity: {"action_type":"commit","decision":"none|low|high","rationale":"<short>"}\n'
    "Checking has a small cost; check only decision-relevant factors (which depend on earlier answers), then commit."
)


def gen_cases(n, rho, seed, families=DEEP_FAMILIES, prefix="ti-"):
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
    lines = [f"Patient: {_FAM_DESC[case['family']]}.", "Confirm the relevant risk factors before choosing intensity."]
    if case["observed"]:
        for n, v in case["observed"].items():
            lines.append(f"Chart: {_REVEAL[n][v]}.")
    else:
        lines.append("Chart: no risk factors documented yet.")
    return "\n".join(lines)


def answer(case, intent):
    item = resolve_ask(intent)
    if item is None:
        return "That is not a recognized factor."
    return f"Finding: {_REVEAL[item][case['assign'][IDX[item]]]}."


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
        print(f"{nm:7s} GOAL={GOAL}: mean V_oracle={mv:.3f} mean oracle_asks={ma:.2f}; "
              f"e.g. {cs[0]['family']} state={cs[0]['true_risk']} asks={cs[0]['oracle_asks']} -> {cs[0]['oracle_action']}")
