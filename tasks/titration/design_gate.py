"""CPU design gate (Gate 0) for the titration task — run BEFORE any GPU work.
Checks the U/factor design yields a non-degenerate adaptive-VOI task:
 (a) BC-gap > 0: adaptive gathering beats the best single FIXED intensity (and beats prior-blind commit);
 (b) no single fixed intensity dominates (oracle action distribution is spread);
 (c) VOI: gathering changes the committed action on a meaningful fraction of cases;
 (d) goal-sensitivity: the full-info optimal intensity flips across goal profiles {balanced, clot_first, bleed_first}.
Pure CPU, analytic; no model, no training.
"""
import statistics as stx
from collections import Counter
import tasks.titration.env as e

N, COST = 600, e.ASK_COST
cs = e.gen_cases(N, rho=0.0, seed=12345)

# --- state + oracle behavior ---
state_dist = Counter(c["true_risk"] for c in cs)
v_oracle = stx.mean(c["v_oracle"] for c in cs)
mean_asks = stx.mean(len(c["oracle_asks"]) for c in cs)
oracle_act = Counter(c["oracle_action"] for c in cs)

# --- (a) best FIXED intensity (no asks): argmax_A mean_c U[true_c][A] ---
fixed_val = {a: stx.mean(e.U[c["true_risk"]][a] for c in cs) for a in e.ACTIONS}
best_fixed_a = max(e.ACTIONS, key=lambda a: fixed_val[a])
best_fixed_v = fixed_val[best_fixed_a]
# prior-blind commit (no info): single action argmax over the prior belief
_, prior_a = e.best_commit("anticoag", {})
prior_blind_v = stx.mean(e.U[c["true_risk"]][prior_a] for c in cs)
bc_gap = v_oracle - best_fixed_v

# --- (c) VOI: gathering changes the committed action vs prior-blind ---
voi_changed = stx.mean(1.0 if c["oracle_action"] != prior_a else 0.0 for c in cs)
# full-info optimal action per case (what you'd pick knowing the true state)
def full_opt(c, wb, wh):
    return max(e.ACTIONS, key=lambda a: e.utility(c["true_risk"], a, wb, wh))
full_changed = stx.mean(1.0 if full_opt(c, 1, 1) != prior_a else 0.0 for c in cs)

# --- (d) goal-sensitivity: optimal flips across goal profiles ---
profiles = {"balanced": (1.0, 1.0), "clot_first": (1.5, 0.7), "bleed_first": (0.7, 1.5)}
opt_by_goal = {g: [full_opt(c, wb, wh) for c in cs] for g, (wb, wh) in profiles.items()}
flip_any = stx.mean(1.0 if len({opt_by_goal[g][i] for g in profiles}) > 1 else 0.0 for i in range(len(cs)))
flip_cb = stx.mean(1.0 if opt_by_goal["clot_first"][i] != opt_by_goal["bleed_first"][i] else 0.0 for i in range(len(cs)))
goal_act = {g: Counter(opt_by_goal[g]) for g in profiles}

# --- safety-critical subset: high clot & high bleed ---
hh = [c for c in cs if c["true_risk"] == "high_high"]

print("==== TITRATION DESIGN GATE (hidden, balanced) ====")
print(f"n={N} cost={COST}")
print(f"state_dist: { {k: round(v/N,3) for k,v in sorted(state_dist.items())} }")
print(f"high_high frac = {state_dist['high_high']/N:.3f}  (safety-critical subset)")
print(f"oracle: mean V={v_oracle:.3f}  mean asks={mean_asks:.2f}  action_dist={ {k: round(v/N,2) for k,v in oracle_act.items()} }")
print(f"\n(a) BC-gap = V_oracle {v_oracle:.3f} - best_fixed[{best_fixed_a}] {best_fixed_v:.3f} = {bc_gap:+.3f}   (>0 ?)")
print(f"    fixed_val per action: { {k: round(v,3) for k,v in fixed_val.items()} }")
print(f"    prior-blind commit = {prior_a} (val {prior_blind_v:.3f})")
print(f"\n(b) oracle action spread: {dict(oracle_act)}  (want >1 action used)")
print(f"\n(c) VOI: gathering changes action vs prior-blind on {voi_changed:.3f} of cases; full-info-opt differs from prior on {full_changed:.3f}")
print(f"\n(d) goal-sensitivity: optimal flips across {{balanced,clot,bleed}} on {flip_any:.3f}; clot_first vs bleed_first differ on {flip_cb:.3f}")
for g in profiles:
    print(f"    {g:11} action_dist={ {k: round(v/N,2) for k,v in goal_act[g].items()} }")
print(f"\nsafety-critical high_high: n={len(hh)} ({len(hh)/N:.3f}); balanced opt={full_opt(hh[0],1,1) if hh else '-'}")

# --- verdict heuristics ---
print("\n---- GATE 0 CHECKS ----")
print(f"(a) BC-gap > 0.02 : {'PASS' if bc_gap > 0.02 else 'FAIL'} ({bc_gap:+.3f})")
print(f"(b) >=3 actions used by oracle : {'PASS' if len(oracle_act) >= 3 else 'WEAK'} ({len(oracle_act)})")
print(f"(c) VOI changes action >=15% : {'PASS' if voi_changed >= 0.15 else 'FAIL'} ({voi_changed:.3f})")
print(f"(d) goal flip (clot vs bleed) >=20% : {'PASS' if flip_cb >= 0.20 else 'FAIL'} ({flip_cb:.3f})")
