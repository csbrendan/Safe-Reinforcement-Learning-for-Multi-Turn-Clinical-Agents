"""Task 3 — Long-term care (longitudinal management, competing objectives). SCAFFOLD — not yet implemented.

Target shape (conform to framework/envs/base.py): chronic-disease management over multiple visits with
partially-hidden, evolving state. The model gathers status (ASK) then commits a management action; multiple
actions are often defensible. Verifiable reward = multi-objective outcome (efficacy vs side-effects vs
adherence vs monitoring burden) — the hardest reward to design; sequence this task LAST. Expected position on
the demonstrability axis: LOW, with genuine trade-offs (set-valued acceptable actions).

Build order: define state/actions/U + analytic helpers (mirror tasks/rx_triage/env.py), then Gate 1
(`framework.gates.bc_gap`) until BC-gap >= ~0.10 with safety + over-escalation priced in, then Gate 2, then GRPO.
"""
from __future__ import annotations

ACTIONS = ()          # TODO: management decisions
ASK_INTENTS = []      # TODO: status items gatherable per visit
DEEP_FAMILIES = ()    # TODO
ASK_COST = 0.0
EPS = 0.03
SYS = ""              # TODO: action schema prompt

_NOT_IMPL = "tasks.longterm_care.env is a scaffold; implement per docs/experiment_plan.md before use."

def gen_cases(*a, **k): raise NotImplementedError(_NOT_IMPL)
def user_msg(*a, **k): raise NotImplementedError(_NOT_IMPL)
def answer(*a, **k): raise NotImplementedError(_NOT_IMPL)
def resolve_ask(*a, **k): raise NotImplementedError(_NOT_IMPL)
def score(*a, **k): raise NotImplementedError(_NOT_IMPL)
def oracle_value(*a, **k): raise NotImplementedError(_NOT_IMPL)
def oracle_rollout(*a, **k): raise NotImplementedError(_NOT_IMPL)
def best_commit(*a, **k): raise NotImplementedError(_NOT_IMPL)
def true_risk(*a, **k): raise NotImplementedError(_NOT_IMPL)
def rel_items(*a, **k): raise NotImplementedError(_NOT_IMPL)
def acceptable(*a, **k): raise NotImplementedError(_NOT_IMPL)
