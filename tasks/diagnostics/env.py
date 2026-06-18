"""Task 2 — Diagnostics (sequential work-up under uncertainty). SCAFFOLD — not yet implemented.

Target shape (conform to framework/envs/base.py): a visible presentation + many hidden findings/tests; only
a few are decision-relevant per case, and WHICH depends on revealed results (adaptive value-of-information).
The model orders tests (ASK) then commits a diagnosis/management action. Verifiable reward = outcome utility
with a steep penalty for missing a dangerous diagnosis; ask-cost calibrated so adaptive selection beats any
fixed panel. Expected position on the demonstrability axis: LOW (information value is the whole task).

Build order: define ITEMS/P1/FAMILIES(trees)/U + the analytic helpers (mirror tasks/rx_triage/env.py), then
Gate 1 (`framework.gates.bc_gap`) until BC-gap >= ~0.10 with safety priced in, then Gate 2, then GRPO.
"""
from __future__ import annotations

ACTIONS = ()          # TODO: diagnosis/management decisions
ASK_INTENTS = []      # TODO: orderable tests/findings
DEEP_FAMILIES = ()    # TODO
ASK_COST = 0.0
EPS = 0.03
SYS = ""              # TODO: action schema prompt

_NOT_IMPL = "tasks.diagnostics.env is a scaffold; implement per docs/experiment_plan.md before use."

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
