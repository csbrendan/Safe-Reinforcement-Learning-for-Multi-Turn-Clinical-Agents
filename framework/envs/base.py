"""The Env interface every task module must satisfy for the shared gates / SFT / probe / GRPO to run.

This is a structural (duck-typed) contract — a task module (e.g. tasks/rx_triage/env.py) is passed wherever
an `Env` is expected. It does NOT need to subclass anything; it just needs these names. The `EnvProtocol`
below documents and (optionally) type-checks the contract.

A task is a verifiable POMDP-style decision problem:
  * a hidden state (some items observed up front w.p. rho, the rest must be gathered by ASKing);
  * a verifiable reward computable from the TRUE state (so RL's contribution is cleanly attributable);
  * an analytic oracle (so the BC-gap / demonstrability can be measured a priori, before any training).

Required module-level attributes
--------------------------------
ACTIONS        : tuple[str]            commit decisions
ASK_INTENTS    : list[str]             askable item names
DEEP_FAMILIES  : tuple[str]            case families that are the RL focus (largest BC-gap)
ASK_COST, EPS  : float                 per-ask cost; oracle-grade tolerance
SYS            : str                   system prompt describing the action schema

Required functions
------------------
gen_cases(n, rho, seed, families=..., prefix="") -> list[case dict]
    each case dict has: case_id, family, assign(list[int]), observed(dict), true_risk, acceptable(list),
    v_oracle(float), oracle_asks(list), oracle_action(str).
user_msg(case) -> str                          the chart shown to the model (hidden state excluded)
answer(case, intent) -> str                    reveal the true value of a queried item
resolve_ask(intent) -> str | None              canonicalize a model intent to an item (or None)
score(case, asks, decision) -> (reward, oracle_grade, valid, acceptable_hit)
oracle_value(fam, cost, revealed=None) -> float       analytic adaptive value (for BC-gap)
oracle_rollout(fam, assign, cost, revealed0=None) -> (asks, action, reward)
best_commit(fam, revealed) -> (value, action)         argmax commit under the posterior
true_risk(fam, assign) -> str ; rel_items(fam) -> set[str] ; acceptable(risk) -> list[str]
clear_caches() -> None                         reset memoization when the reward table changes
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class EnvProtocol(Protocol):
    ACTIONS: tuple
    ASK_INTENTS: list
    DEEP_FAMILIES: tuple
    ASK_COST: float
    EPS: float
    SYS: str

    def gen_cases(self, n, rho, seed, families=..., prefix=""): ...
    def user_msg(self, case): ...
    def answer(self, case, intent): ...
    def resolve_ask(self, intent): ...
    def score(self, case, asks, decision): ...
    def oracle_value(self, fam, cost, revealed=None): ...
    def oracle_rollout(self, fam, assign, cost, revealed0=None): ...
    def best_commit(self, fam, revealed): ...
    def true_risk(self, fam, assign): ...
    def rel_items(self, fam): ...
    def acceptable(self, risk): ...


REQUIRED = ["ACTIONS", "ASK_INTENTS", "DEEP_FAMILIES", "ASK_COST", "EPS", "SYS",
            "gen_cases", "user_msg", "answer", "resolve_ask", "score", "oracle_value",
            "oracle_rollout", "best_commit", "true_risk", "rel_items", "acceptable"]


def check_env(mod):
    """Return the list of missing required names on a task env module (empty == conforms)."""
    return [name for name in REQUIRED if not hasattr(mod, name)]
