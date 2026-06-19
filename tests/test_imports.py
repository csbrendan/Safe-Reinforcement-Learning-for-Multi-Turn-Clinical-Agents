"""Smoke tests: framework imports, env-interface conformance, and a fast Gate-1 sanity on rx_triage.

CPU-only, no heavy deps (no transformers/vllm) — run with: python -m pytest -q  (or `python tests/test_imports.py`).
"""
from __future__ import annotations
import importlib


def test_framework_imports():
    for m in ("framework.parsing", "framework.envs.base", "framework.gates.bc_gap",
              "framework.gates.reward_calibration", "framework.report", "framework.plotting"):
        importlib.import_module(m)


def test_parse_action_basic():
    from framework.parsing import parse_action
    assert parse_action('{"action_type":"ask","intent":"potassium_high"}') == ("ask", "potassium_high")
    kind, payload = parse_action('{"action_type":"commit","decision":"escalate"}')
    assert kind == "commit"
    assert parse_action("not json at all")[0] == "invalid"
    # list-valued intent must be rejected, not crash
    assert parse_action('{"action_type":"ask","intent":["a","b"]}')[0] == "invalid"


def test_rx_triage_env_conformance():
    from framework.envs import base
    env = importlib.import_module("tasks.rx_triage.env")
    missing = base.check_env(env)
    assert not missing, f"rx_triage env missing interface members: {missing}"


def test_rx_triage_gate1_sanity():
    """Oracle must dominate and the deep BC-gap must clear the ~0.10 bar (the whole premise of Task 1)."""
    from framework.gates import bc_gap
    env = importlib.import_module("tasks.rx_triage.env")
    R, gap, per_fam, BEST, checks = bc_gap.run(env, env.ASK_COST)
    assert R["oracle"]["all"] >= max(R[p]["all"] for p in R if p != "oracle") - 1e-9, "oracle not highest"
    assert gap > 0.0, f"BC-gap not positive: {gap}"
    assert checks["best_fixed below oracle by >= ~0.10"], f"BC-gap below bar: {gap:+.3f}"
    assert all(checks.values()), f"Gate-1 checks failing: {[k for k, v in checks.items() if not v]}"


def test_scaffold_envs_conform_names():
    """longterm_care is still a stub but must import; diagnostics is now a full env (checked separately)."""
    for name in ("tasks.diagnostics.env", "tasks.longterm_care.env"):
        importlib.import_module(name)  # import must not raise


def test_diagnostics_env_conformance():
    from framework.envs import base
    env = importlib.import_module("tasks.diagnostics.env")
    missing = base.check_env(env)
    assert not missing, f"diagnostics env missing interface members: {missing}"


def test_diagnostics_gate1_sanity():
    """Task 2 (diagnostics) must clear Gate 1 with rx_triage-like geometry: oracle highest, BC-gap >= ~0.10."""
    from framework.gates import bc_gap
    env = importlib.import_module("tasks.diagnostics.env")
    R, gap, per_fam, BEST, checks = bc_gap.run(env, env.ASK_COST)
    assert R["oracle"]["all"] >= max(R[p]["all"] for p in R if p != "oracle") - 1e-9, "oracle not highest"
    assert checks["best_fixed below oracle by >= ~0.10"], f"BC-gap below bar: {gap:+.3f}"
    assert all(checks.values()), f"Gate-1 checks failing: {[k for k, v in checks.items() if not v]}"


if __name__ == "__main__":
    test_framework_imports()
    test_parse_action_basic()
    test_rx_triage_env_conformance()
    test_rx_triage_gate1_sanity()
    test_scaffold_envs_conform_names()
    test_diagnostics_env_conformance()
    test_diagnostics_gate1_sanity()
    print("all smoke tests passed")
