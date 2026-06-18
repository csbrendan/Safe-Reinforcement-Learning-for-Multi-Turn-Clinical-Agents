#!/usr/bin/env python
"""Thin CPU runner for Gate 1 on any task env. Wrapper over `python -m framework.gates.bc_gap`.

Usage: python scripts/run_gate1.py tasks.rx_triage.env [cost]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from framework.gates import bc_gap

if __name__ == "__main__":
    argv = sys.argv[1:]
    new = ["bc_gap", "--env", argv[0] if argv else "tasks.rx_triage.env"]
    if len(argv) > 1:
        new += ["--cost", argv[1]]
    sys.argv = new
    bc_gap.main()
