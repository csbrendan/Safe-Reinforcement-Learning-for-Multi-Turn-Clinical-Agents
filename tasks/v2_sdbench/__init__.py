"""V2: SD-Bench environment for the three-arm GRPO comparison (see V2/sdbench_execution_plan.md).

SD-Bench (microsoft/sdbench) lives in its own clone and venv (~/v2_bench/sdbench). We import it read-only by
putting the clone on sys.path (`sdb.py`), so the shared training venv is never modified.
"""
