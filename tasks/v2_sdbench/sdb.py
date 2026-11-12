"""Read-only import shim for the SD-Bench clone. Nothing is installed into the training venv.

Set V2_SDBENCH_DIR to override the clone location (default ~/v2_bench/sdbench, commit 304ce96).
"""
from __future__ import annotations
import os
import sys

SDBENCH_DIR = os.path.expanduser(os.environ.get("V2_SDBENCH_DIR", "~/v2_bench/sdbench"))
if not os.path.isdir(os.path.join(SDBENCH_DIR, "sdbench")):
    raise ImportError(f"SD-Bench clone not found at {SDBENCH_DIR} (set V2_SDBENCH_DIR)")
# Data root defaults to the clone's data/ (SD-Bench otherwise resolves it against the CWD). Local overrides are
# disabled so nothing in our working directory can replace SD-Bench's prompts or config.
os.environ.setdefault("SDBENCH_DATA_PATH", os.path.join(SDBENCH_DIR, "data"))
os.environ.setdefault("SDBENCH_LOCAL_DIR", "disable")
if SDBENCH_DIR not in sys.path:
    sys.path.insert(0, SDBENCH_DIR)

from sdbench import case_data, gatekeeper, judge, config as sdb_config  # noqa: E402
from sdbench.agents import baseline  # noqa: E402
from sdbench.harness import harness  # noqa: E402

__all__ = ["SDBENCH_DIR", "case_data", "gatekeeper", "judge", "sdb_config", "baseline", "harness"]
