"""Case loading and frozen splits.

Split design (V2/sdbench_execution_plan.md):
  train    = official `train` set: the ONLY cases GRPO arms B and C ever see.
  dev      = n_dev cases drawn from official `validation` by a fixed seed: lambda calibration + checkpoint selection.
  eval_val = the rest of `validation`: primary evaluation.
  test     = official `test`: sealed until all arms are frozen (see `assert_not_test`).

The split file stores case ids plus a content hash per case, so a silently changed dataset file is detected.
Do NOT commit split files for credentialed datasets until we've confirmed case ids aren't restricted data.

CLI:
  python -m tasks.v2_sdbench.cases freeze --dataset mimic --n-dev 54 --seed 20261111
  python -m tasks.v2_sdbench.cases show   --dataset synthetic
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from typing import Any

from tasks.v2_sdbench import sdb

SPLIT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "V2", "sdbench_splits")
OFFICIAL = {"train": "train", "validation": "validation", "test": "test"}


def case_hash(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def official_sets(dataset: str) -> dict[str, list[dict]]:
    return {k: list(v) for k, v in sdb.case_data._get_case_sets(dataset).items()}


def get_case(dataset: str, case_id: str) -> dict:
    return sdb.case_data.get_case(dataset, str(case_id))


def freeze_splits(dataset: str, n_dev: int = 54, seed: int = 20261111, out_dir: str = SPLIT_DIR) -> dict:
    sets = official_sets(dataset)
    missing = [s for s in OFFICIAL.values() if s not in sets]
    if missing:
        raise ValueError(f"{dataset}: official sets {missing} not found (have {sorted(sets)}). "
                         "Splits can only be frozen on a dataset with official train/validation/test sets.")
    val = sorted(sets["validation"], key=lambda c: str(c["case_id"]))
    rng = random.Random(seed)
    dev_ids = set(str(c["case_id"]) for c in rng.sample(val, n_dev))
    split: dict[str, Any] = {"dataset": dataset, "seed": seed, "n_dev": n_dev, "splits": {}, "hashes": {}}
    groups = {
        "train": sets["train"],
        "dev": [c for c in val if str(c["case_id"]) in dev_ids],
        "eval_val": [c for c in val if str(c["case_id"]) not in dev_ids],
        "test": sets["test"],
    }
    seen: dict[str, str] = {}
    for name, cs in groups.items():
        ids = sorted(str(c["case_id"]) for c in cs)
        for i in ids:
            if i in seen and seen[i] != name:
                raise ValueError(f"case {i} is in both {seen[i]} and {name}")
            seen[i] = name
        split["splits"][name] = ids
        for c in cs:
            split["hashes"][str(c["case_id"])] = case_hash(c)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists; splits are frozen. Delete it deliberately to re-freeze.")
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    return split


def load_split(dataset: str, name: str, out_dir: str = SPLIT_DIR, verify: bool = True) -> list[dict]:
    """Cases in a frozen split, verified against stored hashes. `synthetic` has no splits: plumbing only."""
    if dataset == "synthetic":
        if name != "train":
            raise ValueError("synthetic has only the 5 smoke-test cases (official set 'train'); plumbing only")
        return list(official_sets("synthetic")["train"])
    with open(os.path.join(out_dir, f"{dataset}.json")) as f:
        split = json.load(f)
    cases = [get_case(dataset, i) for i in split["splits"][name]]
    if verify:
        bad = [str(c["case_id"]) for c in cases if case_hash(c) != split["hashes"][str(c["case_id"])]]
        if bad:
            raise ValueError(f"{len(bad)} case(s) changed since splits were frozen: {bad[:5]}")
    return cases


def assert_not_test(dataset: str, case_ids, out_dir: str = SPLIT_DIR, allow_test: bool = False) -> None:
    """Guard for training and dev code paths: refuse any sealed test case."""
    if dataset == "synthetic" or allow_test:
        return
    with open(os.path.join(out_dir, f"{dataset}.json")) as f:
        test = set(json.load(f)["splits"]["test"])
    leaked = sorted(set(map(str, case_ids)) & test)
    if leaked:
        raise PermissionError(f"refusing to use {len(leaked)} sealed test case(s): {leaked[:5]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "show"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-dev", type=int, default=54)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    if a.cmd == "freeze":
        s = freeze_splits(a.dataset, a.n_dev, a.seed)
        print({k: len(v) for k, v in s["splits"].items()})
    else:
        print({k: len(v) for k, v in official_sets(a.dataset).items()})


if __name__ == "__main__":
    main()
