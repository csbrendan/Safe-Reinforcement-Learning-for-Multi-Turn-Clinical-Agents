"""Append-only JSONL key-value cache, safe across threads in one process.

Used for gatekeeper results, judge scores, test costs and reader scores, so that the same request always gets
the same answer within and across training and evaluation runs (determinism), and nothing is paid for twice.
"""
from __future__ import annotations
import hashlib
import json
import os
import threading
from typing import Any, Callable


def make_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class JsonlCache:
    def __init__(self, path: str | None):
        self.path = path
        self._d: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # tolerate a torn final line after a crash
                    self._d[row["k"]] = row["v"]

    def __len__(self) -> int:
        return len(self._d)

    def get(self, key: str, default: Any = None) -> Any:
        return self._d.get(key, default)

    def put(self, key: str, value: Any, meta: dict | None = None) -> None:
        with self._lock:
            if key in self._d:
                return  # first write wins, keeps answers stable
            self._d[key] = value
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps({"k": key, "v": value, **({"meta": meta} if meta else {})},
                                       ensure_ascii=False) + "\n")

    def get_or_compute(self, key: str, fn: Callable[[], Any], meta: dict | None = None) -> Any:
        if key in self._d:
            self.hits += 1
            return self._d[key]
        with self._lock:
            kl = self._key_locks.setdefault(key, threading.Lock())
        with kl:  # concurrent requests for the same key compute once
            if key in self._d:
                self.hits += 1
                return self._d[key]
            self.misses += 1
            v = fn()
            self.put(key, v, meta)
            return self._d[key]
