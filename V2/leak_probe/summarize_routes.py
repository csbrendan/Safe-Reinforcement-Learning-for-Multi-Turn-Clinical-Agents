"""Summarise v2_leak_routes.py output: per route, what the gatekeeper revealed and what the guard did."""
import json, sys
from collections import Counter, defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
by = defaultdict(list)
for r in rows:
    by[r["route"]].append(r)
print(f"{len(rows)} requests, cases {sorted({r['case_id'] for r in rows})}")
for route, rs in by.items():
    leak = [r for r in rs if r["dx_terms"] or r["course"]]
    shown = [r for r in rs if r["dx_terms_shown"] or r["course_shown"]]
    print(f"{route:12s} n={len(rs):2d} gatekeeper revealed dx/course={len(leak):2d}  "
          f"still shown after guard={len(shown):2d}  kinds={dict(Counter(r['kind'] for r in rs))}")
print("\nShown to the agent despite a dx/course signal:")
for r in rows:
    if r["dx_terms_shown"] or r["course_shown"]:
        print(f"  [{r['case_id']}] {r['request'][:70]} kind={r['kind']} -> {r['shown'][:200]!r}")
if "--info" in sys.argv:
    print("\nAll answers passed as info (manual read for misses):")
    for r in rows:
        if r["kind"] == "info":
            print(f"  [{r['case_id']}] {r['request'][26:90]}\n      {r['shown'][:300]!r}")
