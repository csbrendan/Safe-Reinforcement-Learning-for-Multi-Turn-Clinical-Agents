"""Summarise a first_turn.py parts file (works on partial runs)."""
import itertools, json, statistics as st, sys
from collections import defaultdict
rows = defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line)
    rows[(r["v"], r["t"])].append((r["cid"], r["outs"]))
for (v, t), jobs in sorted(rows.items()):
    outs = [o for _, os_ in jobs for o in os_]
    n = len(outs)
    k = lambda x: sum(o["kind"] == x for o in outs) / n  # noqa: E731
    div = [1 - len(set(a["reqs"]) & set(b["reqs"])) / len(set(a["reqs"]) | set(b["reqs"]))
           for _, os_ in jobs for a, b in itertools.combinations(os_, 2) if set(a["reqs"]) | set(b["reqs"])]
    print(f"{v:20s} T={t} cases={len(jobs)} n={n:3d} mixed={k('mixed'):.2f} question={k('question'):.2f} "
          f"test={k('test'):.2f} dx={k('diagnosis'):.2f} none={k('none'):.2f} "
          f"trunc={sum(o['truncated'] for o in outs)/n:.2f} reqs={st.mean(o['n_req'] for o in outs):.1f} "
          f"div={st.mean(div):.2f}")
