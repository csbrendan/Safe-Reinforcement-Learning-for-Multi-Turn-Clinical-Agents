"""First-turn format probe (no gatekeeper needed): sample N opening replies per synthetic case for each policy_format
variant and temperature, and measure SD-Bench turn-format adherence and diversity.

    PYTHONPATH=$PWD python V2/format_probe/first_turn.py --n 32 --temps 0.7,1.0 \
        --out results/v2_format/first_turn.json

Per (variant, temperature): mixed (SD-Bench would reject the whole turn), no-tag, truncated (token cap),
question-only / test-only / diagnosis share, mean requests per turn, and diversity = mean pairwise Jaccard distance
between the request sets of two samples of the same case (1 = no overlap) plus distinct-request count.
Synthetic cases only.
"""
from __future__ import annotations
import argparse
import itertools
import json
import os
import re
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from tasks.v2_sdbench import turn_format
from tasks.v2_sdbench.cases import load_split
from tasks.v2_sdbench.episode import SYSTEM_PROMPT, END_OF_TURN_ID, clean_reply, first_user_message

QT = re.compile(r"<question>(.*?)</question>", re.S | re.I)
TT = re.compile(r"<test>(.*?)</test>", re.S | re.I)
DT = re.compile(r"<diagnosis>(.*?)</diagnosis>", re.S | re.I)


def classify(text: str) -> dict:
    q = [x.strip().lower() for x in QT.findall(text)]
    t = [x.strip().lower() for x in TT.findall(text)]
    d = DT.findall(text)
    kind = "mixed" if q and t else "question" if q else "test" if t else "diagnosis" if d else "none"
    return {"kind": kind, "reqs": sorted(set(q + t)), "n_req": len(q) + len(t)}


def jaccard_dist(a: list, b: list) -> float:
    A, B = set(a), set(b)
    return 1 - len(A & B) / len(A | B) if A | B else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.100.2:8100/v1")
    ap.add_argument("--model", default="base")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temps", default="0.7,1.0")
    ap.add_argument("--variants", default=",".join(turn_format.VARIANTS))
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2, help="concurrent requests (each has n sequences); keep low on a Spark")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    client = OpenAI(base_url=a.url, api_key="x", timeout=600)
    cases = load_split("synthetic", "train")
    jobs = [(v, float(t), c) for v in a.variants.split(",") for t in a.temps.split(",") for c in cases]

    def run(job):
        v, t, c = job
        msgs = [{"role": "system", "content": turn_format.system_prompt(SYSTEM_PROMPT, v)},
                {"role": "user", "content": first_user_message(c)}]
        r = client.chat.completions.create(
            model=a.model, messages=msgs, n=a.n, temperature=t, max_tokens=a.max_tokens,
            extra_body={"stop_token_ids": [END_OF_TURN_ID], "skip_special_tokens": False,
                        **turn_format.openai_extra(v)})
        outs = []
        for ch in r.choices:
            text, _ = clean_reply(ch.message.content or "")
            outs.append({**classify(text), "truncated": ch.finish_reason == "length", "text": text})
        return v, t, str(c["case_id"]), outs

    # incremental + resumable: one JSON line per (variant, temp, case) job, so a watchdog trip loses nothing
    part = a.out + ".parts.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            done[(r["v"], r["t"], r["cid"])] = r["outs"]
    todo = [j for j in jobs if (j[0], j[1], str(j[2]["case_id"])) not in done]
    print(f"[probe] {len(jobs) - len(todo)} jobs done, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(a.workers) as pool, open(part, "a") as f:
        for v, t, cid, outs in pool.map(run, todo):
            f.write(json.dumps({"v": v, "t": t, "cid": cid, "outs": outs}) + "\n")
            f.flush()
            done[(v, t, cid)] = outs
    res = [(v, t, cid, outs) for (v, t, cid), outs in done.items()]

    summary, samples = {}, []
    for v, t in itertools.product(a.variants.split(","), [float(x) for x in a.temps.split(",")]):
        rows = [(cid, o) for vv, tt, cid, outs in res if vv == v and tt == t for o in outs]
        n = len(rows)
        kinds = [o["kind"] for _, o in rows]
        div, distinct = [], []
        for cid in {cid for cid, _ in rows}:
            rq = [o["reqs"] for c2, o in rows if c2 == cid]
            div += [jaccard_dist(x, y) for x, y in itertools.combinations(rq, 2)]
            distinct.append(len(set().union(*map(set, rq))))
        summary[f"{v}@T{t}"] = {
            "n": n, **{f"frac_{k}": kinds.count(k) / n for k in ("mixed", "question", "test", "diagnosis", "none")},
            "frac_truncated": sum(o["truncated"] for _, o in rows) / n,
            "mean_requests": st.mean(o["n_req"] for _, o in rows),
            "diversity_jaccard": st.mean(div) if div else None,
            "distinct_requests_per_case": st.mean(distinct)}
        samples += [{"variant": v, "temp": t, "case_id": cid, **o} for cid, o in rows]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "samples": samples}, open(a.out, "w"), indent=1)
    for k, s in summary.items():
        print(f"{k:28s} mixed={s['frac_mixed']:.2f} q={s['frac_question']:.2f} t={s['frac_test']:.2f} "
              f"dx={s['frac_diagnosis']:.2f} none={s['frac_none']:.2f} trunc={s['frac_truncated']:.2f} "
              f"reqs={s['mean_requests']:.1f} div={s['diversity_jaccard']:.2f} "
              f"distinct/case={s['distinct_requests_per_case']:.0f}")


if __name__ == "__main__":
    main()
