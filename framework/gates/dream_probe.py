"""Composed-policy probe for the DREAM-inspired decoupled explore/exploit policy (diagnostics deephedge).

Evaluates ONLY the deployed composed policy: explore-mode ask-only dialogue -> gather findings -> exploit-mode
one-turn commit over ONLY the gathered findings. The commit phase NEVER sees the hidden severity (mu) or any
unrevealed item (env.assert_no_leak enforces this before every commit). Emits per-rollout records in the EXACT
schema of framework.gates.support_probe (reusing its summarize()), so framework.report runs unchanged.

support_probe.py is imported, NOT modified. env.score / U / the analytic gates are untouched.

Usage:
  VLLM_ATTENTION_BACKEND=TORCH_SDPA PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DX_VARIANT=deephedge \
  python -m framework.gates.dream_probe --env tasks.diagnostics.env_rv --model <base> --adapter <lora> \
    --tag dream_final --k 12 --n-cases 60 --conditions hidden,anchor --max-turns 12 --out results/probe_dream.json
"""
from __future__ import annotations
import argparse, importlib, json
from pathlib import Path
from framework.parsing import parse_action
from framework.gates.support_probe import summarize  # reuse identical metrics/record schema

MAX_TURNS = 12


def run_condition_composed(llm, tok, lreq, env, cases, k, SamplingParams, temp=1.0):
    """Two-phase composed deployment. Returns support_probe-shaped state dicts (case/asks/decision/valid/turns)."""
    st = []
    for c in cases:
        for _ in range(k):
            st.append({"case": c, "asks": [], "decision": None, "valid": 0, "turns": 0,
                       "msgs": [{"role": "system", "content": env.SYS},
                                {"role": "user", "content": env.explore_user_msg(c)}], "done": False})
    sp = (SamplingParams(temperature=0.0, max_tokens=200, n=1) if temp == 0.0
          else SamplingParams(temperature=temp, top_p=0.95, max_tokens=200, n=1))

    # ---- EXPLORE phase: ask-only (commit/invalid ends exploration; no commit here) ----
    for _t in range(MAX_TURNS):
        active = [i for i, s in enumerate(st) if not s["done"]]
        if not active:
            break
        prompts = [tok.apply_chat_template(st[i]["msgs"], tokenize=False, add_generation_prompt=True) for i in active]
        outs = llm.generate(prompts, sp, lora_request=lreq, use_tqdm=False)
        for i, o in zip(active, outs):
            txt = o.outputs[0].text; s = st[i]; s["turns"] += 1
            s["msgs"].append({"role": "assistant", "content": txt})
            kind, val = parse_action(txt)
            canon = env.resolve_ask(val) if kind == "ask" else None
            if kind == "ask" and canon:
                s["valid"] += 1; s["asks"].append(canon)
                s["msgs"].append({"role": "user", "content": env.answer(s["case"], canon)})
            else:
                s["done"] = True

    # ---- EXPLOIT phase: one commit turn over ONLY gathered findings (leak-guarded) ----
    commit_prompts, idxs = [], []
    for i, s in enumerate(st):
        cu = env.commit_user_msg(s["case"], s["asks"])
        env.assert_no_leak(s["case"], s["asks"], cu)   # hard guard: no mu / unrevealed-finding leak
        msgs = [{"role": "system", "content": env.SYS}, {"role": "user", "content": cu}]
        commit_prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)); idxs.append(i)
    outs = llm.generate(commit_prompts, sp, lora_request=lreq, use_tqdm=False)
    for i, o in zip(idxs, outs):
        s = st[i]; s["turns"] += 1
        kind, val = parse_action(o.outputs[0].text)
        if kind == "commit":
            dec = val[0] if isinstance(val, tuple) else val
            s["valid"] += 1 if dec in env.ACTIONS else 0
            s["decision"] = dec
    return st


def main():
    global MAX_TURNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tasks.diagnostics.env_rv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--conditions", default="hidden,anchor")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    MAX_TURNS = args.max_turns
    env = importlib.import_module(args.env)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    RHO = {"hidden": 0.0, "anchor": 1.0}
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, enable_lora=bool(args.adapter), max_lora_rank=32, dtype="bfloat16",
              gpu_memory_utilization=0.75, max_model_len=4096)
    lreq = LoRARequest("sft", 1, str(Path(args.adapter).resolve())) if args.adapter else None
    results = {"tag": args.tag, "env": args.env, "model": args.model, "adapter": args.adapter or None,
               "max_turns": MAX_TURNS, "composed_policy": True}
    records = []
    for cond in args.conditions.split(","):
        cases = env.gen_cases(args.n_cases, RHO[cond], seed=100)
        st = run_condition_composed(llm, tok, lreq, env, cases, args.k, SamplingParams)
        summ, _ = summarize(env, st, args.k); results[cond] = summ
        for s in st:
            c = s["case"]; r, ok, _, acc = env.score(c, s["asks"], s["decision"])
            records.append({"cond": cond, "case_id": c["case_id"], "family": c.get("family"),
                            "true_risk": c.get("true_risk"), "oracle_asks": c.get("oracle_asks"),
                            "acceptable": c.get("acceptable"), "v_oracle": c.get("v_oracle"),
                            "n_asks": sum(1 for a in s["asks"] if env.resolve_ask(a)), "asks": s["asks"],
                            "decision": s["decision"], "reward": r, "oracle_grade": ok, "acceptable_hit": acc})
        print(f"\n=== [{args.tag}] {cond} composed (rho={RHO[cond]}, n={args.n_cases}, k={args.k}) ===")
        for key in ("support_at_k", "mean_reward", "parse_validity", "mean_asks", "oracle_grade_blind_rate",
                    "zero_ask_commit_rate", "ask_hist", "commit_dist"):
            print(f"  {key:24s} {summ[key]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    rp = str(args.out).replace(".json", ".records.jsonl")
    with open(rp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[saved] {args.out}  +  {rp} ({len(records)} records)")


if __name__ == "__main__":
    main()
