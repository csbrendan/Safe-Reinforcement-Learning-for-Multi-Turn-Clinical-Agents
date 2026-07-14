"""Gate 2 — the LATENT-SUPPORT axis (pass@k), env-agnostic. GPU inference (not training).

Does the model EVER sample oracle-grade gather->commit behavior? RL can only elicit capability that already
exists. Runs k rollouts/prompt against a task env; reports support@k, mean reward, parse validity, ask
distribution, and the adaptive-vs-blind discriminators (oracle_grade_blind_rate, oracle_grade_ask_hist) so a
"support" that is really blind-commit coincidence is caught. Dumps per-rollout records for offline analysis.

Usage:
  VLLM_ATTENTION_BACKEND=TORCH_SDPA PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m framework.gates.support_probe --env tasks.rx_triage.env --model <base> [--adapter <lora>] \
    --tag base --k 12 --n-cases 60 --conditions hidden,anchor --out results/probe_base.json
"""
from __future__ import annotations
import argparse, importlib, json, statistics as stx
from collections import Counter
from pathlib import Path
from framework.parsing import parse_action

MAX_TURNS = 10  # dialogue-horizon cap; override via --max-turns. Modest headroom so a full best-fixed
                # checklist (<=6 asks) can still commit, without making ask-all (priced out by ask-cost) free.


def run_condition(llm, tok, lreq, env, cases, k, SamplingParams, temp=1.0):
    st = []
    for c in cases:
        for _ in range(k):
            st.append({"case": c, "msgs": [{"role": "system", "content": env.SYS},
                                            {"role": "user", "content": env.user_msg(c)}],
                       "asks": [], "decision": None, "done": False, "turns": 0, "valid": 0})
    sp = (SamplingParams(temperature=0.0, max_tokens=200, n=1) if temp == 0.0
          else SamplingParams(temperature=temp, top_p=0.95, max_tokens=200, n=1))
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
            if kind == "commit":
                dec = val[0] if isinstance(val, tuple) else val
                s["valid"] += 1 if dec in env.ACTIONS else 0
                s["decision"] = dec; s["done"] = True
            elif kind == "ask" and canon:
                s["valid"] += 1; s["asks"].append(canon)
                s["msgs"].append({"role": "user", "content": env.answer(s["case"], canon)})
            else:
                s["done"] = True
    return st


def summarize(env, st, k):
    by_case, rewards, n_asks, vt, tt = {}, [], [], 0, 0
    ask_c, dec_c = Counter(), Counter()
    for s in st:
        r, ok, _, _ = env.score(s["case"], s["asks"], s["decision"])
        rewards.append(r); by_case.setdefault(s["case"]["case_id"], []).append((r, ok, s))
        vt += s["valid"]; tt += s["turns"]
        for a in s["asks"]: ask_c[a] += 1
        if s["decision"] is not None: dec_c[s["decision"]] += 1
        n_asks.append(sum(1 for a in s["asks"] if env.resolve_ask(a)))
    og = [n for n, s in [(sum(1 for a in s["asks"] if env.resolve_ask(a)), s) for s in st]
          if env.score(s["case"], s["asks"], s["decision"])[1]]
    def hist(xs):
        h = {"0": 0, "1": 0, "2": 0, "3+": 0}
        for x in xs: h["0" if x == 0 else "1" if x == 1 else "2" if x == 2 else "3+"] += 1
        return {kk: round(v / max(len(xs), 1), 3) for kk, v in h.items()}
    supp = sum(1 for rs in by_case.values() if any(o for _, o, _ in rs)) / max(len(by_case), 1)
    return {
        "support_at_k": round(supp, 4), "mean_reward": round(stx.mean(rewards), 4) if rewards else 0.0,
        "reward_std_overall": round(stx.pstdev(rewards), 4) if len(rewards) > 1 else 0.0,
        "parse_validity": round(vt / max(tt, 1), 4), "mean_asks": round(sum(n_asks) / max(len(n_asks), 1), 3),
        "ask_hist": hist(n_asks), "oracle_grade_ask_hist": hist(og),
        "oracle_grade_blind_rate": round(sum(1 for x in og if x == 0) / len(og), 3) if og else 0.0,
        "zero_ask_commit_rate": round(sum(1 for s in st if sum(1 for a in s["asks"] if env.resolve_ask(a)) == 0
                                           and s["decision"] is not None) / max(len(st), 1), 4),
        "ask_dist": dict(ask_c), "commit_dist": dict(dec_c), "n_cases": len(by_case), "k": k,
    }, by_case


def main():
    global MAX_TURNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tasks.rx_triage.env")
    ap.add_argument("--model", required=True, help="base model path/id served by vLLM")
    ap.add_argument("--adapter", default="", help="optional LoRA adapter dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--conditions", default="hidden,anchor")
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS, help="dialogue-horizon cap")
    ap.add_argument("--seed", type=int, default=100,
                    help="eval case-generation seed (100 = the held-out TEST set; use a different seed for a "
                         "VALIDATION set when selecting alpha/checkpoints)")
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
               "max_turns": MAX_TURNS}
    records = []
    for cond in args.conditions.split(","):
        cases = env.gen_cases(args.n_cases, RHO[cond], seed=args.seed)
        st = run_condition(llm, tok, lreq, env, cases, args.k, SamplingParams)
        summ, _ = summarize(env, st, args.k); results[cond] = summ
        for s in st:
            c = s["case"]; r, ok, _, acc = env.score(c, s["asks"], s["decision"])
            records.append({"cond": cond, "case_id": c["case_id"], "family": c.get("family"),
                            "true_risk": c.get("true_risk"), "oracle_asks": c.get("oracle_asks"),
                            "acceptable": c.get("acceptable"), "v_oracle": c.get("v_oracle"),
                            "n_asks": sum(1 for a in s["asks"] if env.resolve_ask(a)), "asks": s["asks"],
                            "decision": s["decision"], "reward": r, "oracle_grade": ok, "acceptable_hit": acc})
        print(f"\n=== [{args.tag}] {cond} (rho={RHO[cond]}, n={args.n_cases}, k={args.k}) ===")
        for key in ("support_at_k", "mean_reward", "parse_validity", "mean_asks", "oracle_grade_blind_rate",
                    "ask_hist", "oracle_grade_ask_hist", "commit_dist"):
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
