"""Decoding-temperature diagnostic: is an SFT checkpoint's 'under-gathering' a learned policy or a temp-1.0
sampling artifact? Loads a model(+adapter) once and runs the HIDDEN probe at several temperatures, reporting
mean asks / 1-ask rate / no-commit / reward / support and the high-risk subset per temperature. Greedy (temp 0)
shows what the policy 'wants' to do; comparing to temp 1.0 (the Gate-2 probe setting) isolates sampling effects.

No env/reward/Gate changes. Usage:
  VLLM_ATTENTION_BACKEND=TORCH_SDPA PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m framework.gates.temp_sweep --env tasks.rx_triage.env --model checkpoints/medgemma4b_text \
    --adapter checkpoints/sft_rx_v3/final --tag v3 --temps 0.0,0.5,1.0 --n-cases 40 --k 8 --out results/temp_sweep_v3.json
"""
from __future__ import annotations
import argparse, importlib, json, statistics as stx
from pathlib import Path
import framework.gates.support_probe as SP
from framework.gates.support_probe import run_condition


def metrics(env, st):
    by_case, rew, n_asks = {}, [], []
    hi_rew, hi_by_case = [], {}
    vt = tt = 0
    for s in st:
        c = s["case"]; r, ok, _, _ = env.score(c, s["asks"], s["decision"])
        na = sum(1 for a in s["asks"] if env.resolve_ask(a))
        rew.append(r); n_asks.append(na); vt += s["valid"]; tt += s["turns"]
        by_case.setdefault(c["case_id"], []).append(ok)
        if c.get("true_risk") == "high":
            hi_rew.append(r); hi_by_case.setdefault(c["case_id"], []).append(ok)
    supp = sum(1 for v in by_case.values() if any(v)) / max(len(by_case), 1)
    hi_supp = sum(1 for v in hi_by_case.values() if any(v)) / max(len(hi_by_case), 1) if hi_by_case else float("nan")
    return dict(
        mean_asks=round(stx.mean(n_asks), 2), one_ask=round(sum(1 for x in n_asks if x == 1) / len(n_asks), 3),
        no_commit=round(sum(1 for s in st if s["decision"] is None) / len(st), 3),
        reward=round(stx.mean(rew), 3), support=round(supp, 3),
        high_reward=round(stx.mean(hi_rew), 3) if hi_rew else float("nan"), high_support=round(hi_supp, 3),
        parse=round(vt / max(tt, 1), 3), n_roll=len(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tasks.rx_triage.env")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--temps", default="0.0,0.5,1.0")
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--k", type=int, default=8, help="rollouts/case for temp>0 (temp=0 forced to k=1, greedy)")
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    SP.MAX_TURNS = args.max_turns
    env = importlib.import_module(args.env)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, enable_lora=bool(args.adapter), max_lora_rank=32, dtype="bfloat16",
              gpu_memory_utilization=0.75, max_model_len=4096)
    lreq = LoRARequest("ad", 1, str(Path(args.adapter).resolve())) if args.adapter else None
    cases = env.gen_cases(args.n_cases, 0.0, seed=100)   # hidden, same seed as the Gate-2 probe
    temps = [float(t) for t in args.temps.split(",")]
    rows = {}
    print(f"=== temp sweep [{args.tag}] (hidden, n={args.n_cases}, max_turns={args.max_turns}) ===")
    print(f"  {'temp':>5} {'k':>2} {'mAsks':>6} {'1ask':>6} {'noCmt':>6} {'reward':>7} {'supp':>5} {'hiRew':>7} {'hiSup':>6} {'parse':>6}")
    for t in temps:
        k = 1 if t == 0.0 else args.k
        st = run_condition(llm, tok, lreq, env, cases, k, SamplingParams, temp=t)
        m = metrics(env, st); rows[str(t)] = m
        print(f"  {t:>5.2f} {k:>2} {m['mean_asks']:>6.2f} {m['one_ask']:>6.3f} {m['no_commit']:>6.3f} "
              f"{m['reward']:>+7.3f} {m['support']:>5.2f} {m['high_reward']:>+7.3f} {m['high_support']:>6.2f} {m['parse']:>6.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": args.tag, "model": args.model, "adapter": args.adapter or None,
               "max_turns": args.max_turns, "by_temp": rows}, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
