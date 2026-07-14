"""Targeted high-risk rollout collection for the enriched safety validation (VALIDATION seed, GPU, in-process
vLLM). Rolls out base MedGemma / plain-GRPO / info-gain-GRPO on high-risk (emergent) deephedge cases with
stochastic sampling, so the judged set contains real unsafe/premature AND safe/oracle-like high-risk
trajectories. Writes analytic-labeled records (arm tag kept for later per-policy analysis; the JUDGE never sees it).

  DX_VARIANT=deephedge PYTHONPATH=$PWD python -m framework.procreward.collect_highrisk \
      --val-seed 202 --pool 400 --cap-high 40 --k 12
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

os.environ.setdefault("DX_VARIANT", "deephedge")
from tasks.diagnostics import env_rv as E
import framework.gates.support_probe as SP

BASE = "checkpoints/medgemma4b_text"
ARMS = {"base": None,
        "plain_grpo": "checkpoints/dh_grpo_base/final",
        "infogain_grpo": "checkpoints/dh_grpo_infogain/final"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-seed", type=int, default=202)   # separate from TEST seed 100
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--cap-high", type=int, default=40)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--out", default="results/procreward_pilot/highrisk_rollouts.jsonl")
    args = ap.parse_args()
    assert args.val_seed != 100, "must not use the final TEST seed 100 for validation"
    SP.MAX_TURNS = args.max_turns

    pool = E.gen_cases(args.pool, rho=0.0, seed=args.val_seed, families=E.DEEP_FAMILIES)
    high = [c for c in pool if c["true_risk"] == "high"][:args.cap_high]
    print(f"[collect] val_seed={args.val_seed}: {len(high)}/{len(pool)} high-risk cases "
          f"(families: {', '.join(sorted({c['family'] for c in high}))})", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(BASE)
    llm = LLM(model=BASE, enable_lora=True, max_lora_rank=32, dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=4096)

    records = []
    for ai, (arm, adapter) in enumerate(ARMS.items(), start=1):
        lreq = LoRARequest(arm, ai, str(Path(adapter).resolve())) if adapter else None
        st = SP.run_condition(llm, tok, lreq, E, high, args.k, SamplingParams, temp=args.temp)
        for s in st:
            c = s["case"]
            r, ok, _, acc = E.score(c, s["asks"], s["decision"])
            records.append({"arm": arm, "case_id": c["case_id"], "family": c["family"],
                            "true_risk": c["true_risk"], "asks": s["asks"], "decision": s["decision"],
                            "n_asks": sum(1 for a in s["asks"] if E.resolve_ask(a)),
                            "oracle_asks": c["oracle_asks"], "v_oracle": c["v_oracle"],
                            "acceptable": c["acceptable"], "reward": r, "oracle_grade": ok, "acceptable_hit": acc})
        nun = sum(1 for s in st if s["decision"] == "admit" or (s["decision"] not in ("emergent_tx", None)))
        print(f"[collect] {arm}: {len(st)} rollouts, ~unsafe(admit/other)≈{nun}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[collect] wrote {len(records)} high-risk rollouts -> {args.out}")


if __name__ == "__main__":
    main()
