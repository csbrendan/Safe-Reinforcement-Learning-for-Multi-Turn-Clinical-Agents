"""GRPO policy-update microbenchmark for MedGemma-4B text + LoRA (the V2 trainer's settings), one sequence at a time.

    python V2/hw_stability/trainer_mem_bench.py --model ~/v2_bench/models/medgemma4b_text \
        --lengths 2048,4096,8192,12288 --mem-fraction 0.28 --out bench.json

Per sequence length L: one reference forward (adapter disabled, no grad; beta > 0 needs it) and one policy
forward + backward with gradient checkpointing, the per-token log-probs computed with TRL's own
selective_log_softmax, then an AdamW step on the LoRA parameters. Reports wall time and peak allocated memory.
--mem-fraction caps this process's share of the unified memory so an over-size case raises a CUDA OOM inside the
process instead of pushing the whole machine into swap (the inference servers are co-resident).
"""
from __future__ import annotations
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="2048,4096,8192,12288")
    ap.add_argument("--mem-fraction", type=float, default=0.28)
    ap.add_argument("--completion-frac", type=float, default=0.7, help="share of tokens that are policy tokens")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from trl.trainer.utils import selective_log_softmax

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    lora = LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    load_s = time.time() - t0
    base_mem = torch.cuda.memory_allocated() / 2**30
    vocab = model.config.get_text_config().vocab_size if hasattr(model.config, "get_text_config") else 262208
    out = {"load_s": load_s, "weights_gib": base_mem, "mem_fraction": a.mem_fraction,
           "cap_gib": a.mem_fraction * torch.cuda.get_device_properties(0).total_memory / 2**30, "runs": []}
    print(json.dumps(out), flush=True)

    def logps(m, ids, n_comp):
        logits = m(input_ids=ids).logits[:, :-1, :]
        logits = logits[:, -n_comp:, :]
        return selective_log_softmax(logits, ids[:, -n_comp:])

    for L in [int(x) for x in a.lengths.split(",")]:
        for rep in range(a.repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(10, vocab - 10, (1, L), device="cuda")
            n_comp = int(L * a.completion_frac)
            row = {"L": L, "rep": rep}
            try:
                torch.cuda.synchronize()
                t = time.time()
                with torch.no_grad(), model.disable_adapter():
                    ref = logps(model, ids, n_comp)
                torch.cuda.synchronize()
                row["ref_fwd_s"] = time.time() - t
                t = time.time()
                lp = logps(model, ids, n_comp)
                kl = torch.exp(ref - lp) - (ref - lp) - 1
                loss = (-(lp * 0.5) + 0.02 * kl).mean()
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                row["policy_fwd_bwd_step_s"] = time.time() - t
                row["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
                row["ok"] = True
            except torch.cuda.OutOfMemoryError as e:
                row.update({"ok": False, "error": "OOM", "peak_gib": torch.cuda.max_memory_allocated() / 2**30})
                del e
            out["runs"].append(row)
            print(json.dumps(row), flush=True)
            ids = ref = lp = kl = loss = None
            if not row["ok"]:
                break
        if not out["runs"][-1]["ok"]:
            break
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
