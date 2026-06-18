"""Prepare a TEXT-ONLY causal-LM base from a Gemma3 multimodal checkpoint (e.g. google/medgemma-4b-it).

The pipeline (sft_warmup / support_probe / grpo) is text-only and uses AutoModelForCausalLM + vLLM. A raw
Gemma3 *-it checkpoint carries a SigLIP vision tower + multimodal projector that AutoModelForCausalLM can't
load and that crashes vLLM weight-sync. This extracts the language_model into a plain Gemma3ForCausalLM and
saves it (bf16) — a clean foundation base, NO domain training. Lossless for text: the vision path is dropped,
not approximated. Runs on CPU to keep the GPU free. The output is a derived weight: keep it out of git.

Usage:
  python -m framework.infra.prep_text_base --src google/medgemma-4b-it --out checkpoints/medgemma4b_text
"""
from __future__ import annotations
import argparse
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration, Gemma3ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="google/medgemma-4b-it")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[prep] loading {args.src} (CPU, bf16) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.src)
    mm = Gemma3ForConditionalGeneration.from_pretrained(args.src, torch_dtype=torch.bfloat16, device_map="cpu")
    base = mm.model if hasattr(mm, "model") else mm
    lm = getattr(base, "language_model", None) or getattr(mm, "language_model", None)
    if lm is None:
        raise RuntimeError("could not locate language_model submodule on the multimodal checkpoint")
    print(f"[prep] language_model type: {type(lm).__name__}", flush=True)

    text_cfg = mm.config.text_config
    causal = Gemma3ForCausalLM(text_cfg).to(torch.bfloat16)
    miss, unexp = causal.model.load_state_dict(lm.state_dict(), strict=False)
    print(f"[prep] load into causal.model -> missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if miss[:5]:
        print("  sample missing:", miss[:5], flush=True)
    if unexp[:5]:
        print("  sample unexpected:", unexp[:5], flush=True)
    causal.tie_weights()
    mm_head = getattr(mm, "lm_head", None)
    if mm_head is not None and not getattr(text_cfg, "tie_word_embeddings", True):
        causal.lm_head.load_state_dict(mm_head.state_dict())
        print("[prep] copied untied lm_head", flush=True)

    emb_norm = causal.get_input_embeddings().weight.float().norm().item()
    print(f"[prep] embed norm={emb_norm:.1f} (sanity, expect >0)", flush=True)
    causal.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[prep] DONE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
