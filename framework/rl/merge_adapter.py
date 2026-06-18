"""Merge a LoRA adapter into its base and save a standalone model (for serving / as an RL init)."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    print(f"[merge] base={args.base}\n[merge] adapter={args.adapter}\n[merge] out={args.out}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    print("[merge] DONE", flush=True)


if __name__ == "__main__":
    main()
