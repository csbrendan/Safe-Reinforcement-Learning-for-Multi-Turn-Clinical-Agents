"""Generic multi-turn protocol SFT warmup (LoRA). Teaches the ask-then-commit protocol, not new medicine.

Supervises EVERY assistant turn (the asks AND the commit) — masking everything but the last turn would teach
the model only to commit, never to gather. Init = a (text-only) base model + a fresh all-linear LoRA. Data is
a jsonl of {"messages": [...]} trajectories produced by a task's data-gen script.

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m framework.rl.sft_warmup \
    --base <base_model> --data <traj.jsonl> --out <adapter_dir> --epochs 2 --seed 1
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, json, shutil, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE, DTYPE = "cuda:0", torch.bfloat16


class MultiTurnSFT(Dataset):
    """Supervise every assistant turn; mask system/user turns (prefix-tokenization spans)."""
    def __init__(self, path, tok, max_len):
        self.tok, self.max_len = tok, max_len
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self): return len(self.rows)

    def _ids(self, msgs, gen):
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def __getitem__(self, idx):
        msgs = self.rows[idx]["messages"]
        full = self._ids(msgs, False)
        labels = [-100] * len(full)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            pre = self._ids(msgs[:i], True)
            upto = self._ids(msgs[:i + 1], False)
            for j in range(len(pre), min(len(upto), len(full))):
                labels[j] = full[j]
        full, labels = full[:self.max_len], labels[:self.max_len]
        return {"input_ids": torch.tensor(full), "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(full), dtype=torch.long)}


def collate(batch, pad_id):
    n = max(b["input_ids"].size(0) for b in batch)
    ii = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lb = torch.full((len(batch), n), -100, dtype=torch.long)
    am = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = b["input_ids"].size(0)
        ii[i, :k] = b["input_ids"]; lb[i, :k] = b["labels"]; am[i, :k] = b["attention_mask"]
    return {"input_ids": ii, "labels": lb, "attention_mask": am}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random as _random
    _random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=DTYPE, device_map=DEVICE)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
                                             task_type="CAUSAL_LM", target_modules="all-linear"))
    model.print_trainable_parameters(); model.train()

    ds = MultiTurnSFT(args.data, tok, args.max_seq_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    total = args.epochs * len(dl); warm = max(1, int(0.05 * total))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    print(f"[sft] {args.epochs} ep x {len(ds)} cases @ batch {args.batch} = {total} steps", flush=True)
    t0, step = time.time(), 0
    for ep in range(args.epochs):
        for batch in dl:
            out = model(input_ids=batch["input_ids"].to(DEVICE), attention_mask=batch["attention_mask"].to(DEVICE),
                        labels=batch["labels"].to(DEVICE))
            out.loss.backward(); opt.step(); sched.step(); opt.zero_grad(); step += 1
            if step % 10 == 0 or step == 1:
                print(f"[ep {ep} step {step}/{total}] loss={out.loss.item():.4f} cum={time.time()-t0:.0f}s", flush=True)
    fin = out_dir / "final"
    if fin.exists(): shutil.rmtree(fin)
    fin.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(fin)); tok.save_pretrained(str(fin))
    print(f"[sft] done {time.time()-t0:.0f}s -> {fin}", flush=True)


if __name__ == "__main__":
    main()
