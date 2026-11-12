"""Disaggregated on-policy GRPO over a task env (env-agnostic). Trainer on node A, vLLM rollout server on B.

The model generates the ask/commit dialogue on-policy; the task simulator answers; the verifiable task reward
applies. env_mask trains only model-generated tokens. importance-sampling correction off (ratio=1), EOS set so
turns terminate at <end_of_turn>. Per-rollout diagnostics (mean_asks / parse_validity / committed / reward)
are printed to complement trl's reward/std/kl/clip/entropy logging.

Run (server on B must be up, e.g. via framework/infra/serve_b.sh):
  python -m framework.rl.grpo_disagg --env tasks.rx_triage.env --base <merged_init> \
    --server-host 192.168.100.2 --max-steps 80 --lr 1e-5 --beta 0.0 --rollout-temp 1.25 --out <ckpt_dir>
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
import argparse, importlib, json, re, time, statistics as stx
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from framework.parsing import parse_action

MAX_DIALOG_TURNS = 10  # default; override with --max-turns to match the Gate-2 probe horizon for the task
CASES: dict = {}
_CID = re.compile(r"\[case ([\w\-]+)\]")


def _ids(tok, msgs, gen):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    return tok(text, add_special_tokens=False)["input_ids"]


def _env_mask(tok, msgs, prompt_len, comp_len):
    mask = [0] * comp_len
    for j, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        pre = len(_ids(tok, msgs[:j], True)); upto = len(_ids(tok, msgs[:j + 1], False))
        for k in range(max(pre - prompt_len, 0), min(upto - prompt_len, comp_len)):
            mask[k] = 1
    return mask


def _entropy(dist):
    import math
    return -sum(p * math.log(p) for p in dist.values() if p > 0)


def traj_infogain(env, case, asks):
    """DREAM-inspired single-episode info-gain proxy (RL shaping only; NOT in env.score / the gates).
    Sums the reduction in entropy of the true-severity posterior `env.risk_dist(fam, revealed)` as each NEW,
    relevant ask reveals its value. Treats the hidden case (true severity) as the task identity mu and the ask
    turns as the exploration trajectory: an ask is rewarded exactly when it reduces uncertainty about the correct
    disposition. Irrelevant/repeat asks reveal nothing -> 0 gain (so this also discourages over-asking)."""
    fam, assign = case["family"], case["assign"]
    revealed, prev, gain = {}, _entropy(env.risk_dist(case["family"], {})), 0.0
    for a in asks:
        it = env.resolve_ask(a)
        if it is None:
            continue
        i = env.IDX[it]
        if i in revealed:
            continue
        revealed[i] = assign[i]
        cur = _entropy(env.risk_dist(fam, revealed))
        gain += max(prev - cur, 0.0); prev = cur
    return gain


def make_rollout(tok, env, rollout_temp, infogain_coef=0.0, claude_judge=None, claude_alpha=0.0,
                 claude_norm="zscore_clip", claude_log=None):
    def rollout_func(prompts, trainer):
        client = trainer.vllm_generation.vllm_client
        st = []
        for p in prompts:
            mc = _CID.search(p[-1]["content"])
            st.append({"msgs": list(p), "asks": [], "dec": None, "done": False, "vt": 0, "nt": 0,
                       "case": CASES.get(mc.group(1) if mc else None)})
        for _turn in range(MAX_DIALOG_TURNS):
            active = [i for i, s in enumerate(st) if not s["done"] and s["case"] is not None]
            if not active:
                break
            out = client.chat([st[i]["msgs"] for i in active], n=1, temperature=rollout_temp, max_tokens=128, logprobs=0)
            texts = tok.batch_decode(out["completion_ids"], skip_special_tokens=True)
            for i, txt in zip(active, texts):
                s = st[i]; s["msgs"].append({"role": "assistant", "content": txt}); s["nt"] += 1
                kind, val = parse_action(txt)
                canon = env.resolve_ask(val) if kind == "ask" else None
                if kind == "commit":
                    s["dec"] = val[0] if isinstance(val, tuple) else val
                    s["vt"] += 1 if s["dec"] in env.ACTIONS else 0; s["done"] = True
                elif kind == "ask" and canon:
                    s["vt"] += 1; s["asks"].append(canon)
                    s["msgs"].append({"role": "user", "content": env.answer(s["case"], canon)})
                else:
                    s["done"] = True
        prompt_ids, completion_ids, logprobs, env_mask, rewards, igains = [], [], [], [], [], []
        for s in st:
            pid = _ids(tok, s["msgs"][:2], True); full = _ids(tok, s["msgs"], False); comp = full[len(pid):]
            prompt_ids.append(pid); completion_ids.append(comp); logprobs.append([0.0] * len(comp))
            env_mask.append(_env_mask(tok, s["msgs"], len(pid), len(comp)))
            r = 0.0 if s["case"] is None else env.score(s["case"], s["asks"], s["dec"])[0]
            ig = 0.0 if (s["case"] is None or infogain_coef == 0.0) else traj_infogain(env, s["case"], s["asks"])
            igains.append(ig)
            rewards.append(r + infogain_coef * ig)   # base reward = clinical reward + lambda * info-gain
        # --- Claude process reward (disabled unless claude_judge is set AND alpha > 0) -------------------
        # total = base_reward + alpha * within-case-normalized Claude score. Judge sees student-visible info
        # ONLY; the analytic reward is never replaced. Raw/normalized/analytic/total are logged.
        if claude_judge is not None and claude_alpha > 0.0:
            from framework.procreward.reward import assemble
            claude_judge.reset_stats()
            _t0 = time.time()
            claude_raw = [None if s["case"] is None else claude_judge.score(s["case"], s["asks"], s["dec"])
                          for s in st]
            _score_wall = time.time() - _t0
            case_ids = [(s["case"]["case_id"] if s["case"] else f"none-{i}") for i, s in enumerate(st)]
            lengths = [len(s["asks"]) for s in st]
            totals, normd, log, gstats = assemble(rewards, claude_raw, case_ids, claude_alpha, claude_norm,
                                                  extra=[{"n_asks": n} for n in lengths])
            _cr = [c for c in claude_raw if c is not None]
            js = claude_judge.stats
            _api = js["api_calls"] or 1
            # correlation of raw process score with trajectory length (asks)
            _pairs = [(c, l) for c, l in zip(claude_raw, lengths) if c is not None]
            _corr = float("nan")
            if len(_pairs) > 2:
                cs, ls = [p[0] for p in _pairs], [p[1] for p in _pairs]
                mc, ml = stx.mean(cs), stx.mean(ls)
                den = (sum((c-mc)**2 for c in cs)*sum((l-ml)**2 for l in ls))**0.5
                _corr = (sum((c-mc)*(l-ml) for c, l in _pairs)/den) if den else float("nan")
            print(f"[claude-reward] alpha={claude_alpha} norm={claude_norm} "
                  f"mean_analytic={stx.mean(rewards):.3f} mean_total={stx.mean(totals):.3f} "
                  f"claude_raw[mean={(stx.mean(_cr) if _cr else float('nan')):.2f} "
                  f"sd={(stx.pstdev(_cr) if len(_cr)>1 else 0):.2f}] "
                  f"mean|alpha*norm|={stx.mean(abs(a) for a in [claude_alpha*n for n in normd]):.4f} "
                  f"nonzero_adv_groups={gstats['nonzero_advantage_groups']}/{gstats['n_groups']}", flush=True)
            print(f"[claude-judge] scored={len(_cr)}/{len(st)} cache_hits={js['cache_hits']}/{js['calls']} "
                  f"api_calls={js['api_calls']} retries={js['retries']} parse_fail={js['parse_fail']} "
                  f"score_wall={_score_wall:.1f}s api_latency/call={js['api_latency_s']/_api:.2f}s "
                  f"corr(score,len)={_corr:+.2f}", flush=True)
            if claude_log:
                with open(claude_log, "a") as f:
                    for row in log:
                        f.write(json.dumps(row) + "\n")
            rewards = totals
        nt = sum(s["nt"] for s in st) or 1
        print(f"[rollout-diag] mean_asks={stx.mean(len(s['asks']) for s in st):.2f} "
              f"parse_validity={sum(s['vt'] for s in st)/nt:.3f} "
              f"committed={sum(1 for s in st if s['dec'] is not None)/max(len(st),1):.3f} "
              f"mean_reward={stx.mean(rewards):.3f} mean_infogain={stx.mean(igains):.3f}", flush=True)
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids,
                "logprobs": logprobs, "env_mask": env_mask, "traj_reward": rewards}
    return rollout_func


def _gen_turn(client, tok, msgs_list, rollout_temp):
    out = client.chat(msgs_list, n=1, temperature=rollout_temp, max_tokens=128, logprobs=0)
    return tok.batch_decode(out["completion_ids"], skip_special_tokens=True)


def make_dream_rollout(tok, env, rollout_temp):
    """DREAM-inspired decoupled explore/exploit (single network, mode token). Both phases are generated EVERY
    step (for full logging); the gradient update alternates by step parity -- explore steps train the ask tokens
    on the recoverability reward (traj_infogain over env.risk_dist), exploit steps train the commit token on the
    pure env.score of a commit made over ONLY the explore-gathered findings. Decoupling: the two rewards never
    share a trajectory and are advantage-normalised within their own phase's step. env.score / U / gates untouched.
    """
    state = {"step": 0}

    def rollout_func(prompts, trainer):
        client = trainer.vllm_generation.vllm_client
        cids, cases = [], []
        for p in prompts:
            mc = _CID.search(p[-1]["content"]); cid = mc.group(1) if mc else None
            cids.append(cid); cases.append(CASES.get(cid))

        # ---- EXPLORE phase: ask-only dialogue (a commit/invalid ends exploration, no commit credit) ----
        exp = [{"msgs": ([{"role": "system", "content": env.SYS},
                          {"role": "user", "content": f"[case {cid}]\n" + env.explore_user_msg(case)}]
                         if case is not None else None),
                "asks": [], "done": case is None, "nt": 0, "vt": 0, "case": case}
               for cid, case in zip(cids, cases)]
        for _turn in range(MAX_DIALOG_TURNS):
            active = [i for i, s in enumerate(exp) if not s["done"]]
            if not active:
                break
            texts = _gen_turn(client, tok, [exp[i]["msgs"] for i in active], rollout_temp)
            for i, txt in zip(active, texts):
                s = exp[i]; s["msgs"].append({"role": "assistant", "content": txt}); s["nt"] += 1
                kind, val = parse_action(txt)
                canon = env.resolve_ask(val) if kind == "ask" else None
                if kind == "ask" and canon:
                    s["vt"] += 1; s["asks"].append(canon)
                    s["msgs"].append({"role": "user", "content": env.answer(s["case"], canon)})
                else:
                    s["done"] = True   # ask-only phase: anything else terminates explore

        # ---- COMMIT phase: one commit turn over ONLY explore-gathered findings (leak-guarded) ----
        com = []
        for cid, s in zip(cids, exp):
            case = s["case"]
            if case is None:
                com.append({"msgs": None, "dec": None, "case": None, "asks": []}); continue
            cu = env.commit_user_msg(case, s["asks"])
            env.assert_no_leak(case, s["asks"], cu)   # hard guard: commit context exposes only gathered findings
            com.append({"msgs": [{"role": "system", "content": env.SYS},
                                 {"role": "user", "content": f"[case {cid}]\n" + cu}],
                        "dec": None, "case": case, "asks": s["asks"]})
        active = [i for i, c in enumerate(com) if c["case"] is not None]
        if active:
            texts = _gen_turn(client, tok, [com[i]["msgs"] for i in active], rollout_temp)
            for i, txt in zip(active, texts):
                c = com[i]; c["msgs"].append({"role": "assistant", "content": txt})
                kind, val = parse_action(txt)
                if kind == "commit":
                    c["dec"] = val[0] if isinstance(val, tuple) else val

        explore_r = [traj_infogain(env, s["case"], s["asks"]) if s["case"] is not None else 0.0 for s in exp]
        exploit_r = [env.score(c["case"], c["asks"], c["dec"])[0] if c["case"] is not None else 0.0 for c in com]

        phase = "explore" if state["step"] % 2 == 0 else "exploit"
        state["step"] += 1
        prompt_ids, completion_ids, logprobs, env_mask, rewards = [], [], [], [], []
        for idx, (s, c) in enumerate(zip(exp, com)):
            if phase == "explore":
                msgs, rew = s["msgs"], explore_r[idx]
            else:
                msgs, rew = (c["msgs"] if (c["case"] is not None and len(c["msgs"]) >= 3) else None), exploit_r[idx]
            if msgs is None:                       # degenerate (no case / no commit produced): empty no-op sample
                msgs = [{"role": "system", "content": env.SYS}, {"role": "user", "content": f"[case {cids[idx]}]"},
                        {"role": "assistant", "content": ""}]
                rew = 0.0
            pid = _ids(tok, msgs[:2], True); full = _ids(tok, msgs, False); comp = full[len(pid):]
            prompt_ids.append(pid); completion_ids.append(comp); logprobs.append([0.0] * len(comp))
            env_mask.append(_env_mask(tok, msgs, len(pid), len(comp)))
            rewards.append(rew)
        nt = sum(s["nt"] for s in exp) or 1
        print(f"[dream-diag] phase={phase} "
              f"mean_explore_reward={stx.mean(explore_r):.3f} mean_exploit_reward={stx.mean(exploit_r):.3f} "
              f"mean_total_reward={stx.mean([a + b for a, b in zip(explore_r, exploit_r)]):.3f} "
              f"mean_asks={stx.mean(len(s['asks']) for s in exp):.2f} "
              f"committed={sum(1 for c in com if c['dec'] is not None) / max(len(com), 1):.3f} "
              f"parse_validity={sum(s['vt'] for s in exp) / nt:.3f}", flush=True)
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids,
                "logprobs": logprobs, "env_mask": env_mask, "traj_reward": rewards}
    return rollout_func


def reward_echo(completions, traj_reward=None, **kw):
    return traj_reward if traj_reward is not None else [0.0] * len(completions)


def main():
    global MAX_DIALOG_TURNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="tasks.rx_triage.env")
    ap.add_argument("--base", required=True)
    ap.add_argument("--server-host", default="192.168.100.2")
    ap.add_argument("--server-port", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--n-cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--rollout-temp", type=float, default=1.25)
    ap.add_argument("--save-steps", type=int, default=40)
    ap.add_argument("--max-turns", type=int, default=MAX_DIALOG_TURNS, help="dialogue horizon (match the probe)")
    ap.add_argument("--infogain-coef", type=float, default=0.0,
                    help="lambda for DREAM-inspired info-gain shaping (0 = baseline; shaping is RL-only, not in env.score/gates)")
    ap.add_argument("--dream", action="store_true",
                    help="DREAM-inspired decoupled explore/exploit (single net, mode token); overrides --infogain-coef")
    # --- Claude process reward (DISABLED BY DEFAULT: alpha 0.0). total = analytic + alpha*norm(claude) ---
    ap.add_argument("--claude-alpha", type=float, default=0.0,
                    help="weight on the within-case-normalized Claude process reward (0 = OFF, the default)")
    ap.add_argument("--claude-norm", choices=["zscore_clip", "centered_rank"], default="zscore_clip",
                    help="within-case normalization for the Claude score")
    ap.add_argument("--claude-cache", default="results/procreward_pilot/grpo_judge_cache.jsonl",
                    help="JSONL cache for judge calls (reused across steps/arms)")
    ap.add_argument("--claude-log", default="", help="optional JSONL log of raw/norm/analytic/total per rollout")
    ap.add_argument("--out", required=True)
    # --- V2: envs that bring their own prompts + rollout (e.g. tasks.v2_sdbench.env). V1 envs ignore these. ---
    ap.add_argument("--env-config", default=None, help="JSON settings for envs defining build_train_prompts")
    ap.add_argument("--max-completion-length", type=int, default=512,
                    help="trl max_completion_length (V1 default 512; SD-Bench transcripts need ~12k+)")
    args = ap.parse_args()
    MAX_DIALOG_TURNS = args.max_turns
    env = importlib.import_module(args.env)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    t0 = time.time()
    custom_env = hasattr(env, "build_train_prompts")
    if custom_env:
        ds = Dataset.from_list(env.build_train_prompts(args))
    else:
        cases = env.gen_cases(args.n_cases, rho=0.0, seed=12345, families=env.DEEP_FAMILIES)
        for c in cases:
            CASES[c["case_id"]] = c
        ds = Dataset.from_list([{"prompt": [{"role": "system", "content": env.SYS},
                                            {"role": "user", "content": f"[case {c['case_id']}]\n" + env.user_msg(c)}]}
                                for c in cases])
    print(f"[grpo] env={args.env} {len(ds)} cases init={Path(args.base).name} "
          f"lr={args.lr} beta={args.beta} temp={args.rollout_temp} steps={args.max_steps} "
          f"{'dream=ON' if args.dream else f'infogain={args.infogain_coef}'}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    model.generation_config.eos_token_id = [1, 106]
    lora = LoraConfig(r=32, lora_alpha=64, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = GRPOConfig(
        output_dir=args.out, per_device_train_batch_size=8, num_generations=8, gradient_accumulation_steps=4,
        max_completion_length=args.max_completion_length, temperature=args.rollout_temp, beta=args.beta, learning_rate=args.lr,
        max_steps=args.max_steps, logging_steps=1, bf16=True, report_to=[], seed=args.seed,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=4,
        use_vllm=True, vllm_mode="server", vllm_server_host=args.server_host, vllm_server_port=args.server_port,
        vllm_importance_sampling_correction=False)
    claude_judge = None
    if args.claude_alpha > 0.0:
        from framework.procreward.rollout_judge import RolloutJudge
        claude_judge = RolloutJudge(env, args.claude_cache)
        print(f"[grpo] Claude process reward ON: alpha={args.claude_alpha} norm={args.claude_norm} "
              f"cache={args.claude_cache}", flush=True)
    if custom_env:
        rollout = env.make_rollout_func(tok, args, {"ids": _ids, "env_mask": _env_mask})
    else:
        rollout = (make_dream_rollout(tok, env, args.rollout_temp) if args.dream
                   else make_rollout(tok, env, args.rollout_temp, args.infogain_coef,
                                     claude_judge=claude_judge, claude_alpha=args.claude_alpha,
                                     claude_norm=args.claude_norm, claude_log=(args.claude_log or None)))
    trainer = GRPOTrainer(model=model, reward_funcs=[reward_echo], args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=lora, rollout_func=rollout)
    print("[grpo] trainer built; train()", flush=True)
    trainer.train()
    trainer.save_model(str(Path(args.out) / "final"))
    print(f"[grpo] DONE steps={args.max_steps} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
