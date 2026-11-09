#!/usr/bin/env python3
"""Bounded HealthBench plumbing test using the official simple-evals code (HealthBenchEval).
Grader is local Qwen3-8B, an UNVALIDATED substitute for the official GPT-4.1 grader: scores here prove the pipeline
runs and must never be reported as HealthBench scores. Downloads the public 60 MB JSONL via the official code path."""
import importlib, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/v2_bench")); os.environ.setdefault("OPENAI_API_KEY", "local")
from openai import OpenAI
hb = importlib.import_module("simple-evals.healthbench_eval"); cs = importlib.import_module("simple-evals.sampler.chat_completion_sampler")
class Local(cs.ChatCompletionSampler):
    def __init__(self, url, model, max_tokens, extra=None, temperature=0.05):
        super().__init__(model=model, max_tokens=max_tokens, temperature=temperature); self.client = OpenAI(base_url=url, api_key="local", timeout=240, max_retries=1); self.extra = extra
    def __call__(self, ml):
        r = self.client.chat.completions.create(model=self.model, messages=ml, temperature=self.temperature, max_tokens=self.max_tokens, extra_body=self.extra)
        return cs.SamplerResponse(response_text=r.choices[0].message.content or "", response_metadata={"usage": None}, actual_queried_message_list=ml)
n, out, model = int(sys.argv[1]), sys.argv[2], sys.argv[3]
policy = Local("http://127.0.0.1:8100/v1", model, 700, temperature=0.0)
grader = Local("http://192.168.100.2:8101/v1", "aux", 400, {"chat_template_kwargs": {"enable_thinking": False}})
ev = hb.HealthBenchEval(grader_model=grader, num_examples=n, n_threads=2, input_path=os.path.expanduser("~/v2_bench/healthbench_data/oss_eval.jsonl"))
t = time.time(); res = ev(policy)
rows = []
for c in res.convos: rows.append(c)
json.dump({"model": model, "n": n, "secs": round(time.time() - t, 1), "grader": "local Qwen3-8B (UNOFFICIAL)", "metrics": res.metrics, "score": res.score}, open(out, "w"), indent=1, default=str)
print(open(out).read()[:1500])
