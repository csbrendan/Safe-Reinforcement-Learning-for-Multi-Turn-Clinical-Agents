"""Diagnostic hedging task, held-out hidden, pure env.score, by risk subset:
base / plain GRPO / credit-assignment GRPO / decoupled GRPO. Plain GRPO and credit-assignment GRPO are averaged
over 3 seeds with +/- std error bars; base and decoupled GRPO are single-seed."""
import json, statistics as stx
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ARMS = [
    ("base", ["results/probe_dh_base_t12.records.jsonl"], "#888888"),
    ("plain GRPO", ["results/probe_dh_grpo_final_t12.records.jsonl",
                    "results/probe_dh_ctl_s2_t12.records.jsonl",
                    "results/probe_dh_ctl_s3_t12.records.jsonl"], "#d1495b"),
    ("credit-assignment GRPO", ["results/probe_dh_ig_final_t12.records.jsonl",
                                "results/probe_dh_ig_s2_t12.records.jsonl",
                                "results/probe_dh_ig_s3_t12.records.jsonl"], "#2e7d32"),
    ("decoupled GRPO", ["results/probe_dream_s40_t12.records.jsonl"], "#3b6fb6"),
]
SUBSETS = [("high", "high"), ("moderate", "moderate"), ("none", "benign")]
def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"] == "hidden"]
def supp(recs, key):
    rs=[r for r in recs if r["true_risk"]==key]; by={}
    for r in rs: by.setdefault(r["case_id"],[]).append(r)
    return stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0 for v in by.values()) if by else 0.0
def val(recs, key):
    rs=[r for r in recs if r["true_risk"]==key]; return stx.mean(r["reward"] for r in rs) if rs else 0.0
def stat(files, key, fn):
    vals=[fn(load(f), key) for f in files]
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0)
fig,(axS,axV)=plt.subplots(1,2,figsize=(13,4.8)); x=np.arange(len(SUBSETS)); w=0.2
for k,(name,files,c) in enumerate(ARMS):
    sm=[stat(files,key,supp) for key,_ in SUBSETS]; vm=[stat(files,key,val) for key,_ in SUBSETS]; n=len(files)
    lab=f"{name} ({n} seed{'s' if n>1 else ''})"
    axS.bar(x+(k-1.5)*w,[m for m,_ in sm],w,yerr=[e for _,e in sm],capsize=3,label=lab,color=c,error_kw=dict(lw=1,ecolor="#333"))
    axV.bar(x+(k-1.5)*w,[m for m,_ in vm],w,yerr=[e for _,e in vm],capsize=3,label=lab,color=c,error_kw=dict(lw=1,ecolor="#333"))
for ax,t,yl in [(axS,"Oracle-grade support@12","support@12"),(axV,"Clinical value","mean reward")]:
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in SUBSETS]); ax.set_title(t); ax.set_ylabel(yl)
    ax.axhline(0,color="k",lw=0.6); ax.grid(axis="y",alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Diagnostic hedging task by risk subset (held-out hidden, pure reward). "
             "Plain GRPO & credit-assignment GRPO: mean$\\pm$std over 3 seeds.", fontsize=10.5)
plt.tight_layout(rect=[0,0,1,0.95]); out="results/figs/dx_4arm_seeds.png"; plt.savefig(out,dpi=130); print("saved",out)
for name,files,_ in ARMS:
    hm,he=stat(files,"high",supp); print(f"  {name:24} high-supp={hm:.2f}+/-{he:.2f} ({len(files)}s)")
