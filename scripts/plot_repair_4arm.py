"""SFT-coverage 4-arm repair bar: biased-SFT vs info-gain-repair vs DREAM-repair, by risk subset.
Neither RL variant restores high-risk support, and both erode benign/moderate. Held-out hidden (n=200), pure env.score.
"""
import json, statistics as stx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [("biased SFT","results/probe_dx_biased30_n200_t12.records.jsonl","#1f77b4"),
        ("info-gain repair","results/probe_dx_repair_final_t12.records.jsonl","#d1495b"),
        ("DREAM repair","results/probe_dx_repdream_final_t12.records.jsonl","#2e7d32")]
SUBSETS=[("overall",lambda s:True),("high",lambda s:s=="high"),("moderate",lambda s:s=="moderate"),("benign",lambda s:s=="none")]
def load(p): return [json.loads(l) for l in open(p) if json.loads(l)["cond"]=="hidden"]
def supp(recs,pred):
    rs=[r for r in recs if pred(r["true_risk"])]
    by={}
    for r in rs: by.setdefault(r["case_id"],[]).append(r)
    return stx.mean(1.0 if any(x["oracle_grade"] for x in v) else 0 for v in by.values()) if by else 0
def val(recs,pred): rs=[r for r in recs if pred(r["true_risk"])]; return stx.mean(r["reward"] for r in rs) if rs else 0
data={n:load(p) for n,p,_ in ARMS}

fig,(axS,axV)=plt.subplots(1,2,figsize=(13,4.6))
x=np.arange(len(SUBSETS)); w=0.26
for k,(n,_,c) in enumerate(ARMS):
    axS.bar(x+(k-1)*w,[supp(data[n],p) for _,p in SUBSETS],w,label=n,color=c)
    axV.bar(x+(k-1)*w,[val(data[n],p) for _,p in SUBSETS],w,label=n,color=c)
for ax,t,yl in [(axS,"Oracle-grade support@12","support@12"),(axV,"Clinical value","mean reward")]:
    ax.set_xticks(x); ax.set_xticklabels([s for s,_ in SUBSETS]); ax.set_title(t); ax.set_ylabel(yl)
    ax.axhline(0,color="k",lw=0.6); ax.grid(axis="y",alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Biased-SFT high-risk hole is NOT repaired by RL: info-gain collapses support, DREAM over-gathers "
             "and damages benign/moderate (held-out hidden, n=200)", fontsize=10.5)
plt.tight_layout(rect=[0,0,1,0.95])
out="results/figs/repair_4arm.png"; plt.savefig(out,dpi=130); print("saved",out)
for n,_,_ in ARMS:
    print(f"\n{n}:")
    for s,p in SUBSETS: print(f"  {s:9} supp={supp(data[n],p):.2f} val={val(data[n],p):+.3f}")
