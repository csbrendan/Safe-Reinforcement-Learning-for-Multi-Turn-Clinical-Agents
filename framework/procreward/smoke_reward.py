"""CPU smoke test for the Claude process-reward integration math (no GPU, no API).

Verifies the guarantees the plan requires:
  - within-CASE normalization (z-score+clip and centered-rank), zero-mean per group, clipping honored;
  - the process score is ADDED to, never REPLACES, the analytic terminal reward;
  - alpha scales the process term; alpha=0 => total == analytic exactly;
  - degenerate groups (all-equal or <2 scored) contribute ZERO shaping;
  - None (parse-fail) rollouts are handled without corrupting the group;
  - the per-rollout log carries raw / normalized / analytic / total.

  PYTHONPATH=$PWD python -m framework.procreward.smoke_reward
"""
from __future__ import annotations
import statistics as stx
from framework.procreward.reward import normalize_group, assemble

P = []
def check(name, cond, detail=""):
    P.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


# ---- normalize_group -------------------------------------------------------
z = normalize_group([1.0, 2.0, 3.0, 4.0], "zscore_clip")
check("zscore zero-mean per group", abs(stx.mean(z)) < 1e-9, f"mean={stx.mean(z):.2e}")
z2 = normalize_group([0.0, 0.0, 0.0, 100.0], "zscore_clip", clip=2.0)
check("zscore clipping honored", max(abs(x) for x in z2) <= 2.0 + 1e-9, f"max|z|={max(abs(x) for x in z2):.2f}")
r = normalize_group([5.0, 1.0, 3.0, 9.0], "centered_rank")
check("centered_rank in [-1,1] with min=-1,max=+1", min(r) == -1.0 and max(r) == 1.0, str(r))
check("centered_rank order matches value order", r[3] == 1.0 and r[1] == -1.0, str(r))
check("degenerate all-equal -> zeros", normalize_group([2.0, 2.0, 2.0], "zscore_clip") == [0, 0, 0])
check("degenerate <2 scored -> zeros", normalize_group([None, 4.0], "zscore_clip") == [0.0, 0.0])
zn = normalize_group([1.0, 2.0, 3.0, None], "zscore_clip")
check("None filled to group mean (~0 after centering)", abs(zn[3]) < 1e-9, f"none->{zn[3]:.2e}")

# ---- assemble (grouping by case_id, non-replacement, alpha) -----------------
analytic = [0.80, 0.20, 0.79, 0.10]          # 2 cases x 2 rollouts
claude = [4.0, -3.0, 3.0, -2.0]              # judge prefers the higher-analytic rollout in each case
cids = ["cA", "cA", "cB", "cB"]

tot0, _, log0, _g0 = assemble(analytic, claude, cids, alpha=0.0)
check("alpha=0 => total == analytic exactly", tot0 == analytic)
check("log preserves analytic (non-replacement)", all(abs(l["analytic"] - a) < 1e-9 for l, a in zip(log0, analytic)))
check("log carries raw/norm/analytic/total keys",
      all(set(l) >= {"claude_raw", "claude_norm", "analytic", "total", "alpha"} for l in log0))

tot, normd, log, _g = assemble(analytic, claude, cids, alpha=0.1, method="zscore_clip")
check("process ADDS, not replaces (total = analytic + alpha*norm)",
      all(abs(t - (a + 0.1 * n)) < 1e-9 for t, a, n, in zip(tot, analytic, normd)))
check("normalization is WITHIN case (each case zero-mean)",
      abs(normd[0] + normd[1]) < 1e-9 and abs(normd[2] + normd[3]) < 1e-9)
check("judge nudges better-analytic rollout UP within each case",
      tot[0] > analytic[0] and tot[1] < analytic[1] and tot[2] > analytic[2] and tot[3] < analytic[3])
check("process term magnitude << analytic scale at small alpha",
      max(abs(0.1 * n) for n in normd) < max(analytic), f"max|alpha*norm|={max(abs(0.1*n) for n in normd):.3f}")

# a case where the judge is wrong-but-bounded: analytic still dominates the sign
aw = [0.9, 0.1]; cw = [-5.0, 5.0]; cid = ["z", "z"]
tw, nw, _l, _gw = assemble(aw, cw, cid, alpha=0.1)
check("bounded process cannot flip a large analytic gap at small alpha", tw[0] > tw[1],
      f"totals={[round(x,3) for x in tw]}")

print(f"\n=== {sum(P)}/{len(P)} checks passed ===")
import sys
sys.exit(0 if all(P) else 1)
