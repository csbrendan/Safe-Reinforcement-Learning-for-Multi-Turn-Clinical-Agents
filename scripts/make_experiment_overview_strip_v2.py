#!/usr/bin/env python3
"""Experiment overview (v2) — editorial redesign of the v1 strip for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

Hierarchy, not a row of equal boxes:
  ZONE 1  a large "Clinical multi-turn environment" container = the focal process
          (MedGemma <-> patient simulator, hidden state, commit -> clinical action;
           oracle + verifier as small secondary badges attached to it)
  ZONE 2  GRPO training -> fans OUT into three reward-variant pills (reward differs)
  ZONE 3  the three variants CONVERGE back into one common held-out evaluation
          (same pure verifier, shaping off) -> metrics

Conceptual schematic; no equations/results/trajectories. Not Figure 1.
Outputs: figs/experiment_overview_strip_v2.{png,svg,pdf}. Run inside ~/spark1/.venv.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

INK="#20303f"; MUTE="#66707b"; FAINT="#98a0aa"; HAIR="#d8dde2"
TEAL="#207d8b"; TEAL_L="#e2f0f2"
GRAYB="#c6ccd2"; GRAY_L="#eef1f3"
GOLD="#b0851f"; GOLD_L="#f6edcf"
VIO="#5f5a7a"; VIO_L="#e9e7f0"
RED="#bd4657"; RED_L="#f6e3e6"
GREEN="#2e7d32"; GREEN_L="#e2efe3"
BLUE="#3b6fb6"; BLUE_L="#e5edf7"
ENV_BG="#f1f6f8"

FIG_W, FIG_H = 16.6, 8.4
BY = 4.55                      # main-flow baseline
os.makedirs("figs", exist_ok=True)
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=GRAYB, lw=1.3, r=0.10, z=2, alpha=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1, linestyle=ls))

def arrow(p1, p2, color="#a4abb3", lw=1.7, rad=0.0, z=3, ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z, linestyle=ls))

def dline(p1, p2, color=GOLD, lw=1.1):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=(0, (3, 3)), zorder=1.6)

def cap(cx, cy, w, h, title, sub, ec, fc, r=0.30, tfs=9.4, sfs=7.2):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=r, z=2.6, lw=1.6)
    ax.text(cx, cy + 0.12, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=ec, zorder=3)
    ax.text(cx, cy - 0.15, sub, ha="center", va="center", fontsize=sfs, color=MUTE, zorder=3)

def box(cx, cy, w, h, title, sub=None, ec=GRAYB, fc="white", tc=INK, tfs=10.5, sfs=7.8, r=0.11, lw=1.4):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=r, z=2.4, lw=lw)
    if sub:
        ax.text(cx, cy + 0.16, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=tc, zorder=3)
        ax.text(cx, cy - 0.19, sub, ha="center", va="center", fontsize=sfs, color=MUTE, zorder=3, linespacing=0.95)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=tc, zorder=3)

# tiny title
ax.text(0.34, FIG_H - 0.26, "Experimental overview", ha="left", va="top", fontsize=12.5,
        fontweight="bold", color=INK)

# ============================ ZONE 1: environment ============================
EX0, EX1, EY0, EY1 = 0.4, 6.75, 0.85, 7.05
rrect(EX0, EY0, EX1 - EX0, EY1 - EY0, fc=ENV_BG, ec="#cdd7db", r=0.16, z=0.7, lw=1.4)
ax.text(EX0 + 0.24, EY1 - 0.22, "Clinical multi-turn environment", ha="left", va="top", fontsize=12.5,
        fontweight="bold", color=INK)
ax.text(EX0 + 0.25, EY1 - 0.58, "refill triage · acute diagnostics · dosing titration", ha="left", va="top",
        fontsize=8.2, color=MUTE)

# hidden-state banner above the simulator
rrect(3.28, 5.66, 3.20, 0.42, fc="white", ec="#cdb3b6", r=0.10, z=2.2)
ax.text(4.88, 5.87, "hidden state:  diagnosis · severity · medication", ha="center", va="center",
        fontsize=7.2, color=RED, zorder=3)

# policy <-> simulator (the focal interaction)
box(1.83, BY + 0.30, 2.02, 1.02, "MedGemma-4B-it", "LoRA-adapted policy", ec=TEAL, fc=TEAL_L, tc=TEAL, r=0.30)
box(4.88, BY + 0.30, 2.05, 1.02, "Patient simulator", "hidden clinical state", ec=GRAYB, fc="white")
arrow((2.72, BY + 0.52), (3.73, BY + 0.52), color=TEAL, rad=-0.32, ms=11)
ax.text(3.22, BY + 0.92, "ask", ha="center", va="center", fontsize=7.6, color=TEAL, fontweight="bold")
arrow((3.73, BY + 0.16), (2.72, BY + 0.16), color="#a4abb3", rad=-0.32, ms=11)
ax.text(3.22, BY - 0.20, "finding revealed", ha="center", va="center", fontsize=7.0, color=MUTE)

# commit -> clinical action
arrow((1.83, BY - 0.21), (1.83, BY - 0.86), color="#a4abb3", ms=11)
ax.text(2.06, BY - 0.54, "commit", ha="left", va="center", fontsize=7.2, color=INK)
box(1.83, BY - 1.20, 1.86, 0.52, "clinical action", ec="#c6ccd2", fc="#f4f6f8", tfs=8.6)
ax.text(4.88, BY - 0.70, "ask again  or  commit", ha="center", va="center", fontsize=7.0, color=FAINT, style="italic")

# oracle + verifier: small secondary badges attached to the environment
def badge(cx, cy, title, sub, ec, fc):
    w, h = 2.20, 0.70
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=0.13, z=2.5, lw=1.4)
    ax.text(cx - w/2 + 0.20, cy + 0.13, "◆", ha="center", va="center", fontsize=7.5, color=ec, zorder=3)
    ax.text(cx + 0.08, cy + 0.13, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=ec, zorder=3)
    ax.text(cx, cy - 0.15, sub, ha="center", va="center", fontsize=7.0, color=MUTE, zorder=3)
badge(2.45, 1.62, "Exact adaptive oracle", "reference procedure", GOLD, GOLD_L)
badge(4.95, 1.62, "Deterministic verifier", "terminal outcome scorer", VIO, VIO_L)
dline((2.75, 1.97), (3.95, BY - 0.05), color=GOLD)
dline((4.95, 1.97), (4.88, BY - 0.21), color=VIO)
ax.text(3.62, 1.05, "analytic properties of the controlled simulator", ha="center", va="center",
        fontsize=6.8, color=FAINT, style="italic")

# ============================ ZONE 2: GRPO -> fan out ========================
arrow((EX1, BY), (7.35, BY), color="#8f97a0", ms=13, lw=1.9)
ax.text((EX1 + 7.35) / 2, BY + 0.22, "policy rollouts", ha="center", va="center", fontsize=7.2, color=MUTE)
box(8.25, BY, 1.72, 1.02, "GRPO training", "reward varies\nby method", ec=INK, fc="white", tfs=10.5, sfs=7.6)

pills = [(RED, RED_L, "Outcome-only", "terminal reward", BY + 1.30),
         (GREEN, GREEN_L, "Credit assignment", "+ information credit", BY),
         (BLUE, BLUE_L, "Decoupled", "separate gather / commit", BY - 1.30)]
PX = 10.75
for ec, fc, t, s, cy in pills:
    arrow((9.13, BY), (PX - 1.08, cy), color="#b7bdc4", ms=9, lw=1.4, rad=(0.0 if cy == BY else (0.12 if cy > BY else -0.12)))
    cap(PX, cy, 2.10, 0.66, t, s, ec, fc)
# "training reward differs" bracket over the pills
bx0, bx1, byk = PX - 1.15, PX + 1.15, BY + 1.86
ax.plot([bx0, bx0, bx1, bx1], [byk - 0.10, byk, byk, byk - 0.10], color="#c3c9d0", lw=1.2)
ax.text(PX, byk + 0.16, "training reward differs", ha="center", va="bottom", fontsize=8.6,
        fontweight="bold", color=INK)

# ============================ ZONE 3: converge -> common eval ================
MERGE = 12.30
for _, _, _, _, cy in pills:
    arrow((PX + 1.08, cy), (MERGE, BY), color="#b7bdc4", ms=9, lw=1.4,
          rad=(0.0 if cy == BY else (-0.12 if cy > BY else 0.12)))
EXC = 14.15
arrow((MERGE, BY), (EXC - 1.20, BY), color="#8f97a0", ms=13, lw=1.9)
box(EXC, BY, 2.30, 0.66, "Held-out simulator cases", "unseen patients", tfs=9.6, sfs=7.2)
arrow((EXC, BY - 0.33), (EXC, BY - 0.72), color="#a4abb3", ms=11)
# prominent shared verifier
rrect(EXC - 1.28, BY - 1.66, 2.56, 0.92, fc=VIO_L, ec=VIO, r=0.14, z=2.6, lw=2.2)
ax.text(EXC, BY - 1.01, "Same pure verifier", ha="center", va="center", fontsize=11, fontweight="bold",
        color=VIO, zorder=3)
ax.text(EXC, BY - 1.30, "one terminal scorer, every method", ha="center", va="center", fontsize=7.4,
        color=MUTE, zorder=3)
rrect(EXC - 1.05, BY - 2.14, 2.10, 0.34, fc="white", ec=RED, r=0.10, z=2.8)
ax.text(EXC, BY - 1.97, "reward shaping off", ha="center", va="center", fontsize=7.8, fontweight="bold",
        color=RED, zorder=3)
arrow((EXC, BY - 2.16), (EXC, BY - 2.50), color="#a4abb3", ms=10)
# metrics (2x2)
rrect(EXC - 1.55, BY - 3.55, 3.10, 0.92, fc="white", ec=HAIR, r=0.11, z=2.3)
ax.text(EXC, BY - 2.78, "evaluation metrics", ha="center", va="center", fontsize=8.2, fontweight="bold",
        color=INK, zorder=3)
for lab, mx, my in [("overall value", EXC - 0.76, BY - 3.10), ("high-risk value", EXC + 0.76, BY - 3.10),
                    ("question burden", EXC - 0.76, BY - 3.42), ("oracle-grade support", EXC + 0.76, BY - 3.42)]:
    ax.text(mx, my, "· " + lab, ha="center", va="center", fontsize=6.9, color=MUTE, zorder=3)
# "common evaluation" bracket over the eval column
cbx0, cbx1, cby = EXC - 1.30, EXC + 1.30, BY + 0.62
ax.plot([cbx0, cbx0, cbx1, cbx1], [cby - 0.10, cby, cby, cby - 0.10], color="#cdbf9a", lw=1.2)
ax.text(EXC, cby + 0.16, "common evaluation — held constant", ha="center", va="bottom", fontsize=8.6,
        fontweight="bold", color=GOLD)

# optional disaggregated-RL inset (small, secondary), under GRPO
ix, iy = 8.25, 2.05
rrect(ix - 1.30, iy - 0.62, 2.60, 1.16, fc="white", ec=HAIR, r=0.10, z=1.8)
ax.text(ix, iy + 0.38, "Disaggregated RL", ha="center", va="center", fontsize=7.8, fontweight="bold", color=MUTE)
box(ix - 0.66, iy - 0.14, 1.02, 0.40, "trainer", ec="#c9ced4", fc="#f4f6f8", tfs=7.2, lw=1.0)
box(ix + 0.66, iy - 0.14, 1.02, 0.40, "rollout", ec="#c9ced4", fc="#f4f6f8", tfs=7.2, lw=1.0)
arrow((ix - 0.13, iy - 0.06), (ix + 0.13, iy - 0.06), color="#c3c8ce", ms=7, lw=1.0)
arrow((ix + 0.13, iy - 0.22), (ix - 0.13, iy - 0.22), color="#c3c8ce", ms=7, lw=1.0)
ax.text(ix, iy - 0.50, "LoRA updates  ·  vLLM generation", ha="center", va="center", fontsize=6.4, color=FAINT)

for ext in ("png", "svg", "pdf"):
    out = f"figs/experiment_overview_strip_v2.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
