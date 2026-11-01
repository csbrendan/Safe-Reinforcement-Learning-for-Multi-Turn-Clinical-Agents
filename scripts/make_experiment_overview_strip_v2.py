#!/usr/bin/env python3
"""Experiment overview (v2, Nature-style pass) for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

Editorial reduction of the earlier v2: three zones (pale panels for the environment and evaluation,
an unpaneled training section), strong outlines reserved for the few objects that matter
(MedGemma, GRPO training, Same pure verifier), oracle/verifier reduced to annotation badges,
reward variants as a clean pill row under GRPO, a vertical evaluation stack, and a subordinate
disaggregated two-node inset. Conceptual; not Figure 1.

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

INK="#22303c"; MUTE="#7c848d"; FAINT="#a3aab2"; HAIR="#e0e4e8"
TEAL="#1f7f8c"; TEAL_L="#e6f1f3"
GRAY_L="#f1f4f6"; GRAYB="#d6dbe0"
GOLD="#a9821f"; GOLD_L="#f6eed4"
VIO="#5f5a7a"; VIO_L="#e9e7f0"
RED="#bd4657"; RED_L="#f7e6e9"
GREEN="#2e7d32"; GREEN_L="#e4efe4"
BLUE="#3b6fb6"; BLUE_L="#e6edf7"
ENV_BG="#f4f7f8"; EVAL_BG="#f7f8f9"

FIG_W, FIG_H = 16.0, 8.2
BY = 5.35
os.makedirs("figs", exist_ok=True)
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=HAIR, lw=1.1, r=0.10, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color="#aab0b8", lw=1.4, rad=0.0, z=3, ms=11, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))

def dline(p1, p2, color=FAINT, lw=1.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=(0, (3, 3)), zorder=1.6)

def obj(cx, cy, w, h, title, sub=None, ec=GRAYB, fc="white", tc=INK, tfs=10.5, sfs=7.2, r=0.11, lw=1.2, bold=True):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=r, z=2.4, lw=lw)
    if sub:
        ax.text(cx, cy + 0.15, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "medium"), color=tc, zorder=3)
        ax.text(cx, cy - 0.18, sub, ha="center", va="center", fontsize=sfs, color=MUTE, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "medium"), color=tc, zorder=3)

# ============================ title + zone panels ============================
ax.text(0.36, FIG_H - 0.28, "Experimental overview", ha="left", va="top", fontsize=13.5,
        fontweight="bold", color=INK)
EX0, EX1, EY0, EY1 = 0.35, 6.25, 0.80, 7.02
VX0, VX1 = 11.35, 15.65
rrect(EX0, EY0, EX1 - EX0, EY1 - EY0, fc=ENV_BG, ec="none", r=0.16, z=0.6)
rrect(VX0, EY0, VX1 - VX0, EY1 - EY0, fc=EVAL_BG, ec="none", r=0.16, z=0.6)
ax.text(EX0 + 0.26, EY1 - 0.20, "Clinical multi-turn environment", ha="left", va="top", fontsize=11.5,
        fontweight="bold", color=INK)
ax.text(EX0 + 0.27, EY1 - 0.52, "refill triage · acute diagnostics · dosing titration", ha="left", va="top",
        fontsize=7.8, color=MUTE)
ax.text(VX0 + 0.26, EY1 - 0.20, "Held-out evaluation", ha="left", va="top", fontsize=11.5,
        fontweight="bold", color=INK)

# ============================ ZONE A: environment ============================
# hidden-state pill (small, light) above the simulator
rrect(3.05, BY + 0.70, 2.55, 0.32, fc="white", ec=HAIR, r=0.14, z=2.2)
ax.text(4.33, BY + 0.86, "hidden state · diagnosis · severity · dose", ha="center", va="center",
        fontsize=6.7, color=MUTE, zorder=3)
# model <-> simulator (only MedGemma is strongly outlined)
obj(1.74, BY, 1.92, 1.00, "MedGemma-4B-it", "LoRA-adapted policy", ec=TEAL, fc=TEAL_L, tc=TEAL, lw=2.0, r=0.28)
obj(4.33, BY, 1.94, 1.00, "Patient simulator", "hidden clinical state", ec=GRAYB, fc=GRAY_L, lw=1.0, bold=False)
arrow((2.62, BY + 0.20), (3.44, BY + 0.20), color=TEAL, rad=-0.32, ms=10)
ax.text(3.03, BY + 0.50, "ask", ha="center", va="center", fontsize=7.2, color=TEAL, fontweight="bold")
arrow((3.44, BY - 0.16), (2.62, BY - 0.16), color="#aab0b8", rad=-0.32, ms=10)
ax.text(3.03, BY - 0.44, "finding", ha="center", va="center", fontsize=6.8, color=MUTE)
# commit -> clinical action
arrow((1.74, BY - 0.50), (1.74, BY - 0.92), color="#aab0b8", ms=10)
ax.text(1.98, BY - 0.71, "commit", ha="left", va="center", fontsize=6.8, color=MUTE)
obj(1.74, BY - 1.26, 1.60, 0.44, "clinical action", ec=HAIR, fc="white", tfs=8.2, bold=False)
# oracle + verifier as small annotation badges
def badge(cx, cy, title, sub, ec, fc):
    w, h = 1.98, 0.56
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=0.13, z=2.4, lw=1.0)
    ax.text(cx - w/2 + 0.18, cy + 0.10, "◆", ha="center", va="center", fontsize=6.6, color=ec, zorder=3)
    ax.text(cx + 0.09, cy + 0.10, title, ha="center", va="center", fontsize=8.0, fontweight="bold", color=ec, zorder=3)
    ax.text(cx, cy - 0.14, sub, ha="center", va="center", fontsize=6.6, color=MUTE, zorder=3)
badge(1.95, 2.30, "Exact adaptive oracle", "reference procedure", GOLD, GOLD_L)
badge(4.28, 2.30, "Deterministic verifier", "outcome scorer", VIO, VIO_L)
dline((2.20, 2.58), (3.30, BY - 0.52), color=GOLD)
dline((4.28, 2.58), (4.33, BY - 0.52), color=VIO)

# ============================ ZONE B: training ==============================
arrow((EX1, BY), (7.42, BY), color="#9aa1a9", ms=12, lw=1.6)
ax.text((EX1 + 7.42) / 2, BY + 0.20, "policy rollouts", ha="center", va="center", fontsize=6.8, color=MUTE)
obj(8.35, BY, 1.66, 0.90, "GRPO training", ec=INK, fc="white", tc=INK, lw=2.0, tfs=10.5)
# reward variants: a clean pill row beneath GRPO (branching implied by a light fork, no arcs)
PY = BY - 1.42
pills = [(7.20, RED, RED_L, "Outcome-only", "terminal reward"),
         (8.35, GREEN, GREEN_L, "Credit assignment", "+ info credit"),
         (9.50, BLUE, BLUE_L, "Decoupled", "gather / commit")]
dline((8.35, BY - 0.45), (8.35, PY + 0.46), color=GRAYB, lw=1.0)
ax.plot([7.20, 9.50], [PY + 0.40, PY + 0.40], color=GRAYB, lw=1.0, zorder=1.5)
for cx, ec, fc, t, s in pills:
    ax.plot([cx, cx], [PY + 0.40, PY + 0.26], color=GRAYB, lw=1.0, zorder=1.5)
    rrect(cx - 0.58, PY - 0.26, 1.16, 0.52, fc=fc, ec=ec, r=0.26, z=2.5, lw=1.3)
    ax.text(cx, PY + 0.055, t, ha="center", va="center", fontsize=7.6, fontweight="bold", color=ec, zorder=3)
    ax.text(cx, PY - 0.145, s, ha="center", va="center", fontsize=6.0, color=MUTE, zorder=3)
ax.text(8.35, PY + 0.62, "training reward differs", ha="center", va="center", fontsize=7.8,
        fontweight="bold", color=MUTE)
# trained policies -> common evaluation
arrow((9.18, BY), (12.18, BY), color="#9aa1a9", ms=12, lw=1.6)
ax.text((9.18 + 12.18) / 2, BY + 0.20, "trained policies · common evaluation", ha="center", va="center",
        fontsize=7.0, color=MUTE)

# disaggregated two-node inset (subordinate, tied to GRPO/training column)
IX0, IX1, IY0, IY1 = 6.98, 9.72, 1.66, 2.98
dline((8.35, PY - 0.26), (8.35, IY1), color=HAIR, lw=1.0)
rrect(IX0, IY0, IX1 - IX0, IY1 - IY0, fc=EVAL_BG, ec=HAIR, r=0.10, z=1.7)
ax.text((IX0 + IX1) / 2, IY1 - 0.15, "Disaggregated RL", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=FAINT, zorder=3)
for cx, name, sub in [(7.62, "Trainer node", "GRPO · LoRA"), (9.08, "Rollout node", "vLLM · 8 rollouts")]:
    rrect(cx - 0.60, 2.06, 1.20, 0.52, fc="white", ec="#d3d8dd", r=0.10, z=2.0, lw=1.0)
    ax.text(cx, 2.44, name, ha="center", va="center", fontsize=6.4, fontweight="bold", color=MUTE, zorder=3)
    ax.text(cx, 2.24, sub, ha="center", va="center", fontsize=5.6, color=FAINT, zorder=3)
ax.add_patch(FancyArrowPatch((8.22, 2.32), (8.48, 2.32), arrowstyle="<|-|>", mutation_scale=7,
             linewidth=1.0, color="#b3b9c0", shrinkA=1, shrinkB=1, zorder=2.4))
ax.text((IX0 + IX1) / 2, 1.82, "two DGX Spark nodes · updates ↔ rollouts", ha="center", va="center",
        fontsize=5.6, color=FAINT, zorder=3)

# ============================ ZONE C: evaluation ============================
obj(13.5, BY, 2.50, 0.60, "Held-out simulator cases", "unseen patients", ec=GRAYB, fc="white",
    tfs=9.4, sfs=6.8, lw=1.0, bold=False)
arrow((13.5, BY - 0.30), (13.5, 4.36), color="#aab0b8", ms=10)
obj(13.5, 3.96, 2.60, 0.80, "Same pure verifier", "common scorer", ec=VIO, fc=VIO_L, tc=VIO,
    lw=2.0, tfs=10.5, sfs=7.0)
arrow((13.5, 3.56), (13.5, 3.30), color="#aab0b8", ms=9)
rrect(13.5 - 0.86, 3.00, 1.72, 0.30, fc="white", ec=RED, r=0.12, z=2.6)
ax.text(13.5, 3.15, "reward shaping off", ha="center", va="center", fontsize=7.2, fontweight="bold", color=RED, zorder=3)
ax.text(13.5, 2.55, "metrics", ha="center", va="center", fontsize=7.2, fontweight="bold", color=MUTE, zorder=3)
ax.text(13.5, 2.29, "overall value   ·   high-risk value", ha="center", va="center", fontsize=7.0, color=MUTE, zorder=3)
ax.text(13.5, 2.07, "question burden   ·   oracle-grade support", ha="center", va="center", fontsize=7.0, color=MUTE, zorder=3)

for ext in ("png", "svg", "pdf"):
    out = f"figs/experiment_overview_strip_v2.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
