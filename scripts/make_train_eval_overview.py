#!/usr/bin/env python3
"""Train-vs-evaluate overview — a standalone conceptual figure for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

One point, made visually obvious: RL methods use DIFFERENT training reward structures, but every method is
evaluated on the SAME held-out clinical environments with the SAME unshaped deterministic verifier.

Left half = Training (interaction loop + oracle/verifier + reward varies by method).
Right half = Held-out evaluation (frozen policy -> held-out cases -> same verifier -> metrics).

Not Figure 1; a banner to sit above it. Conceptual, no equations/results/trajectories.
Outputs: figs/train_eval_overview.{png,svg,pdf}. Run inside ~/spark1/.venv.
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
LEFT_BG="#f3f7f9"; RIGHT_BG="#f7f9fa"

FIG_W, FIG_H = 16.0, 8.6
os.makedirs("figs", exist_ok=True)
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=GRAYB, lw=1.3, r=0.10, z=2, alpha=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1, linestyle=ls))

def arrow(p1, p2, color="#a9afb7", lw=1.7, rad=0.0, z=3, ms=12, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z, linestyle=ls))

def dline(p1, p2, color=GOLD, lw=1.2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=(0, (3, 3)), zorder=1.6)

def card(cx, cy, w, h, title, sub=None, ec=GRAYB, fc="white", tc=INK, tfs=10.5, sfs=8.0, r=0.10, lw=1.3, bold=True):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=r, z=2.4, lw=lw)
    if sub:
        ax.text(cx, cy + 0.15, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "normal"), color=tc, zorder=3, linespacing=0.95)
        ax.text(cx, cy - 0.22, sub, ha="center", va="center", fontsize=sfs, color=MUTE, zorder=3, linespacing=0.95)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "normal"), color=tc, zorder=3, linespacing=0.95)

# ===================== panel backgrounds + titles =====================
PY0, PY1 = 0.45, 7.65
rrect(0.30, PY0, 8.85, PY1 - PY0, fc=LEFT_BG, ec=HAIR, r=0.12, z=0.6)
rrect(9.75, PY0, 5.95, PY1 - PY0, fc=RIGHT_BG, ec=HAIR, r=0.12, z=0.6)
dline((9.47, PY0 + 0.2), (9.47, PY1 - 0.2), color="#c3c9d0", lw=1.3)
ax.text(0.62, PY1 - 0.30, "Training", ha="left", va="top", fontsize=17, fontweight="bold", color=INK)
ax.text(10.02, PY1 - 0.30, "Held-out evaluation", ha="left", va="top", fontsize=17, fontweight="bold", color=INK)

# ===================== LEFT: training interaction =====================
# MedGemma policy (agent, outside the environment)
card(1.78, 5.85, 2.05, 1.05, "MedGemma-4B-it", "LoRA-adapted policy", ec=TEAL, fc=TEAL_L, tc=TEAL, r=0.30, tfs=10.5)

# environment container
rrect(3.15, 4.55, 5.75, 2.45, fc="white", ec=GRAYB, r=0.12, z=1.4)
ax.text(3.35, 6.86, "Clinical multi-turn environment", ha="left", va="center", fontsize=9.6,
        fontweight="bold", color=INK, zorder=3)
ax.text(3.35, 6.60, "refill triage · diagnostics · titration", ha="left", va="center", fontsize=7.8,
        color=MUTE, zorder=3)
# patient simulator (inside env)
card(4.85, 5.60, 2.05, 1.00, "Patient simulator", "hidden clinical state", ec=GRAYB, fc=GRAY_L, tfs=10)
rrect(4.10, 4.92, 1.50, 0.30, fc="white", ec="#cdb3b6", r=0.08, z=2.6)
ax.text(4.85, 5.07, "state hidden from agent", ha="center", va="center", fontsize=6.6, color=RED, zorder=3)
# oracle + verifier (attached to env, distinct roles)
card(7.55, 6.28, 2.30, 0.70, "Exact adaptive oracle", "reference policy", ec=GOLD, fc=GOLD_L, tfs=9.2, sfs=7.2)
card(7.55, 5.28, 2.30, 0.70, "Deterministic verifier", "terminal outcome scorer", ec=VIO, fc=VIO_L, tfs=9.2, sfs=7.2)
dline((5.88, 5.75), (6.40, 6.28)); dline((5.88, 5.55), (6.40, 5.28))
ax.text(6.15, 6.62, "reference", ha="center", va="center", fontsize=6.4, color=GOLD, style="italic", zorder=3)

# interaction arrows (ask / finding revealed / commit)
arrow((2.83, 6.02), (3.79, 6.02), color=TEAL, ms=11)
ax.text(3.31, 6.20, "ask", ha="center", va="center", fontsize=7.6, color=TEAL, fontweight="bold")
arrow((3.79, 5.45), (2.83, 5.45), color="#a9afb7", ms=11)
ax.text(3.31, 5.28, "finding revealed", ha="center", va="center", fontsize=7.0, color=MUTE)
ax.text(1.78, 5.02, "ask again  or  commit", ha="center", va="center", fontsize=7.2, color=INK)

# ===================== LEFT: reward varies by method =====================
arrow((4.85, 4.55), (4.85, 4.18), color="#a9afb7", ms=11)
ax.text(5.02, 4.36, "rollouts", ha="left", va="center", fontsize=7.0, color=MUTE)
ax.text(4.72, 3.96, "Training reward varies by method", ha="center", va="center", fontsize=11,
        fontweight="bold", color=INK)
meth = [(2.28, RED, RED_L, "Outcome-only GRPO", "terminal utility\n− question cost"),
        (4.72, GREEN, GREEN_L, "Credit-assignment GRPO", "terminal reward\n+ information credit"),
        (7.16, BLUE, BLUE_L, "Decoupled explore/exploit", "separate gathering &\ncommitment objectives")]
for cx, ec, fc, t, s in meth:
    arrow((4.72, 3.70), (cx, 3.42), color="#b7bdc4", ms=9, lw=1.3, rad=0.0)
    card(cx, 2.92, 2.32, 1.02, t, s, ec=ec, fc=fc, tc=ec, tfs=9.2, sfs=7.6, lw=1.5)
ax.text(4.72, 2.16, "same task family — only the training reward differs", ha="center", va="center",
        fontsize=7.8, color=MUTE, style="italic")

# optional disaggregated-RL inset (small, secondary)
rrect(0.55, 0.66, 3.35, 1.12, fc="white", ec=HAIR, r=0.09, z=1.6)
ax.text(0.72, 1.60, "Disaggregated RL", ha="left", va="center", fontsize=7.8, fontweight="bold", color=MUTE)
for gx in (0.80, 1.02):
    ax.add_patch(FancyBboxPatch((gx, 1.00), 0.16, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor="#c3c8ce", zorder=2.2))
ax.text(1.30, 1.08, "trainer node · GRPO + LoRA", ha="left", va="center", fontsize=6.8, color=FAINT)
for gx in (0.80, 1.02):
    ax.add_patch(FancyBboxPatch((gx, 0.76), 0.16, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor="#c3c8ce", zorder=2.2))
ax.text(1.30, 0.84, "rollout server · vLLM generation", ha="left", va="center", fontsize=6.8, color=FAINT)

# ===================== TRANSITION =====================
arrow((9.05, 6.30), (11.05, 6.30), color=INK, ms=14, lw=2.0)
ax.text(10.05, 6.54, "freeze & select policy", ha="center", va="center", fontsize=8.2,
        color=INK, fontweight="bold")
ax.text(10.05, 6.08, "trained checkpoint", ha="center", va="center", fontsize=7.4, color=MUTE)

# ===================== RIGHT: evaluation flow =====================
EX = 12.72
card(EX, 6.30, 3.05, 0.72, "Trained policy", "frozen checkpoint", ec=TEAL, fc=TEAL_L, tc=TEAL, tfs=10, sfs=7.4)
arrow((EX, 5.94), (EX, 5.66), color="#a9afb7", ms=11)
card(EX, 5.28, 3.45, 0.74, "Held-out clinical cases", "unseen simulated patients", tfs=10, sfs=7.4)
arrow((EX, 4.91), (EX, 4.58), color="#a9afb7", ms=11)
# prominent shared verifier
rrect(EX - 1.90, 3.62, 3.80, 0.94, fc=VIO_L, ec=VIO, r=0.12, z=2.6, lw=2.2)
ax.text(EX, 4.30, "Same deterministic verifier", ha="center", va="center", fontsize=11,
        fontweight="bold", color=VIO, zorder=3)
ax.text(EX, 4.02, "one scorer for every method", ha="center", va="center", fontsize=8.0, color=MUTE, zorder=3)
rrect(EX - 1.55, 3.16, 3.10, 0.36, fc="white", ec=RED, r=0.10, z=2.8)
ax.text(EX, 3.34, "reward shaping off  ·  no auxiliary training rewards", ha="center", va="center",
        fontsize=7.6, color=RED, fontweight="bold", zorder=3)
arrow((EX, 3.10), (EX, 2.80), color="#a9afb7", ms=11)
# metrics
rrect(EX - 1.95, 1.30, 3.90, 1.42, fc="white", ec=GRAYB, r=0.11, z=2.4)
ax.text(EX, 2.48, "Evaluation metrics", ha="center", va="center", fontsize=10, fontweight="bold", color=INK, zorder=3)
mets = [("overall value", EX - 0.98, 2.02), ("high-risk value", EX + 0.98, 2.02),
        ("question burden", EX - 0.98, 1.58), ("oracle-grade support", EX + 0.98, 1.58)]
for lab, mx, my in mets:
    rrect(mx - 0.88, my - 0.17, 1.76, 0.34, fc="#f4f6f8", ec=HAIR, r=0.09, z=2.6)
    ax.text(mx, my, lab, ha="center", va="center", fontsize=7.4, color=INK, zorder=3)

# key caption under the eval half
ax.text(EX, 0.86, "Same pure verifier for all methods", ha="center", va="center", fontsize=9.4,
        fontweight="bold", color=VIO)

for ext in ("png", "svg", "pdf"):
    out = f"figs/train_eval_overview.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
