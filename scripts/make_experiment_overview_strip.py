#!/usr/bin/env python3
"""Experiment overview strip — a standalone, journal-style horizontal infographic that introduces the
experimental object for "Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

It is NOT Figure 1; it is a compact banner that may later sit above Figure 1.

Flow (left -> right):
  MedGemma-4B-it (policy)  <->  clinical multi-turn environment  ->  patient simulator (hidden state)
  ->  exact adaptive oracle + deterministic verifier  ->  GRPO training (reward varies by method)
  ->  held-out evaluation (same pure verifier)

Conceptual, not a full methods diagram. Outputs: figs/experiment_overview_strip.{png,svg,pdf}.
Run inside ~/spark1/.venv.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

INK  = "#20303f"; MUTE = "#66707b"; FAINT = "#9aa2ac"; HAIR = "#dfe3e7"
GRAYB = "#c9ced4"                       # generic card border
TEAL  = "#207d8b"; TEAL_L = "#e2f0f2"   # MedGemma / policy
GOLD  = "#b0851f"; GOLD_L = "#f6edcf"   # oracle + verifier + shared eval
RED   = "#bd4657"; RED_L  = "#f6e3e6"
GREEN = "#2e7d32"; GREEN_L= "#e2efe3"
BLUE  = "#3b6fb6"; BLUE_L = "#e5edf7"

FIG_W, FIG_H = 16.0, 4.3
os.makedirs("figs", exist_ok=True)

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=GRAYB, lw=1.3, r=0.10, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color="#aeb4bc", lw=1.8, rad=0.0, z=3, ms=13, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z, linestyle=ls))

CY = 2.55
W, H = 2.15, 1.72
C = [1.42, 4.04, 6.66, 9.28, 11.90, 14.52]   # block centers

def card(cx, title, sub=None, tiny=None, ec=GRAYB, fc="white", tc=INK, r=0.10,
         tfs=11, cy=CY, h=H, sfs=8.3):
    rrect(cx - W / 2, cy - h / 2, W, h, fc=fc, ec=ec, r=r, z=2)
    ty = cy + h / 2 - 0.36
    ax.text(cx, ty, title, ha="center", va="center", fontsize=tfs, fontweight="bold", color=tc,
            linespacing=0.98, zorder=3)
    if sub:
        nlines = title.count("\n") + 1
        ax.text(cx, ty - 0.30 * nlines - 0.10, sub, ha="center", va="center", fontsize=sfs,
                color=MUTE, linespacing=1.0, zorder=3)
    if tiny:
        ax.text(cx, cy - h / 2 + 0.24, tiny, ha="center", va="center", fontsize=7.4, color=FAINT, zorder=3)


# ---------------- heading ----------------
ax.text(0.35, FIG_H - 0.34, "Experimental setup", ha="left", va="top", fontsize=12.5,
        fontweight="bold", color=INK)
ax.text(0.35, FIG_H - 0.70, "an interactive clinical RL environment with an exact adaptive oracle and a shared "
        "held-out verifier", ha="left", va="top", fontsize=9.2, color=MUTE)

# ---------------- 1. MedGemma (policy) ----------------
card(C[0], "MedGemma-4B-it", "policy model", tiny="LoRA-adapted", ec=TEAL, fc=TEAL_L, tc=TEAL, r=0.42)

# interaction (acts / observes) between policy and environment
gx0, gx1 = C[0] + W / 2, C[1] - W / 2
arrow((gx0 + 0.02, CY + 0.16), (gx1 - 0.02, CY + 0.16), color=TEAL, ms=12)
arrow((gx1 - 0.02, CY - 0.16), (gx0 + 0.02, CY - 0.16), color="#aeb4bc", ms=12)
ax.text((gx0 + gx1) / 2, CY + 0.40, "acts", ha="center", va="center", fontsize=7.4, color=TEAL)
ax.text((gx0 + gx1) / 2, CY - 0.40, "observes", ha="center", va="center", fontsize=7.4, color=MUTE)

# ---------------- 2. Clinical multi-turn environment ----------------
card(C[1], "Clinical multi-turn\nenvironment", "interactive · multi-turn",
     tiny="refill triage · diagnostics · titration")

# ---------------- 3. Patient simulator (hidden state) ----------------
card(C[2], "Patient simulator", "hidden diagnosis /\nseverity / med state")
# mini ask -> reveal -> commit loop beneath
ly = 1.35
ax.text(C[2] - 0.86, ly, "ask", ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")
arrow((C[2] - 0.66, ly), (C[2] - 0.16, ly), color="#b8bec4", ms=9, lw=1.4)
ax.text(C[2] + 0.14, ly, "finding", ha="center", va="center", fontsize=7.6, color=INK)
arrow((C[2] + 0.52, ly), (C[2] + 0.86, ly), color="#b8bec4", ms=9, lw=1.4)
ax.text(C[2] + 1.16, ly, "commit", ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")
arrow((C[2] + 0.10, ly + 0.14), (C[2] - 0.80, ly + 0.14), color=TEAL, rad=-0.6, ms=8, lw=1.2)
ax.text(C[2] - 0.34, ly + 0.46, "ask again", ha="center", va="center", fontsize=6.8, color=TEAL)

# ---------------- 4. Oracle + Verifier (two stacked) ----------------
sub_h = 0.78
rrect(C[3] - W / 2, CY + 0.06, W, sub_h, fc=GOLD_L, ec=GOLD, r=0.09, z=2)
ax.text(C[3], CY + 0.56, "Exact adaptive oracle", ha="center", va="center", fontsize=9.2,
        fontweight="bold", color=GOLD, zorder=3)
ax.text(C[3], CY + 0.28, "reference policy", ha="center", va="center", fontsize=7.6, color=MUTE, zorder=3)
rrect(C[3] - W / 2, CY - sub_h - 0.06, W, sub_h, fc="white", ec=GOLD, r=0.09, z=2)
ax.text(C[3], CY - 0.30, "Deterministic verifier", ha="center", va="center", fontsize=9.2,
        fontweight="bold", color=INK, zorder=3)
ax.text(C[3], CY - 0.58, "common outcome scorer", ha="center", va="center", fontsize=7.6, color=MUTE, zorder=3)

# ---------------- 5. GRPO training ----------------
card(C[4], "GRPO training", "reward varies by method", tiny="(during training only)")
chips = [("outcome-only", RED, RED_L), ("credit assignment", GREEN, GREEN_L), ("decoupled", BLUE, BLUE_L)]
chx = [C[4] - 0.86, C[4] + 0.10, C[4] + 0.98]
cw = [0.98, 1.30, 0.92]
cyl = 1.35
for (lab, ec, fc), x, w in zip(chips, chx, cw):
    rrect(x - w / 2, cyl - 0.17, w, 0.34, fc=fc, ec=ec, r=0.09, z=2.5, lw=1.2)
    ax.text(x, cyl, lab, ha="center", va="center", fontsize=6.6, color=ec, fontweight="bold", zorder=3)
# tiny optional disaggregated-RL inset (secondary)
gy = 0.92
ax.add_patch(FancyBboxPatch((C[4] - 0.62, gy - 0.10), 0.20, 0.20, boxstyle="round,pad=0,rounding_size=0.03",
             linewidth=0, facecolor="#c3c8ce", zorder=2))
ax.add_patch(FancyBboxPatch((C[4] - 0.38, gy - 0.10), 0.20, 0.20, boxstyle="round,pad=0,rounding_size=0.03",
             linewidth=0, facecolor="#c3c8ce", zorder=2))
ax.text(C[4] - 0.06, gy, "disaggregated RL: trainer + rollout server", ha="left", va="center",
        fontsize=6.6, color=FAINT, zorder=3)

# ---------------- 6. Held-out evaluation (shared verifier) ----------------
card(C[5], "Held-out evaluation", "same pure verifier\nfor all methods",
     tiny="reward shaping off", ec=GOLD, fc=GOLD_L, tc=INK)

# ---------------- main flow arrows (2->3->4->5->6) ----------------
for a, b in [(1, 2), (2, 3), (3, 4), (4, 5)]:
    arrow((C[a] + W / 2 + 0.02, CY), (C[b] - W / 2 - 0.02, CY), color="#aeb4bc", ms=13)

# shared-verifier cue: gold matches between block 4 (verifier) and block 6 (eval)
ax.text((C[3] + C[5]) / 2, CY + H / 2 + 0.30, "the same deterministic verifier scores every method at evaluation",
        ha="center", va="center", fontsize=7.6, color=GOLD, style="italic")

for ext in ("png", "svg", "pdf"):
    out = f"figs/experiment_overview_strip.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
