#!/usr/bin/env python3
"""Figure 1 (v4) — refinement of v3 (scripts/make_figure1_v3.py). NOT a redesign.

Three panels, same verified data:
  A  compact setup strip: incomplete clinical information (unknown -> query -> revealed)   (~12% height)
  B  HERO: same patient, different information-gathering procedures (oracle + 3 learned)     (dominant)
  C  quantitative confirmation: high-risk value + mean asks; replication status made visible

v4 changes vs v3: Panel A shrunk to a horizontal strip; oracle vs learned distinction sharpened;
decoupled non-decisive findings visually dimmed + bracketed; gathering-depth scale anchored on the oracle
(oracle & credit both at 4 asks); Panel C marks single-seed references (†, hatch) vs 3-seed mean±SD; squarer
finding cards + prominent action pills + stronger type hierarchy. Clean export has NO title.
Outputs: figure1_overview_v4.{svg,pdf,png}.  Run inside ~/spark1/.venv.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
})

INK   = "#20303f"; ANNO = "#586170"; ANNO2 = "#8b93a0"; HAIR = "#e2e5e9"
GRAY  = "#8a8f98"
RED   = "#bd4657"; RED_L  = "#f6e3e6"
GREEN = "#2e7d32"; GREEN_L= "#e2efe3"
BLUE  = "#3b6fb6"; BLUE_L = "#e5edf7"
GOLD  = "#b0851f"; GOLD_L = "#f6edcf"
TEAL  = "#0f8a7e"
TINT  = {RED: RED_L, GREEN: GREEN_L, BLUE: BLUE_L, GOLD: GOLD_L}

TITLED = os.environ.get("FIG1_TITLE") == "1"
FIG_W, FIG_H = 14.0, (13.0 if TITLED else 12.6)
YTOP = FIG_H - (0.9 if TITLED else 0.3)

LAB = {
    "abd_rigidity":  "Abdominal\nrigidity", "rebound_tender":"Rebound\ntenderness",
    "ct_free_air":   "CT free\nair",        "lactate_high":  "Lactate\nelevated",
    "guarding":      "Guarding",            "radiation_arm": "Radiating\npain",
    "dyspnea_severe":"Severe\ndyspnea",     "hypotension":   "Blood\npressure",
    "wbc_high":      "White cell\ncount",   "ecg_ischemia":  "ECG",
}
ACT = {"emergent_tx": "Emergency\ntreatment", "admit": "Admit"}
PRESENT = {"abd_rigidity":1,"rebound_tender":1,"ct_free_air":1,"lactate_high":1,"guarding":1,
           "radiation_arm":1,"dyspnea_severe":1,"hypotension":0,"wbc_high":0,"ecg_ischemia":0}

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=HAIR, lw=1.1, r=0.06, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color="#b9bec4", lw=1.4, rad=0.0, z=1.6, ms=9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))

def head(x, y, letter, title, sub=None):
    ax.text(x, y, letter, ha="left", va="top", fontsize=15.5, fontweight="bold", color=INK)
    ax.text(x + 0.52, y - 0.02, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    if sub:
        ax.text(x + 0.53, y - 0.40, sub, ha="left", va="top", fontsize=9.4, color=ANNO)

def patient_glyph(cx, cy, s=1.0, color="#b6bcc4"):
    ax.add_patch(Circle((cx, cy + 0.16 * s), 0.17 * s, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Wedge((cx, cy - 0.14 * s), 0.34 * s, 20, 160, facecolor=color, edgecolor="none", zorder=3))

# trajectory chips: squarer, smaller
CW, CH, GAP = 0.86, 0.58, 0.20
TRACK_X0 = 3.35
def chip_x(i):
    return TRACK_X0 + CW / 2 + i * (CW + GAP)

def finding_chip(cx, cy, key, present, dim=False):
    ec = "#d9dde2" if dim else "#c4c9cf"
    a = 0.5 if dim else 1.0
    rrect(cx - CW / 2, cy - CH / 2, CW, CH, fc="white", ec=ec, r=0.035, z=2.4, alpha=a, lw=1.0)
    tc = "#a7adb5" if dim else INK
    ax.text(cx, cy + 0.075, LAB.get(key, key), ha="center", va="center", fontsize=6.4,
            color=tc, zorder=3, linespacing=0.95, alpha=a)
    if not dim:
        mc = TEAL if present else "#bcc2c8"
        ax.add_patch(Circle((cx - 0.25, cy - 0.185), 0.042, facecolor=(mc if present else "white"),
                     edgecolor=("none" if present else "#bcc2c8"), linewidth=1.0, zorder=3.2))
        ax.text(cx - 0.165, cy - 0.185, "present" if present else "normal", ha="left", va="center",
                fontsize=5.1, color=(TEAL if present else ANNO2), zorder=3.2)

def action_pill(cx, cy, key, color, w=1.30, h=0.66):
    rrect(cx - w / 2, cy - h / 2, w, h, fc=TINT.get(color, "white"), ec=color, lw=2.0, r=0.16, z=2.6)
    ax.text(cx, cy, ACT.get(key, key), ha="center", va="center", fontsize=7.4, color=color,
            fontweight="bold", zorder=3, linespacing=0.95)
    return cx + w / 2

def reward_tag(cx, cy, value, color):
    ax.text(cx, cy + 0.10, f"{value:+.2f}", ha="center", va="center", fontsize=14, fontweight="bold",
            color=color, zorder=3)
    ax.text(cx, cy - 0.22, "reward", ha="center", va="center", fontsize=6.6, color=ANNO2, zorder=3)

def trajectory(y, keys, action_key, reward, method, color, sub, sub2, oracle=False,
               dim_from=None, bracket_label=None):
    col = GOLD if oracle else color
    ax.add_patch(FancyBboxPatch((0.55, y - 0.33), 0.11, 0.66, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor=col, zorder=3))
    ax.text(0.80, y + 0.16, method, ha="left", va="center", fontsize=10.5, fontweight="bold", color=col)
    ax.text(0.80, y - 0.13, sub, ha="left", va="center", fontsize=8.0, color=ANNO)
    ax.text(0.80, y - 0.40, sub2, ha="left", va="top", fontsize=7.2, color=ANNO2)
    n = len(keys)
    for i, k in enumerate(keys):
        cx = chip_x(i)
        dim = dim_from is not None and i >= dim_from
        finding_chip(cx, y, k, PRESENT.get(k, 1), dim=dim)
        if i > 0:
            arrow((chip_x(i - 1) + CW / 2, y), (cx - CW / 2, y), color="#c6cbd1", lw=1.1, ms=7)
    # bracket over the dimmed (non-decisive) findings
    if dim_from is not None and bracket_label:
        bx0 = chip_x(dim_from) - CW / 2 - 0.03
        bx1 = chip_x(n - 1) + CW / 2 + 0.03
        by = y + CH / 2 + 0.13
        ax.plot([bx0, bx0, bx1, bx1], [by - 0.07, by, by, by - 0.07], color="#b3b9c0", lw=1.0, zorder=2)
        ax.text((bx0 + bx1) / 2, by + 0.11, bracket_label, ha="center", va="bottom", fontsize=6.8,
                color=ANNO2, style="italic")
    last_r = chip_x(n - 1) + CW / 2
    apx = last_r + 0.62
    arrow((last_r, y), (apx - 0.66, y), color="#c6cbd1", lw=1.2, ms=8)
    right = action_pill(apx, y, action_key, col)
    reward_tag(right + 0.60, y, reward, col)


# ================================ TITLE (preview only) =======================
if TITLED:
    ax.text(FIG_W / 2, FIG_H - 0.48, "Reward Design Governs Information Gathering in Clinical "
            "Multi-Turn Agents", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)

# =============================== PANEL A (compact strip) =====================
head(0.5, YTOP, "A", "Incomplete clinical information")
ax_y = YTOP - 0.78
patient_glyph(0.72, ax_y + 0.02, s=0.52)
ax.text(1.06, ax_y + 0.14, "Adult with acute abdominal pain", ha="left", va="center",
        fontsize=9.4, fontweight="bold", color=INK)
ax.text(1.06, ax_y - 0.17, "underlying severity hidden from the agent", ha="left", va="center",
        fontsize=7.8, color=RED)
arrow((4.05, ax_y), (4.45, ax_y), color="#b9bec4", ms=10)

def a_hidden(cx, label):
    w, h = 1.30, 0.52
    rrect(cx - w / 2, ax_y - h / 2, w, h, fc="#eef0f2", ec="#d5d9dd", r=0.04, z=2.2)
    ax.text(cx - w / 2 + 0.16, ax_y, "?", ha="center", va="center", fontsize=11, color="#aab0b8",
            fontweight="bold")
    ax.text(cx + 0.06, ax_y, label, ha="center", va="center", fontsize=7.6, color="#6a727c")

for cx, lab in [(5.15, "Rigidity"), (6.60, "CT free air"), (8.05, "Lactate")]:
    a_hidden(cx, lab)
ax.text(6.60, ax_y + 0.46, "order one finding at a time", ha="center", va="center", fontsize=6.8, color=ANNO2)
arrow((8.85, ax_y), (9.28, ax_y), color=TEAL, ms=10, lw=1.6)
rrect(9.42, ax_y - 0.31, 2.42, 0.62, fc="white", ec=TEAL, lw=1.9, r=0.05, z=2.6)
ax.text(10.63, ax_y + 0.10, "Abdominal rigidity", ha="center", va="center", fontsize=7.8,
        color=INK, fontweight="bold")
ax.text(10.63, ax_y - 0.15, "revealed:  PRESENT", ha="center", va="center", fontsize=7.6,
        color=TEAL, fontweight="bold")
ax.text(0.80, YTOP - 1.30, "Selectively reveal findings  ·  each request costs 0.03  ·  commit when enough "
        "evidence is available", ha="left", va="center", fontsize=8.4, color=ANNO)
ax.plot([0.5, 13.5], [YTOP - 1.62, YTOP - 1.62], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL B (hero) ==============================
by = YTOP - 1.95
head(0.5, by, "B", "Same patient, different information-gathering procedures",
     sub="same presentation · same hidden severity · same final verifier · only the training reward differs")
lx0 = 10.75
ax.add_patch(Circle((lx0, by - 0.14), 0.045, facecolor=TEAL, edgecolor="none", zorder=3))
ax.text(lx0 + 0.10, by - 0.14, "present", ha="left", va="center", fontsize=6.9, color=ANNO)
ax.add_patch(Circle((lx0 + 1.05, by - 0.14), 0.045, facecolor="white", edgecolor="#bcc2c8", lw=1.0, zorder=3))
ax.text(lx0 + 1.15, by - 0.14, "normal", ha="left", va="center", fontsize=6.9, color=ANNO)

r0, r1, r2, r3 = by - 1.30, by - 2.62, by - 3.94, by - 5.32
trajectory(r0, ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high"], "emergent_tx", 0.83,
           "Reference: exact adaptive oracle", GOLD,
           "4 targeted findings → emergency treatment", "not a learned policy", oracle=True)
trajectory(r1, ["abd_rigidity"], "admit", 0.17, "Outcome-only GRPO", RED,
           "1 finding · premature commitment", "under-gathering — misses the emergency")
trajectory(r2, ["abd_rigidity", "guarding", "rebound_tender", "ct_free_air"], "emergent_tx", 0.83,
           "Credit-assignment GRPO", GREEN,
           "sampled learned trajectory · 4 findings",
           "successful learned rollout; reward within oracle tolerance, evidence order differs from oracle")
trajectory(r3, ["abd_rigidity", "rebound_tender", "guarding", "radiation_arm", "dyspnea_severe",
                "hypotension", "wbc_high", "ecg_ischemia"], "emergent_tx", 0.71,
           "Decoupled explore/exploit", BLUE,
           "8 findings vs oracle 4 · correct action",
           "excessive work-up; reward cut by extra question cost",
           dim_from=3, bracket_label="additional non-decisive findings")

# --- gathering-depth scale, anchored on the oracle ---
gy = by - 6.45
def xn(n):
    return chip_x(n - 1)
x_lo, x_hi = xn(1) - 0.45, xn(8) + 0.45
rrect(x_lo, gy - 0.05, x_hi - x_lo, 0.10, fc="#eef0f2", ec="none", r=0.05, z=1.3)
rrect(xn(3) - 0.20, gy - 0.05, (xn(5) + 0.20) - (xn(3) - 0.20), 0.10, fc="#dcecdd", ec="none", r=0.05, z=1.35)
ax.plot([xn(4), xn(4)], [gy - 0.22, gy + 0.30], color=GOLD, lw=1.8, zorder=2.6)
ax.text(xn(4), gy + 0.40, "oracle", ha="center", va="bottom", fontsize=7.2, color=GOLD, fontweight="bold")
mk = [(xn(1), RED, "1"), (xn(4) + 0.17, GREEN, "4"), (xn(8), BLUE, "8")]
for mx, col, num in mk:
    ax.add_patch(Circle((mx, gy), 0.075, facecolor=col, edgecolor="white", linewidth=1.2, zorder=3))
    ax.text(mx, gy + 0.20, num, ha="center", va="bottom", fontsize=6.8, color=col, fontweight="bold")
ax.text(x_lo, gy - 0.30, "too little evidence", ha="left", va="center", fontsize=7.2, color=ANNO)
ax.text(xn(4), gy - 0.30, "appropriate depth", ha="center", va="center", fontsize=7.2, color=GREEN)
ax.text(x_hi, gy - 0.30, "excessive work-up", ha="right", va="center", fontsize=7.2, color=ANNO)
ax.text(xn(4) + 0.17, gy + 0.20, "", ha="center")  # (green "4" label already drawn above)

ax.plot([0.5, 13.5], [by - 6.95, by - 6.95], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL C =====================================
cyh = by - 7.28
head(0.5, cyh, "C", "Quantitative confirmation")

METHODS = ["Base†", "Plain", "Credit", "Decoupled†"]
MCOLORS = [GRAY, RED, GREEN, BLUE]
SINGLE = [True, False, False, True]

def mini_bars(rect, title, vals, errs, ylim, yticks, fmt, hline=None, hlabel=None):
    axb = fig.add_axes(rect)
    xs = np.arange(4)
    bars = axb.bar(xs, vals, width=0.66, color=MCOLORS, zorder=2,
                   yerr=[[0]*4, errs], capsize=3, error_kw=dict(ecolor="#4a525c", lw=1.1))
    for i, b in enumerate(bars):
        if SINGLE[i]:
            b.set_hatch("////"); b.set_edgecolor("white"); b.set_linewidth(0.0)
    if hline is not None:
        axb.axhline(hline, ls=(0, (4, 3)), lw=1.4, color=GOLD, zorder=3)
        axb.text(-0.44, hline + ylim[1]*0.015, hlabel, ha="left", va="bottom", fontsize=6.8, color=GOLD)
    for xi, v, e in zip(xs, vals, errs):
        axb.text(xi, v + ylim[1]*0.035 + (e or 0), fmt.format(v), ha="center", va="bottom",
                 fontsize=7.2, color=INK, fontweight="bold")
    axb.set_ylim(*ylim); axb.set_yticks(yticks)
    axb.set_xticks(xs); axb.set_xticklabels(METHODS, fontsize=8.0)
    axb.tick_params(axis="y", labelsize=7.0, length=2.5); axb.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#c4c9cf")
    axb.set_title(title, fontsize=10, color=INK, fontweight="bold", pad=7)
    axb.margins(x=0.08)

bt, bh = 0.090, 0.110
mini_bars([0.128, bt, 0.335, bh], "High-risk value",
          [0.160, 0.098, 0.157, 0.248], [0, 0.037, 0.057, 0], (0, 0.31), [0, 0.1, 0.2, 0.3], "{:.3f}")
mini_bars([0.560, bt, 0.335, bh], "Mean questions asked",
          [2.50, 1.82, 2.08, 4.40], [0, 0.11, 0.30, 0], (0, 5.2), [0, 2, 4], "{:.2f}",
          hline=3.9, hlabel="oracle ≈ 3.9")

ax.text(FIG_W / 2, 0.60, "Held-out diagnostics; pure verifier at evaluation.   High-risk value = rare emergent "
        "subset.   † single-seed reference;  Plain / Credit = mean ± SD, n = 3 seeds.",
        ha="center", va="center", fontsize=8.0, color=ANNO)

# ================================ SAVE ======================================
for ext in ("svg", "pdf", "png"):
    out = f"figure1_overview_v4.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
