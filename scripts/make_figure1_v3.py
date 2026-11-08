#!/usr/bin/env python3
"""Figure 1 (v3) — simplified 3-panel opening figure for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

Reads as one story:  incomplete patient  ->  same patient, different learned procedures  ->
quantitative confirmation.  Simplification of v2 (scripts/make_figure1_v2.py); same verified data.

  A  the clinical information-gathering problem  (~22% height)
  B  same patient, different procedures — oracle + 3 learned rows  (dominant, ~52%)
  C  quantitative confirmation — high-risk value + mean asks  (~22%)

Clean publication export has NO title (starts at panel A). Set env FIG1_TITLE=1 for a titled preview.
All trajectories/values verified in figure1_v3_design_notes.md against stored probe records.
Outputs: figure1_overview_v3.{svg,pdf,png}.  Run inside ~/spark1/.venv.
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
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

INK   = "#233140"; MUTE = "#7a828c"; FAINT = "#aeb4bc"; HAIR = "#e4e7ea"
GRAY  = "#8a8f98"
RED   = "#c14b5a"
GREEN = "#2e7d32"
BLUE  = "#3b6fb6"
GOLD  = "#b7902e"; GOLD_L = "#f7efd6"
TEAL  = "#0f8a7e"

TITLED = os.environ.get("FIG1_TITLE") == "1"
FIG_W, FIG_H = 14.0, (13.2 if TITLED else 12.6)
YTOP = FIG_H - (0.85 if TITLED else 0.25)   # top of panel A content

LAB = {
    "abd_rigidity":  "Abdominal\nrigidity", "rebound_tender":"Rebound\ntenderness",
    "ct_free_air":   "CT free\nair",        "lactate_high":  "Lactate\nelevated",
    "guarding":      "Guarding",            "radiation_arm": "Radiating\npain",
    "dyspnea_severe":"Severe\ndyspnea",     "hypotension":   "Blood\npressure",
    "wbc_high":      "White cell\ncount",   "ecg_ischemia":  "ECG",  "fever_high": "Fever",
}
ACT = {"emergent_tx": "Emergency\ntreatment", "admit": "Admit"}
PRESENT = {"abd_rigidity":1,"rebound_tender":1,"ct_free_air":1,"lactate_high":1,"guarding":1,
           "radiation_arm":1,"dyspnea_severe":1,"hypotension":0,"wbc_high":0,"ecg_ischemia":0,"fever_high":1}

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


def rrect(x, y, w, h, fc="white", ec=HAIR, lw=1.1, r=0.06, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color=FAINT, lw=1.5, rad=0.0, z=1.6, ms=10):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))

def panel_head(x, y, letter, title, sub=None):
    ax.text(x, y, letter, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    ax.text(x + 0.52, y - 0.02, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    if sub:
        ax.text(x + 0.53, y - 0.38, sub, ha="left", va="top", fontsize=9.2, color=MUTE)

def patient_glyph(cx, cy, s=1.0, color="#b8bec6"):
    ax.add_patch(Circle((cx, cy + 0.16 * s), 0.17 * s, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Wedge((cx, cy - 0.14 * s), 0.34 * s, 20, 160, facecolor=color, edgecolor="none", zorder=3))

CW, CH, GAP = 0.88, 0.66, 0.20
TRACK_X0 = 3.25
def chip_x(i):
    return TRACK_X0 + CW / 2 + i * (CW + GAP)

def finding_chip(cx, cy, key, present):
    rrect(cx - CW / 2, cy - CH / 2, CW, CH, fc="white", ec="#c7ccd2", r=0.07, z=2.4)
    ax.text(cx, cy + 0.09, LAB.get(key, key), ha="center", va="center", fontsize=6.4,
            color=INK, zorder=3, linespacing=0.95)
    mc = TEAL if present else "#b8bec6"
    ax.add_patch(Circle((cx - 0.26, cy - 0.20), 0.045, facecolor=(mc if present else "white"),
                 edgecolor=("none" if present else "#b8bec6"), linewidth=1.0, zorder=3.2))
    ax.text(cx - 0.17, cy - 0.20, "present" if present else "normal", ha="left", va="center",
            fontsize=5.2, color=(TEAL if present else MUTE), zorder=3.2)

def action_pill(cx, cy, key, color, w=1.28, h=0.62):
    rrect(cx - w / 2, cy - h / 2, w, h, fc="white", ec=color, lw=1.8, r=0.14, z=2.6)
    ax.text(cx, cy, ACT.get(key, key), ha="center", va="center", fontsize=7.2, color=color,
            fontweight="bold", zorder=3, linespacing=0.95)
    return cx + w / 2

def reward_tag(cx, cy, value, color):
    ax.text(cx, cy + 0.11, f"{value:+.2f}", ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=color, zorder=3)
    ax.text(cx, cy - 0.22, "reward", ha="center", va="center", fontsize=6.6, color=MUTE, zorder=3)

def trajectory(y, keys, action_key, reward, method, method_color, notes, oracle=False):
    col = GOLD if oracle else method_color
    ax.add_patch(FancyBboxPatch((0.55, y - 0.34), 0.10, 0.68, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor=col, zorder=3))
    ax.text(0.78, y + 0.15, method, ha="left", va="center", fontsize=9.6, fontweight="bold", color=col)
    ax.text(0.78, y - 0.16, notes[0], ha="left", va="center", fontsize=7.4, color=MUTE)
    n = len(keys)
    for i, k in enumerate(keys):
        cx = chip_x(i)
        finding_chip(cx, y, k, PRESENT.get(k, 1))
        if i > 0:
            arrow((chip_x(i - 1) + CW / 2, y), (cx - CW / 2, y), color="#c3c8ce", lw=1.2, ms=8)
    last_r = chip_x(n - 1) + CW / 2
    apx = last_r + 0.62
    arrow((last_r, y), (apx - 0.64, y), color="#c3c8ce", lw=1.3, ms=9)
    right = action_pill(apx, y, action_key, col)
    reward_tag(right + 0.62, y, reward, col)
    if len(notes) > 1:
        ax.text(0.78, y - 0.44, notes[1], ha="left", va="top", fontsize=6.8, color=FAINT)


# ================================ TITLE (preview only) =======================
if TITLED:
    ax.text(FIG_W / 2, FIG_H - 0.45, "Reward Design Governs Information Gathering in Clinical "
            "Multi-Turn Agents", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)

# =============================== PANEL A ====================================
panel_head(0.5, YTOP, "A", "The clinical information-gathering problem")
ay = YTOP - 1.05                      # patient/cards baseline

# -- patient (left) --
patient_glyph(1.15, ay + 0.10, s=1.15)
ax.text(1.15, ay - 0.55, "Adult with acute\nabdominal pain", ha="center", va="top", fontsize=9.0,
        color=INK, fontweight="bold", linespacing=1.0)
rrect(0.30, ay - 1.42, 1.72, 0.38, fc="#fdeef0", ec="#eccace", r=0.09, z=2)
ax.text(1.16, ay - 1.23, "severity: hidden from agent", ha="center", va="center", fontsize=7.4, color=RED)

# -- findings: one being revealed + five covered --
def hidden_card(cx, cy, key, w, h):
    rrect(cx - w / 2, cy - h / 2, w, h, fc="#eef0f2", ec="#d3d7db", r=0.08, z=2.2)
    ax.text(cx, cy + 0.16, "?", ha="center", va="center", fontsize=15, color="#b3b9c0", fontweight="bold")
    ax.text(cx, cy - 0.24, LAB[key].replace("\n", " "), ha="center", va="center", fontsize=6.0, color="#9aa0a8")

def reveal_card(cx, cy, key, w, h):
    rrect(cx - w / 2, cy - h / 2, w, h, fc="white", ec=TEAL, lw=1.8, r=0.08, z=2.6)
    ax.text(cx, cy + 0.26, LAB[key].replace("\n", " "), ha="center", va="center", fontsize=6.6,
            color=INK, fontweight="bold")
    ax.text(cx - 0.42, cy - 0.16, "?", ha="center", va="center", fontsize=13, color="#b3b9c0", fontweight="bold")
    arrow((cx - 0.22, cy - 0.16), (cx + 0.02, cy - 0.16), color=TEAL, lw=1.4, ms=9)
    rrect(cx + 0.06, cy - 0.34, 0.62, 0.36, fc="#e5f3f0", ec=TEAL, r=0.07, z=2.8)
    ax.text(cx + 0.37, cy - 0.16, "Present", ha="center", va="center", fontsize=6.6, color=TEAL, fontweight="bold")

cw, ch, cg = 1.62, 0.98, 0.16
cards = ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high", "fever_high", "wbc_high"]
cx0 = 2.75
for i, k in enumerate(cards):
    cx = cx0 + cw / 2 + i * (cw + cg)
    if i == 0:
        reveal_card(cx, ay, k, cw, ch)
    else:
        hidden_card(cx, ay, k, cw, ch)
ax.text(0.5 * (cx0 + cw / 2 + (cx0 + cw / 2 + 5 * (cw + cg))), ay - 0.86,
        "Choose which finding to reveal; each request costs 0.03, then commit when enough evidence is available.",
        ha="center", va="center", fontsize=8.6, color=MUTE)

ax.plot([0.4, 13.6], [YTOP - 2.55, YTOP - 2.55], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL B ====================================
by = YTOP - 2.95
panel_head(0.5, by, "B", "Same patient, different information-gathering procedures",
           sub="same presentation · same hidden severity · same final verifier · only the training reward differs")
# present/normal legend (top-right)
lx0 = 10.75
ax.add_patch(Circle((lx0, by - 0.12), 0.045, facecolor=TEAL, edgecolor="none", zorder=3))
ax.text(lx0 + 0.10, by - 0.12, "present", ha="left", va="center", fontsize=6.8, color=MUTE)
ax.add_patch(Circle((lx0 + 1.05, by - 0.12), 0.045, facecolor="white", edgecolor="#b8bec6", lw=1.0, zorder=3))
ax.text(lx0 + 1.15, by - 0.12, "normal", ha="left", va="center", fontsize=6.8, color=MUTE)

r0, r1, r2, r3 = by - 1.30, by - 2.55, by - 3.80, by - 5.05
trajectory(r0, ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high"], "emergent_tx", 0.83,
           "Exact adaptive oracle", GOLD,
           ["4 targeted findings → emergency treatment", "reference procedure, not a learned policy"], oracle=True)
trajectory(r1, ["abd_rigidity"], "admit", 0.17, "Outcome-only GRPO", RED,
           ["1 finding · premature commitment", "under-gathering — misses the emergency"])
trajectory(r2, ["abd_rigidity", "guarding", "rebound_tender", "ct_free_air"], "emergent_tx", 0.83,
           "Credit-assignment GRPO", GREEN,
           ["4 findings · successful sampled trajectory",
            "oracle-grade by reward; evidence sequence differs from oracle"])
trajectory(r3, ["abd_rigidity", "rebound_tender", "guarding", "radiation_arm", "dyspnea_severe",
                "hypotension", "wbc_high", "ecg_ischemia"], "emergent_tx", 0.71,
           "Decoupled explore/exploit", BLUE,
           ["8 findings vs oracle 4 · correct action, excessive work-up", "reward reduced by question cost"])

# small conceptual scale (kept subtle)
gy = by - 5.95
def xn(n):
    return chip_x(n - 1)
x_lo, x_hi = xn(1) - 0.45, xn(8) + 0.45
rrect(x_lo, gy - 0.055, x_hi - x_lo, 0.11, fc="#f0f2f4", ec="none", r=0.055, z=1.3)
rrect(xn(3) - 0.18, gy - 0.055, (xn(5) + 0.18) - (xn(3) - 0.18), 0.11, fc="#dcecdd", ec="none", r=0.055, z=1.35)
for n, col in [(1, RED), (4, GREEN), (8, BLUE)]:
    ax.add_patch(Circle((xn(n), gy), 0.07, facecolor=col, edgecolor="white", linewidth=1.1, zorder=3))
ax.plot([xn(4), xn(4)], [gy - 0.17, gy + 0.17], color=GOLD, lw=1.5, zorder=2.8)
ax.text(x_lo - 0.05, gy - 0.30, "too little evidence", ha="left", va="center", fontsize=6.8, color=MUTE)
ax.text(xn(4), gy - 0.30, "appropriate depth", ha="center", va="center", fontsize=6.8, color=GREEN)
ax.text(x_hi + 0.05, gy - 0.30, "excessive work-up", ha="right", va="center", fontsize=6.8, color=MUTE)

ax.plot([0.4, 13.6], [by - 6.45, by - 6.45], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL C ====================================
cyh = by - 6.72
panel_head(0.5, cyh, "C", "Quantitative confirmation")

METHODS = ["Base", "Plain", "Credit", "Decoupled"]
MCOLORS = [GRAY, RED, GREEN, BLUE]

def mini_bars(rect, title, vals, errs, ylim, yticks, fmt, hline=None, hlabel=None):
    axb = fig.add_axes(rect)
    xs = np.arange(4)
    axb.bar(xs, vals, width=0.66, color=MCOLORS, zorder=2,
            yerr=[[0]*4, errs], capsize=2.5, error_kw=dict(ecolor="#556", lw=1.0))
    if hline is not None:
        axb.axhline(hline, ls=(0, (4, 3)), lw=1.3, color=GOLD, zorder=3)
        axb.text(-0.42, hline + ylim[1]*0.015, hlabel, ha="left", va="bottom", fontsize=6.6, color=GOLD)
    for xi, v, e in zip(xs, vals, errs):
        axb.text(xi, v + ylim[1]*0.03 + (e or 0), fmt.format(v), ha="center", va="bottom",
                 fontsize=6.8, color=INK)
    axb.set_ylim(*ylim); axb.set_yticks(yticks)
    axb.set_xticks(xs); axb.set_xticklabels(METHODS, fontsize=7.2)
    axb.tick_params(axis="y", labelsize=6.6, length=2); axb.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#c7ccd2")
    axb.set_title(title, fontsize=9.2, color=INK, fontweight="bold", pad=6)
    axb.margins(x=0.08)

# figure-fraction rects (two compact plots)
bt = 0.072; bh = 0.108
mini_bars([0.128, bt, 0.335, bh], "High-risk value",
          [0.160, 0.098, 0.157, 0.248], [0, 0.037, 0.057, 0], (0, 0.31), [0, 0.1, 0.2, 0.3], "{:.3f}")
mini_bars([0.560, bt, 0.335, bh], "Mean questions asked",
          [2.50, 1.82, 2.08, 4.40], [0, 0.11, 0.30, 0], (0, 5.2), [0, 2, 4], "{:.2f}",
          hline=3.9, hlabel="oracle ≈ 3.9")

ax.text(FIG_W / 2, 0.52, "Held-out diagnostics · pure verifier · no shaping at evaluation.   "
        "Plain and credit-assignment: mean ± SD across 3 seeds;  base and decoupled: single-seed references.",
        ha="center", va="center", fontsize=7.4, color=MUTE)
ax.text(FIG_W / 2, 0.30, "High-risk value = the rare emergent subset — the safety-critical metric.",
        ha="center", va="center", fontsize=7.4, color=FAINT, style="italic")

# ================================ SAVE ======================================
for ext in ("svg", "pdf", "png"):
    out = f"figure1_overview_v3.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
