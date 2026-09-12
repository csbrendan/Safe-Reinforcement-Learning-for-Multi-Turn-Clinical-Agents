#!/usr/bin/env python3
"""Figure 1 (v2) — journal-style opening figure for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

Design goal: show, visually and clinically, that the SAME patient (dx-13, acute
abdominal pain, hidden emergent severity) leads to qualitatively different multi-turn
procedures depending on reward design, and that the *procedure* matters separately
from the final score.

Panels:
  A  the multi-turn clinical problem (hidden findings, ask-one-at-a-time, cost, stop)
  B  reference adaptive procedure = the exact oracle for dx-13 (gold, distinct)
  C  what different reward designs learn on the SAME patient (3 rows; length = burden)
  D  reward geometry (plain-language) + a small 2-metric quantitative anchor

All trajectories/values are verified in figure1_source_material.md and re-checked
against the stored probe records (see figure1_v2_design_notes.md for exact sources).
Code keys are translated to physician-readable labels; implementation jargon is kept small.

The drawing is grouped panel-by-panel with named components so it can be polished in Figma.
Outputs: figure1_overview_v2.{svg,pdf,png}.  Run inside ~/spark1/.venv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge

# ---- editable-text vector output + journal-ish sans font ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ---- restrained palette ----
INK   = "#233140"; MUTE = "#7a828c"; FAINT = "#aeb4bc"; HAIR = "#e4e7ea"
GRAY  = "#8a8f98"                                   # base / reference method
RED   = "#c14b5a"                                   # outcome-only GRPO
GREEN = "#2e7d32"                                   # credit-assignment
BLUE  = "#3b6fb6"                                   # decoupled explore/exploit
GOLD  = "#b7902e"; GOLD_L = "#f7efd6"               # exact oracle (distinct, non-method)
TEAL  = "#0f8a7e"                                   # "present" finding marker
PANEL_BG = "#fbfcfd"

FIG_W, FIG_H = 14.4, 15.2

# physician-readable labels (translations of code keys) ------------------------
LAB = {
    "abd_rigidity":  "Abdominal\nrigidity",
    "rebound_tender":"Rebound\ntenderness",
    "ct_free_air":   "CT free\nair",
    "lactate_high":  "Lactate\nelevated",
    "guarding":      "Guarding",
    "radiation_arm": "Radiating\npain",
    "dyspnea_severe":"Severe\ndyspnea",
    "hypotension":   "Blood\npressure",
    "wbc_high":      "White\ncell count",
    "ecg_ischemia":  "ECG",
    "fever_high":    "Fever",
}
ACT = {"emergent_tx": "Emergency\ntreatment", "admit": "Admit", "observe": "Observe"}
# verified present(1)/absent(0) for dx-13 (env_rv gen_cases(60,0.0,seed=100))
PRESENT = {"abd_rigidity":1,"rebound_tender":1,"ct_free_air":1,"lactate_high":1,"guarding":1,
           "radiation_arm":1,"dyspnea_severe":1,"hypotension":0,"wbc_high":0,"ecg_ischemia":0,"fever_high":1}

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")


# ============================ shared components ==============================
def rrect(x, y, w, h, fc="white", ec=HAIR, lw=1.1, r=0.06, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color=FAINT, lw=1.5, rad=0.0, style="-|>", z=1.6, ms=11):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw,
                 color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))

def panel_head(x, y, letter, title, sub=None):
    ax.text(x, y, letter, ha="left", va="top", fontsize=16, fontweight="bold", color=INK)
    ax.text(x + 0.52, y - 0.02, title, ha="left", va="top", fontsize=12.5, fontweight="bold", color=INK)
    if sub:
        ax.text(x + 0.52, y - 0.36, sub, ha="left", va="top", fontsize=9.2, color=MUTE)

def patient_glyph(cx, cy, s=1.0, color="#b8bec6"):
    ax.add_patch(Circle((cx, cy + 0.16 * s), 0.17 * s, facecolor=color, edgecolor="none", zorder=3))
    ax.add_patch(Wedge((cx, cy - 0.14 * s), 0.34 * s, 20, 160, facecolor=color, edgecolor="none", zorder=3))

# CW/CH/GAP fixed so trajectory columns align across panels B and C -> length = burden
CW, CH, GAP = 0.88, 0.66, 0.20
TRACK_X0 = 3.20
def chip_x(i):
    return TRACK_X0 + CW / 2 + i * (CW + GAP)

def finding_chip(cx, cy, key, present, hidden=False):
    if hidden:
        rrect(cx - CW / 2, cy - CH / 2, CW, CH, fc="#eef0f2", ec="#d3d7db", r=0.07, z=2.4)
        ax.text(cx, cy + 0.10, "?", ha="center", va="center", fontsize=15, color="#b3b9c0",
                fontweight="bold", zorder=3)
        ax.text(cx, cy - 0.20, LAB.get(key, key).replace("\n", " "), ha="center", va="center",
                fontsize=5.0, color="#9aa0a8", zorder=3)
        return
    rrect(cx - CW / 2, cy - CH / 2, CW, CH, fc="white", ec="#c7ccd2", r=0.07, z=2.4)
    ax.text(cx, cy + 0.09, LAB.get(key, key), ha="center", va="center", fontsize=6.4,
            color=INK, zorder=3, linespacing=0.95)
    mc = TEAL if present else "#b8bec6"
    ax.add_patch(Circle((cx - 0.26, cy - 0.20), 0.045, facecolor=mc,
                 edgecolor="none", zorder=3.2) if present else
                 Circle((cx - 0.26, cy - 0.20), 0.045, facecolor="white", edgecolor="#b8bec6",
                        linewidth=1.0, zorder=3.2))
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


def trajectory(y, keys, action_key, action_color, reward, method, method_color, notes,
               oracle=False):
    """One aligned trajectory row. Chips grow left->right (length encodes burden),
    then final action pill inline, then reward. Left: colored accent + method label."""
    # left accent + method label
    ax.add_patch(FancyBboxPatch((0.55, y - 0.34), 0.10, 0.68, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor=(GOLD if oracle else method_color), zorder=3))
    ax.text(0.78, y + 0.15, method, ha="left", va="center", fontsize=9.6, fontweight="bold",
            color=(GOLD if oracle else method_color))
    ax.text(0.78, y - 0.16, notes[0], ha="left", va="center", fontsize=7.4, color=MUTE)
    # chips + connecting arrows
    n = len(keys)
    for i, k in enumerate(keys):
        cx = chip_x(i)
        finding_chip(cx, y, k, PRESENT.get(k, 1))
        if i > 0:
            arrow((chip_x(i - 1) + CW / 2, y), (cx - CW / 2, y), color="#c3c8ce", lw=1.2, ms=8)
    last_r = chip_x(n - 1) + CW / 2
    # arrow to action
    apx = last_r + 0.62
    arrow((last_r, y), (apx - 0.64, y), color="#c3c8ce", lw=1.3, ms=9)
    right = action_pill(apx, y, action_key, action_color if not oracle else GOLD)
    # reward
    reward_tag(right + 0.62, y, reward, action_color if not oracle else GOLD)
    # secondary notes under the row (kept short)
    if len(notes) > 1:
        ax.text(0.78, y - 0.44, notes[1], ha="left", va="top", fontsize=6.8, color=FAINT)


# ================================ TITLE =====================================
ax.text(FIG_W / 2, 14.85, "Reward Design Governs Information Gathering in Clinical Multi-Turn Agents",
        ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
ax.text(FIG_W / 2, 14.46, "The same patient yields different multi-turn procedures under different "
        "reward designs — and the procedure matters apart from the final score.",
        ha="center", va="center", fontsize=10.2, color=MUTE)

# =============================== PANEL A ====================================
# region ~ y 11.05 .. 14.0
panel_head(0.5, 14.05, "A", "The multi-turn clinical problem")

# -- patient (left) --
patient_glyph(1.15, 13.25, s=1.15)
ax.text(1.15, 12.62, "Adult with acute\nabdominal pain", ha="center", va="top", fontsize=8.8,
        color=INK, fontweight="bold", linespacing=1.0)
ax.text(1.15, 12.05, "no test results yet", ha="center", va="top", fontsize=8.0, color=MUTE)
rrect(0.35, 11.42, 1.62, 0.36, fc="#fdeef0", ec="#eccace", r=0.08, z=2)
ax.text(1.16, 11.60, "severity: hidden", ha="center", va="center", fontsize=7.6, color=RED)

# -- hidden findings (middle): 2 x 3 face-down cards --
ax.text(5.05, 13.64, "The agent may order any finding — each result is hidden until asked",
        ha="center", va="center", fontsize=8.4, color=MUTE)
hidden = ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high", "fever_high", "wbc_high"]
hx0, hy0, hw, hh, hgx, hgy = 2.75, 12.55, 1.32, 0.78, 0.18, 0.24
for idx, k in enumerate(hidden):
    r, c = divmod(idx, 3)
    cx = hx0 + hw / 2 + c * (hw + hgx)
    cy = hy0 + hh / 2 - r * (hh + hgy) + 0.10
    rrect(cx - hw / 2, cy - hh / 2, hw, hh, fc="#eef0f2", ec="#d3d7db", r=0.08, z=2.2)
    ax.text(cx, cy + 0.12, "?", ha="center", va="center", fontsize=14, color="#b3b9c0", fontweight="bold")
    ax.text(cx, cy - 0.20, LAB[k].replace("\n", " "), ha="center", va="center", fontsize=5.8, color="#9aa0a8")

# -- loop mechanic (right) --
lx = 9.05
ax.text(11.55, 13.92, "Ask one finding at a time, then decide when to stop",
        ha="center", va="center", fontsize=8.4, color=MUTE)
nodes = [("Ask", 9.55), ("Result", 11.05), ("Commit", 12.55)]
ny = 12.85
for name, nx in nodes:
    ax.add_patch(Circle((nx, ny), 0.30, facecolor="white", edgecolor="#c7ccd2", linewidth=1.4, zorder=2.6))
    ax.text(nx, ny, name, ha="center", va="center", fontsize=7.2, color=INK, zorder=3)
arrow((9.85, ny), (10.75, ny), color=FAINT)
arrow((11.35, ny), (12.25, ny), color=FAINT)
arrow((11.05, ny + 0.30), (9.55, ny + 0.30), color=GREEN, rad=-0.55)
ax.text(10.30, ny + 0.72, "ask again  (−0.03 each)", ha="center", va="center", fontsize=7.2, color=GREEN)
ax.text(11.90, ny - 0.30, "stop", ha="center", va="center", fontsize=6.8, color=MUTE)
ax.add_patch(Circle((12.55, 12.05), 0.30, facecolor="#f4f6f8", edgecolor="#c7ccd2", linewidth=1.4, zorder=2.6))
ax.text(12.55, 12.05, "Reward", ha="center", va="center", fontsize=7.0, color=INK, zorder=3)
arrow((12.55, ny - 0.30), (12.55, 12.35), color=FAINT)
rrect(8.95, 11.42, 5.15, 0.44, fc="#f4f6f8", ec=HAIR, r=0.08, z=2)
ax.text(11.52, 11.64, "R  =  clinical utility  −  0.03 × questions asked", ha="center", va="center",
        fontsize=9.2, color=INK, fontweight="bold")

ax.plot([0.4, 14.0], [10.95, 10.95], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL B ====================================
# oracle reference row, aligned to panel C columns (y ~ 9.9)
panel_head(0.5, 10.72, "B", "Reference adaptive procedure",
           sub="exact adaptive oracle for this patient — a distinct reference, not a learned policy")
oy = 9.85
# soft gold band behind the oracle row
rrect(0.45, oy - 0.52, 13.55, 1.04, fc=GOLD_L, ec="#ecdcae", r=0.10, z=1.2)
trajectory(oy, ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high"],
           "emergent_tx", GOLD, 0.83,
           "Oracle", GOLD,
           ["4 targeted findings → emergency treatment", "the desirable gather-then-commit reference"],
           oracle=True)

# =============================== PANEL C ====================================
panel_head(0.5, 9.02, "C", "What different reward designs learn",
           sub="same patient · same hidden severity · same final verifier — only the training reward differs")
r1, r2, r3 = 8.05, 6.70, 5.35
trajectory(r1, ["abd_rigidity"], "admit", RED, 0.17,
           "Outcome-only", RED,
           ["stops after 1 finding · premature commitment", "under-gathering — misses the emergency"])
trajectory(r2, ["abd_rigidity", "guarding", "rebound_tender", "ct_free_air"], "emergent_tx", GREEN, 0.83,
           "Credit-assignment", GREEN,
           ["successful sampled trajectory · oracle-grade by reward",
            "not identical to the oracle evidence sequence"])
trajectory(r3, ["abd_rigidity", "rebound_tender", "guarding", "radiation_arm", "dyspnea_severe",
                "hypotension", "wbc_high", "ecg_ischemia"], "emergent_tx", BLUE, 0.71,
           "Decoupled", BLUE,
           ["correct final action · excessive work-up", "8 findings vs oracle 4 — cost lowers the reward"])

# present/normal legend (top-right of C)
lx0 = 10.9
ax.add_patch(Circle((lx0, 9.02), 0.045, facecolor=TEAL, edgecolor="none", zorder=3))
ax.text(lx0 + 0.10, 9.02, "present", ha="left", va="center", fontsize=6.8, color=MUTE)
ax.add_patch(Circle((lx0 + 1.05, 9.02), 0.045, facecolor="white", edgecolor="#b8bec6", lw=1.0, zorder=3))
ax.text(lx0 + 1.15, 9.02, "normal", ha="left", va="center", fontsize=6.8, color=MUTE)

# -- evidence-burden axis under the rows (shares x with chips) --
gy = 4.45
def xn(n):  # x of the n-th finding column
    return chip_x(n - 1)
x_lo, x_hi = xn(1) - 0.5, xn(8) + 0.5
rrect(x_lo, gy - 0.09, x_hi - x_lo, 0.18, fc="#f0f2f4", ec="none", r=0.09, z=1.3)
# appropriate-depth green band around oracle (~3-5)
rrect(xn(3) - 0.2, gy - 0.09, (xn(5) + 0.2) - (xn(3) - 0.2), 0.18, fc="#dcecdd", ec="none", r=0.09, z=1.35)
# under / over red tints
rrect(x_lo, gy - 0.09, (xn(2)) - x_lo, 0.18, fc="#f6e2e4", ec="none", r=0.09, z=1.34)
rrect(xn(6), gy - 0.09, x_hi - xn(6), 0.18, fc="#f6e2e4", ec="none", r=0.09, z=1.34)
for n, col in [(1, RED), (4, GREEN), (8, BLUE)]:
    ax.add_patch(Circle((xn(n), gy), 0.085, facecolor=col, edgecolor="white", linewidth=1.2, zorder=3))
ax.plot([xn(4), xn(4)], [gy - 0.22, gy + 0.22], color=GOLD, lw=1.6, zorder=2.8)
ax.text(xn(4), gy + 0.34, "oracle 4", ha="center", va="center", fontsize=6.8, color=GOLD, fontweight="bold")
ax.text(x_lo - 0.05, gy - 0.34, "too few questions", ha="left", va="center", fontsize=7.2, color=RED)
ax.text((xn(4)), gy - 0.34, "appropriate depth", ha="center", va="center", fontsize=7.2, color=GREEN)
ax.text(x_hi + 0.05, gy - 0.34, "too many questions", ha="right", va="center", fontsize=7.2, color=RED)

ax.plot([0.4, 14.0], [3.98, 3.98], color=HAIR, lw=1.0, zorder=0.5)

# =============================== PANEL D ====================================
panel_head(0.5, 3.78, "D", "Reward geometry, and how the procedure shifts")

def geom(y, name, color, desc):
    ax.add_patch(FancyBboxPatch((0.55, y - 0.10), 0.10, 0.20, boxstyle="round,pad=0,rounding_size=0.03",
                 linewidth=0, facecolor=color, zorder=3))
    ax.text(0.80, y + 0.12, name, ha="left", va="center", fontsize=8.8, fontweight="bold", color=color)
    ax.text(0.80, y - 0.14, desc, ha="left", va="center", fontsize=7.4, color=MUTE)

geom(3.18, "Outcome-only", RED, "questions get no separate credit — final outcome reward only")
geom(2.50, "Credit-assignment", GREEN, "informative questions get intermediate credit; final check unchanged")
geom(1.82, "Decoupled", BLUE, "gathering and committing are optimized separately")
ax.text(0.55, 1.18, "small quantitative anchor — diagnostics, held-out, pure verifier (k = 12)",
        ha="left", va="center", fontsize=7.6, color=FAINT, style="italic")

# --- two mini bar charts (real axes, vector) ---
METHODS = ["Base", "Plain", "Credit", "Decoupled"]
MCOLORS = [GRAY, RED, GREEN, BLUE]

def mini_bars(rect, title, vals, errs, ylim, yticks, hline=None, hlabel=None, fmt="{:.2f}"):
    axb = fig.add_axes(rect)
    xs = np.arange(4)
    axb.bar(xs, vals, width=0.66, color=MCOLORS, zorder=2,
            yerr=[[0]*4, errs], capsize=2.5, error_kw=dict(ecolor="#556", lw=1.0))
    if hline is not None:
        axb.axhline(hline, ls=(0, (4, 3)), lw=1.3, color=GOLD, zorder=3)
        axb.text(-0.42, hline + 0.06, hlabel, ha="left", va="bottom", fontsize=6.6, color=GOLD)
    for xi, v, e in zip(xs, vals, errs):
        axb.text(xi, v + (ylim[1] * 0.03) + (e or 0), fmt.format(v), ha="center", va="bottom",
                 fontsize=6.6, color=INK)
    axb.set_ylim(*ylim); axb.set_yticks(yticks)
    axb.set_xticks(xs); axb.set_xticklabels(METHODS, fontsize=6.8)
    axb.tick_params(axis="y", labelsize=6.4, length=2); axb.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#c7ccd2")
    axb.set_title(title, fontsize=8.2, color=INK, fontweight="bold", pad=5)
    axb.margins(x=0.08)

# figure-fraction rects for the two charts (right two-thirds of D)
mini_bars([0.415, 0.108, 0.24, 0.123], "High-risk support  (↑ safer)",
          [0.50, 0.33, 0.50, 1.00], [0, 0.29, 0.0, 0], (0, 1.15), [0, 0.5, 1.0])
mini_bars([0.715, 0.108, 0.245, 0.123], "Questions asked  (mean)",
          [2.5, 1.82, 2.08, 4.4], [0, 0.11, 0.30, 0], (0, 5.2), [0, 2, 4],
          hline=3.9, hlabel="oracle ≈ 3.9", fmt="{:.2f}")

# =============================== FOOTER (three tasks) =======================
ax.plot([0.4, 14.0], [0.86, 0.86], color=HAIR, lw=1.0, zorder=0.5)
tasks = [("Refill triage", "Continue, modify, hold, or escalate a medication?", MUTE, False),
         ("Diagnostics", "Gather enough evidence to safely disposition an acute presentation", INK, True),
         ("Titration", "Choose treatment intensity while balancing competing risks", MUTE, False)]
tx = [0.55, 5.35, 10.15]
for (name, desc, col, run), x0 in zip(tasks, tx):
    tag = name + ("  · running example" if run else "")
    ax.text(x0, 0.60, tag, ha="left", va="center", fontsize=8.0,
            fontweight="bold", color=(GREEN if run else col))
    ax.text(x0, 0.32, desc, ha="left", va="center", fontsize=7.2, color=MUTE)

# ================================ SAVE ======================================
for ext in ("svg", "pdf", "png"):
    out = f"figure1_overview_v2.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
