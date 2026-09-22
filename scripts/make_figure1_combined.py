#!/usr/bin/env python3
"""Figure 1 (combined) — "Safe Reinforcement Learning for Clinical Multi-Turn Agents
Requires More Than Outcome Rewards."

Merges two previously developed graphics into ONE cohesive full-page figure, rebuilt from
editable vector primitives (not stacked PNGs):

  A  Experimental setup      <- recomposed from scripts/make_experiment_overview_strip_v2.py
                                (three zones: clinical environment | GRPO training | held-out evaluation)
  B  Same patient, different decision procedures  <- recomposed from scripts/make_figure1_v4.py Panel B
                                (dx-13, oracle reference + 3 learned trajectories)   [CENTREPIECE]
  C  Quantitative diagnostics results             <- recomposed from scripts/make_figure1_v4.py Panel C

Harmonisation: one palette (the strip-v2 palette wins on INK/GOLD/TEAL), one type scale, one set of
box/pill/badge primitives, one arrow weight vocabulary, shared method colours across A/B/C.

All Panel B trajectories and Panel C values are re-verified against the stored probe records and
recomputed with tasks.diagnostics.env_rv.score(); see figs/figure1_combined_design_notes.md.

Outputs: figs/figure1_combined.{png,svg,pdf}.  Run inside ~/spark1/.venv.
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

# ============================== unified palette ==============================
# (strip_v2 is the design reference; v4's near-duplicates are folded into these)
INK   = "#22303c"   # primary type
MUTE  = "#7c848d"   # secondary type
FAINT = "#a3aab2"   # tertiary type / subordinate inset
HAIR  = "#e0e4e8"   # hairline borders, dividers
LINE  = "#aab0b8"   # arrows
GRAYB = "#d6dbe0"   # neutral object border
GRAY_L= "#f1f4f6"   # neutral object fill
GRAY  = "#8a8f98"   # base-model bars

TEAL  = "#1f7f8c"; TEAL_L  = "#e6f1f3"   # policy model / revealed finding
GOLD  = "#a9821f"; GOLD_L  = "#f6eed4"   # exact adaptive oracle (reference, never a learned policy)
VIO   = "#5f5a7a"; VIO_L   = "#e9e7f0"   # deterministic verifier
RED   = "#bd4657"; RED_L   = "#f7e6e9"   # outcome-only GRPO
GREEN = "#2e7d32"; GREEN_L = "#e4efe4"   # credit-assignment GRPO
BLUE  = "#3b6fb6"; BLUE_L  = "#e6edf7"   # decoupled explore/exploit GRPO
ENV_BG  = "#f4f7f8"
EVAL_BG = "#f7f8f9"
TINT = {RED: RED_L, GREEN: GREEN_L, BLUE: BLUE_L, GOLD: GOLD_L}

# ============================== type scale ===================================
FS_PANEL   = 15.5   # panel letter
FS_PTITLE  = 13.0   # panel title
FS_PSUB    = 9.0    # panel subtitle
FS_ZONE    = 11.0   # zone title in Panel A
FS_OBJ     = 10.2   # named object (MedGemma, GRPO training, ...)
FS_METHOD  = 10.2   # method name in Panel B
FS_SUB     = 7.8    # object subtitle
FS_ANNO    = 7.8    # annotation
FS_ANNO2   = 7.0    # secondary annotation
FS_CHIP    = 6.4    # clinical finding label
FS_TINY    = 5.8    # subordinate inset

# ============================== canvas =======================================
FIG_W = 14.0
A_TOP, A_H = 0.30, 4.02          # Panel A zone-panel band
B_H, C_H   = 6.85, 3.45
GAP_AB, GAP_BC = 0.50, 0.38
HEAD_A = 0.62                     # heading height above each panel band
FIG_H = A_TOP + HEAD_A + A_H + GAP_AB + B_H + GAP_BC + C_H + 0.30

os.makedirs("figs", exist_ok=True)
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")

def Y(top_down):
    """absolute y from the top of the canvas"""
    return FIG_H - top_down

# ============================== primitives ===================================
def rrect(x, y, w, h, fc="white", ec=HAIR, lw=1.1, r=0.10, z=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, alpha=alpha, mutation_scale=1))

def arrow(p1, p2, color=LINE, lw=1.4, rad=0.0, z=3, ms=11, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))

def dline(p1, p2, color=FAINT, lw=1.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=(0, (3, 3)), zorder=1.6)

def obj(cx, cy, w, h, title, sub=None, ec=GRAYB, fc="white", tc=INK, tfs=FS_OBJ, sfs=FS_SUB,
        r=0.11, lw=1.2, bold=True):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=r, z=2.4, lw=lw)
    if sub:
        ax.text(cx, cy + 0.15, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "medium"), color=tc, zorder=3)
        ax.text(cx, cy - 0.17, sub, ha="center", va="center", fontsize=sfs, color=MUTE, zorder=3)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=tfs,
                fontweight=("bold" if bold else "medium"), color=tc, zorder=3)

def head(x, y, letter, title, sub=None, sub2=None):
    ax.text(x, y, letter, ha="left", va="top", fontsize=FS_PANEL, fontweight="bold", color=INK)
    ax.text(x + 0.50, y - 0.02, title, ha="left", va="top", fontsize=FS_PTITLE, fontweight="bold", color=INK)
    if sub:
        ax.text(x + 0.51, y - 0.38, sub, ha="left", va="top", fontsize=FS_PSUB, color=MUTE)
    if sub2:
        ax.text(x + 0.51, y - 0.63, sub2, ha="left", va="top", fontsize=FS_ANNO2, color=FAINT, style="italic")

# =============================================================================
#  PANEL A — experimental setup (three zones)
# =============================================================================
AY1 = Y(A_TOP + HEAD_A)          # top of the zone-panel band
AY0 = AY1 - A_H                  # bottom
AMY = AY0 + 2.40                 # the model <-> simulator axis (zones align on it)

head(0.42, Y(A_TOP), "A", "Experimental setup",
     sub="partially observed clinical simulator  ·  GRPO with three reward formulations  ·  one common held-out verifier")

EX0, EX1 = 0.42, 6.32            # environment zone
VX0, VX1 = 10.60, 13.62          # evaluation zone
rrect(EX0, AY0, EX1 - EX0, A_H, fc=ENV_BG, ec="none", r=0.16, z=0.6)
rrect(VX0, AY0, VX1 - VX0, A_H, fc=EVAL_BG, ec="none", r=0.16, z=0.6)
ax.text(EX0 + 0.24, AY1 - 0.18, "Clinical multi-turn environment", ha="left", va="top",
        fontsize=FS_ZONE, fontweight="bold", color=INK)
ax.text(EX0 + 0.25, AY1 - 0.48, "refill triage  ·  acute diagnostics  ·  dosing titration", ha="left", va="top",
        fontsize=FS_ANNO2, color=MUTE)
ax.text(VX0 + 0.24, AY1 - 0.18, "Held-out evaluation", ha="left", va="top",
        fontsize=FS_ZONE, fontweight="bold", color=INK)
ax.text(VX0 + 0.25, AY1 - 0.48, "unseen simulator cases", ha="left", va="top",
        fontsize=FS_ANNO2, color=MUTE)

# ---- zone 1: model <-> simulator ----
rrect(3.28, AMY + 0.70, 2.54, 0.32, fc="white", ec=HAIR, r=0.14, z=2.2)
ax.text(4.55, AMY + 0.86, "hidden state  ·  diagnosis · severity · dose", ha="center", va="center",
        fontsize=6.7, color=MUTE, zorder=3)
obj(1.80, AMY, 1.94, 1.00, "MedGemma-4B-it", "LoRA-adapted policy",
    ec=TEAL, fc=TEAL_L, tc=TEAL, lw=2.0, r=0.28)
obj(4.55, AMY, 1.94, 1.00, "Patient simulator", "hidden clinical state",
    ec=GRAYB, fc=GRAY_L, lw=1.0, bold=False)
arrow((2.77, AMY + 0.20), (3.58, AMY + 0.20), color=TEAL, rad=-0.32, ms=10)
ax.text(3.17, AMY + 0.50, "ask", ha="center", va="center", fontsize=FS_ANNO2, color=TEAL, fontweight="bold")
arrow((3.58, AMY - 0.16), (2.77, AMY - 0.16), color=LINE, rad=-0.32, ms=10)
ax.text(3.17, AMY - 0.45, "finding revealed", ha="center", va="center", fontsize=6.6, color=MUTE)
ax.text(3.64, AMY - 0.70, "ask again  ·  or commit", ha="left", va="center", fontsize=6.6,
        color=MUTE, style="italic")

# commit -> clinical action
arrow((1.80, AMY - 0.50), (1.80, AMY - 0.80), color=LINE, ms=10)
ax.text(2.02, AMY - 0.65, "commit", ha="left", va="center", fontsize=6.6, color=MUTE)
obj(1.80, AMY - 1.06, 1.62, 0.42, "clinical action", ec=HAIR, fc="white", tfs=8.2, bold=False, r=0.09)

# ---- oracle + verifier as annotation badges (properties of the simulator, not pipeline stages) ----
def badge(cx, cy, title, sub, ec, fc, w=1.86, h=0.56):
    rrect(cx - w/2, cy - h/2, w, h, fc=fc, ec=ec, r=0.13, z=2.4, lw=1.0)
    ax.text(cx - w/2 + 0.17, cy + 0.10, "◆", ha="center", va="center", fontsize=6.4, color=ec, zorder=3)
    ax.text(cx + 0.09, cy + 0.10, title, ha="center", va="center", fontsize=7.9,
            fontweight="bold", color=ec, zorder=3)
    ax.text(cx, cy - 0.145, sub, ha="center", va="center", fontsize=6.5, color=MUTE, zorder=3)

BGY = AY0 + 0.42
badge(1.76, BGY, "Exact adaptive oracle", "reference procedure", GOLD, GOLD_L)
badge(4.50, BGY, "Deterministic verifier", "terminal outcome scorer", VIO, VIO_L)
dline((2.69, BGY), (3.70, AMY - 0.52), color=GOLD)
dline((5.43, BGY), (5.52, AMY - 0.52), color=VIO)

# ---- zone 2: GRPO training ----
arrow((EX1, AMY), (7.50, AMY), color="#9aa1a9", ms=12, lw=1.6)
ax.text((EX1 + 7.50)/2, AMY + 0.19, "policy rollouts", ha="center", va="center",
        fontsize=FS_ANNO2, color=MUTE)
obj(8.55, AMY, 1.74, 0.92, "GRPO training", ec=INK, fc="white", tc=INK, lw=2.0, tfs=FS_OBJ)
ax.text(8.55, AMY - 0.68, "training reward differs by method", ha="center", va="center",
        fontsize=FS_ANNO, fontweight="bold", color=MUTE)

PY = AMY - 1.06
pills = [(7.30, RED,   RED_L,   "Outcome-only",      "terminal reward only"),
         (8.55, GREEN, GREEN_L, "Credit assignment", "+ per-ask info credit"),
         (9.80, BLUE,  BLUE_L,  "Decoupled",         "gather / commit split")]
dline((8.55, AMY - 0.46), (8.55, PY + 0.44), color=GRAYB, lw=1.0)
ax.plot([7.30, 9.80], [PY + 0.38, PY + 0.38], color=GRAYB, lw=1.0, zorder=1.5)
for cx, ec, fc, t, s in pills:
    ax.plot([cx, cx], [PY + 0.38, PY + 0.26], color=GRAYB, lw=1.0, zorder=1.5)
    rrect(cx - 0.61, PY - 0.26, 1.22, 0.52, fc=fc, ec=ec, r=0.26, z=2.5, lw=1.3)
    ax.text(cx, PY + 0.06, t, ha="center", va="center", fontsize=7.5, fontweight="bold", color=ec, zorder=3)
    ax.text(cx, PY - 0.145, s, ha="center", va="center", fontsize=5.9, color=MUTE, zorder=3)

# ---- subordinate two-node inset ----
IX0, IX1 = 7.06, 10.04
IY1 = AY0 + 1.00
IY0 = AY0 + 0.02
dline((8.55, PY - 0.26), (8.55, IY1), color=HAIR, lw=1.0)
rrect(IX0, IY0, IX1 - IX0, IY1 - IY0, fc=EVAL_BG, ec=HAIR, r=0.10, z=1.7)
ax.text((IX0 + IX1)/2, IY1 - 0.14, "Disaggregated RL", ha="center", va="center",
        fontsize=7.0, fontweight="bold", color=FAINT, zorder=3)
NY = IY0 + 0.46
for cx, name, nsub in [(7.78, "Trainer node", "GRPO · LoRA updates"),
                       (9.32, "Rollout node", "vLLM · 8 rollouts/prompt")]:
    rrect(cx - 0.66, NY - 0.22, 1.32, 0.44, fc="white", ec="#d3d8dd", r=0.10, z=2.0, lw=1.0)
    ax.text(cx, NY + 0.08, name, ha="center", va="center", fontsize=6.3, fontweight="bold",
            color=MUTE, zorder=3)
    ax.text(cx, NY - 0.09, nsub, ha="center", va="center", fontsize=FS_TINY - 0.5, color=FAINT, zorder=3)
ax.add_patch(FancyArrowPatch((8.46, NY), (8.64, NY), arrowstyle="<|-|>", mutation_scale=7,
             linewidth=1.0, color="#b3b9c0", shrinkA=1, shrinkB=1, zorder=2.4))
ax.text((IX0 + IX1)/2, IY0 + 0.12, "policy weights →   ·   ← rollout trajectories",
        ha="center", va="center", fontsize=FS_TINY - 0.3, color=FAINT, zorder=3)

# ---- zone 3: held-out evaluation ----
arrow((9.42, AMY), (VX0, AMY), color="#9aa1a9", ms=12, lw=1.6)
ax.text((9.42 + VX0)/2, AMY + 0.19, "trained policies", ha="center", va="center",
        fontsize=FS_ANNO2, color=MUTE)
VCX = (VX0 + VX1)/2
obj(VCX, AMY + 0.56, 2.58, 0.54, "Held-out simulator cases", ec=GRAYB, fc="white",
    tfs=9.2, lw=1.0, bold=False)
arrow((VCX, AMY + 0.29), (VCX, AMY + 0.12), color=LINE, ms=9)
obj(VCX, AMY - 0.26, 2.66, 0.74, "Same pure verifier", "one terminal score, every method",
    ec=VIO, fc=VIO_L, tc=VIO, lw=2.0, tfs=FS_OBJ, sfs=6.6)
arrow((VCX, AMY - 0.63), (VCX, AMY - 0.80), color=LINE, ms=9)
rrect(VCX - 0.90, AMY - 1.12, 1.80, 0.30, fc="white", ec=RED, r=0.12, z=2.6)
ax.text(VCX, AMY - 0.97, "reward shaping off", ha="center", va="center", fontsize=7.2,
        fontweight="bold", color=RED, zorder=3)
ax.text(VCX, BGY + 0.34, "metrics", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=MUTE, zorder=3)
ax.text(VCX, BGY + 0.13, "overall value   ·   high-risk value", ha="center", va="center",
        fontsize=6.8, color=MUTE, zorder=3)
ax.text(VCX, BGY - 0.08, "question burden   ·   oracle-grade support", ha="center", va="center",
        fontsize=6.8, color=MUTE, zorder=3)

# =============================================================================
#  PANEL B — same patient, different decision procedures  (CENTREPIECE)
# =============================================================================
BHEAD = AY0 - GAP_AB
head(0.42, BHEAD, "B", "Same patient, different decision procedures",
     sub="one held-out high-risk case (acute abdominal pain, hidden emergent severity)  ·  "
         "same simulator, same final verifier  ·  only the training reward differs",
     sub2="single representative rollouts (k = 12 per condition); illustrative of procedure class, "
          "not modal or aggregate behaviour")

# chip grid (shared by all four rows -> trajectory length is directly comparable)
CW, CH, CGAP = 0.84, 0.58, 0.17
TRACK_X0 = 3.05
def chip_x(i):
    return TRACK_X0 + CW/2 + i*(CW + CGAP)

LAB = {
    "abd_rigidity":  "Abdominal\nrigidity", "rebound_tender": "Rebound\ntenderness",
    "ct_free_air":   "CT free\nair",        "lactate_high":   "Lactate\nelevated",
    "guarding":      "Guarding",            "radiation_arm":  "Radiating\npain",
    "dyspnea_severe":"Severe\ndyspnea",     "hypotension":    "Blood\npressure",
    "wbc_high":      "White cell\ncount",   "ecg_ischemia":   "ECG",
}
ACT = {"emergent_tx": "Emergency\ntreatment", "admit": "Admit"}
# verified from tasks.diagnostics.env_rv.gen_cases(60, 0.0, seed=100)['dx-13']['assign']
PRESENT = {"abd_rigidity": 1, "rebound_tender": 1, "ct_free_air": 1, "lactate_high": 1, "guarding": 1,
           "radiation_arm": 1, "dyspnea_severe": 1, "hypotension": 0, "wbc_high": 0, "ecg_ischemia": 0}

def finding_chip(cx, cy, key, present, dim=False):
    ec = "#dde1e5" if dim else "#c4c9cf"
    a  = 0.55 if dim else 1.0
    rrect(cx - CW/2, cy - CH/2, CW, CH, fc="white", ec=ec, r=0.035, z=2.4, alpha=a, lw=1.0)
    ax.text(cx, cy + 0.075, LAB.get(key, key), ha="center", va="center", fontsize=FS_CHIP,
            color=("#a7adb5" if dim else INK), zorder=3, linespacing=0.95, alpha=a)
    if not dim:
        mc = TEAL if present else "#bcc2c8"
        ax.add_patch(Circle((cx - 0.25, cy - 0.185), 0.042, facecolor=(mc if present else "white"),
                     edgecolor=("none" if present else "#bcc2c8"), linewidth=1.0, zorder=3.2))
        ax.text(cx - 0.165, cy - 0.185, "present" if present else "normal", ha="left", va="center",
                fontsize=5.1, color=(TEAL if present else MUTE), zorder=3.2)

def action_pill(cx, cy, key, color, w=1.32, h=0.66):
    rrect(cx - w/2, cy - h/2, w, h, fc=TINT.get(color, "white"), ec=color, lw=2.0, r=0.16, z=2.6)
    ax.text(cx, cy, ACT.get(key, key), ha="center", va="center", fontsize=7.4, color=color,
            fontweight="bold", zorder=3, linespacing=0.95)
    return cx + w/2

def trajectory(y, keys, action_key, reward, method, color, sub, sub2,
               reference=False, dim_from=None, bracket_label=None):
    col = GOLD if reference else color
    if reference:   # pale gold band marks the non-learned reference row (zone-panel language from A)
        rrect(0.42, y - 0.50, 13.20, 1.00, fc=GOLD_L, ec="none", r=0.10, z=0.5, alpha=0.45)
    ax.add_patch(FancyBboxPatch((0.55, y - 0.33), 0.11, 0.66,
                 boxstyle="round,pad=0,rounding_size=0.03", linewidth=0, facecolor=col, zorder=3))
    ax.text(0.80, y + 0.17, method, ha="left", va="center", fontsize=FS_METHOD,
            fontweight="bold", color=col)
    ax.text(0.80, y - 0.12, sub, ha="left", va="center", fontsize=FS_ANNO, color=MUTE)
    ax.text(0.80, y - 0.39, sub2, ha="left", va="top", fontsize=FS_ANNO2, color=FAINT)
    n = len(keys)
    for i, k in enumerate(keys):
        cx = chip_x(i)
        finding_chip(cx, y, k, PRESENT.get(k, 1), dim=(dim_from is not None and i >= dim_from))
        if i > 0:
            arrow((chip_x(i-1) + CW/2, y), (cx - CW/2, y), color="#c6cbd1", lw=1.1, ms=7, z=1.6)
    if dim_from is not None and bracket_label:
        bx0 = chip_x(dim_from) - CW/2 - 0.03
        bx1 = chip_x(n - 1) + CW/2 + 0.03
        by  = y + CH/2 + 0.13
        ax.plot([bx0, bx0, bx1, bx1], [by - 0.07, by, by, by - 0.07], color="#b3b9c0", lw=1.0, zorder=2)
        ax.text((bx0 + bx1)/2, by + 0.10, bracket_label, ha="center", va="bottom",
                fontsize=6.7, color=MUTE, style="italic")
    last_r = chip_x(n - 1) + CW/2
    APW = 1.32
    apx = last_r + 0.30 + APW/2          # real gap: the pill never overlaps the last finding chip
    arrow((last_r, y), (apx - APW/2, y), color="#c6cbd1", lw=1.2, ms=8, z=1.6)
    right = action_pill(apx, y, action_key, col, w=APW)
    ax.text(right + 0.58, y + 0.10, f"{reward:+.2f}", ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=col, zorder=3)
    ax.text(right + 0.58, y - 0.21, "reward", ha="center", va="center", fontsize=6.6,
            color=MUTE, zorder=3)

# present/normal legend
lgx = 10.50
ax.add_patch(Circle((lgx, BHEAD - 0.16), 0.045, facecolor=TEAL, edgecolor="none", zorder=3))
ax.text(lgx + 0.10, BHEAD - 0.16, "finding present", ha="left", va="center", fontsize=6.9, color=MUTE)
ax.add_patch(Circle((lgx + 1.36, BHEAD - 0.16), 0.045, facecolor="white", edgecolor="#bcc2c8", lw=1.0, zorder=3))
ax.text(lgx + 1.46, BHEAD - 0.16, "finding normal", ha="left", va="center", fontsize=6.9, color=MUTE)

R0 = BHEAD - 1.34
r0, r1, r2, r3 = R0, R0 - 1.30, R0 - 2.60, R0 - 3.98
trajectory(r0, ["abd_rigidity", "rebound_tender", "ct_free_air", "lactate_high"], "emergent_tx", 0.83,
           "Exact adaptive oracle", GOLD,
           "reference procedure · not a learned policy",
           "4 targeted findings → emergency treatment", reference=True)
trajectory(r1, ["abd_rigidity"], "admit", 0.17, "Outcome-only GRPO", RED,
           "1 finding · premature commitment", "emergency action missed")
trajectory(r2, ["abd_rigidity", "guarding", "rebound_tender", "ct_free_air"], "emergent_tx", 0.83,
           "Credit-assignment GRPO", GREEN,
           "4 findings · successful sampled trajectory",
           "oracle-grade by reward; evidence order differs from the oracle")
trajectory(r3, ["abd_rigidity", "rebound_tender", "guarding", "radiation_arm", "dyspnea_severe",
                "hypotension", "wbc_high", "ecg_ischemia"], "emergent_tx", 0.71,
           "Decoupled explore/exploit", BLUE,
           "8 findings vs oracle 4 · correct final action",
           "excessive information gathering; reward cut by question cost",
           dim_from=3, bracket_label="additional non-decisive findings")

# ---- gathering-depth scale, anchored on the oracle ----
gy = R0 - 5.05
def xn(n):
    return chip_x(n - 1)
x_lo, x_hi = xn(1) - 0.45, xn(8) + 0.45
rrect(x_lo, gy - 0.05, x_hi - x_lo, 0.10, fc="#eef0f2", ec="none", r=0.05, z=1.3)
rrect(xn(3) - 0.20, gy - 0.05, (xn(5) + 0.20) - (xn(3) - 0.20), 0.10, fc="#dcecdd", ec="none", r=0.05, z=1.35)
ax.plot([xn(4), xn(4)], [gy - 0.22, gy + 0.30], color=GOLD, lw=1.8, zorder=2.6)
ax.text(xn(4), gy + 0.39, "oracle depth", ha="center", va="bottom", fontsize=7.1,
        color=GOLD, fontweight="bold")
for mx, col, num in [(xn(1), RED, "1"), (xn(4) + 0.17, GREEN, "4"), (xn(8), BLUE, "8")]:
    ax.add_patch(Circle((mx, gy), 0.075, facecolor=col, edgecolor="white", linewidth=1.2, zorder=3))
    ax.text(mx, gy + 0.19, num, ha="center", va="bottom", fontsize=6.8, color=col, fontweight="bold")
ax.text(x_lo - 0.10, gy, "questions asked", ha="right", va="center", fontsize=7.1, color=MUTE)
ax.text(x_lo, gy - 0.29, "too little evidence", ha="left", va="center", fontsize=7.1, color=MUTE)
ax.text(xn(4), gy - 0.29, "appropriate depth", ha="center", va="center", fontsize=7.1, color=GREEN)
ax.text(x_hi, gy - 0.29, "excessive work-up", ha="right", va="center", fontsize=7.1, color=MUTE)

# =============================================================================
#  PANEL C — quantitative diagnostics results
# =============================================================================
CHEAD = BHEAD - (HEAD_A + B_H) + 0.20
ax.plot([0.42, 13.62], [CHEAD + 0.30, CHEAD + 0.30], color=HAIR, lw=1.0, zorder=0.5)
head(0.42, CHEAD, "C", "Quantitative diagnostics results",
     sub="held-out diagnostics  ·  same unshaped terminal verifier for every method")

METHODS  = ["Base†", "Outcome-\nonly", "Credit\nassignment", "Decoupled†"]
MCOLORS  = [GRAY, RED, GREEN, BLUE]
SINGLE   = [True, False, False, True]

def mini_bars(rect, title, vals, errs, ylim, yticks, fmt, hline=None, hlabel=None):
    axb = fig.add_axes(rect)
    xs = np.arange(4)
    bars = axb.bar(xs, vals, width=0.64, color=MCOLORS, zorder=2,
                   yerr=[[0]*4, errs], capsize=3, error_kw=dict(ecolor="#4a525c", lw=1.1))
    for i, b in enumerate(bars):
        if SINGLE[i]:
            b.set_hatch("////"); b.set_edgecolor("white"); b.set_linewidth(0.0)
    if hline is not None:
        axb.axhline(hline, ls=(0, (4, 3)), lw=1.4, color=GOLD, zorder=3)
        axb.text(-0.46, hline + ylim[1]*0.018, hlabel, ha="left", va="bottom", fontsize=6.9, color=GOLD)
    for xi, v, e in zip(xs, vals, errs):
        axb.text(xi, v + ylim[1]*0.035 + (e or 0), fmt.format(v), ha="center", va="bottom",
                 fontsize=7.2, color=INK, fontweight="bold")
    axb.set_ylim(*ylim); axb.set_yticks(yticks)
    axb.set_xticks(xs); axb.set_xticklabels(METHODS, fontsize=7.6, color=INK, linespacing=1.0)
    axb.tick_params(axis="y", labelsize=7.0, length=2.5); axb.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axb.spines[sp].set_color("#c4c9cf")
    axb.set_title(title, fontsize=9.8, color=INK, fontweight="bold", pad=6)
    axb.margins(x=0.08)
    return axb

BW, BH_ = 4.30, 1.42
BX1, BX2 = 1.45, 7.85
BY_in = CHEAD - 0.98 - BH_        # bar-chart bottom, in canvas inches from y=0
mini_bars([BX1/FIG_W, BY_in/FIG_H, BW/FIG_W, BH_/FIG_H], "High-risk value",
          [0.160, 0.098, 0.157, 0.248], [0, 0.037, 0.057, 0],
          (0, 0.32), [0, 0.1, 0.2, 0.3], "{:.3f}")
mini_bars([BX2/FIG_W, BY_in/FIG_H, BW/FIG_W, BH_/FIG_H], "Mean questions asked",
          [2.54, 1.82, 2.08, 4.42], [0, 0.11, 0.30, 0],
          (0, 5.3), [0, 2, 4], "{:.2f}", hline=3.78, hlabel="oracle ≈ 3.8")

ax.text(FIG_W/2, BY_in - 0.42,
        "High-risk value = mean verifier reward on the rare emergent subset (a behavioural measure, not a "
        "direct measure of clinical safety).", ha="center", va="center", fontsize=FS_ANNO, color=MUTE)
ax.text(FIG_W/2, BY_in - 0.62,
        "† single-seed reference (hatched).   Outcome-only / Credit assignment: mean ± SD across "
        "3 training seeds.   Dashed line: exact adaptive oracle on the same held-out cases.",
        ha="center", va="center", fontsize=FS_ANNO2, color=FAINT)

# ================================ save =======================================
for ext in ("png", "svg", "pdf"):
    out = f"figs/figure1_combined.{ext}"
    fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    print("saved", out)
print(f"canvas {FIG_W} x {FIG_H:.2f} in")
print(f"panel heights: A={HEAD_A + A_H:.2f}  B={HEAD_A + B_H:.2f}  C={C_H:.2f}")
