#!/usr/bin/env python3
"""Figure 1 — conceptual overview for
"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents."

Trajectory-driven, minimal numbers. Four panels:
  A  ask-to-commit loop (hidden state + verifiable reward)
  B  three environment cards (refill / diagnostics / titration)
  C  post-training branches (outcome-only / credit-assignment / decoupled), reward geometry per branch
  D  centerpiece: the SAME patient (dx-13) under three policies -> same verifier, different procedures

All content is sourced from figure1_source_material.md (verified against the repo/results).
Panel D trajectories are real held-out rollouts on diagnostics-deephedge case dx-13
(credit-assignment: probe_dh_ig_final; outcome-only: probe_dh_grpo_final; decoupled: probe_dream_s40).

Outputs: figure1_overview.{svg,pdf,png}.  Run inside ~/spark1/.venv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- editable-text vector output + journal-ish sans font ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",   # keep text as <text> for Illustrator/Inkscape editing
    "pdf.fonttype": 42,        # embed TrueType in the PDF
    "axes.linewidth": 0,
})

# ---- palette ----
INK    = "#1f2937"
MUTE   = "#6b7280"
BORDER = "#cfd4da"
GRAY   = "#8a8f98"; GRAY_L = "#eef0f2"
RED    = "#c14b5a"; RED_L  = "#f7e6e9"   # outcome-only / plain GRPO
GREEN  = "#2e7d32"; GREEN_L= "#e3efe4"   # credit-assignment
BLUE   = "#3b6fb6"; BLUE_L = "#e6edf7"   # decoupled
AMBER  = "#b07d2b"; AMBER_L= "#fbf5e6"   # diagnostics highlight
DBG    = "#f6f9fc"                          # panel D centerpiece tint

FIG_W, FIG_H = 15.0, 9.5


class Box:
    """A rounded box in inch-coordinates with edge anchor points."""
    def __init__(self, ax, cx, cy, w, h, lines, fc="white", ec=BORDER, tc=INK,
                 fs=9.5, lw=1.2, round_size=0.09, header=None, header_c=None, align="center"):
        self.cx, self.cy, self.w, self.h = cx, cy, w, h
        x0, y0 = cx - w / 2, cy - h / 2
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, mutation_scale=1,
                     boxstyle=f"round,pad=0,rounding_size={round_size}",
                     linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
        ty = cy + h / 2 - 0.30
        if header:
            ax.text(cx, ty, header, ha="center", va="top", fontsize=fs + 1.0,
                    fontweight="bold", color=header_c or tc, zorder=3)
            ty -= 0.34
        if isinstance(lines, str):
            lines = [lines]
        tx = cx if align == "center" else x0 + 0.20
        ha = "center" if align == "center" else "left"
        for ln in lines:
            txt, c, fw, math = ln if isinstance(ln, tuple) else (ln, tc, "normal", False)
            ax.text(tx, ty, txt, ha=ha, va="top", fontsize=fs, color=c,
                    fontweight=fw, zorder=3, usetex=False)
            ty -= 0.30

    @property
    def L(self): return (self.cx - self.w / 2, self.cy)
    @property
    def R(self): return (self.cx + self.w / 2, self.cy)
    @property
    def T(self): return (self.cx, self.cy + self.h / 2)
    @property
    def Bt(self): return (self.cx, self.cy - self.h / 2)


def arrow(ax, p1, p2, color=MUTE, lw=1.4, rad=0.0, label=None, lx=0, ly=0,
          style="-|>", fs=8.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                 linewidth=lw, color=color, connectionstyle=f"arc3,rad={rad}",
                 shrinkA=1.5, shrinkB=1.5, zorder=1.5, linestyle=ls))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color, zorder=4)


def panel(ax, x0, y0, x1, y1, letter, title, bg="white"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, mutation_scale=1,
                 boxstyle="round,pad=0,rounding_size=0.12", linewidth=1.3,
                 edgecolor=BORDER, facecolor=bg, zorder=0.5))
    ax.text(x0 + 0.22, y1 - 0.30, letter, ha="left", va="top", fontsize=15,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x0 + 0.72, y1 - 0.31, title, ha="left", va="top", fontsize=12.5,
            fontweight="bold", color=INK, zorder=3)


def chip(ax, cx, cy, w, h, text, fc, ec, tc=INK, fs=8.5, fw="normal", round_size=0.10, math=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, mutation_scale=1,
                 boxstyle=f"round,pad=0,rounding_size={round_size}", linewidth=1.1,
                 edgecolor=ec, facecolor=fc, zorder=2.5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight=fw, zorder=3)


def ask_track(ax, x0, y, n, color, size=0.22, gap=0.08):
    """A row of n small squares (one per ask). Returns x of the right edge."""
    x = x0
    for _ in range(n):
        ax.add_patch(FancyBboxPatch((x, y - size / 2), size, size, mutation_scale=1,
                     boxstyle="round,pad=0,rounding_size=0.05", linewidth=1.0,
                     edgecolor=color, facecolor=color, alpha=0.85, zorder=2.5))
        x += size + gap
    return x - gap


# =====================================================================
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)
ax.set_aspect("equal"); ax.axis("off")

# ---- title ----
ax.text(FIG_W / 2, 9.24, "Reward Design Governs Information Gathering in Clinical Multi-Turn Agents",
        ha="center", va="center", fontsize=15.5, fontweight="bold", color=INK)
ax.text(FIG_W / 2, 8.86, "Clinical decisions are an ask-to-commit process; the reward geometry — not just the "
        "verifiable final score — decides whether RL learns a safe gathering procedure.",
        ha="center", va="center", fontsize=9.8, color=MUTE)

# =====================================================================
# PANEL A — ask-to-commit loop
ax0, ay0, ax1, ay1 = 0.25, 5.35, 7.05, 8.55
panel(ax, ax0, ay0, ax1, ay1, "A", "Ask-to-commit loop")

pat = Box(ax, 1.42, 7.48, 2.05, 1.15, [("acute presentation", INK, "normal", False),
          ("latent severity  $\\mu$  (hidden)", RED, "normal", False)],
          header="Patient", fs=8.6)
askb = Box(ax, 3.72, 7.48, 1.95, 0.95, [('{"action_type":"ask",', INK, "normal", False),
          ('  "intent":"..."}', INK, "normal", False)], header="ASK", header_c=INK, fs=8.0)
rev = Box(ax, 5.95, 7.48, 1.95, 0.95, [("Result:  <finding>.", INK, "normal", False),
          ("(one binary finding)", MUTE, "normal", False)], header="Simulator", fs=8.0)
com = Box(ax, 5.95, 6.28, 1.95, 1.0, [('{"action_type":"commit",', INK, "normal", False),
          ('  "decision":"..."}', INK, "normal", False)], header="COMMIT", header_c=INK, fs=8.0)
rew = Box(ax, 3.05, 6.30, 2.35, 0.82,
          [("$R = U[\\mu][a] \\; - \\; c\\cdot n_{\\mathrm{asks}}$", INK, "bold", True)],
          header="Verifiable reward", fs=9.5, fc="#f4f6f8")

arrow(ax, pat.R, askb.L, color=MUTE)
arrow(ax, askb.R, rev.L, color=MUTE)
arrow(ax, rev.T, askb.T, color=GREEN, rad=-0.55, label="ask more", ly=0.20, fs=8.2)
arrow(ax, rev.Bt, com.T, color=RED, label="stop &\ndecide", lx=0.62, fs=8.0)
arrow(ax, com.L, rew.R, color=MUTE)
ax.text((ax0 + ax1) / 2, 5.52, "episode ends on:  commit  ·  turn cap  ·  parse error        "
        "per-ask cost  $c=0.03$", ha="center", va="center", fontsize=8.2, color=MUTE)

# =====================================================================
# PANEL B — three environment cards
bx0, by0, bx1, by1 = 7.35, 5.35, 14.75, 8.55
panel(ax, bx0, by0, bx1, by1, "B", "Three ask-to-commit environments")

def env_card(cx, name, sub, dims, highlight=False):
    ec = AMBER if highlight else BORDER
    fc = AMBER_L if highlight else "white"
    lw = 1.8 if highlight else 1.2
    Box(ax, cx, 6.88, 2.15, 2.35, dims, header=name, fs=8.4, ec=ec, fc=fc, lw=lw)
    ax.text(cx, 7.62, sub, ha="center", va="top", fontsize=7.8, color=MUTE, zorder=3)

env_card(8.65, "Refill triage",
         "med-refill workup",
         [("12 findings", INK, "normal", False), ("5 actions", INK, "normal", False),
          ("latent: severity", INK, "normal", False), ("(none / mod / high)", MUTE, "normal", False)])
env_card(11.05, "Diagnostics",
         "acute workup (hedging)",
         [("18 findings", INK, "normal", False), ("5 actions", INK, "normal", False),
          ("latent: severity", INK, "normal", False),
          ("hedge is capped;", AMBER, "bold", False),
          ("emergency = 4-finding AND", AMBER, "bold", False)], highlight=True)
env_card(13.45, "Titration",
         "anticoag intensity",
         [("6 findings", INK, "normal", False), ("3 ordered actions", INK, "normal", False),
          ("latent: clot x bleed", INK, "normal", False), ("(9 joint states)", MUTE, "normal", False)])
ax.text((bx0 + bx1) / 2, 5.50, "same POMDP + exact adaptive oracle    ·    diagnostics (highlighted) is the "
        "running example in D", ha="center", va="center", fontsize=8.2, color=MUTE)

# =====================================================================
# PANEL C — training branches / reward geometry
cx0, cy0, cx1, cy1 = 0.25, 0.30, 5.55, 5.05
panel(ax, cx0, cy0, cx1, cy1, "C", "Post-training: reward geometry")

init = Box(ax, 1.55, 4.28, 2.15, 0.62, [("MedGemma-4b (text)", INK, "normal", False)], fs=8.4)
sft  = Box(ax, 4.05, 4.28, 2.10, 0.62, [("SFT warmup (optional)", INK, "normal", False)], fs=8.4)
grpo = Box(ax, 2.90, 3.42, 3.9, 0.66, [("GRPO   ·   group 8   ·   2-node vLLM", INK, "bold", False)],
           fs=8.6, fc="#f4f6f8")
arrow(ax, init.R, sft.L, color=MUTE)
arrow(ax, sft.Bt, grpo.T, color=MUTE, rad=0.0)

def branch(cy, name, reward, color, fc, math_reward=True):
    ax.add_patch(FancyBboxPatch((0.55, cy - 0.40), 4.55, 0.80, mutation_scale=1,
                 boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.3,
                 edgecolor=color, facecolor=fc, zorder=2))
    ax.add_patch(FancyBboxPatch((0.55, cy - 0.40), 0.16, 0.80, mutation_scale=1,
                 boxstyle="square,pad=0", linewidth=0, facecolor=color, zorder=2.2))
    ax.text(0.82, cy + 0.16, name, ha="left", va="center", fontsize=9.0, fontweight="bold", color=color)
    ax.text(0.82, cy - 0.16, reward, ha="left", va="center", fontsize=8.8, color=INK)

arrow(ax, (2.90, grpo.Bt[1]), (2.90, 2.72), color=MUTE)
branch(2.50, "Outcome-only GRPO", r"$R=\mathrm{env.score}$   (verifier only)", RED, RED_L)
branch(1.62, "Credit-assignment", r"$R=\mathrm{env.score}+\lambda\sum_t \Delta H_t$   (per-ask info-gain)", GREEN, GREEN_L)
branch(0.74, "Decoupled explore/exploit",
       "explore: $\\Delta H$ recoverability  ·  exploit: verifier", BLUE, BLUE_L)
ax.text((cx0 + cx1) / 2, 0.44, "shaping lives only in the RL rollout — env / gates / eval stay pure",
        ha="center", va="center", fontsize=7.8, color=MUTE, style="italic")

# =====================================================================
# PANEL D — centerpiece: same patient, three policies
dx0, dy0, dx1, dy1 = 5.85, 0.30, 14.75, 5.05
panel(ax, dx0, dy0, dx1, dy1, "D", "Same patient, three procedures — one verifier", bg=DBG)

ax.text(dx0 + 0.30, 4.44, "Patient  dx-13:  acute abdominal pain  ·  latent = EMERGENT (hidden)  ·  "
        "correct action = emergent_tx", ha="left", va="center", fontsize=9.0, color=INK, fontweight="bold")
ax.text(dx0 + 0.30, 4.14, "held-out  ·  reward shaping OFF  ·  same env.score verifier for all three",
        ha="left", va="center", fontsize=8.2, color=MUTE)

LBL_X = 6.20; TRK_X = 8.55; COM_X = 11.95; REW_X = 13.75

def traj(y, method, color, n_asks, decision, dec_ok, reward, verdict, findings=None):
    # method label
    ax.text(LBL_X, y + 0.20, method, ha="left", va="center", fontsize=9.0, fontweight="bold", color=color)
    ax.text(LBL_X, y - 0.14, f"{n_asks} ask{'s' if n_asks != 1 else ''}", ha="left", va="center",
            fontsize=8.0, color=MUTE)
    # ask track (squares) + arrow to commit
    end = ask_track(ax, TRK_X, y, n_asks, color)
    arrow(ax, (end + 0.06, y), (COM_X - 0.72, y), color=MUTE, lw=1.2)
    # commit chip
    dec_c = GREEN if dec_ok else RED
    chip(ax, COM_X, y, 1.42, 0.5, decision, fc="white", ec=dec_c, tc=dec_c, fs=8.4, fw="bold")
    # reward + short verdict (mark beside, centered)
    ax.text(REW_X, y + 0.15, reward, ha="center", va="center", fontsize=13.0, fontweight="bold", color=color)
    mark = "✓" if dec_ok else "✗"          # check / ballot-x (rendered via DejaVu)
    mc = GREEN if dec_ok else RED
    ax.text(REW_X - 0.52, y - 0.20, mark, ha="center", va="center", fontsize=8.5, color=mc,
            family="DejaVu Sans", zorder=3)
    ax.text(REW_X - 0.34, y - 0.20, verdict, ha="left", va="center", fontsize=7.8, color=MUTE, zorder=3)
    if findings:
        ax.text(TRK_X, y - 0.36, findings, ha="left", va="center", fontsize=7.0, color=MUTE, style="italic")

# column captions
ax.text(TRK_X, 3.82, "asks", ha="left", va="center", fontsize=7.6, color=MUTE)
ax.text(COM_X, 3.82, "commit", ha="center", va="center", fontsize=7.6, color=MUTE)
ax.text(REW_X, 3.82, "reward", ha="center", va="center", fontsize=7.6, color=MUTE)

traj(3.48, "Credit-assignment", GREEN, 4, "emergent_tx", True, "+0.83",
     "calibrated",
     findings="rigidity → guarding → rebound → CT free-air")
traj(2.42, "Outcome-only GRPO", RED, 1, "admit", False, "+0.17",
     "under-gather",
     findings="stops after one finding — misses the emergency")
traj(1.36, "Decoupled", BLUE, 8, "emergent_tx", True, "+0.71",
     "over-gather",
     findings="8 findings ordered — correct action, ask-cost bleeds reward")

# takeaway banner
ax.add_patch(FancyBboxPatch((dx0 + 0.30, 0.42), (dx1 - dx0) - 0.60, 0.42, mutation_scale=1,
             boxstyle="round,pad=0,rounding_size=0.08", linewidth=0, facecolor="#e7edf3", zorder=1.8))
ax.text((dx0 + dx1) / 2, 0.63, "Reward design changes the procedure — not just the score.",
        ha="center", va="center", fontsize=9.8, fontweight="bold", color=INK, zorder=3)

# =====================================================================
for ext in ("svg", "pdf", "png"):
    out = f"figure1_overview.{ext}"
    fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight",
                facecolor="white")
    print("saved", out)
