#!/usr/bin/env python3
"""Figure: agent <-> patient-simulator dialogues in plain clinical language.

  A  Same patient, different policies — held-out diagnostics case dx-13 (acute abdominal pain,
     hidden high risk): exact adaptive oracle vs the modal outcome-only GRPO rollout.
  B  Oracle procedures in the three tasks (the Appendix B traces): diagnostics dx-42,
     refill triage wk-6, titration ti-0.

Nothing clinical is typed by hand except the plain-language *phrasing*. Every case, ask sequence,
simulator answer, decision and reward is regenerated from the simulators
(env.gen_cases(60, 0.0, seed=100), env.answer, env.score) and the outcome-only rollout is read from
results/probe_dh_grpo_final_t12.records.jsonl. The PHRASE table maps each (finding, revealed value) to
a question and answer; the script asserts that the simulator's own answer text matches the value the
phrase assumes, so a mismatch fails the build instead of producing a wrong figure.

Outputs: paper/figs/fig_dialogues.{pdf,png} (PNG at 300 dpi). Run inside ~/spark1/.venv:
    python figures/make_fig_dialogues.py
"""
import collections
import json
import os
import sys

os.environ["DX_VARIANT"] = "deephedge"      # the paper's diagnostics variant (18 findings)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from tasks.diagnostics import env_rv as DX
from tasks.rx_triage import env as RX
from tasks.titration import env as TI

# ================================ verified content ================================
# (finding, value) -> (plain question, plain answer, the simulator's own wording it must match)
PHRASE = {
    "abd_rigidity":      ("Is the abdomen rigid?",           {1: ("Yes, the abdomen is rigid.", "abdominal rigidity")}),
    "rebound_tender":    ("Is there rebound tenderness?",    {1: ("Yes, rebound tenderness is present.", "rebound tenderness")}),
    "ct_free_air":       ("Does CT show free air?",          {1: ("Yes, CT shows free air.", "CT: free air")}),
    "lactate_high":      ("Is lactate elevated?",            {1: ("Yes, lactate is elevated.", "lactate elevated")}),
    "troponin_high":     ("Is troponin elevated?",           {1: ("Yes, troponin is elevated.", "troponin elevated")}),
    "ecg_ischemia":      ("Does the ECG show ischemia?",     {1: ("Yes, the ECG shows ischemia.", "ECG shows ischemia")}),
    "dyspnea_severe":    ("Is the dyspnea severe?",          {1: ("Yes, the dyspnea is severe.", "severe dyspnea")}),
    "hypotension":       ("Is the patient hypotensive?",     {1: ("Yes, the patient is hypotensive.", "hypotensive")}),
    "renal_decline":     ("Is the eGFR declining?",          {1: ("Yes, the eGFR is declining.", "eGFR declining")}),
    "egfr_severe":       ("Is the eGFR decline severe?",     {0: ("No, the decline is mild.", "decline mild")}),
    "volume_dehydrated": ("Is the patient dehydrated?",      {1: ("Yes, the patient is dehydrated.", "dehydrated")}),
    "bleed_history":     ("Any history of major bleeding?",  {0: ("No, there is no prior bleeding.", "no prior bleeding")}),
    "fall_risk":         ("Is the patient at high fall risk?", {0: ("No, fall risk is low.", "low fall risk")}),
}
DECISION = {"emergent_tx": "Emergency treatment", "admit": "Admit", "dose_adjust": "Adjust dose",
            "high": "Full-intensity anticoagulation"}
RISK = {"high": "high", "moderate": "moderate", "high_low": "high clot, low bleed"}
# opening simulator turn, condensed from env.user_msg(case) (which is asserted to contain the key phrase)
OPENING = {
    "dx-13": ("Adult with acute abdominal pain. No results on file yet.", "acute abdominal pain"),
    "dx-42": ("Adult with acute chest pain. No results on file yet.", "acute chest pain"),
    "wk-6":  ("Refill request: lisinopril (ACE inhibitor). No recent labs on file.", "lisinopril"),
    "ti-0":  ("Older adult with atrial fibrillation. No risk factors documented yet.", "atrial fibrillation"),
}


def load(env, cid):
    return {c["case_id"]: c for c in env.gen_cases(60, 0.0, seed=100)}[cid]


def build(env, cid, asks, decision):
    """Return a verified dialogue dict for one rollout of case `cid`."""
    case = load(env, cid)
    opening, key = OPENING[cid]
    assert key in env.user_msg(case), (cid, env.user_msg(case))
    turns = []
    for a in asks:
        val = case["assign"][env.IDX[a]]
        q, answers = PHRASE[a]
        assert val in answers, f"{cid}:{a} revealed value {val} has no verified phrasing"
        plain, sim_text = answers[val]
        sim = env.answer(case, a)
        assert sim_text in sim, (cid, a, sim, sim_text)
        turns.append((q, plain))
    reward, *_ = env.score(case, asks, decision)
    return dict(cid=cid, opening=opening, turns=turns, decision=DECISION[decision],
                risk=RISK[case["true_risk"]], reward=reward, oracle_asks=case["oracle_asks"])


def oracle(env, cid):
    c = load(env, cid)
    d = build(env, cid, c["oracle_asks"], c["oracle_action"])
    assert abs(d["reward"] - c["v_oracle"]) < 1e-9
    return d


# A: dx-13, oracle vs modal outcome-only GRPO rollout (seed 1, final checkpoint, hidden condition)
recs = [json.loads(l) for l in open(os.path.join(ROOT, "results/probe_dh_grpo_final_t12.records.jsonl"))]
recs = [r for r in recs if r["case_id"] == "dx-13" and r["cond"] == "hidden"]
modal, n_modal = collections.Counter((tuple(r["asks"]), r["decision"]) for r in recs).most_common(1)[0]
assert modal == (("abd_rigidity",), "admit") and n_modal == 6 and len(recs) == 12, (modal, n_modal)
A_OR = oracle(DX, "dx-13")
A_RL = build(DX, "dx-13", list(modal[0]), modal[1])
assert (A_OR["reward"], A_RL["reward"]) == (0.83, 0.17)
NOT_ASKED = [a for a in A_OR["oracle_asks"] if a not in modal[0]]

B = [("Diagnostics", "acute chest pain", oracle(DX, "dx-42")),
     ("Refill triage", "lisinopril, no recent labs", oracle(RX, "wk-6")),
     ("Titration", "older adult with atrial fibrillation", oracle(TI, "ti-0"))]
assert [b[2]["reward"] for b in B] == [0.83, 0.81, 1.34]

# short decision names for footers; plain finding names for the "never asked" note
DECISION_SHORT = {"Emergency treatment": "emergency treatment", "Admit": "admit",
                  "Adjust dose": "adjust dose", "Full-intensity anticoagulation": "full intensity"}
FINDING = {"rebound_tender": "rebound tenderness", "ct_free_air": "CT free air", "lactate_high": "lactate"}

# ==================================== style ====================================
# Drawn at the paper's true \textwidth (5.5 in, nips_2017.sty) so point sizes are print sizes.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
INK, MUTE, HAIR = "#1f2933", "#5f6873", "#d5d9de"
AG_FC, AG_EC = "#dbe6f3", "#9db6d4"        # agent bubble: muted blue
SM_FC, SM_EC = "#eeeff1", "#d6d9dd"        # simulator bubble: light gray
DEC_EC = "#2b4c7e"                         # decision box border (dark blue)
WARN = "#c75a12"                           # callout (Okabe-Ito vermillion, darkened)
FS, FS_HEAD, FS_TITLE, FS_PANEL = 7.0, 7.5, 8.0, 9.0   # nothing below 7 pt
LH = FS * 1.2 / 72                         # line height (in)
PADX, PADY = 0.055, 0.028                  # bubble padding (in)
GAP_QA, GAP_EX = 0.02, 0.035              # gap after a question / after an answer (in)
ROLE_H = FS * 1.2 / 72

W, H = 5.5, 9.0                            # canvas; the saved figure is cropped to its content
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
R = fig.canvas.get_renderer()


def text_w(s, size=FS, weight="normal"):
    t = ax.text(0, 0, s, fontsize=size, fontweight=weight)
    w = t.get_window_extent(R).width / fig.dpi
    t.remove()
    return w


def wrap(s, maxw, size=FS, weight="normal"):
    lines, cur = [], ""
    for word in s.split():
        cand = (cur + " " + word).strip()
        if cur and text_w(cand, size, weight) > maxw:
            lines.append(cur); cur = word
        else:
            cur = cand
    return lines + [cur]


def bubble(x0, x1, y, s, side, maxw, role=None, fc=None, ec=None, bold=False, lw=0.6, r=0.05):
    """Bubble with its top at y; side 'L' hugs x0, 'R' hugs x1. Returns (y below it, bbox)."""
    if role:
        ax.text(x1 if side == "R" else x0, y, role, ha="right" if side == "R" else "left",
                va="top", fontsize=FS, color=MUTE)
        y -= ROLE_H + 0.01
    weight = "bold" if bold else "normal"
    lines = wrap(s, maxw - 2 * PADX, FS, weight)
    bw = max(text_w(l, FS, weight) for l in lines) + 2 * PADX
    bh = len(lines) * LH + 2 * PADY
    bx = x1 - bw if side == "R" else x0
    ax.add_patch(FancyBboxPatch((bx, y - bh), bw, bh, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(bx + PADX, y - bh / 2, "\n".join(lines), ha="left", va="center", fontsize=FS, color=INK,
            fontweight=weight, linespacing=1.2, zorder=3)
    return y - bh, (bx, y - bh, bw, bh)


def dialogue(x0, x1, y, d, maxw):
    y, _ = bubble(x0, x1, y, d["opening"], "L", maxw, role="Simulator (hidden record)", fc=SM_FC, ec=SM_EC)
    y -= GAP_EX
    for i, (q, a) in enumerate(d["turns"]):
        y, _ = bubble(x0, x1, y, q, "R", maxw, role="Agent" if i == 0 else None, fc=AG_FC, ec=AG_EC)
        y -= GAP_QA
        y, _ = bubble(x0, x1, y, a, "L", maxw, fc=SM_FC, ec=SM_EC)
        y -= GAP_EX
    y -= 0.02
    y, box = bubble(x0, x1, y, d["decision"], "R", maxw, role="Decision",
                    fc="white", ec=DEC_EC, bold=True, lw=1.0, r=0.02)
    return y, box


def footer(x0, x1, y, d):
    """Under a hairline: hidden risk, decision, and reward (bold), one per line."""
    ax.plot([x0, x1], [y, y], color=HAIR, lw=0.6)
    y -= 0.05
    step = FS * 1.3 / 72
    for i, (s, w) in enumerate([(f"Hidden risk: {d['risk']}", "normal"),
                                (f"Decision: {DECISION_SHORT[d['decision']]}", "normal"),
                                (f"Reward {d['reward']:+.2f}", "bold")]):
        ax.text(x0, y - i * step, s, ha="left", va="top", fontsize=FS, color=INK, fontweight=w)
    return y - 3 * step


def panel_label(x, y, letter, title, sub):
    ax.text(x, y, letter, ha="left", va="top", fontsize=FS_PANEL, fontweight="bold", color=INK)
    ax.text(x + 0.17, y - 0.01, title, ha="left", va="top", fontsize=FS_TITLE, fontweight="bold", color=INK)
    ax.text(x + 0.17, y - 0.17, sub, ha="left", va="top", fontsize=FS, color=MUTE)


def col_head(x0, x1, y, title, sub):
    ax.text(x0, y, title, ha="left", va="top", fontsize=FS_HEAD, fontweight="bold", color=INK)
    ax.text(x0, y - 0.14, sub, ha="left", va="top", fontsize=FS, color=MUTE)
    ax.plot([x0, x1], [y - 0.30, y - 0.30], color=HAIR, lw=0.6)
    return y - 0.37


M = 0.04                                   # outer margin
# ------------------------------------ Row A ------------------------------------
yA = H - 0.04
panel_label(M, yA, "A", "Same patient, different policies",
            "Held-out diagnostics case with hidden high (emergency) risk. Replies come from a rule-based simulator\n"
            "that looks up each requested finding in the case's hidden record (present or absent, no numeric values).")
GAP_A = 0.30
CW_A = (W - 2 * M - GAP_A) / 2
cols = [(M, M + CW_A), (W - M - CW_A, W - M)]
yc = yA - 0.50
ys = [col_head(*cols[0], yc, "Exact adaptive oracle", "reference procedure: 4 questions"),
      col_head(*cols[1], yc, "Outcome-only GRPO", f"modal rollout ({n_modal} of 12 samples): 1 question")]
MAXW_A = CW_A
y_or, _ = dialogue(*cols[0], ys[0], A_OR, MAXW_A)
y_rl, box = dialogue(*cols[1], ys[1], A_RL, MAXW_A)

# callout: premature commitment (thin bracket + italic note beside the decision box)
bx, by, bw, bh = box
cx = bx - 0.08
ax.plot([cx + 0.04, cx, cx, cx + 0.04], [by + bh, by + bh, by, by], color=WARN, lw=0.8)
ax.text(cx - 0.05, by + bh / 2, "premature commitment", ha="right", va="center", fontsize=FS,
        color=WARN, style="italic")
# what the oracle checked on this patient that the policy never asked (derived from the oracle path)
na = wrap("Never asked: " + ", ".join(FINDING[a] for a in NOT_ASKED) + ".", CW_A)
ax.text(cols[1][0], y_rl - 0.08, "\n".join(na), ha="left", va="top", fontsize=FS, color=MUTE,
        style="italic", linespacing=1.2)

yfA = min(y_or, y_rl) - 0.07
endA = min(footer(x0, x1, yfA, d) for (x0, x1), d in zip(cols, (A_OR, A_RL)))
ax.plot([W / 2] * 2, [yc + 0.02, endA], color=HAIR, lw=0.6)          # column divider

# ------------------------------------ Row B ------------------------------------
yB = endA - 0.16
ax.plot([M, W - M], [yB + 0.09, yB + 0.09], color="#b9bfc6", lw=0.7)
panel_label(M, yB, "B", "Oracle procedures in the three tasks",
            "One held-out case per task, exact adaptive oracle trajectory (Appendix B).")
GAP_B = 0.16
CW_B = (W - 2 * M - 2 * GAP_B) / 3
ytop = yB - 0.38
ends, xs = [], []
for i, (task, sub, d) in enumerate(B):
    x0 = M + i * (CW_B + GAP_B); x1 = x0 + CW_B
    xs.append((x0, x1))
    y = col_head(x0, x1, ytop, task, sub)
    y, _ = dialogue(x0, x1, y, d, 0.88 * CW_B)
    ends.append(y)
yfB = min(ends) - 0.07
bottom = min(footer(x0, x1, yfB, d) for (x0, x1), (_, _, d) in zip(xs, B)) - 0.03
assert bottom > 0, f"canvas too short by {-bottom:.2f} in"

out = os.path.join(ROOT, "paper", "figs", "fig_dialogues")
crop = matplotlib.transforms.Bbox([[0, bottom], [W, H]])
fig.savefig(out + ".pdf", bbox_inches=crop)
fig.savefig(out + ".png", dpi=300, bbox_inches=crop)
print(f"wrote {out}.pdf/.png  ({W:.2f} x {H - bottom:.2f} in)")
