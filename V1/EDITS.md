# v1 replacement text (paste into Overleaf)

All numbers recomputed from `results/probe_*.records.jsonl` on 2026-11-01. Text uses hyphenated compounds
("high-risk", "outcome-only") — apply the same in the global hyphenation pass. Each block says **where** it goes.

---

## 1. High-risk subgroup sizes (n=2 on diagnostics)

Facts: held-out high-risk cases = diagnostics **2/60** (`dx-13` abdominal, `dx-42` chest), titration high-clot
**6/60**, refill **7/60**, coverage repair **14/200**. Every diagnostics arm that reports high-risk support 0.50
supports the **same** case, `dx-13`; `dx-42` is reached only by decoupled GRPO at step 40.

### 1a. §3 "Safety focused evaluation" — replace the first sentence

Old: *"Aggregate reward can conceal clinically important failures because high risk cases represent only
approximately 5–10% of the generated distribution."*

```latex
Aggregate reward can conceal clinically important failures because high-risk cases are rare: the held-out sets
contain 2 of 60 high-risk cases for diagnostics, 6 of 60 high-clot cases for titration, 7 of 60 high-risk cases
for refill triage, and 14 of 200 for the supervised coverage experiment.
```

### 1b. Table 2 (`tab:master`) caption — append

```latex
High-risk subgroup sizes are small (diagnostics $n{=}2$, titration high-clot $n{=}6$, refill $n{=}7$ cases);
with $k{=}12$ rollouts per case, diagnostics high-risk support can take only the values $0$, $0.5$, and $1$.
```

### 1c. Table 3 (`tab:dx`) caption — append

```latex
The held-out high-risk subgroup contains only two cases; every policy with high-risk support of $0.50$ supports
the same abdominal case, so high-risk values in this table are illustrative rather than precise estimates.
```

### 1d. §5.1 — replace the paragraph beginning "The variance across seeds is as important as the mean."

```latex
The variance across seeds is informative, but it must be read against a very small subgroup. The held-out
diagnostics set contains only two high-risk cases, so high-risk support of $0.50$ means that one of the two
cases is reached by at least one of twelve rollouts. The base model supports one abdominal case; one of the
three plain GRPO seeds loses it, falling to $0.0$. Mean questioning also falls from $2.5$ at initialization to
$1.82\pm0.11$, well below the oracle level. The policy is therefore improving its aggregate score partly by
reducing the amount of evidence it gathers. I treat the high-risk collapse as an illustrative failure rather
than a precise estimate; the moderate subgroup ($n{=}35$) and overall value carry the statistically more
informative evidence.
```

### 1e. §5.2 — replace the paragraph beginning "The clearest benefit is the reduction in high risk instability."

```latex
High-risk support is consistent with, but cannot establish, greater stability. Credit-assignment GRPO
maintains high-risk support of $0.50{\pm}0.00$ across all three seeds, in each case the same single abdominal
case that the base model supports, whereas one plain GRPO seed loses that case. With two high-risk cases this
difference is suggestive only. The clearer gains are on the moderate subgroup ($n{=}35$), where support rises
from $0.38{\pm}0.18$ to $0.67{\pm}0.10$, and on overall value, where the improvement from $+0.285{\pm}0.027$ to
$+0.313{\pm}0.024$ is modest and the seed ranges overlap.
```

### 1f. §5.2 — replace the last paragraph ("These results support a limited but important conclusion. …")

```latex
These results support a limited conclusion. Dense information credit improves overall value and moderate
support under the unshaped verifier and does not show the high-risk seed failure observed under plain GRPO,
but the high-risk evidence rests on two cases and three seeds. A larger stratified evaluation is needed before
treating the high-risk stabilization as established.
```

---

## 2. Decoupled GRPO checkpoint selection

Facts: step-40 ckpt chosen on the held-out set. Final ckpt (step 60): overall +0.304, high val +0.158, high
supp 0.50, moderate supp 0.74, asks 5.03, benign supp 0.09.

### 2a. Table 3 (`tab:dx`) — rename the decoupled row and add a row below it

```latex
decoupled GRPO (step 40$^\S$) & $+0.329$ & $\mathbf{+0.248}$ & $\mathbf{1.00}$ & $0.63$ & $4.4$ \\
decoupled GRPO (final, step 60) & $+0.304$ & $+0.158$ & $0.50$ & $0.74$ & $5.0$ \\
```

Append to the Table 3 caption:

```latex
$^\S$The step-40 checkpoint was selected on the held-out evaluation set and is therefore an optimistic estimate;
the final checkpoint is shown for comparison.
```

In Table 2 (`tab:master`), rename the row to `decoupled GRPO (step 40$^\S$)` and add the same footnote under the
table.

### 2b. §5.3 — insert after the paragraph ending "Overall value reaches $+0.329$."

```latex
This checkpoint was selected on the held-out set, so it is an optimistic estimate of the method. By the final
checkpoint, high-risk value returns to $+0.158$ and high-risk support to $0.50$, matching credit-assignment
GRPO, while mean questions rise further to $5.0$. The high-risk advantage of decoupling is therefore transient
in this run, whereas its over-gathering persists.
```

### 2c. Figure 1 caption, panel C — append

```latex
Decoupled GRPO is shown at its step-40 checkpoint, selected on held-out data; its final checkpoint does not
retain the high-risk advantage (Table~\ref{tab:dx}).
```

---

## 3. Coverage repair (§5.6 + Table 5)

Facts (mild bias, 30% of high-risk SFT examples kept; n=200, 14 high-risk): init high val −0.285 but high supp
1.00 (asks 2.46 on high-risk). Credit: high val +0.092, supp 0.14, high-risk asks 1.39. Decoupled: +0.040,
0.36, asks 3.71.

### 3a. §5.6 — replace the two paragraphs beginning "Passing the gate does not lead to successful repair." and "Decoupled GRPO fails in the opposite direction."

```latex
The mildly biased SFT policy, which serves as the initialization for repair, has a reliability gap rather than
an availability gap. Its high-risk support is $1.00$: for every high-risk case, at least one of twelve rollouts
is oracle grade. Its typical high-risk rollout is nevertheless poor, with high-risk value of $-0.285$.

Both RL methods change this profile without repairing it. Credit-assignment GRPO raises high-risk value to
$+0.092$ but reduces high-risk support from $1.00$ to $0.14$; mean questions fall from $3.0$ to $2.2$ overall
and to $1.4$ on high-risk cases, and overall support falls from $0.93$ to $0.45$. The policy becomes more
consistently mediocre: it avoids the worst outcomes more often but rarely reaches the oracle-grade procedure
that the initialization could still sample.

Decoupled GRPO moves in the opposite direction on gathering. Mean questions rise to $4.3$, high-risk value
reaches $+0.040$, and high-risk support falls to $0.36$. Benign support falls from $0.92$ to $0.16$, and overall
support falls to $0.41$. Neither method converts the sampled oracle-grade behaviour into reliable behaviour.
```

### 3b. Table 5 (`tab:repair`) caption — replace

```latex
Attempted repair of a mildly biased diagnostics SFT policy ($30\%$ of high-risk SFT examples retained), in the
held-out hidden condition with $n{=}200$ cases ($14$ high-risk) and the pure verifier reward. The SFT
initialization samples an oracle-grade trajectory for every high-risk case (support $1.00$) but performs poorly
on a typical high-risk rollout (value $-0.285$). Both RL methods raise high-risk value but sharply reduce
high-risk support, and both reduce overall support. Credit-assignment GRPO under-gathers; decoupled GRPO
over-gathers and damages benign support. Neither converts the sampled behaviour into a reliable procedure.
``supp'' denotes oracle-grade support@12.
```

---

## 4. Refill triage: SFT then RL (§5.5, abstract, summary sentences)

Facts: base overall +0.386 / high +0.151 / high supp 0.00 / asks 2.2. SFT +0.318 / +0.195 / 1.00 / 2.8 (benign
value falls 0.752→0.546). Plain GRPO +0.431 / +0.032 / 1.00† / 1.3. Credit +0.415 / +0.043 / 1.00† / 1.6.

### 4a. §5.5 — replace the paragraph beginning "The result indicates that the supervised warmup had already captured …"

```latex
The refill experiment reproduces the diagnostics pattern from a supervised starting point. The SFT warmup buys
its high-risk performance with additional questioning: overall value falls below the base model ($+0.318$
versus $+0.386$), largely because benign cases are over-investigated. RL then recovers and exceeds the base
model's aggregate value ($+0.431$ for plain GRPO, $+0.415$ for credit assignment) by trimming questions, while
high-risk value falls below even the untrained base model ($+0.032$ and $+0.043$ versus $+0.151$). Reducing the
number of questions was the easiest way to raise the terminal reward, and the high-risk procedure the warmup
had learned was the cost.
```

### 4b. §5.5 — replace the final paragraph ("A small or negative gain beyond SFT should therefore not be …")

```latex
An apparent gain from RL beyond SFT should therefore be audited by subgroup. In this experiment, RL improved
aggregate reward after SFT while eroding the high-risk behaviour that SFT had established; the remaining reward
pressure favoured efficiency over the learned safety procedure.
```

### 4c. Abstract — replace

Old: *"reinforcement learning provides little benefit when supervised fine tuning has already exhausted the
available headroom;"*

```latex
after a supervised warmup that already gathers carefully, reinforcement learning raises aggregate reward by
trimming questions while degrading high-risk performance;
```

### 4d. Other sentences with the same old claim — insert one word, **"safe"**

Change "RL adds little" → "RL adds little **safe** improvement" (or "little safe headroom") in: Introduction
(paragraph beginning "The results show that a verifiable terminal reward…"), contributions paragraph ("strong
supervised models may leave little useful headroom" → "little safe headroom"), §5 opening paragraph, §5.7, and
the Conclusion ("RL does not reliably add value beyond a strong SFT policy" → "does not reliably add safe value
beyond a strong SFT policy"). The section title "A strong SFT warmup leaves little safe RL headroom" already
fits.

---

## 5. Oracle question counts (define, don't change)

Facts: 3.9 (diagnostics) and 2.77 (titration) are the oracle's **expected** asks over the case distribution
(Gate 1/0 analytic). Realized on the held-out cases: **3.78** and **2.97**. Figures draw the 3.9 / 2.8 lines, so
keep those values and define them once.

### 5a. §5.1 — replace "The exact oracle requests approximately $3.9$ findings per case."

```latex
The exact oracle requests $3.9$ findings per case in expectation over the case distribution ($3.8$ on the
held-out set); I use the expected value as the reference oracle level in figures.
```

### 5b. §5.4 — replace "substantially above the oracle level of approximately $2.77$"

```latex
substantially above the oracle level ($2.77$ questions in expectation over the case distribution; $3.0$ on the
held-out set)
```

---

## 6. Credit assignment still under-gathers

### 6a. §5.2 — insert after the paragraph replaced in 1e

```latex
Credit assignment does not, however, restore oracle-level gathering. Its policies ask $2.08{\pm}0.30$
questions on average, compared with $1.82{\pm}0.11$ for plain GRPO, $2.5$ for the base model, and an oracle
level of approximately $3.9$; on the high-risk cases they ask between $1.1$ and $1.5$. The improvement is
therefore better allocation of a still-shortened procedure rather than recovery of the full gather-then-commit
behaviour.
```

### 6b. Figure 9 (`fig:traincurves`) caption — replace

Old: *"…outcome only GRPO moves below the oracle gathering level of approximately $3.9$, credit assignment
preserves an intermediate level, and decoupled GRPO gathers the most."*

```latex
outcome-only GRPO and credit-assignment GRPO both settle well below the oracle gathering level of
approximately $3.9$, with credit assignment slightly higher, while decoupled GRPO rises above it.
```

---

## 7. Intro addition (from the 2026-11-01 discussion)

Place after the RLVR paragraph ("Reinforcement learning with verifiable rewards, or RLVR, is a natural candidate
…"), before "This distinction creates an important safety problem."

```latex
Many language agents now trained with reinforcement learning share a gather-then-commit structure: search agents
decide what to retrieve before answering, coding agents what to inspect before editing, and tool-use agents which
calls to make before acting, yet the dominant recipe rewards only the final outcome. Whether outcome-only training
teaches a sound information-gathering procedure or a reward-exploiting shortcut is therefore a general open
question, and clinical decision-making is its high-stakes instance.
```

---

## Remaining v1 checklist (not drafted here)

- [ ] Figure 1: `[t]` → `[p]`; `\FloatBarrier` before `\section{Background}`; larger fonts in the figure script
- [ ] `nips_2017.sty`: define `\@nipsordinal{Preprint.}`, empty `\@nipsyear` / `\@nipslocation`
- [ ] Addendum → Appendix C; `\FloatBarrier`s per results subsection
- [ ] Appendix B: delete "Future versions should…" / "A more complete qualitative appendix should…"; delete
      `% Optional callout` comment
- [ ] Table 7 seed row: "1 (diagnostics plain/credit: 1–3)"
- [ ] Hyphenation + UK/US spelling pass
- [ ] Bib: arXiv `url`s, title braces, Feng year, MedGemma first author, Med-RLVR authors, EHRMind ID; add
      ED-Copilot, AMIE, a reward-hacking cite, MediQ
- [ ] Repo public + README rewrite + PDF at `paper/` before the Code link goes live
