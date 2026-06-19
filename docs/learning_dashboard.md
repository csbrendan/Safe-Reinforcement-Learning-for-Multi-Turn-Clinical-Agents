# RX-triage learning dashboard (provisional / internal debug)

First-pass visualizations of the gated-RL experiments so far. **These are internal learning/debug plots, not
final paper figures.** Built only from committed probe records (`results/probe_*_t10.records.jsonl`), committed
summary numbers, and the analytic Gate-1 baselines — no new training, no env/reward changes. Regenerate with
`python scripts/make_learning_plots.py`; figures live in `results/figs/learning_dashboard/`.

All model numbers are the canonical MAX_TURNS=10 / deterministic-ordering Gate-2 probes (hidden condition).

## Figures

1. **[Gate 1 — policy values](../results/figs/learning_dashboard/01_gate1_policy_values.png)** — oracle 0.854 >
   best-fixed 0.749 (the BC-gap) > every shortcut (ask-all 0.747, 1-ask 0.608, no-ask 0.601, blind-escalate
   0.376). *Lesson: the widened env genuinely requires selective adaptive questioning; shortcuts are priced out.*
2. **[Gate 1 — asks vs value](../results/figs/learning_dashboard/02_gate1_asks_vs_value.png)** — the
   value-vs-cost frontier: 0-ask is cheap but unsafe, ask-all (~6.5) overpays, the oracle sits at the knee
   (~2.9 asks). *Lesson: adaptive SELECTION wins, not simply more/fewer asks.*
3. **[Model comparison](../results/figs/learning_dashboard/03_model_comparison.png)** — small multiples of
   reward / support@12 / mean-asks / 1-ask / no-commit / high-risk reward / high-risk support. *Lesson:
   aggregate reward+support are misleading — GRPO #1 & improved-SFT show high support@12 while asks collapse
   to ~1 and high-risk reward craters.*
4. **[Ask-behavior composition](../results/figs/learning_dashboard/04_ask_behavior_composition.png)** — stacked
   {no-commit / 0 / 1 / 2 / 3+ asks} from records. *Lesson: smoke-SFT = big no-commit band + over-asking;
   GRPO #1 = near-total 1-ask collapse; improved-SFT commits but is still mostly 1-ask (under-gathers).*
5. **[Failure-mode heatmap](../results/figs/learning_dashboard/05_failure_mode_heatmap.png)** — severity
   (controlled/mild/severe) across non-termination, 1-ask shortcut, high-risk regression, benign
   over-escalation, over-gathering, under-gathering. *Lesson: each stage fixes one pathology and exposes
   another; v3 target row is all-controlled.*
6. **[Curriculum progression](../results/figs/learning_dashboard/06_curriculum_progression.png)** — smoke →
   improved trajectories with green **v3 target bands** (no fabricated v3 values). *Lesson: smoke learned to ask
   but not commit; improved learned to commit but under-gathers; v3 must teach both.*
7. **[Gate / safety status matrix](../results/figs/learning_dashboard/07_gate_status_matrix.png)** — ✓/✗/—
   grid over BC-gap, latent support, parse, no-commit, 1-ask, high-risk support, high-risk reward, benign
   over-escalation, exploit verdict. *Lesson: both gates pass; every learned stage trips ≥1 check; v3 is the
   all-green target before any GRPO #2.*

Severity/status thresholds used (documented for reproducibility): 1-ask controlled = one_ask < 0.6; no-commit
controlled < 0.05; high-risk support present > 0; high-risk reward acceptable ≳ −0.1; benign over-escalation
controlled < 0.4; over-gathering = mean asks ≫ oracle (~ask-all); under-gathering = mean asks ≪ oracle 2.9.

## Takeaways

- **Gate 1 passes:** widened RX triage requires adaptive questioning — oracle (0.854) beats best-fixed (0.749)
  and all shortcuts are below best-fixed (figs 1–2).
- **Gate 2 shows latent support exists:** even the base model has support@12 ≈ 0.58, and every stage ≥ 0.58
  (fig 3) — the capability to reach oracle-grade is present to elicit.
- **GRPO #1 is correctly labeled EXPLOIT:** it improved support (0.60→0.85) and termination (no-commit
  0.304→0.004) but **collapsed to a 1-ask shortcut** (1-ask 0.98, mean asks 1.10) and **regressed high-risk
  reward to −0.77** (figs 3–5). The support@12 rise is a k=12 sampling artifact over a degenerate policy.
- **Improved-SFT is better than smoke-SFT but not a clean baseline:** termination fixed (no-commit 0.010, parse
  0.998), reward up to 0.409, high-risk reward −0.08 and high-risk support 1.0 — **yet it still under-gathers**
  (mean asks 1.33, 1-ask 0.90 → fails the anti-shortcut guard; reward 0.409 ≪ best-fixed 0.749) (figs 4–6).
- **v3-SFT (now measured) fixed gathering + safety but slipped on termination:** mean asks 1.33→**2.78**
  (risk-graded: high 3.4 / benign 2.4), 1-ask 0.90→**0.539** (passes guard), high-risk reward −0.08→**+0.195**
  (best of all), high-risk support 1.0 — BUT no-commit 0.01→**0.196** and parse 0.998→**0.9499** (≥0.95 FAIL),
  so reward fell to 0.318. The ~5% per-turn invalid rate compounds over longer trajectories into ~20% no-commit.
- **The SFT oscillates on the termination↔gathering axis** (smoke → improved → v3); no stage is yet all-green
  (fig 7). **v4 must hold both** (reliable commit AND adaptive gathering, reward toward 0.749) before any GRPO #2.
- **Caveat:** these are provisional internal plots for intuition/debugging, not paper evidence. v4 rows are
  *targets*, not measured results.
