# Results (fresh — populated only by new runs in this repo)

Each task records its results here as it clears the gated workflow. Prior-project numbers are NOT results of this
repo; see [lessons_learned.md](lessons_learned.md) for conceptual carry-over.

> **Case study 1 — rx_triage: COMPLETE & FROZEN.** Paper-ready writeup:
> **[rx_triage_case_study.md](rx_triage_case_study.md)**. Headline: Gate 1 (BC-gap +0.105) and Gate 2 (latent
> support 0.58) both PASS, but neither SFT (v2–v5) nor two bounded GRPO runs realize adaptive gathering — both
> collapse to a 1-ask shortcut under the ask-cost. **Gates are necessary but not sufficient**; reward geometry
> (concentrated vs diffuse gathering benefit) is the third factor. Visual dashboard:
> [learning_dashboard.md](learning_dashboard.md). Next: replicate the pattern on Task 2 (diagnostics).

## Headline (to be filled): RL gain vs demonstrability
Plot: `results/figs/rl_gain_vs_bcgap.png` — one point per (task, reward/scale setting): x = BC-gap,
y = RL − SFT held-out value, only where latent support was confirmed.

| task | model | reward | BC-gap (deep) | latent support@k | RL − SFT (held-out) | verdict |
|---|---|---|---|---|---|---|
| rx_triage | medgemma-4b (text) | v3 0.318 / GRPO2 0.431 | +0.105 | base 0.58 / v3 0.97 / GRPO2 0.97 | GRPO2: asks 2.78→1.31, hi-rew 0.195→0.032 | **GRPO#1 & #2 EXPLOIT (1-ask collapse); SFT can't stably gather. Gates necessary, NOT sufficient under this reward** |
| diagnostics | medgemma-4b (text) | — | **+0.117** | — | — | **Gate 1 PASS** (emergent ~12%); Gate 2/SFT/GRPO pending |
| longterm_care | — | — | — | — | — | (pending) |

## Per-task detail (template)
For each task, record: Gate-1 (BC-gap + policy/subset table + checks), Gate-2 (base & SFT pass@k, blind-rate,
ask-hist), the bounded GRPO config + training trajectory (KL, frac_reward_zero_std), the held-out diagnostic
(SFT vs RL by subset, blind-rate, high-risk support, benign over-escalation), and the four-way verdict.

### rx_triage
- **Gate 1: PASS** at env build (BC-gap +0.105; oracle 0.854 vs best-fixed 0.749; anchor ≈0). Regenerate:
  `python -m framework.gates.bc_gap --env tasks.rx_triage.env`.
- **Gate 2: PASS** (2026-06-18). Base = text-only medgemma-4b-it (vision tower dropped via
  `framework.infra.prep_text_base`; clean foundation, no domain SFT). Smoke SFT = 160 best-fixed-checklist
  trajectories (`tasks.rx_triage.gen_data`), 2 ep / 80 steps. Probe k=12, n=60 cases/condition, hidden
  prioritized. Driver: `scripts/run_gate2.sh`; report: `framework.gates.gate2_report`.

  **Canonical run: MAX_TURNS=10, deterministic oracle ordering** (remediation `4879cf2` after the audit below).
  Hidden-condition, by risk subset (supp@12 / mean reward / mean asks):

  | run | overall | high | moderate | benign | relevant | parse | reward std | high-risk supp | benign over-esc | no-commit |
  |---|---|---|---|---|---|---|---|---|---|---|
  | base | 0.58 / +0.386 / 2.19 | 0.00 / +0.15 / 3.0 | 0.50 / +0.16 / 2.3 | 0.87 / +0.75 / 1.9 | 0.41 / +0.16 / 2.4 | 0.994 | 0.519 | **0.00** | 0.181 (overesc) | 0.085 |
  | smoke-SFT | 0.60 / +0.168 / 4.48 | 0.14 / −0.43 / 4.7 | 0.43 / −0.06 / 5.1 | 0.96 / +0.65 / 3.6 | 0.38 / −0.13 / 5.0 | 0.985 | 0.544 | **0.143** | 0.022 | **0.304** |

  Analytic baselines (hidden, cost 0.03): oracle 0.854, best-fixed 0.749, ask-all 0.747, no-ask 0.601,
  1-ask 0.608, blind-escalate 0.376. `oracle_grade_blind_rate = 0.0` for BOTH runs → every oracle-grade
  success used real gathering (not blind-commit coincidence). All 10 Gate-2 checks PASS.

  **Anomaly audit + remediation** (full: [docs/eval_runs/gate2_audit_rx_triage.md](eval_runs/gate2_audit_rx_triage.md)).
  Independent re-score (1440 rollouts) cleared any reward/env/parser/prompt bug (0/720 reward mismatches, 0
  leaks). base>SFT mean reward is the demonstrability trap (base banks 79% of its reward from benign cheap-
  `continue` + a blind `hold_for_labs` hedge on high that scores 0.40 without being oracle-grade). Two confounds
  were addressed: (i) a `v_oracle` ±0.03 wobble from `PYTHONHASHSEED`-dependent set ordering in the oracle path
  (fixed: sort `rel_items`; affects only the oracle-grade threshold, never reward); (ii) the SFT no-commit rate.
  **Raising MAX_TURNS 8→10 did NOT reduce no-commit (0.318→0.304); the model just asked more (4.05→4.48)** — so
  no-commit is a genuine non-termination pathology of the undertrained smoke SFT, not a turn-budget artifact.
  Net: the SFT reward shortfall is dominated by behavior RL must fix (commit reliably; stop asking earlier),
  with a real safety/support signal underneath (high-risk support 0→0.14, noisy on 7 cases; benign over-esc
  0.181→0.022). The earlier t8 run (high-risk support 0.43) is retained in git history (`b34341c`) for provenance.

  **Verdict — latent support confirmed, capability latent not expressed.** Support@12 is non-trivial (0.55 →
  0.62) and clean (blind-rate 0). Smoke-SFT's mean reward (0.180) sits far below best-fixed (0.749) and oracle
  (0.854): SFT learned the gather→commit protocol (mean asks 2.14→4.05) but over-asks (pays ask-cost) and
  mis-commits high/moderate, so reward *fell* vs base — because the base exploits the demonstrability trap
  (fast "continue" on a benign-heavy mix scores high without solving the hard subsets). The discriminator that
  matters moved the right way: **high-risk support 0.00 → 0.43** (base never reached oracle-grade on a single
  high-risk case; SFT does, always via 3+ asks — high-risk og_ask_hist = 100% at 3+), and benign over-escalation
  0.167 → 0.025. This is the safety subset where the BC-gap lives becoming reachable. Reward std 0.55 gives
  ample rollout diversity for an RL signal, and there is no degenerate collapse (no-ask-commit 0, 1-ask 0.44,
  mean asks 4.05 < rel-union 7, blind-escalate 0). → An ideal RL starting point: the policy *can* reach
  oracle-grade (0.62) but doesn't *reliably* (reward 0.18); RL has a clear, safe gradient.
- **GRPO #1 (bounded, 60 steps): ran — verdict Exploit/termination, NOT thesis-positive.** Full report:
  [docs/eval_runs/grpo_report_rx_triage.md](eval_runs/grpo_report_rx_triage.md). GRPO cleanly fixed termination
  (no-commit 0.304→**0.004**, parse→1.0) but **collapsed to a 1-ask shortcut** (mean asks 4.48→**1.10**, 98%
  one-ask), so overall reward rose only to 0.261 (< base 0.386 ≪ best-fixed 0.749) and **high-risk reward
  regressed −0.43→−0.77** (1 ask can't serve high-risk; high-risk support@12 0.14→0.43 is a k=12 sampling
  artifact). This is the expected consequence of RL optimizing against a smoke-SFT init confounded by ~30%
  non-termination — RL's easiest win was commit-after-one-ask, not adaptive gathering. **Next: rebuild the SFT
  baseline (curriculum generator, commit `cb3bce0`) to teach ask→stop→commit, re-run Gate 2, then re-evaluate;
  no second GRPO until that review.** (`framework.report` mislabels this THESIS-POSITIVE — it lacks an
  ask-collapse / action-quality guard; see the report.)
- **Improved-SFT (curriculum) Gate 2:** full report
  [docs/eval_runs/sft_v2_gate2_rx_triage.md](eval_runs/sft_v2_gate2_rx_triage.md). The curriculum fixed
  termination (no-commit 0.304→**0.010**, parse 0.998) and lifted reward to **+0.409** (best non-oracle; high-risk
  reward −0.43→−0.08, high-risk support→1.0) — BUT **over-corrected into under-gathering**: mean asks 1.33,
  1-ask rate 0.90 → **Gate 2 FAILS** the anti-1-ask-shortcut check, and reward 0.409 ≪ best-fixed 0.749 (it
  under-fit the gathering it was trained to clone). Not yet a clean baseline. **Next: curriculum v3** (more rho=0
  full-checklist-then-commit, fewer commit-now/short) so the SFT both gathers AND commits before any GRPO.
  **No GRPO #2 proposed/launched.**
- **Curriculum v3 SFT Gate 2:** full report
  [docs/eval_runs/sft_v3_gate2_rx_triage.md](eval_runs/sft_v3_gate2_rx_triage.md). v3 **fixed the under-gathering**
  (mean asks 1.33→**2.78**, risk-graded high 3.4/benign 2.4; 1-ask 0.90→**0.539** passes guard) and delivered the
  **best high-risk safety yet** (high-risk reward −0.08→**+0.195**, support 1.0) — BUT **termination regressed**
  (no-commit 0.01→**0.196**, parse 0.998→**0.9499** → Gate-2 FAIL), pulling reward to 0.318. Mechanism: ~5%
  per-turn invalid rate compounds over v3's longer trajectories. The SFT is oscillating on the
  termination↔gathering axis (smoke→v2→v3); no stage is yet all-green. **Next: v4** (hold both gather AND commit:
  no-commit≈0, parse≈1.0, asks ~2.8–4 risk-graded, reward→0.749). **No GRPO proposed/launched** — v3 did not pass
  cleanly.
- **Curriculum v4 SFT Gate 2:** full report
  [docs/eval_runs/sft_v4_gate2_rx_triage.md](eval_runs/sft_v4_gate2_rx_triage.md). v4 (v3 mix + 2× training)
  **fixed v3's validity+termination** (parse 0.9499→**0.9896**, no-commit 0.196→**0.029**) and kept positive
  high-risk reward (+0.197) + risk-grading — **but the extra training reverted gathering to 1-ask** (mean asks
  2.78→**1.53**, 1-ask 0.539→**0.796** → Gate-2 FAIL, the only failing check). **Finding:** an SFT oscillation —
  more training fixes validity/termination but collapses gathering into the 1-ask basin; less training gathers
  but degrades validity. No (mix, epochs) point hits gather≈2.8 AND parse≈0.99 AND no-commit≈0 together — direct
  evidence that oracle-depth adaptive gathering is hard to *instill by demonstration* (the demonstrability
  difficulty). **No GRPO.** Next decision: (a) v5 = v4 data @ 2 epochs (sweet-spot search), or (b) treat v3/v4 as
  the baseline frontier and run a guard-instrumented bounded GRPO. Awaiting direction.
- **v5 SFT + temperature diagnostic:** full report
  [docs/eval_runs/sft_v5_and_temp_diagnostic_rx_triage.md](eval_runs/sft_v5_and_temp_diagnostic_rx_triage.md).
  v5 (v4 data @ 2 ep) **FAILS** (1-ask 0.918, gathered *least* of all — refutes the interpolation idea). The
  **temperature diagnostic is decisive**: v4/v5 commit after exactly 1 ask at **greedy** (temp 0) — under-gathering
  is a **learned policy, not a sampling artifact** (higher temp *adds* gathering); v3 gathers at all temps. SFT
  frontier: only **v3** learned to gather (greedy 2.23, risk-graded); v2/v4/v5 collapsed to greedy 1-ask —
  **adaptive gathering is unstable/rare under SFT**, direct evidence that demonstrations under-specify the optimal
  policy. **Decision: SFT search capped; proceed to a guarded GRPO #2 from v3** (KL-anchored β=0.02; ask-collapse
  + high-risk-reward guards) — **proposed, NOT launched**, awaiting review.
- **Guarded GRPO #2 (from v3, β=0.02, temp 1.0):** full report
  [docs/eval_runs/grpo2_rx_triage.md](eval_runs/grpo2_rx_triage.md). Held-out trajectory: asks declined
  monotonically 2.78→1.81(s20)→1.38(s40)→**1.31**(s60), 1-ask 0.54→**0.87**; reward 0.318→**0.431**, parse
  0.95→**0.998**, no-commit 0.196→**0.004**; but **high-risk acceptable-decisions fell 0.50→0.43** and high-risk
  reward 0.195→**0.032**. Guarded verdict **EXPLOIT / under-gathering — NOT thesis-positive** (no mid-run peak;
  KL anchor + gatherer init *mitigated* the collapse vs GRPO #1's −0.77 high-risk reward, but didn't prevent it).
  **Conclusion: BC-gap + latent support are necessary but NOT sufficient** for a bounded GRPO to realize the
  gain here — the ask-cost vs rare-high-risk-benefit credit assignment is a strong 1-ask attractor for both SFT
  and RL. Realizing the gain likely needs reward redesign (less-diffuse gathering benefit), not more RL.

### diagnostics / longterm_care
- pending env build.
