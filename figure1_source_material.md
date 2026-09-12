# Figure 1 Source Material

Verified source material for redesigning **Figure 1** of the paper (working retitle:
*"Reward Design Governs Information Gathering in Clinical Multi-Turn Agents"*).

**Rules honored in this document:** nothing is inferred; every value carries a `file:line` (or record/case)
citation; where the manuscript prose and the repo/results disagree, the **repo/results value is preferred** and
the conflict is flagged in §8. Values that exist only in CLAUDE.md (not in code) are labeled as such.

**Scope reminder for the figure:** Figure 1 should convey the *experimental logic*, not the Results section.
Keep to the four panels (A–D) in §Figure-1 asset checklist. Do **not** pack every experiment/number in.

> ⚠️ **Read-before-drawing flags (full list in §8):**
> 1. **Diagnostics = `tasks/diagnostics/env_rv.py` with `DX_VARIANT=deephedge`, 18 findings, turn cap 12.** The
>    other file `tasks/diagnostics/env.py` is an *older 14-finding* variant that does **not** reproduce the paper
>    results — do not cite it for the figure.
> 2. **Three different paper titles are in play** (repo `.tex`, CLAUDE.md, and the user's new title). Use the
>    user's new title on the figure.
> 3. **The `.tex` manuscript lives in the repo** at `paper/main.tex` — contradicting CLAUDE.md's claim that it is
>    Overleaf-only.
> 4. **Prefer the manuscript tables / CLAUDE.md over manuscript prose** — several prose numbers are stale
>    single-seed values.

---

## 1. Core ask-to-commit loop

The three tasks share one POMDP interface (`framework/envs/base.py:8-25`). Diagnostics deephedge
(`tasks/diagnostics/env_rv.py`) is used as the reference below since it is the paper's main task.

### 1.1 Model-visible input structure
Each turn the model receives a two-message chat: a fixed **system** prompt + a **user** turn that is the case
chart, then the growing dialogue. Built in the trainer at `framework/rl/grpo_disagg.py:291-293`:
```python
ds = Dataset.from_list([{"prompt": [{"role": "system", "content": env.SYS},
                                    {"role": "user", "content": f"[case {c['case_id']}]\n" + env.user_msg(c)}]}
                        for c in cases])
```
**System prompt** (`env_rv.SYS`, `tasks/diagnostics/env_rv.py:274-281`, verbatim):
```
You are an acute-care triage assistant. The decision-relevant findings depend on what you order, so gather only
what you need before deciding. Each turn output EXACTLY ONE action as JSON (one item per turn).
To order a test/finding: {"action_type":"ask","intent":"<item>"} where <item> is one of: troponin_high,
ecg_ischemia, hypotension, dyspnea_severe, radiation_arm, pleuritic_pain, prior_cad, abd_rigidity,
rebound_tender, lactate_high, fever_high, wbc_high, guarding, ct_free_air, renal_impair, age_high,
anticoag_use, copd
To decide disposition: {"action_type":"commit","decision":"discharge|symptomatic_tx|observe|admit|emergent_tx",
"rationale":"<short>"}
Ordering has a small cost; order only decision-relevant tests (which depend on earlier results), then commit.
```
**User turn (t=0)** (`env_rv.user_msg`, `tasks/diagnostics/env_rv.py:299-306`). With RL/eval `rho=0` the chart is
empty, e.g. for an abdominal case (verbatim, deterministic):
```
Presentation: adult with acute abdominal pain.
Establish the work-up before deciding disposition.
Chart: no results on file yet.
```
Each turn the assistant text is appended as `assistant`; a valid ask's answer is appended as `user`; the whole
list is re-sent (`framework/rl/grpo_disagg.py:85-96`).

### 1.2 Allowed `ask` action format
`framework/parsing.py:19-20` — any JSON object with `action_type:"ask"` and a non-empty string `intent`:
```python
if o.get("action_type") == "ask" and isinstance(o.get("intent"), str) and o.get("intent"):
    return ("ask", o["intent"])
```
The parser takes the first `{...}` JSON object (`framework/parsing.py:10,14-18`); the intent is canonicalized by
`resolve_ask` (lowercase, strip `order_/ask_/test_`, must match an `ITEMS` key) — `tasks/diagnostics/env_rv.py`
(same logic as `env.py:169-177`).

### 1.3 Simulator response format
Natural-language text, **not JSON** (`env_rv.answer`, `tasks/diagnostics/env_rv.py:308-312`):
```python
def answer(case, intent):
    item = resolve_ask(intent)
    if item is None:
        return "That is not a recognized test."
    return f"Result: {_REVEAL[item][case['assign'][IDX[item]]]}."
```
`_REVEAL` maps each item to a `(negative, positive)` phrase pair (`env_rv.py:253-272`), e.g.
`troponin_high → ("troponin normal","troponin elevated")`. Example returned text: `Result: troponin elevated.`

### 1.4 Allowed `commit` action format
`framework/parsing.py:21-24` — JSON with `action_type:"commit"` and a string `decision` (+ `rationale`):
```python
if o.get("action_type") == "commit" and isinstance(o.get("decision"), str):
    ...
    return ("commit", (o.get("decision"), act))
```
Valid diagnostics decisions (`tasks/diagnostics/env_rv.py:45`):
`ACTIONS = ("discharge", "symptomatic_tx", "observe", "admit", "emergent_tx")`. A legacy `<answer>{...}</answer>`
form is also accepted (`framework/parsing.py:25-34`).

### 1.5 Reward equation
`env_rv.score` (`tasks/diagnostics/env_rv.py:314-320`, same shape as `env.py:241-247`):
```python
reward = U[case["true_risk"]][decision] - ASK_COST * n_asks   # n_asks counts only recognized asks
# invalid/absent commit -> 0.0 ; oracle_grade = reward >= v_oracle - EPS
```
So **`R = U[μ][a] − c · n_asks`**. Manuscript statements match: `R = U[μ][a] − c·n_asks`
(`paper/main.tex:134-135, 230-231`).

### 1.6 Ask cost `c`
`tasks/diagnostics/env_rv.py:52` → `ASK_COST = 0.015 if DX_VARIANT in ("lowcost","ah_lc") else 0.03`; under
**deephedge this is `0.03`**. Same 0.03 in refill (`rx_triage/env.py:30`) and titration (`titration/env.py:44`).
Oracle-grade tolerance `EPS = 0.03` (`env_rv.py:53`). Manuscript: `c=0.03`, `EPS=0.03` (`paper/main.tex:230-233`).

### 1.7 What is hidden from the model
- Latent severity `true_risk ∈ {none, moderate, high}` — computed from the full binary finding vector, never shown
  (`env_rv.py`; base contract `framework/envs/base.py:25` "hidden state excluded").
- The full binary assignment `assign` (true value of every finding); only `observed` items appear, and RL/eval use
  `rho=0` so `observed={}` (`framework/rl/grpo_disagg.py:288`; `support_probe.py:114`).
- The analytic `U` / `risk_dist` / oracle (ground truth for reward + gates only), never exposed.

### 1.8 What ends an episode
`framework/rl/grpo_disagg.py:87-98` — three terminating conditions:
1. **Commit** parsed → `done=True` (`:91-93`).
2. **Parse failure / unrecognized ask** (invalid, or ask whose `resolve_ask` is `None`) → `done=True` (`:97-98`).
3. **Turn cap** — loop bound `MAX_DIALOG_TURNS` (default 10, `:25`), overridden by `--max-turns`; **diagnostics = 12**.
No commit within the cap ⇒ `decision=None` ⇒ reward `0.0`.

---

## 2. Three task summaries

Shared: per-ask cost 0.03, EPS 0.03, latent severity walked from a per-family binary tree, exact adaptive oracle.
**Turn caps and BC-gaps are NOT env constants** — turn cap is a runtime `--max-turns` flag; BC-gap is computed by
`framework/gates/bc_gap.py` (values below are from CLAUDE.md / manuscript, cited per task). **No natural-language
question text exists in code** — the model asks by finding *key* (`intent="<item>"`); each key has a
`(negative, positive)` result-reveal tuple, listed below as the representative questions.

### 2.1 Refill triage — `tasks/rx_triage/env.py`
- **One-sentence description:** medication-refill triage as an adaptive-workup POMDP (`rx_triage/env.py:1`); SYS
  "You are a medication-refill triage assistant." (`:196`).
- **Latent state:** severity `none/moderate/high` (keys of `U`, `rx_triage/env.py:36-40`).
- **# queryable findings: 12** (`rx_triage/env.py:20-21`): `renal_decline, potassium_high, volume_dehydrated,
  bp_low, nsaid_use, adherence_poor, acute_symptoms, egfr_severe, sodium_low, gi_bleed_risk, lft_abnormal, a1c_high`.
- **Terminal actions (5):** `continue, dose_adjust, bridge, hold_for_labs, escalate` (`rx_triage/env.py:28`).
- **Families (2):** `ACEi_ARB`, `NSAID` (`DEEP_FAMILIES`, `rx_triage/env.py:25`).
- **Turn cap:** 10 (runtime default; CLAUDE.md + `tab:envs` `paper/main.tex:249`).
- **BC-gap:** +0.105 (CLAUDE.md; `tasks/rx_triage/design.md:22`; **not** in the manuscript — see §8 Flag 7).
- **Example case:** latent `high` → oracle `escalate` (U 0.95, `rx_triage/env.py:39`). (`none`→`continue` 1.00;
  `moderate`→`dose_adjust` 0.90.)
- **Representative questions (key → reveal, `rx_triage/env.py:181-194`):** `renal_decline`
  ("eGFR stable"/"eGFR declining"); `potassium_high` ("potassium normal"/"potassium elevated");
  `volume_dehydrated` ("euvolemic"/"dehydrated"); `nsaid_use` ("no NSAID use"/"ongoing NSAID use");
  `gi_bleed_risk` ("low GI-bleed risk"/"high GI-bleed risk").

### 2.2 Diagnostics (deephedge) — `tasks/diagnostics/env_rv.py`, `DX_VARIANT=deephedge`
- **One-sentence description:** acute-presentation diagnostic work-up as an adaptive value-of-information POMDP —
  order tests, then commit a disposition (`env_rv.py:1,3-4`); SYS "You are an acute-care triage assistant." (`:275`).
- **Latent state:** severity `none/moderate/high` = benign/urgent/emergent (`env_rv.py:66-73`).
- **# queryable findings: 18** (ask space; `env_rv.py:19-24`): 14 base
  (`troponin_high, ecg_ischemia, hypotension, dyspnea_severe, radiation_arm, pleuritic_pain, prior_cad,
  abd_rigidity, rebound_tender, lactate_high, fever_high, wbc_high, guarding, ct_free_air`) + 4 deephedge extras
  (`renal_impair, age_high, anticoag_use, copd`). Decision trees use 10 (chest) / 9 (abd) of them.
- **Terminal actions (5):** `discharge, symptomatic_tx, observe, admit, emergent_tx` (`env_rv.py:45`); hedge =
  `admit`, escalation = `emergent_tx`.
- **Families (2):** `chest_pain`, `abd_pain` (`env_rv.py:42`).
- **Turn cap:** 12 (script override, e.g. `scripts/run_procreward_arms.sh:23`; `tab:envs` `paper/main.tex:250`).
- **BC-gap:** +0.161 (CLAUDE.md + prose `paper/main.tex:421`; **not** a code constant — `design.md:22`'s +0.117
  is the *base* env, not deephedge).
- **Deephedge trap (key for the figure):** the hedge `admit` is capped (moderate 0.45, high **0.20** vs
  `emergent_tx` 0.95, `env_rv.py:67-73`); emergent is a **4-finding conjunction** so no single finding confirms it
  (chest: troponin&ecg&hypotension&dyspnea `:117`; abd: rigidity&rebound&lactate&ct_free_air `:132`).
- **Example case:** latent `high`(emergent) → oracle `emergent_tx` (U 0.95, `env_rv.py:71-72`).
  (`none`→`discharge` 1.00; `moderate`→`symptomatic_tx` 0.90.)
- **Representative questions (key → reveal, `env_rv.py:253-272`):** `troponin_high`
  ("troponin normal"/"troponin elevated"); `ecg_ischemia` ("ECG without ischemia"/"ECG shows ischemia");
  `hypotension` ("blood pressure stable"/"hypotensive"); `dyspnea_severe` ("no significant dyspnea"/"severe
  dyspnea"); `ct_free_air` ("CT: no free air"/"CT: free air").

### 2.3 Titration — `tasks/titration/env.py`
- **One-sentence description:** risk-benefit titration as an adaptive-VOI POMDP — choose where on an ordered
  intensity scale (none < low < high) to treat under a clot-vs-bleed trade (`titration/env.py:1-5`); family
  "older adult with atrial fibrillation, deciding antithrombotic intensity" (`:184`).
- **Latent state:** **joint clot × bleed, 9 states** (`titration/env.py:57`):
  `low_low, low_mod, low_high, mod_low, mod_mod, mod_high, high_low, high_mod, high_high`. `age_high` and `htn` are
  shared across both axes (`:32-34`), the source of the interesting VOI.
- **# queryable findings: 6** (`titration/env.py:26`): `prior_clot, age_high, chf, htn, bleed_history, fall_risk`.
- **Terminal actions (3, ordered):** `none, low, high` (`titration/env.py:40`); hedge = `low`, escalation = `high`.
- **Families (1):** `anticoag` (`titration/env.py:36`).
- **Turn cap:** 10 (runtime default; `tab:envs` `paper/main.tex:251`).
- **BC-gap:** +0.075 (CLAUDE.md + prose `paper/main.tex:511`; computed live in `tasks/titration/design_gate.py`).
- **Example case:** latent `high_low` (high clot, low bleed) → oracle `high` (U 1.4, computed `titration/env.py:72`).
  Contrast `low_high` → `none` (U 0.0 > −1.5); `high_high` → `low` (the hedge is genuinely optimal, U 0.19).
- **Representative questions (key → reveal, `titration/env.py:186-197`):** `prior_clot` ("no prior
  thromboembolism"/"prior stroke/VTE"); `age_high` ("younger patient"/"elderly (>=75)"); `chf` ("no heart
  failure"/"heart failure"); `bleed_history` ("no prior bleeding"/"prior major bleed"); `fall_risk` ("low fall
  risk"/"high fall risk").

---

## 3. Training / initialization pipeline

- **Base model:** text-only **`google/medgemma-4b-it`** with the SigLIP vision tower + projector dropped into a
  `Gemma3ForCausalLM`, saved bf16 (`framework/infra/prep_text_base.py:20,26-49`). Supplied at runtime via
  `--base`; no weights in repo.
- **LoRA (GRPO):** `r=32, lora_alpha=64`, targets `q/k/v/o_proj, gate/up/down_proj`, dropout default 0.0
  (`framework/rl/grpo_disagg.py:300-301`). **LoRA (SFT):** `r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
  target_modules="all-linear"` (`framework/rl/sft_warmup.py:82-83`).
- **SFT stage (where applicable):** generic multi-turn protocol warmup; supervises **every assistant turn** (asks
  + commit), masks system/user, standard causal-LM CE (`framework/rl/sft_warmup.py:1-6,39-46,95-97`). Defaults
  `--epochs 2 --batch 4 --lr 1e-4 --max-seq-len 2048` (`:68-72`). Data = jsonl of `{"messages":[...]}` oracle/
  checklist trajectories (CLAUDE.md: 800 trajectories, seed 7).
- **GRPO stage:** TRL `GRPOTrainer` with a custom `rollout_func` + echo reward `reward_echo`
  (`framework/rl/grpo_disagg.py:251-252,319-320`). Coded config `per_device_train_batch_size=8,
  num_generations=8, gradient_accumulation_steps=4, max_completion_length=512, bf16, use_vllm=True,
  vllm_mode="server", vllm_importance_sampling_correction=False` (`:302-308`). CLI defaults `--max-steps 80
  --n-cases 400 --seed 1 --lr 1e-5 --beta 0.0 --rollout-temp 1.25 --save-steps 40` (`:262-268`). **The diagnostics
  headline run overrides** (CLAUDE.md common commands): `--max-steps 60 --lr 1e-5 --beta 0.02 --rollout-temp 1.0
  --save-steps 20 --n-cases 300 --seed 1 --max-turns 12`.
- **Rollout group size:** **8** rollouts/prompt (`num_generations=8`, `framework/rl/grpo_disagg.py:303`).
- **How MedGemma is served during RL:** disaggregated 2-node — trainer (node A) calls a vLLM **server** on node B
  via `trl vllm-serve --model <dir> --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.55 --dtype bfloat16
  --max-model-len 4096` (`framework/infra/serve_b.sh`); client `client.chat(..., n=1,
  temperature=rollout_temp, max_tokens=128)` (`framework/rl/grpo_disagg.py:75,85`). Node B default
  `192.168.100.2:8000` (`:260-261`); node A `192.168.100.1` is in CLAUDE.md only (not in code).
- **Simulator tokens masked from the gradient:** `_env_mask` sets mask=1 **only** for `assistant`-message tokens;
  all system/user tokens — including the simulator's `env.answer` replies — stay 0 and are excluded
  (`framework/rl/grpo_disagg.py:35-43`, returned as `"env_mask"` at `:103,154`).

---

## 4. Experimental branches

All training rewards are assembled in the GRPO `rollout_func`; **`env.score` is never modified** — shaping is added
on top. Shared reward loop (`framework/rl/grpo_disagg.py:100-107`):
```python
r  = env.score(s["case"], s["asks"], s["dec"])[0]          # verifiable clinical reward
ig = 0.0 if infogain_coef == 0.0 else traj_infogain(env, s["case"], s["asks"])
rewards.append(r + infogain_coef * ig)
```

- **Outcome-only / plain GRPO** (`--infogain-coef 0.0`, the control): **`R_train = env.score[0]`** (pure verifier;
  `ig` forced to 0, `grpo_disagg.py:105-107,270-271`).
- **Credit-assignment GRPO** (`--infogain-coef 0.2`; **0.1** for titration): **`R_train = env.score + λ·Σ_t ΔH_t`**.
  `ΔH_t = max(H(risk_dist(fam, revealed_before)) − H(risk_dist(fam, revealed_after)), 0)`, summed only over **new,
  relevant** asks; unrecognized or repeated asks `continue` → 0 credit (`traj_infogain`, `grpo_disagg.py:51-69`;
  entropy `:46-48`). Manuscript potential: `Φ = −ΔH(p(μ|revealed))` (`paper/main.tex:307-308`).
- **Decoupled explore/exploit GRPO** (`--dream`, DREAM-inspired): two phases alternated by step parity
  (`grpo_disagg.py:163-248`). **Explore** (ask-only) reward = `traj_infogain` (recoverability); **exploit**
  (commit-only, over gathered findings) reward = pure `env.score[0]` (`:220-221`). **Hard leak guard:** commit
  prompt = `env.commit_user_msg(case, asks)` and `env.assert_no_leak(...)` raises if any un-gathered/privileged
  finding appears (`:200-210`). `tasks/diagnostics/env_rv.py` is the revealed-set entry point used by `--dream`.
- **SFT → GRPO headroom experiment** (refill): SFT warmup (`framework/rl/sft_warmup.py`, via
  `scripts/run_sft_gate2.sh`) → GRPO from the merged init → guarded report. Tests whether a strong/saturated
  warmup leaves RL any room. No single dedicated script; recorded in `docs/eval_runs/sft_v*_gate2_rx_triage.md`.
- **Biased-SFT → RL repair experiment** (diagnostics): `tasks/diagnostics/gen_sft_biased.py` builds oracle SFT
  data that under-samples high-risk (`--keep-high`, drop rule `:49-50,62-63`; severe=0.0, mild=0.30→21%
  `emergent_tx`). Two repair arms from that init — **info-gain repair** (credit-assignment GRPO) and **DREAM
  repair** (decoupled GRPO) (`scripts/plot_repair_4arm.py:8-10`). Result: neither restores high-risk support.
- **Claude process supervision** — **OPTIONAL / NOT IN THE CURRENT PAPER; a LEARNED process reward, not
  deterministic RLVR.** After each rollout, **Claude Opus (`claude-opus-4-8`, temp 0)** scores the student-visible
  transcript against a JSON rubric → clamped scalar `overall_process_score ∈ [−5,5]`
  (`framework/procreward/rollout_judge.py:29-31,82-87`); within-case z-score/clip normalization then
  **`training_reward = analytic_terminal_reward + α·normalized_claude_process_reward`**
  (`framework/procreward/reward.py:3,39-56`). Wired into GRPO but **off by default** (`--claude-alpha` default 0.0,
  `grpo_disagg.py:108-147,275-276`). **Status (corrected per author, ~2026-09; supersedes the repo README's
  "sweep running" text, which is stale):** the α-sweep is **COMPLETE and is a NEGATIVE result** — the matched
  α=0 control was run, **α=0.05 is inert and α=0.10 is harmful**; the learned Claude process reward did **not**
  beat the analytic info-gain credit. It is a **secondary negative experiment, not in the manuscript**
  (`paper/main.tex`, confirmed absent). → **Leave out of Figure 1 entirely** (the author agrees; the figure's
  story is reward *geometry*, and this is a secondary negative). Do not reuse the README's outdated
  "running/awaiting review" status in any manuscript revision.

---

## 5. Common held-out evaluation

- **Shaping is OFF at eval — verified.** Evaluation runs through `framework/gates/support_probe.py`, which imports
  **only `env.score`** (no `traj_infogain`, no `infogain_coef`, no Claude) and computes reward with pure
  `env.score` (`support_probe.py:59-60,118`); `framework/report.py` then reads the `probe_*.records.jsonl` and
  reports on the stored pure-verifier `reward`. Neither file references any shaping term.
- **Exact eval reward:** `reward = env.score(case, asks, decision)[0]` = `U[true_risk][decision] − 0.03·n_asks`
  (else 0.0), `n_asks` = count of resolvable asks.
- **Overall value:** mean reward over the subset (`report.py:29`, `value=mean(r["reward"])`).
- **High-risk value:** same on the `true_risk=="high"` subset (`report.py:20-21,54`).
- **support@k (k=12):** fraction of cases with ≥1 of k rollouts oracle-grade
  (`report.py:26-27`; `support_probe.py:71`; k default 12 `:91`). `oracle_grade = reward ≥ v_oracle − EPS`.
- **Mean asks:** `mean(r["n_asks"])` (`report.py:29`; `support_probe.py:75`).
- **Premature commitment / commit-before-enough-gathered:** *no single named scalar* in `report.py`; proxied by
  the `HR_SUPP_COLLAPSE` guard (`report.py:66,80-81`) and by `blind_og`. A dedicated `premature` metric lives in
  `framework/procreward/eval_metrics.py` (real-rollout report cites "premature −0.93 vs +0.8",
  `procreward/README.md:48`). → **NOT FOUND as a core-eval scalar.**
- **Unsupported escalation / blind commit:** `blind_og` = oracle-grade reached with 0 asks (`report.py:30-31`);
  probe `oracle_grade_blind_rate` + `zero_ask_commit_rate` (`support_probe.py:77-79`).
- **Over-gathering:** *no single scalar*; detected via `d_asks = ro["asks"] − so["asks"]` (`report.py:56`) and the
  probe `ask_hist` (`support_probe.py:76`). A dedicated "over-gather" metric is in `eval_metrics.py` only.
- **1-ask collapse:** `one_ask = frac(n_asks==1)` (`report.py:32`); guard `ASK_COLLAPSE = one_ask ≥ 0.85 or
  asks ≤ 1.3` (`report.py:62`).
- **Greedy / pass@1:** the probe supports greedy `temperature=0.0` sampling (`support_probe.py:30-31`), but there
  is **no named pass@1 metric**. → **NOT FOUND** as a named metric.
- **Guarded four-way verdict** (`report.py:55-96`), thresholds: **THESIS-POSITIVE** = `d_value>0.02` AND
  `d_support>0.03` AND no high-risk-support regression; **ASK-COST EXPLOITATION** = `one_ask≥0.85` or `asks≤1.3`
  or high-risk value drop `>0.05` or high-risk support drop `>0.15` (or reward-up-with-support-flat-and-`d_asks<−0.2`);
  **TERMINATION-ONLY** = `d_no_commit<−0.05` AND `d_value>0.02`; **OPTIMIZATION-ONLY** = `|d_value|>0.02` or
  `|d_support|>0.05`; else **INCONCLUSIVE**.

---

## 6. Representative trajectories for Figure 1 — shared case `dx-13`

**One patient case covers all three procedures.** Diagnostics deephedge, `case_id = dx-13`, `family = abd_pain`
(adult with acute abdominal pain), `true_risk = high` (emergent), `v_oracle = 0.83`, `acceptable = ['emergent_tx']`,
`oracle_asks = ['abd_rigidity','rebound_tender','ct_free_air','lactate_high']` (4). Metadata stored in every
`probe_dh_*`/`probe_dream_*` record; hidden `assign` reconstructed via
`DX_VARIANT=deephedge gen_cases(60, 0.0, seed=100)` (validated: all stored derived fields match). Reward geometry:
`U["high"] = {discharge:−1.00, symptomatic_tx:−0.50, observe:−0.60, admit:0.20, emergent_tx:0.95}`,
cost 0.03 (`env_rv.py:67-73,314-320`). **All rewards below were independently recomputed via `env_rv.score()` and
match the stored `reward` field.**

**Initial prompt (identical for all three):** system = `env_rv.SYS` (§1.1); user = the abd_pain chart (§1.1,
"Chart: no results on file yet."). Per-ask answers below are reconstructed from `env_rv.answer` + `_REVEAL`
(`env_rv.py:253-272,308-312`) on the recovered `assign` (NOT stored).

### (a) Desirable — gather-then-commit  ✔
- **Method / file:** credit-assignment GRPO (λ=0.2), final ckpt — `results/probe_dh_ig_final_t12.records.jsonl`.
- **Asks (verbatim, in order):** `abd_rigidity → guarding → rebound_tender → ct_free_air`
  - `abd_rigidity` → *Result: abdominal rigidity.*
  - `guarding` → *Result: abdominal guarding.*
  - `rebound_tender` → *Result: rebound tenderness.*
  - `ct_free_air` → *Result: CT: free air.*
- **Final action:** `emergent_tx` · **Reward:** **0.830** (= 0.95 − 0.03·4) · `oracle_grade=True`, `n_asks=4`.
- **Process class:** **oracle-grade gather-then-commit** (4 asks = oracle depth; confirmed the emergent
  conjunction; committed the only acceptable action).
- **Cleaner gold alternative (if a purer trace is wanted):** *base* policy
  (`results/probe_dh_base_t12.records.jsonl`) on dx-13 asks `abd_rigidity → rebound_tender → ct_free_air` →
  `emergent_tx`, reward **0.860**, `oracle_grade=True` (exactly on the tree path). The exact oracle reference is
  `abd_rigidity → rebound_tender → ct_free_air → lactate_high → emergent_tx`, reward 0.83.

### (b) Under-gathering — premature commit to the capped hedge  ✘
- **Method / file:** plain (outcome-only) GRPO, seed 1, final ckpt — `results/probe_dh_grpo_final_t12.records.jsonl`.
- **Asks (verbatim):** `abd_rigidity` (one only)
  - `abd_rigidity` → *Result: abdominal rigidity.*
- **Final action:** `admit` (the capped hedge) · **Reward:** **0.170** (= 0.20 − 0.03·1) · `oracle_grade=False`,
  `n_asks=1`.
- **Process class:** **premature-commit / under-gather** (stopped after 1 finding; hedged instead of treating the
  emergency). Modal plain-GRPO behavior on dx-13: 9/12 rollouts are 1-ask; **3/12 commit `observe` after 1 ask →
  reward −0.630** (catastrophic). Use the `admit`=0.170 trace as the clean illustration, or the `observe`=−0.630
  trace (same file) to emphasize danger.

### (c) Over-gathering — exhaustive gather, still wrong  ✘
- **Method / file:** decoupled GRPO (DREAM), final ckpt — `results/probe_dream_final_t12.records.jsonl`.
  *(Decoupled is the documented over-gatherer; the plain/credit files don't over-gather on this case.)*
- **Asks (12 = turn cap, verbatim):** `abd_rigidity → rebound_tender → guarding → fever_high → wbc_high →
  lactate_high → age_high → anticoag_use → prior_cad → dyspnea_severe → hypotension → radiation_arm`
- **Final action:** `admit` · **Reward:** **−0.160** (= 0.20 − 0.03·12) · `oracle_grade=False`, `n_asks=12`.
- **Process class:** **over-gather** (hit the turn cap, cost-crushed into negative reward, still hedged wrong).
- **Positive over-gather alternative (same case, DREAM ckpt-40, `results/probe_dream_s40_t12.records.jsonl`):**
  8 asks → `emergent_tx`, reward **0.710**, `acceptable_hit=True` (correct action but 8 ≫ 4 bleeds below
  oracle-grade) — useful if you want over-gathering that still gets the right answer at a cost.

> **For the figure:** the cleanest didactic triple on `dx-13` is
> **(a) base 3-ask → emergent_tx, +0.86** · **(b) plain GRPO 1-ask → admit, +0.17** (or → observe, −0.63) ·
> **(c) DREAM 12-ask → admit, −0.16**. All same patient, same hidden state, same verifier.

> **Note:** the model's raw generated JSON/rationale text per turn is **not** stored in the probe records (only the
> parsed `asks`/`decision`). If verbatim rationale bubbles are wanted in the figure they'd be illustrative, not
> from logs — label them as such.

---

## 7. Figure-1-safe headline findings

Short claims, each with the preferred (table/CLAUDE.md) number and citation. Keep the headline short; the numbers
are backup, not all for the figure face.

1. **Terminal reward can rise while the procedure degrades.** Diagnostics plain GRPO: overall value
   +0.251 → +0.285, but high-risk value +0.160 → +0.098 and high-risk support 0.50 → 0.33±0.29 (one seed → 0.00).
   (`tab:dx` `paper/main.tex:464-465`; CLAUDE.md.)
2. **Credit assignment stabilizes diagnostics.** high-risk support held at 0.50±0.00 (vs 0.33±0.29), overall
   +0.313±0.024, moderate support 0.26 → 0.67. (`tab:dx:466`; CLAUDE.md.)
3. **Decoupled exploration maximizes high-risk safety at an over-gathering cost.** high-risk support 0.50 → 1.00,
   high-risk value +0.248, but mean asks 4.4 (oracle ≈3.9) and benign support 0.48 → 0.04. (`tab:dx:467`; CLAUDE.md.)
4. **The credit coefficient λ is a task-specific safety dial.** Titration: λ=0 asks 1.4 (under), λ=0.2 asks 4.10
   (over), predicted-and-confirmed λ=0.1 asks 2.67 ≈ oracle 2.77. (`tab:ti` `paper/main.tex:566-570`; CLAUDE.md.)
5. **Better gathering ≠ value win — headroom (BC-gap) governs payoff.** Titration λ=0.1 stays INCONCLUSIVE
   (overall −0.036) at BC-gap +0.075; diagnostics converts at BC-gap +0.161. (`tab:ti:569`; prose `:538-548,421`.)
6. **RL does not reliably repair an SFT coverage gap even when Gate 2 passes.** Biased SFT overall +0.337 /
   support 0.93 hides high-risk value −0.285; with 21% `emergent_tx` sampling (Gate 2 passes), credit-assignment
   collapses high-risk support 1.00 → 0.14 and decoupled over-gathers to 0.36. (`tab:repair`
   `paper/main.tex:649-651`; CLAUDE.md.)

---

## 8. Provenance & inconsistencies

Provenance is cited inline throughout §1–§7. This section collects the **conflicts** (user directive: prefer
repo/results over manuscript prose).

**Manuscript ↔ repo structure**
- **F0 — the `.tex` is in the repo.** `paper/main.tex` (938 lines, git-tracked) is the live manuscript;
  CLAUDE.md says the `.tex` is Overleaf-only "not here." Repo wins — use `paper/main.tex`.
- **F-title — three titles.** repo `.tex`: *"When Verifiable RL Helps Clinical Information Gathering: Capability
  Gates, SFT Coverage, and Reward Geometry"* (`paper/main.tex:25-26`); CLAUDE.md: *"RLVR for Clinical Multi-Turn
  Agents: When Verifiable Rewards Improve or Degrade Information Gathering"*; user's new title: *"Reward Design
  Governs Information Gathering in Clinical Multi-Turn Agents."* → Use the **user's new title** on the figure.
- **F-dx-env — 14 vs 18 findings.** `tasks/diagnostics/env.py` defines 14 `ITEMS`; the **paper** uses
  `tasks/diagnostics/env_rv.py` deephedge with **18** (matches `tab:envs` `paper/main.tex:250`). Use 18 / `env_rv.py`.

**Manuscript prose ↔ its own tables / CLAUDE.md** (all appear to be stale single-seed prose; prefer tables/CLAUDE.md)
- **F1** — credit-assignment overall value: prose "+0.255 → +0.334" (`:441`) vs table/CLAUDE.md **+0.251 → +0.313**.
- **F2** — moderate support: prose "0.26 → 0.77" (`:442`) vs table/CLAUDE.md **0.26 → 0.67**.
- **F3** — plain-GRPO high-risk value: prose "+0.160 → +0.095" (`:423`) vs table/CLAUDE.md **+0.098**.
- **F4** — plain-GRPO mean asks: prose "2.5 → 1.9" (`:426`) vs table **1.8** / CLAUDE.md **1.82**.
- **F5** — decoupled high-risk value baseline: prose "+0.139 → +0.248" (`:491`); the +0.139 start is not the base
  high-risk value (+0.160) — unexplained reference point. Endpoint +0.248 is fine.

**Gate thresholds & missing values**
- **F6 — BC-gap threshold.** Manuscript prose states the Gate-1 threshold as **">0"** (`paper/main.tex:150-154`);
  CLAUDE.md's operational screen is **≥~0.10** (code `bc_gap.py:94` uses `deep_gap >= 0.095`). If the figure names a
  gate threshold, use ~0.10 (the operational number).
- **F7 — refill BC-gap absent from manuscript.** Refill BC-gap +0.105 exists only in CLAUDE.md / `rx_triage/
  design.md:22` (no BC-gap column in `tab:envs`). Source it from CLAUDE.md if shown.
- **Not-found scalars (per §5):** named `premature`, `over-gather`, and `pass@1` metrics are **not** in the
  core eval (`report.py`/`support_probe.py`); a `premature`/`over-gather` pair exists only in
  `framework/procreward/eval_metrics.py`. Don't attribute them to the main eval on the figure.
- **Node A IP 192.168.100.1** and **SFT 800-trajectories/seed-7** are CLAUDE.md-only, not in the read code.

---

## Figure 1 asset checklist (everything to draw the first draft)

**Layout:** wide 4-panel (A | B | C | D). Title = user's new title (F-title). Deterministic RLVR only in the main
branch panel; Claude supervision at most a faded "planned" chip (§4).

**Panel A — Ask-to-commit loop.** Incomplete patient card ("Chart: no results on file yet.") → **ASK**
`{"action_type":"ask","intent":"<finding>"}` → simulator reveals `Result: <finding>.` → ASK/STOP → **COMMIT**
`{"action_type":"commit","decision":"…"}` → verifiable reward **R = U[μ][a] − 0.03·n_asks**. Callouts: μ (latent
severity) hidden; episode ends on commit / turn cap / parse-fail. (§1.)

**Panel B — Three environment cards** (§2). Refill triage: 12 findings · 5 actions · 2 families · severity ·
cap 10 · BC-gap +0.105. Diagnostics (hedging): 18 findings · 5 actions · 2 families · severity(none/mod/high) ·
cap 12 · BC-gap +0.161 · *hedge `admit` capped, emergent = 4-finding conjunction*. Titration: 6 findings · 3
ordered actions · 1 family · joint clot×bleed (9 states) · cap 10 · BC-gap +0.075. Each card: one example
"latent → oracle action" (high→escalate / high→emergent_tx / high_low→high).

**Panel C — Post-training branches** (§3–§4). Pipeline: `medgemma-4b-it (text-only) → [LoRA SFT warmup] → GRPO
(group=8, disaggregated 2-node vLLM)`. Three reward geometries, signal shown per branch:
- Outcome-only GRPO → `R = env.score`
- Credit-assignment GRPO → `R = env.score + λ·Σ ΔH_t` (dense per-ask info-gain credit)
- Decoupled explore/exploit → explore rewarded by recoverability (ΔH), exploit by verifier; leak-guarded commit.
(Optional faded chip: "planned: Claude process reward, R += α·Ĉ".)

**Panel D — Common eval + takeaway** (§5–§6). "shaping OFF → same held-out verifier (pure env.score, k=12)."
Three trajectories on the **same patient `dx-13`** (abd_pain, emergent):
- Calibrated gather-then-commit: 3–4 asks → `emergent_tx`, **R ≈ +0.83–0.86** ✔ (base / credit-assignment).
- Under-gathering: 1 ask → `admit`, **R = +0.17** (or → `observe`, **−0.63**) ✘ (plain GRPO).
- Over-gathering: 12 asks → `admit`, **R = −0.16** ✘ (decoupled).
Takeaway strip: *audit the rare high-risk subset — aggregate reward can rise while high-risk safety falls.*

**Numbers block (use these exact values, table/CLAUDE.md-preferred):** diagnostics base +0.251 / high +0.160 /
high-supp 0.50 / asks 2.5; plain GRPO +0.285 / high +0.098 / high-supp 0.33±0.29 / asks 1.8; credit-assignment
+0.313 / high +0.157 / high-supp 0.50±0.00 / mod-supp 0.67 / asks 2.1; decoupled +0.329 / high +0.248 /
high-supp 1.00 / asks 4.4. (§7.)

**Colors (reuse existing figure palette for consistency):** base `#888888`, plain/outcome GRPO `#d1495b`,
credit-assignment `#2e7d32`, decoupled `#3b6fb6`/`#6a4c93` (`scripts/plot_dx_4arm_seeds.py:9-16`).

**Reproducibility:** figure script → `scripts/make_figure1.py`; outputs → `figure1_overview.{svg,pdf,png}`;
env `~/spark1/.venv` (matplotlib 3.10.8, fonts DejaVu Sans / Nimbus Sans).
