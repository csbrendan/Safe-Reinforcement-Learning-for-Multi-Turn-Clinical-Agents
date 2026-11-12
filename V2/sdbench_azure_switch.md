# V2 SD-Bench: auxiliary roles moved to Azure AI Foundry (2026-11-12)

The gatekeeper, judge and pricer move from local Qwen3-8B to Azure AI Foundry deployments, matching SD-Bench's own
default configuration. The training-environment guard and corrected cost from
[sdbench_env_validation.md](sdbench_env_validation.md) are unchanged.

**Status: code ready and tested offline. Nothing has been sent to Azure yet: there are no Azure credentials on
this machine.** The GPT-5 side of the calibration is waiting on credentials. MIMIC use is blocked on a data-policy
question (section 3).

## 1. What changed

| Role | Before | Now (`V2/sdbench_configs/azure_synthetic.json`) | SD-Bench official default |
|---|---|---|---|
| Judge | Qwen3-8B, temperature 0, 800-token cap | Azure `gpt-5`, prompt `v4_reasoning_with_tests` | `gpt-5`, `v4_reasoning_with_tests` |
| Gatekeeper | Qwen3-8B | Azure `gpt-5` (recommended; `gpt-5-mini` is a config change) | `gpt-5` |
| Pricer | Qwen3-8B | Azure `gpt-5-mini` | `gpt-5-mini` |

- **`tasks/v2_sdbench/llm.py`:** each role picks a `provider`, `local` or `azure`. Azure calls go out the way
  SD-Bench sends them:
  - only the deployment name and `model_args` (empty by default);
  - no temperature, token cap or seed. GPT-5 rejects temperature 0 and `max_tokens`, and SD-Bench sends neither.
  - The code refuses those three arguments so they can't creep in.
- **Prompts:** the judge prompt comes from SD-Bench's own `check_diagnosis` code and is unchanged. The judge
  default is now `v4_reasoning_with_tests`.
- **Rate limits and retries:**
  - a per-role cap on requests in flight (16 for gatekeeper and judge, 8 for pricer);
  - backoff on 429s, timeouts and 5xx errors, honouring `Retry-After`;
  - no retry on 400s.
  - Token usage, including reasoning tokens, is counted per role.
- **Credentials:** read from `~/.secrets/azure.env` into the process only, like the Anthropic key, and never
  exported. Needed: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and optionally `AZURE_OPENAI_API_VERSION`
  (default `2025-01-01-preview`, SD-Bench's).
- **MIMIC guard (`env.check_data_policy`):** if the dataset is `mimic` and the gatekeeper or judge is on Azure,
  configuration fails unless `azure_mimic_human_review_opt_out` is true. The pricer is exempt because it only
  receives test names; SD-Bench runs its own pricer without zero data retention for the same reason.
- **Calibration script:** `tasks/v2_sdbench/calibrate_judge.py` (section 2).
- **Tests:** `tests/test_v2_sdbench_azure.py`, 14 offline tests with a fake client:
  - the official call shape;
  - refused arguments;
  - retry and give-up;
  - credentials read without being exported;
  - the MIMIC guard.

  The full V2 suite has 72 tests, all passing.

**Consequences:**
- **Not deterministic.** GPT-5 output isn't deterministic. The caches (first write wins) keep a run reproducible,
  so arms B and C, development and evaluation must share caches, and a cold re-run won't match bit for bit.
- **Old caches unusable.** Cache keys include the model, so the Qwen caches are never mixed in.
- **Overnight leak and classifier findings must be re-checked.** They were measured with Qwen as gatekeeper, and
  the classifier was tuned on Qwen's answer style. Both need re-running on GPT-5 answers before training: the leak
  probe, and an out-of-sample classifier audit. This is the open caveat from the overnight report, now testable.
- **Gatekeeper requests are sent one at a time** (`previous_tests=[]`), so they can be cached. SD-Bench sends up
  to 10 per call. That was already the case with Qwen. On Azure it means more input tokens (section 4).

## 2. Calibration

`calibrate_judge.py` scores the **same stored episodes** with the reference judge (Qwen3-8B) and the candidate
(Azure GPT-5), both with `v4_reasoning_with_tests`.
- **Inputs:** the final diagnosis and the test log the agent saw.
- **Reward recomputation:** r_out is recomputed by swapping only the judge term, so cost and format terms are held
  fixed.
- **Reported:**
  - score histograms;
  - exact, within-1 and pass/fail agreement;
  - quadratic-weighted κ;
  - Spearman correlation;
  - mean reward shift;
  - within-group reward spread, which is what GRPO learns from.
- **GPT-5 retest:** each episode is scored twice with GPT-5 (`--repeats 1`), because without temperature control
  it isn't deterministic.
- **Spend cap:** `--max-calls`.
- **Data:** synthetic cases only.

**Episodes:**
- **Count:** 30 unique (27 with a diagnosis to judge), from the overnight runs on SD-Bench's 5 public synthetic cases (`results/v2_sdbench/bench/*`
  and the first smoke run).
- **Groups:** 4 groups of 4–8 samples on the same case, like GRPO groups.
- **Mostly invalid** (section 3 of the validation report), so judge scores skew low. This calibrates the judge on
  the base policy's actual output, not on good diagnoses.

### Baseline done: Qwen against itself (`results/v2_sdbench/calibration/qwen_vs_qwen.json`)

Before comparing GPT-5 with Qwen, the noise floor: the same 27 judged episodes re-scored by Qwen3-8B at temperature
0 with fresh caches (two scorings), and compared with the scores logged overnight.

| Comparison | Exact | Within ±1 | Pass/fail agree | Weighted κ | Spearman |
|---|---|---|---|---|---|
| Qwen re-score vs Qwen re-score | 0.81 | 1.00 | 1.00 | 0.96 | 0.89 |
| Qwen (fresh cache A) vs Qwen (fresh cache B) | 0.74 | 1.00 | 1.00 | 0.94 | 0.88 |
| Qwen re-score vs score logged overnight | 0.63 | – | – | – | – |

- **Reward shift from re-scoring alone:**
  - mean |Δr_out| = 0.058, max 0.25 (one judge point);
  - Spearman correlation of rewards 0.91.
- **Within-group reward std** (what GRPO learns from):
  - mean 0.171 → 0.124;
  - one group (T = 0.7, case 1) went from 0.108 to exactly 0.

**Reading.** Even at temperature 0 the local judge isn't reproducible: vLLM batching isn't batch-invariant.
- The noise is always within one point and never flips pass/fail.
- But one point is 0.25 reward, the same size as the within-group differences in these groups. So judge noise is a
  real share of GRPO's signal here, whichever judge is used.
- The cache hides this inside a run, because each (case, diagnosis, test log) is judged once.
- Across arms it is controlled only if arms share the judge cache.

**What GPT-5 has to show.** A Qwen-vs-GPT-5 difference counts as a real shift only if it exceeds this floor:
- exact agreement well below ≈ 0.75;
- any pass/fail disagreements;
- κ below ≈ 0.94;
- mean |Δr| above ≈ 0.06.

GPT-5's own retest agreement sets its noise floor; it can't be assumed lower than Qwen's.

### Pending: Qwen vs Azure GPT-5

Waiting for credentials (command in section 5). The same script fills in the same table plus GPT-5's retest agreement,
token usage and cost.

## 3. Data policy for MIMIC: the stateless default is not enough

Azure's default handling:
- prompts aren't stored in the model, aren't sent to OpenAI and aren't used for training;
- but abuse monitoring can retain flagged prompts for human review.

PhysioNet's guidance on using credentialed data with online services
([physionet.org/news/post/gpt-responsible-use](https://physionet.org/news/post/gpt-responsible-use), 2023-04-18) is
explicit for Azure OpenAI: "You'll need to opt out of human review of the data". The opt-out is the
limited-access **modified abuse monitoring** application. PhysioNet suggests stating that you are processing
sensitive data where harmful outputs are unlikely, and that the data use agreement doesn't allow you to let
Microsoft process the data for abuse detection. The same page says the credentialed DUA prohibits sharing the data
with third parties, sending it through company APIs included.

SD-Bench's own code says the same: its Azure path is for "your own endpoint in AI Foundry with ZDR", and the
deployment it ships against is described as "inherently ZDR".

**So I can't confirm the default is acceptable. It isn't, by PhysioNet's own guidance.** Before any MIMIC case
goes to an Azure gatekeeper or judge:
1. Get modified abuse monitoring approved for the subscription. It needs an organizational email: your Gmail won't
   qualify. Options:
   - an institutional Azure subscription;
   - Agustina's institution. She holds the PhysioNet credential, but DUA terms are per person, so she would need to
     be the one running MIMIC data through it.
2. Then set `azure_mimic_human_review_opt_out: true` in the MIMIC config. The code refuses until then.

Unaffected:
- **Synthetic cases:** public.
- **Pricer:** receives test names only.

If approval isn't possible, the MIMIC gatekeeper and judge have to stay local (Qwen or a larger local model). That
breaks comparability with SD-Bench's published GPT-5 numbers.

## 4. Quota and cost

**Quota: not checked.** It needs the Foundry portal (Manage → Quota) or `az`, and this machine has neither
credentials nor the Azure CLI.

Projected load against the Tier 1 defaults you quoted (250k tokens per minute, 2,500 requests per minute, Global
Standard). Prompt sizes are measured with `o200k_base`; output sizes are assumed until the calibration measures
them.

| Call | Input tokens | Output (assumed, includes reasoning) |
|---|---|---|
| Gatekeeper (one request) | ≈ 1.4k prompt + 9–11k case text ≈ **11k** | ≈ 0.5–1.5k |
| Judge | ≈ 1.9k prompt + reference + test log ≈ 3–6k | ≈ 1–3k |
| Pricer | < 0.5k | small |

- **Per GRPO step:**
  - 32 episodes, about 15 uncached gatekeeper calls each (measured with cold caches);
  - ≈ 480 calls × 11k ≈ **5M input tokens**.
  - One synchronised turn sends ≈ 128 gatekeeper calls at once, ≈ 1.4M tokens.
- **At 250k TPM:**
  - the gatekeeper is capped at ≈ 20 calls per minute, so ≈ 20+ min of throttling per step from tokens alone;
  - ≈ 10+ h per 30-step arm.
  - Requests per minute (2,500) aren't the constraint.
- **Recommendation:** request an increase to ≥ 1–2M TPM for the GPT-5 gatekeeper deployment before a training run.
  Tier 1 is fine for the calibration and the leak probe.

**Cost:** rough figures, assuming list prices of GPT-5 $1.25/$10 and GPT-5-mini $0.25/$2 per 1M input/output
tokens. Check the Azure rate card.

| Item | GPT-5 gatekeeper | GPT-5-mini gatekeeper |
|---|---|---|
| Calibration (≈ 54 judge calls) + GPT-5 leak probe (≈ 25 gatekeeper calls) | ≈ $2–5 | – |
| Per training arm (960 episodes; gatekeeper upper bound, caching lowers it) | ≈ $300–400 + judge ≈ $25 | ≈ $70–90 + judge ≈ $25 |
| Development + final evaluation (≈ 1,130 episodes) | ≈ $350–450 | ≈ $90–110 |

The gatekeeper dominates, because each request re-sends the whole case. GPT-5 matches the official configuration.
GPT-5-mini is about 5× cheaper, but it's a different gatekeeper from the published SD-Bench numbers and would need
its own leak probe.

## 5. Blockers

1. **Azure credentials:**
   - put the key and endpoint in `~/.secrets/azure.env`;
   - create deployments named `gpt-5` and `gpt-5-mini`, or edit the config names.
2. **Spend approval for calibration plus leak probe** (≈ $2–5). Run commands below.
3. **MIMIC data policy** (section 3): modified abuse monitoring approval with an organizational email, or keep MIMIC
   roles local.
4. **Quota increase** for training throughput (section 4). Only you can file it, from the portal.
5. **Unchanged from the overnight report:** the base policy's format failure under sampling and trainer placement
   still block RL, whatever the judge.

### Run once credentials exist (synthetic only)

```bash
cd ~/rl-gains-in-clinical-tasks && source ~/spark1/.venv/bin/activate
B=results/v2_sdbench/bench
PYTHONPATH=$PWD python -m tasks.v2_sdbench.calibrate_judge \
  --episodes $B/before.jsonl $B/after1.jsonl $B/after2.jsonl $B/conc16_guarded.jsonl $B/t07_guarded.jsonl \
             results/v2_sdbench/smoke/eval_A_synth.jsonl \
  --ref V2/sdbench_configs/calib_qwen_ref.json --cand V2/sdbench_configs/azure_synthetic.json \
  --repeats 1 --max-calls 80 --out results/v2_sdbench/calibration/qwen_vs_gpt5.json
```
