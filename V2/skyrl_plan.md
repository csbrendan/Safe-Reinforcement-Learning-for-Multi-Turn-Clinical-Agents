# V2 on Azure: SkyRL trainer plan

Status 2026-11-19: code written and CPU-tested on Spark 2; **nothing has run on a GPU or VM yet**.

## What exists

| File | What it is | Tested |
|---|---|---|
| `tasks/v2_sdbench/skyrl_env.py` | SkyRL gym env `sdbench`: wraps the same `Services` + `Episode` as the TRL path; `build-data` CLI writes JSONL rows | `tests/test_v2_sdbench_skyrl.py`, 6 CPU tests (transcript identical to a bare `Episode`, forced-diagnosis path, overflow penalty, refusal of bad rows) |
| `V2/skyrl/main_sdbench.py` | Ray entrypoint that registers the env and runs SkyRL's `BasePPOExp` | Syntax only (SkyRL isn't installed on Spark, by design) |
| `V2/skyrl/run_pilot_27b.sh` | 2-step MedGemma 27B LoRA GRPO pilot, 1x H100 NVL, colocated (vLLM sleep mode) | Shell syntax only |

VM provisioning: [`azure/README.md`](azure/README.md) (`vm.sh` create/push/setup/delete, `setup_remote.sh`).

The protocol, reward, caches and arm B/C configs are unchanged; the env config file is passed in `V2_ENV_CONFIG`.

## Matching the TRL arm runs

- LoRA r32 / alpha 64, all linear layers; lr 1e-5; no KL; temperature 1.25; 8 rollouts per case, 4 cases per step.
- Stop tokens `[1, 106]` via `generator.sampling_params.additional_kwargs`. Check on the VM that vLLM gets them.
- The `Episode` turn limit (2 x `max_turns` replies) and context guard (`max_context_tokens`) fire before SkyRL's own
  (`generator.max_turns=25`, `max_input_length=16000`). Over-long episodes get the same overflow penalty as in TRL.
  The guard needs `"skyrl": {"tokenizer": "<policy path>"}` in the env config.

## Check on the pilot (not yet verified)

1. Peak GPU memory for 27B LoRA, with trainer and vLLM sharing one H100 NVL (estimate 70–80 GB; fallback 2x A100 80 GB).
2. Step time, and gatekeeper/judge throughput against the gpt-5 tokens-per-minute quota.
3. SkyRL builds the sequence turn by turn. Compare its token IDs and loss mask with
   `grpo_disagg._ids` / `_env_mask` on one transcript (Gemma's chat template should agree, but check).
4. Ray workers need the repo on `PYTHONPATH` (runtime env `working_dir` or install the repo).

## Blockers

- **GPU quota:** 0 for every GPU family. Automated requests on 2026-11-19 gave these results:
  - regional vCPUs 10 → 300: granted;
  - H100 NVL (160 vCPU, and 40 vCPU in both eastus2 and eastus): declined with `ContactSupport`;
  - A100 96 vCPU in eastus2: declined with `QuotaNotAvailableForResource`;
  - spot: the request didn't register.

  A portal support ticket is open for H100 NVL in West US 3 or West US 2 (160 vCPU preferred, 40 minimum), with A100 as the fallback.
- **gpt-5 quota:** granted 2026-11-22 (250K TPM) and `gpt-5` is deployed. Enough for the pilot; training wants 1–2M TPM.
- **Grounded costs:** working end to end (`tasks/v2_sdbench/grounded_costs.py`, 7 CPU tests). First run 2026-11-22 on
  2 synthetic smoke episodes: 3 of 5 workup tests matched a CPT price, 2 fell back to the LLM estimate. The matcher's
  embeddings use an OpenAI key (Azure isn't accepted); its packages live outside the Spark training env.
- **MedGemma 27B weights:** license accepted and weights downloaded on Spark 2 2026-11-22 (`generation_config` eos `[1, 106]`).
