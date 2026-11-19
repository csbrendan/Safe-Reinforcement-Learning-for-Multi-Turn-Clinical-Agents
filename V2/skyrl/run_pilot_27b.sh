#!/usr/bin/env bash
# 27B pilot on ONE Azure GPU VM (1x H100 NVL 94 GB, NC40ads_H100_v5). NOT for Spark.
# Purpose: measure peak memory and step time for 2 GRPO steps; synthetic cases only. Needs Brendan's approval
# (VM ~$7/h on-demand, ~$20-40 total) and gpt-5 quota for the gatekeeper/judge (or a local-aux config for a
# plumbing-only run).
#
# Settings mirror the TRL arm runs (framework/rl/grpo_disagg.py): LoRA r32/alpha64 all linear layers, lr 1e-5,
# no KL (beta 0), temperature 1.25, 8 rollouts per case, 4 cases per step (V2/sdbench_execution_plan.md).
# Turn/context limits: generator.max_turns >= 2*max_turns+1 and max_input_length > max_context_tokens, so the
# Episode's own force/overflow rules always fire first (tasks/v2_sdbench/skyrl_env.py).
#
# Usage (on the VM, from the repo root, inside the SkyRL env):
#   V2_ENV_CONFIG=V2/sdbench_configs/azure_synthetic.json bash V2/skyrl/run_pilot_27b.sh
set -euo pipefail
: "${V2_ENV_CONFIG:?set V2_ENV_CONFIG}"
MODEL="${MODEL:-google/medgemma-27b-text-it}"
DATA="${DATA:-results/v2_skyrl/synthetic_train.jsonl}"
CKPT="${CKPT:-$HOME/ckpts/v2_skyrl_pilot_27b}"
STEPS="${STEPS:-2}"

python -m tasks.v2_sdbench.skyrl_env build-data --env-config "$V2_ENV_CONFIG" --out "$DATA"
export V2_ENV_CONFIG

python -m V2.skyrl.main_sdbench \
  data.train_data="['$DATA']" \
  data.val_data="['$DATA']" \
  environment.env_class=sdbench \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL" \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=64 \
  trainer.policy.optimizer_config.lr=1.0e-5 \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.batched=false \
  generator.use_conversation_multi_turn=true \
  generator.max_turns=25 \
  generator.max_input_length=16000 \
  generator.n_samples_per_prompt=8 \
  generator.sampling_params.temperature=1.25 \
  generator.sampling_params.max_generate_length=400 \
  generator.sampling_params.additional_kwargs="{stop_token_ids: [1, 106]}" \
  trainer.max_prompt_length=1024 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=1 \
  trainer.max_training_steps="$STEPS" \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.logger=console \
  trainer.project_name=v2_sdbench \
  trainer.run_name="pilot_27b_$(date +%Y%m%d_%H%M)" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT" \
  "$@"
