#!/usr/bin/env bash
set -euo pipefail
set -x

ENGINE="${1:-vllm}"
if [[ $# -gt 0 ]]; then
    shift
fi

: "${MODEL_PATH:?Please export MODEL_PATH=/path/to/model_or_hf_id before running this script.}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export RAY_BACKEND_LOG_LEVEL="${RAY_BACKEND_LOG_LEVEL:-DEBUG}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export WANDB_NAME="${WANDB_NAME:-webshop_grpo_qwen2.5_1.7b_sft_skills_test}"

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
val_batch_size="${VAL_BATCH_SIZE:-8}"
train_data_size="${TRAIN_DATA_SIZE:-8}"
val_data_size="${VAL_DATA_SIZE:-64}"
num_gpus="${NUM_GPUS:-8}"
tp_size="${TP_SIZE:-2}"
max_steps="${MAX_STEPS:-15}"
max_prompt_length="${MAX_PROMPT_LENGTH:-6000}"
max_response_length="${MAX_RESPONSE_LENGTH:-768}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}"
skills_json_path="${SKILLS_JSON_PATH:-/home/dataset-assist-0/xiaxu/code/SkillRL/memory_data/webshop/claude_style_skills.json}"
default_local_dir="${DEFAULT_LOCAL_DIR:-outputs/test_webshop_skills}"

if (( train_data_size % num_gpus != 0 )); then
    echo "TRAIN_DATA_SIZE (${train_data_size}) must be divisible by NUM_GPUS (${num_gpus})." >&2
    exit 1
fi

if (( val_data_size % val_batch_size != 0 )); then
    echo "VAL_DATA_SIZE (${val_data_size}) must be divisible by VAL_BATCH_SIZE (${val_batch_size})." >&2
    exit 1
fi

if (( tp_size > num_gpus )); then
    echo "TP_SIZE (${tp_size}) must be <= NUM_GPUS (${num_gpus})." >&2
    exit 1
fi

# The trainer still constructs a train dataloader in val_only mode, so keep a
# small train split for initialization even though no optimizer step is run.
python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOME/data/verl-agent/text/train.parquet" \
    data.val_files="$HOME/data/verl-agent/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_data_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${tp_size}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps="${max_steps}" \
    env.rollout.n=1 \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    +env.use_skills_only_memory=True \
    +env.skills_only_memory.skills_json_path="${skills_json_path}" \
    +env.skills_only_memory.top_k=6 \
    +env.skills_only_memory.enable_dynamic_update=False \
    trainer.critic_warmup=0 \
    "trainer.logger=['console']" \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name=test_webshop_skills \
    trainer.n_gpus_per_node="${num_gpus}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=0 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.log_val_generations=0 \
    trainer.default_local_dir="${default_local_dir}" \
    "$@"
