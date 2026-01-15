#!/bin/bash

set -x

NUM_NODES=2
JOB_NAME="general_sa-verl.dapo-qwen3-8B"


# Container and mount (required)

CONTAINER=verlai/verl:vllm012.latest
MOUNTS="/lustre/fsw/general_sa:/lustre/fsw/general_sa"


# verl recipe script env vars
RAY_DATA_HOME=/lustre/fsw/general_sa/xueh/rl/guide/verl
MODEL_PATH="/lustre/fsw/general_sa/xueh/rl/models/Qwen3-8B-Base"
TRAIN_FILE="/lustre/fsw/general_sa/xueh/rl/datasets/dapo_data/dapo-math-17k.parquet"
TEST_FILE="/lustre/fsw/general_sa/xueh/rl/datasets/dapo_data/aime-2024.parquet"

VERL_COMMAND="
cd /lustre/fsw/general_sa/xueh/rl/guide/verl && \
NNODES=${NUM_NODES} bash recipe/dapo/test_dapo_qwen3_8b_vllm_fsdp_bf16.sh
"


# Your API keys
#export WANDB_API_KEY=
#export HF_TOKEN=
export HF_HOME=/lustre/fsw/general_sa/xueh/hf
export HF_DATASETS_CACHE=/lustre/fsw/general_sa/xueh/hf/datasets


export CONTAINER="$CONTAINER"
export MOUNTS="$MOUNTS"
export VERL_COMMAND="$VERL_COMMAND"
export WANDB_API_KEY="$WANDB_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export HF_HOME="$HF_HOME"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
export RAY_DATA_HOME="$RAY_DATA_HOME"
export MODEL_PATH="$MODEL_PATH"
export TRAIN_FILE="$TRAIN_FILE"
export TEST_FILE="$TEST_FILE"
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"


sbatch \
   --nodes=${NUM_NODES} \
   --account=general_sa \
   --job-name=${JOB_NAME} \
   --partition=batch \
   --time=0:29:00 \
   verl_ray.sub
