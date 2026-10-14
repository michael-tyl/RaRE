#!/usr/bin/env bash
# Source this file before running any rar commands.
# It points HF / wandb / torch / uv caches at fast node-local storage
# ($RAR_LOCAL) and keeps durable outputs on shared storage ($RAR_STAGE).

# Load .env from the repo root first so it can set API keys (OPENAI_API_KEY,
# ...), HF_TOKEN, WANDB_API_KEY, and override RAR_LOCAL / RAR_STAGE below.
# Format: one KEY=value per line, no `export` prefix needed, comments with #.
REPO_ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$REPO_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    set -a
    source "$REPO_ENV_FILE"
    set +a
fi

# Node-local scratch (fast SSD): venv, caches, in-progress checkpoints.
export RAR_LOCAL="${RAR_LOCAL:-/tmp/${USER}/deep_rl}"

# Durable storage visible from every node: final checkpoints, eval JSONs.
export RAR_STAGE="${RAR_STAGE:-/workspace/home/${USER}/deep_rl_stage}"

# Working directories on local SSD.
export RAR_DATA="$RAR_LOCAL/data"
export RAR_CKPT="$RAR_LOCAL/checkpoints"
export RAR_OUT="$RAR_LOCAL/outputs"

# HuggingFace (HF_HOME covers hub + datasets caches). For gated datasets
# (GPQA Diamond), set HF_TOKEN in .env so it reaches every compute node.
export HF_HOME="$RAR_LOCAL/hf"
export HF_HUB_CACHE="$RAR_LOCAL/hf/hub"
export HF_DATASETS_CACHE="$RAR_LOCAL/hf/datasets"

# Weights & Biases
export WANDB_DIR="$RAR_LOCAL/wandb"
export WANDB_CACHE_DIR="$RAR_LOCAL/wandb/cache"
export WANDB_PROJECT="${WANDB_PROJECT:-rar-qwen3}"

# vLLM / torch caches
export VLLM_CACHE_ROOT="$RAR_LOCAL/vllm"
export TORCH_HOME="$RAR_LOCAL/torch"
export TRITON_CACHE_DIR="$RAR_LOCAL/triton"

# torchinductor compile cache: kept on local SSD for speed. It grows across
# runs; bootstrap_node.sh prunes it when /tmp passes a usage threshold.
export TORCHINDUCTOR_CACHE_DIR="$RAR_LOCAL/torchinductor"

# Reduce CUDA allocator fragmentation. Setting both old and new names because
# different torch versions check different names.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Tell uv to put the project venv on local SSD instead of next to pyproject.toml
export UV_PROJECT_ENVIRONMENT="$RAR_LOCAL/.venv"
export UV_CACHE_DIR="$RAR_LOCAL/uv-cache"

# Make sure those dirs exist
mkdir -p "$RAR_DATA" "$RAR_CKPT" "$RAR_OUT" \
         "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$UV_CACHE_DIR" \
         "$RAR_STAGE/checkpoints" "$RAR_STAGE/eval" "$RAR_STAGE/data"

# Activate the project venv (also lives on local SSD)
if [[ -f "$UV_PROJECT_ENVIRONMENT/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$UV_PROJECT_ENVIRONMENT/bin/activate"
fi
