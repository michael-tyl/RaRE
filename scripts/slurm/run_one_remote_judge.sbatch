#!/usr/bin/env bash
#SBATCH --job-name=rar-rj
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# GRPO training with remote (API) judges. All GPUs go to the policy; the
# judges named in the config are called over OpenAI-compatible APIs using
# the keys named by `api_key_env:` (loaded from .env by scripts/env.sh).
#
# Usage (from the repo root):
#   sbatch --export=ALL scripts/slurm/run_one_remote_judge.sbatch \
#       --config configs/run_jvr_4o_dsmini_softlikert_drop030.yaml \
#       --train-gpus 8 [--seed N]
#
# --seed N overrides the config's seed (and the 1,000-prompt training
# subset, which follows it) and appends _sN to the run name for N != 0.
set -euo pipefail

CONFIG="configs/run_jvr_4o_dsmini_softlikert_drop030.yaml"
TRAIN_GPUS=8     # No local judge here; all GPUs go to training.
SEED_ARGS=()
SEED_SUFFIX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--train-gpus|--seed)
            if [[ $# -lt 2 ]]; then
                echo "[slurm] ERROR: $1 needs a value" >&2
                exit 1
            fi ;;&
        --config)     CONFIG="$2"; shift 2 ;;
        --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
        --seed)
            if ! [[ "$2" =~ ^(0|[1-9][0-9]*)$ ]]; then
                echo "[slurm] ERROR: --seed must be a non-negative integer, got '$2'" >&2
                exit 1
            fi
            SEED_ARGS=(--seed "$2")
            [[ "$2" != 0 ]] && SEED_SUFFIX="_s$2"
            shift 2 ;;
        *)
            echo "[slurm] ERROR: unknown argument '$1'" >&2
            exit 1 ;;
    esac
done

# Run from the repo root: sbatch from the repo directory (or set REPO_DIR).
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source scripts/env.sh
bash scripts/slurm/bootstrap_node.sh
# shellcheck disable=SC1091
source scripts/env.sh

# Sanity-check API key reached the job.
KEY_VAR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('judge_api_key_env', ''))")
if [[ -z "$KEY_VAR" ]]; then
    echo "[slurm] ERROR: $CONFIG has no judge_api_key_env field" >&2
    exit 2
fi
KEY_VALUE="${!KEY_VAR:-}"
if [[ -z "$KEY_VALUE" ]]; then
    echo "[slurm] ERROR: \$$KEY_VAR is empty in this job. Did you --export=ALL?" >&2
    exit 3
fi
echo "[slurm] config=$CONFIG  train_gpus=$TRAIN_GPUS  remote-judge env-var=$KEY_VAR (length=${#KEY_VALUE})"

mkdir -p logs "$RAR_OUT"

# Reduce CUDA fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# All visible GPUs go to the trainer (no local judge).
TRAIN_GPU_LIST=$(seq -s, 0 $((TRAIN_GPUS - 1)))
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST" \
    accelerate launch \
        --num_processes "$TRAIN_GPUS" \
        --mixed_precision bf16 \
        -m rar.train_grpo \
        --config "$CONFIG" \
        ${SEED_ARGS[@]+"${SEED_ARGS[@]}"}

echo "[slurm] training complete"

# Copy the final checkpoint to $RAR_STAGE; merge LoRA adapters into a full model.
RUN_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")$SEED_SUFFIX
CKPT_LOCAL="$RAR_CKPT/$RUN_NAME"
CKPT_SHARED="$RAR_STAGE/checkpoints/$RUN_NAME"
if [[ -f "$CKPT_LOCAL/config.json" || -f "$CKPT_LOCAL/adapter_config.json" ]]; then
    echo "[slurm] rsyncing $CKPT_LOCAL -> $CKPT_SHARED"
    mkdir -p "$(dirname "$CKPT_SHARED")"
    rsync -a --delete "$CKPT_LOCAL/" "$CKPT_SHARED/"
    echo "[slurm] checkpoint copied to $CKPT_SHARED"
    if [[ ! -f "$CKPT_LOCAL/config.json" && -f "$CKPT_LOCAL/adapter_config.json" ]]; then
        MERGED_DIR="${CKPT_SHARED}_merged"
        echo "[slurm] LoRA detected — merging adapter into base model -> $MERGED_DIR"
        python scripts/merge_lora.py \
            --adapter "$CKPT_SHARED" \
            --output "$MERGED_DIR" || \
            echo "[slurm] WARNING: merge_lora failed"
    fi
else
    echo "[slurm] WARNING: $CKPT_LOCAL/{config.json,adapter_config.json} not found"
fi
