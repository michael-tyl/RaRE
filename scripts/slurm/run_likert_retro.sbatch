#!/usr/bin/env bash
#SBATCH --job-name=rar-lretro
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
# Retro-GRPO-aux training (paper appendix): drop-on-disagreement main loss plus
# a retrospective auxiliary loss on the masked rollouts (scripts/likert_retro_train.py).
#
# Usage (from the repo root):
#   sbatch --export=ALL scripts/slurm/run_likert_retro.sbatch \
#       --config configs/run_jvr_4o_dsmini_likert_retro_k10.yaml --train-gpus 8
set -euo pipefail

CONFIG=""
TRAIN_GPUS=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|--train-gpus)
            if [[ $# -lt 2 ]]; then
                echo "[lretro-slurm] ERROR: $1 needs a value" >&2
                exit 1
            fi ;;&
        --config)     CONFIG="$2"; shift 2 ;;
        --train-gpus) TRAIN_GPUS="$2"; shift 2 ;;
        *)
            echo "[lretro-slurm] ERROR: unknown argument '$1'" >&2
            exit 1 ;;
    esac
done
if [[ -z "$CONFIG" ]]; then
    echo "[lretro-slurm] ERROR: --config required" >&2
    exit 2
fi

# Run from the repo root: sbatch from the repo directory (or set REPO_DIR).
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source scripts/env.sh
bash scripts/slurm/bootstrap_node.sh
# shellcheck disable=SC1091
source scripts/env.sh

# Every API key named by the config must reach the job.
for KEY_VAR in $(python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
keys = {c.get('judge_api_key_env')} | {j.get('api_key_env') for j in c.get('judges') or []}
print(' '.join(sorted(k for k in keys if k)))"); do
    if [[ -z "${!KEY_VAR:-}" ]]; then
        echo "[lretro-slurm] ERROR: \$$KEY_VAR is empty in this job. Set it in .env (and use --export=ALL)." >&2
        exit 3
    fi
done
echo "[lretro-slurm] config=$CONFIG  train_gpus=$TRAIN_GPUS"

mkdir -p logs "$RAR_OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_GPU_LIST=$(seq -s, 0 $((TRAIN_GPUS - 1)))
CUDA_VISIBLE_DEVICES="$TRAIN_GPU_LIST" \
    accelerate launch \
        --num_processes "$TRAIN_GPUS" \
        --mixed_precision bf16 \
        scripts/likert_retro_train.py \
        --config "$CONFIG"

echo "[lretro-slurm] training complete"

# Copy the final checkpoint to $RAR_STAGE; merge LoRA adapters into a full model.
RUN_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
CKPT_LOCAL="$RAR_CKPT/$RUN_NAME"
CKPT_SHARED="$RAR_STAGE/checkpoints/$RUN_NAME"
if [[ -f "$CKPT_LOCAL/config.json" || -f "$CKPT_LOCAL/adapter_config.json" ]]; then
    echo "[lretro-slurm] rsyncing $CKPT_LOCAL -> $CKPT_SHARED"
    mkdir -p "$(dirname "$CKPT_SHARED")"
    rsync -a --delete "$CKPT_LOCAL/" "$CKPT_SHARED/"
    if [[ ! -f "$CKPT_LOCAL/config.json" && -f "$CKPT_LOCAL/adapter_config.json" ]]; then
        MERGED_DIR="${CKPT_SHARED}_merged"
        echo "[lretro-slurm] LoRA detected; merging adapter into base model -> $MERGED_DIR"
        python scripts/merge_lora.py \
            --adapter "$CKPT_SHARED" \
            --output "$MERGED_DIR" || \
            echo "[lretro-slurm] WARNING: merge_lora failed"
    fi
else
    echo "[lretro-slurm] WARNING: $CKPT_LOCAL/{config.json,adapter_config.json} not found"
fi
