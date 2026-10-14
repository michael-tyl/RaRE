#!/usr/bin/env bash
#SBATCH --job-name=gpqa-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Out-of-distribution GPQA Diamond eval. MCQ accuracy; no LLM judge needed.
# Each positional arg is an HF model id or local checkpoint path.
#
# Usage (from the repo root; paper protocol uses --max-tokens 8192):
#   sbatch --export=ALL scripts/slurm/gpqa_eval.sbatch --max-tokens 8192 \
#       Qwen/Qwen3-4B-Base \
#       $RAR_STAGE/checkpoints/jvr_sci_mini
#
# Output: $RAR_STAGE/eval/gpqa[_mt<max_tokens>]_<checkpoint>.json
set -euo pipefail

N=198
MAX_TOKENS=1024
SEED=0
STRATIFY=0
CKPTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n)                   N="$2"; shift 2 ;;
        --max-tokens)          MAX_TOKENS="$2"; shift 2 ;;
        --seed)                SEED="$2"; shift 2 ;;
        --stratify-subdomain)  STRATIFY=1; shift ;;
        *)                     CKPTS+=("$1"); shift ;;
    esac
done
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    CKPTS=("Qwen/Qwen3-4B-Base")
fi

# Run from the repo root: sbatch from the repo directory (or set REPO_DIR).
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source scripts/env.sh

echo "[gpqa-eval] bootstrap on $(hostname)"
bash scripts/slurm/bootstrap_node.sh
# shellcheck disable=SC1091
source scripts/env.sh

mkdir -p logs "$RAR_OUT"
SHARED_EVAL="$RAR_STAGE/eval"
mkdir -p "$SHARED_EVAL"

# Use a per-job triton/torchinductor cache so concurrent evals on the
# same node don't corrupt a shared cache index (manifests as cubin-missing
# InductorError on later jobs).
export TRITON_CACHE_DIR="$RAR_LOCAL/triton_${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="$RAR_LOCAL/torchinductor_${SLURM_JOB_ID:-local}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

EXTRA_ARGS=()
if [[ "$STRATIFY" == "1" ]]; then
    EXTRA_ARGS+=(--stratify-by-subdomain)
fi
if [[ "$MAX_TOKENS" -ne 1024 ]]; then
    LEN_TAG="_mt${MAX_TOKENS}"
else
    LEN_TAG=""
fi
# Answer-order permutation k (--seed k, k >= 1) gets its own file; seed 0 keeps
# the default name read by scripts/holdout_tau_select.py.
if [[ "$SEED" != 0 ]]; then
    LEN_TAG="${LEN_TAG}_perm${SEED}"
fi

for ckpt in "${CKPTS[@]}"; do
    base=$(basename "$ckpt")
    out="$RAR_OUT/gpqa${LEN_TAG}_${base}.json"
    echo "[gpqa-eval] === n=$N max_tokens=$MAX_TOKENS  $ckpt -> $out ==="
    # Uses the GPU Slurm allocated to this job (inherited CUDA_VISIBLE_DEVICES).
    python -m rar.eval_gpqa \
            --checkpoint "$ckpt" \
            --n "$N" \
            --seed "$SEED" \
            --max-tokens "$MAX_TOKENS" \
            --output "$out" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.6 \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    cp "$out" "$SHARED_EVAL/gpqa${LEN_TAG}_${base}.json"
    echo "[gpqa-eval] copied to $SHARED_EVAL/gpqa${LEN_TAG}_${base}.json"
done

echo "[gpqa-eval] all checkpoints done"
