#!/usr/bin/env bash
#SBATCH --job-name=hb-remote
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# HealthBench eval with a remote grader (default: gpt-4o-mini via the OpenAI
# API, as in the paper). Uses 1 local GPU for policy generation; the grader
# is called over the network with the key in $JUDGE_API_KEY_ENV.
#
# Example (from the repo root; paper protocol):
#   sbatch --export=ALL scripts/slurm/healthbench_remote_judge.sbatch \
#       --subset oss --n 5000 --max-tokens 8192 \
#       Qwen/Qwen3-4B-Base \
#       $RAR_STAGE/checkpoints/jvr_4o_dsmini_softlikert_drop030
#
# Output: $RAR_STAGE/eval/hb_pc_j<grader>[_mt<max_tokens>]_<checkpoint>.json
set -euo pipefail

# ---------------------------------------------------------------- args
SCORING="per_criterion"
JUDGE_URL="https://api.openai.com/v1"
JUDGE_MODEL="gpt-4o-mini"
JUDGE_API_KEY_ENV="OPENAI_API_KEY"
SUBSET="oss"
N=200
MAX_TOKENS=1024
ENGLISH_ONLY=0
STRATIFY_BY=()
PROMPT_IDS_FILE=""
JUDGE_MAX_CONCURRENT=48
TP_SIZE=1
RESUME_FROM_COMPLETIONS=0
GRADER_ARGS=()
CKPTS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --scoring)               SCORING="$2"; shift 2 ;;
        --judge-url)             JUDGE_URL="$2"; shift 2 ;;
        --judge-model)           JUDGE_MODEL="$2"; shift 2 ;;
        --judge-api-key-env)     JUDGE_API_KEY_ENV="$2"; shift 2 ;;
        --subset)                SUBSET="$2"; shift 2 ;;
        --n)                     N="$2"; shift 2 ;;
        --max-tokens)            MAX_TOKENS="$2"; shift 2 ;;
        --english-only)          ENGLISH_ONLY=1; shift ;;
        --stratify-by)           STRATIFY_BY+=("$2"); shift 2 ;;
        --prompt-ids-file)       PROMPT_IDS_FILE="$2"; shift 2 ;;
        --judge-max-concurrent)  JUDGE_MAX_CONCURRENT="$2"; shift 2 ;;
        --tensor-parallel-size)  TP_SIZE="$2"; shift 2 ;;
        --resume-from-completions) RESUME_FROM_COMPLETIONS=1; shift ;;
        --judge-extra-body|--judge-max-tokens|--judge-timeout)
                                 GRADER_ARGS+=("$1" "$2"); shift 2 ;;
        *)                       CKPTS+=("$1"); shift ;;
    esac
done
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[hb-remote] no checkpoints given" >&2
    exit 1
fi

# Run from the repo root: sbatch from the repo directory (or set REPO_DIR).
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source scripts/env.sh

if [[ -z "${!JUDGE_API_KEY_ENV:-}" ]]; then
    echo "[hb-remote] ERROR: \$$JUDGE_API_KEY_ENV is empty. Did you forget --export=ALL?" >&2
    exit 2
fi

echo "[hb-remote] bootstrap on $(hostname)"
bash scripts/slurm/bootstrap_node.sh
# shellcheck disable=SC1091
source scripts/env.sh

mkdir -p logs "$RAR_OUT"
SHARED_EVAL="$RAR_STAGE/eval"
mkdir -p "$SHARED_EVAL"

# Filename tag: slash-to-dash, lowercase (e.g., openai/gpt-4o-mini -> openai-gpt-4o-mini).
JUDGE_TAG=$(echo "$JUDGE_MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
case "$SCORING" in
    per_criterion) SUFFIX="hb_pc" ;;
    likert)        SUFFIX="hb_likert" ;;
    *)             SUFFIX="hb_${SCORING}" ;;
esac

if [[ "$MAX_TOKENS" -ne 1024 ]]; then
    LEN_TAG="_mt${MAX_TOKENS}"
else
    LEN_TAG=""
fi
# Forward stratify-by + english-only + prompt-ids-file as CLI flags.
EXTRA_FLAGS=()
if [[ "$ENGLISH_ONLY" -eq 1 ]]; then
    EXTRA_FLAGS+=(--english-only)
    SUFFIX="${SUFFIX}_en"
fi
for tag in "${STRATIFY_BY[@]:-}"; do
    [[ -n "$tag" ]] && EXTRA_FLAGS+=(--stratify-by "$tag")
done
if [[ -n "$PROMPT_IDS_FILE" ]]; then
    EXTRA_FLAGS+=(--prompt-ids-file "$PROMPT_IDS_FILE")
    manifest_tag=$(basename "$PROMPT_IDS_FILE" .json | sed 's/healthbench_//')
    SUFFIX="${SUFFIX}_${manifest_tag}"
fi
if [[ -n "$JUDGE_MAX_CONCURRENT" ]]; then
    EXTRA_FLAGS+=(--judge-max-concurrent "$JUDGE_MAX_CONCURRENT")
fi

# Force unbuffered stdout so [hb-eval] progress lines flush immediately
# rather than waiting for the script to terminate. Critical for diagnosing
# stalls (otherwise judge-phase progress is invisible until the run ends).
export PYTHONUNBUFFERED=1

for ckpt in "${CKPTS[@]}"; do
    base=$(basename "$ckpt")
    out="$RAR_OUT/${SUFFIX}_j${JUDGE_TAG}${LEN_TAG}_${base}.json"
    cache="$RAR_STAGE/eval/completions_cache_${SUFFIX}${LEN_TAG}_${base}.json"
    RESUME_FLAGS=()
    [[ "$RESUME_FROM_COMPLETIONS" -eq 1 ]] && RESUME_FLAGS+=(--resume-from-completions)
    echo "[hb-remote] === scoring=$SCORING judge=$JUDGE_MODEL max_tokens=$MAX_TOKENS  $ckpt -> $out (cache=$cache) ==="
    # Uses the GPU Slurm allocated to this job (inherited CUDA_VISIBLE_DEVICES).
    python -m rar.eval_healthbench \
            --checkpoint "$ckpt" \
            --judge-url "$JUDGE_URL" \
            --judge-model "$JUDGE_MODEL" \
            --judge-api-key-env "$JUDGE_API_KEY_ENV" \
            --scoring "$SCORING" \
            --subset "$SUBSET" \
            --n "$N" \
            --max-tokens "$MAX_TOKENS" \
            --output "$out" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization 0.7 \
            --completions-cache "$cache" \
            ${RESUME_FLAGS[@]+"${RESUME_FLAGS[@]}"} \
            ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"} \
            ${GRADER_ARGS[@]+"${GRADER_ARGS[@]}"}
    cp "$out" "$SHARED_EVAL/${SUFFIX}_j${JUDGE_TAG}${LEN_TAG}_${base}.json"
    echo "[hb-remote] copied to $SHARED_EVAL/${SUFFIX}_j${JUDGE_TAG}${LEN_TAG}_${base}.json"
done

echo "[hb-remote] all checkpoints done"
