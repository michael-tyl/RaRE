#!/usr/bin/env bash
# Prepare node-local storage ($RAR_LOCAL) on whichever Slurm node a job lands on:
#   0. Prune regeneratable caches if local disk is filling up
#   1. Create the local working directories
#   2. (Optional) Mirror model weights from a shared HF cache to local disk
#   3. (Optional) Copy an HF token from shared storage
#   4. Run `uv sync` to materialize the local venv
#
# Idempotent: safe to call before every job (the sbatch launchers do).
#
# Optional environment variables (set them in .env):
#   RAR_SHARED_HF_HUB    shared HF hub cache to mirror Qwen3-4B(-Base) from
#   RAR_SHARED_HF_TOKEN  shared HF token file to copy into $HF_HOME
# Alternatively set HF_TOKEN in .env; huggingface_hub reads it directly.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
# shellcheck disable=SC1091
source scripts/env.sh

NODE=$(hostname)
echo "[bootstrap] node=$NODE  RAR_LOCAL=$RAR_LOCAL"

# 0. Disk hygiene ----------------------------------------------------------
# The torchinductor, vLLM compile and Triton caches grow across runs. Prune
# them when /tmp is over 60% full so the node's local disk does not fill up.
TMP_USE_PCT=$(df -P /tmp 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5+0}')
echo "[bootstrap] /tmp at ${TMP_USE_PCT:-?}% used"
if [[ -n "${TMP_USE_PCT:-}" && "$TMP_USE_PCT" -gt 60 ]]; then
    echo "[bootstrap] pruning regeneratable caches (>60% threshold)"
    rm -rf "$TORCHINDUCTOR_CACHE_DIR"/* 2>/dev/null || true
    rm -rf "/tmp/torchinductor_${USER}"/* 2>/dev/null || true
    rm -rf "$VLLM_CACHE_ROOT/torch_compile_cache" 2>/dev/null || true
    rm -rf "$TRITON_CACHE_DIR"/* 2>/dev/null || true
    # Old wandb runs (already uploaded; keep last 24h for log inspection)
    find "$WANDB_DIR" -maxdepth 2 -type d -name 'run-*' -mmin +1440 \
        -exec rm -rf {} + 2>/dev/null || true
    find "$UV_CACHE_DIR" -maxdepth 2 -type f -mmin +1440 \
        -delete 2>/dev/null || true
    AFTER_USE_PCT=$(df -P /tmp 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5+0}')
    echo "[bootstrap] /tmp now at ${AFTER_USE_PCT}% used"
fi

# Abort rather than fill the disk during model load.
TMP_USE_PCT=$(df -P /tmp 2>/dev/null | awk 'NR==2 {gsub("%",""); print $5+0}')
if [[ -n "${TMP_USE_PCT:-}" && "$TMP_USE_PCT" -gt 90 ]]; then
    echo "[bootstrap] ERROR: /tmp at ${TMP_USE_PCT}% even after pruning; aborting" >&2
    df -h /tmp >&2
    exit 5
fi

# 1. Working dirs ----------------------------------------------------------
mkdir -p "$RAR_DATA" "$RAR_CKPT" "$RAR_OUT" \
         "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" \
         "$VLLM_CACHE_ROOT" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
         "$UV_CACHE_DIR"

# 2. Mirror model weights from a shared HF cache (optional) ----------------
# Without RAR_SHARED_HF_HUB, vLLM / transformers download the models on first use.
if [[ -n "${RAR_SHARED_HF_HUB:-}" ]]; then
    for model in models--Qwen--Qwen3-4B models--Qwen--Qwen3-4B-Base; do
        src="$RAR_SHARED_HF_HUB/$model"
        dst="$HF_HUB_CACHE/$model"
        if [[ ! -d "$src" ]]; then
            echo "[bootstrap] $model not in shared cache; it will be downloaded"
            continue
        fi
        # A local copy that is much smaller than the source is symlinks-only
        # (reads would still go to shared storage), so re-copy with -L.
        if [[ -d "$dst" ]]; then
            src_real=$(du -sBM --apparent-size "$src/" 2>/dev/null | awk '{print $1+0}')
            dst_real=$(du -sBM "$dst/" 2>/dev/null | awk '{print $1+0}')
            if [[ -n "$src_real" && -n "$dst_real" && "$dst_real" -ge $((src_real / 2)) ]]; then
                echo "[bootstrap] $model already on local disk (${dst_real}M), skipping"
                continue
            fi
            rm -rf "$dst"
        fi
        echo "[bootstrap] mirroring $model from shared cache to local disk..."
        rsync -aL --info=progress2 "$src/" "$dst/" 2>&1 | tail -3 || true
    done
fi

# 3. HF auth token (optional) ----------------------------------------------
if [[ -n "${RAR_SHARED_HF_TOKEN:-}" && -f "$RAR_SHARED_HF_TOKEN" ]]; then
    echo "[bootstrap] copying HF token to $HF_HOME/"
    cp "$RAR_SHARED_HF_TOKEN" "$HF_HOME/token"
fi

# 4. uv sync ---------------------------------------------------------------
uv sync --python 3.12

echo "[bootstrap] node $NODE ready"
