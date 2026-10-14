"""Merge a saved LoRA adapter into its base model and save as a full
HF checkpoint that vLLM / eval scripts can load directly.

Called automatically by scripts/slurm/run_one_remote_judge.sbatch for LoRA
runs, which writes $RAR_STAGE/checkpoints/<run_name>_merged. Evaluate that
directory, not the adapter directory. Manual usage:
    python scripts/merge_lora.py \\
        --adapter $RAR_STAGE/checkpoints/<run_name> \\
        --output $RAR_STAGE/checkpoints/<run_name>_merged

Reads the base-model name from the adapter's `adapter_config.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Path to write merged full model")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)

    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"[merge_lora] adapter_config.json not found at {cfg_path}")
    with cfg_path.open() as f:
        cfg = json.load(f)
    base_model = cfg["base_model_name_or_path"]
    print(f"[merge_lora] base model: {base_model}")
    print(f"[merge_lora] adapter:    {adapter_dir}")
    print(f"[merge_lora] output:     {output_dir}")

    # Lazy imports — heavy.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    print("[merge_lora] loading base model in bf16...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("[merge_lora] applying LoRA adapter...")
    merged = PeftModel.from_pretrained(base, adapter_dir)
    merged = merged.merge_and_unload()

    print(f"[merge_lora] saving merged model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Copy tokenizer files alongside the merged weights.
    tok = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    tok.save_pretrained(output_dir)

    # Sanity: list output files.
    print("[merge_lora] output files:")
    for p in sorted(output_dir.iterdir()):
        size = p.stat().st_size if p.is_file() else "<dir>"
        print(f"  {p.name}  {size}")
    print(f"[merge_lora] done. eval pipeline can now load {output_dir} directly.")


if __name__ == "__main__":
    main()
