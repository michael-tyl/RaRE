"""Build a balanced 1000-prompt subset of HealthBench-OSS English-only.

Proportional stratified sample by `theme:*` example_tag, deterministic via
seed=42. Writes a JSON manifest of prompt_ids to $RAR_STAGE/data/.

Reusable across evals so per-checkpoint numbers stay apples-to-apples.

Usage:
    python scripts/build_hb_balanced1k.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from rar.eval_healthbench import _likely_english


def prompt_text(r: dict) -> str:
    parts = [t.get("content", "") for t in r["prompt"] if t.get("role") == "user"]
    return "\n\n".join(p for p in parts if p)


def get_theme(r: dict) -> str | None:
    for t in r.get("example_tags", []):
        if t.startswith("theme:"):
            return t
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Target subset size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", choices=("oss", "hard", "consensus"), default="oss")
    parser.add_argument("--output", default=None,
                        help="Output JSON path. Defaults to "
                             "$RAR_STAGE/data/healthbench_<subset>_en_balanced<N>_seed<seed>.json")
    args = parser.parse_args()

    files = {
        "oss":       "2025-05-07-06-14-12_oss_eval.jsonl",
        "hard":      "hard_2025-05-08-21-00-10.jsonl",
        "consensus": "consensus_2025-05-09-20-00-46.jsonl",
    }
    fp = hf_hub_download(
        repo_id="openai/healthbench", filename=files[args.subset], repo_type="dataset",
    )
    with open(fp) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    print(f"[build] loaded {args.subset}: {len(rows)} prompts")

    english = [r for r in rows if _likely_english(prompt_text(r))]
    print(f"[build] English-only: {len(english)} ({100*len(english)/len(rows):.1f}%)")

    # Bucket by theme
    by_theme: dict[str, list[dict]] = defaultdict(list)
    untagged = 0
    for r in english:
        t = get_theme(r)
        if t is None:
            untagged += 1
            continue
        by_theme[t].append(r)
    if untagged:
        print(f"[build] WARNING: {untagged} prompts had no theme tag; excluded")

    # Proportional allocation
    total_with_theme = sum(len(v) for v in by_theme.values())
    quotas: dict[str, int] = {}
    for theme, items in by_theme.items():
        quotas[theme] = round(args.n * len(items) / total_with_theme)
    # Fix rounding drift
    drift = args.n - sum(quotas.values())
    if drift != 0:
        # Adjust the largest bucket
        biggest = max(by_theme, key=lambda t: len(by_theme[t]))
        quotas[biggest] += drift

    # Sample
    rng = random.Random(args.seed)
    selected: list[dict] = []
    print("[build] theme allocation:")
    for theme in sorted(by_theme.keys()):
        items = list(by_theme[theme])
        rng.shuffle(items)
        take = quotas[theme]
        if take > len(items):
            print(f"  WARN: quota {take} > available {len(items)} for {theme}; taking all")
            take = len(items)
        selected.extend(items[:take])
        short = theme.replace("theme:", "")
        print(f"  {short:<25s}  pool={len(items):>4d}  take={take:>4d}  "
              f"({100*take/args.n:.1f}%)")
    print(f"[build] total selected: {len(selected)}")

    # Output manifest
    if args.output is None:
        stage = os.environ["RAR_STAGE"]  # set by scripts/env.sh
        args.output = f"{stage}/data/healthbench_{args.subset}_en_balanced{args.n}_seed{args.seed}.json"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": f"healthbench_{args.subset}_en_balanced{args.n}_seed{args.seed}",
        "subset": args.subset,
        "english_only": True,
        "n": len(selected),
        "seed": args.seed,
        "theme_quotas": quotas,
        "prompt_ids": [r["prompt_id"] for r in selected],
    }
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build] wrote {out_path}")
    print(f"[build] manifest name: {manifest['name']}")


if __name__ == "__main__":
    main()
