"""GPQA Diamond accuracy over ten answer-order permutations (paper appendix).

Each policy is evaluated ten times on all 198 questions, permutation k using
`gpqa_eval.sbatch --seed k` (k = 0 is the default answer order). This script
reports the mean accuracy over the ten trials with mean +/- 2 standard errors,
under the corrected answer parser and the legacy parser the paper used.

Usage:
    for k in 0 1 2 3 4 5 6 7 8 9; do
        sbatch --export=ALL scripts/slurm/gpqa_eval.sbatch --max-tokens 8192 --seed $k <checkpoints...>
    done
    python scripts/gpqa_permutations.py jvr_sci_dsmini_softlikert_drop040 jvr_sci_mini Qwen3-4B-Base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics

EVAL_DIR = os.path.join(os.environ.get("RAR_STAGE", f"/workspace/home/{os.environ.get('USER', '')}/deep_rl_stage"), "eval")


def trial_path(run: str, k: int, max_tokens: int) -> str:
    len_tag = "" if max_tokens == 1024 else f"_mt{max_tokens}"
    perm_tag = "" if k == 0 else f"_perm{k}"
    return os.path.join(EVAL_DIR, f"gpqa{len_tag}{perm_tag}_{run}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="checkpoint basenames, e.g. jvr_sci_mini")
    ap.add_argument("--permutations", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()

    print(f"{'run':45s} {'parser':10s} {'mean':>7s} {'+/- 2SE':>8s}  per-permutation accuracies")
    for run in args.runs:
        trials = []
        for k in range(args.permutations):
            path = trial_path(run, k, args.max_tokens)
            if not os.path.exists(path):
                raise SystemExit(f"missing {path} (run gpqa_eval.sbatch --seed {k})")
            d = json.load(open(path))
            assert d["n"] == 198, f"{path}: expected all 198 questions, got n={d['n']}"
            trials.append(d)
        for parser, key in (("corrected", "mean_corrected_parser"), ("legacy", "mean_legacy_parser")):
            accs = [t[key] for t in trials]
            mean = statistics.fmean(accs)
            se = statistics.stdev(accs) / math.sqrt(len(accs))
            print(f"{run:45s} {parser:10s} {100 * mean:7.2f} {200 * se:8.2f}  "
                  + " ".join(f"{100 * a:.1f}" for a in accs))


if __name__ == "__main__":
    main()
