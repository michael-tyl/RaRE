"""English-only HealthBench scores (the paper's headline metric) from full
HealthBench-OSS eval JSONs.

Each eval JSON has `prompt_ids` and `rewards` (per-example, aligned). A
prompt is English if langdetect assigns `en` with probability >= 0.90 to its
user turns (3,931 of the 5,000 OSS prompts). For each file this prints the
full, English-only and non-English means, with a 95% percentile-bootstrap
CI (1,000 resamples) for the English-only mean.

Usage:
    python scripts/hb5k_split_by_language.py $RAR_STAGE/eval/hb_pc_jgpt-4o-mini_mt8192_*.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download
from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

OSS_FILE = "2025-05-07-06-14-12_oss_eval.jsonl"
ENGLISH_MIN_PROB = 0.90


def load_hb_oss() -> dict[str, str]:
    """Return prompt_id -> all user turns (joined) for HB OSS."""
    fp = hf_hub_download(repo_id="openai/healthbench", filename=OSS_FILE, repo_type="dataset")
    out: dict[str, str] = {}
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("prompt_id", "")
            user_turns = [t.get("content", "") for t in row.get("prompt", []) if t.get("role") == "user"]
            out[pid] = "\n\n".join(p for p in user_turns if p)
    return out


def is_english(text: str) -> bool:
    if not text or not text.strip():
        return False
    try:
        cands = detect_langs(text)
    except LangDetectException:
        return False
    if not cands:
        return False
    top = cands[0]
    return top.lang == "en" and top.prob >= ENGLISH_MIN_PROB


def bootstrap_ci(vals: list[float], n_resamples: int = 1000) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    arr = np.array(vals, float)
    rng = np.random.default_rng(0)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    print("loading HB OSS prompt index...")
    pid_to_q = load_hb_oss()
    print(f"  {len(pid_to_q)} prompts")

    print("language-detecting once (shared across all eval files)...")
    pid_lang_en = {pid: is_english(q) for pid, q in pid_to_q.items()}
    en_count = sum(1 for v in pid_lang_en.values() if v)
    print(f"  english: {en_count} / {len(pid_lang_en)}  non-english: {len(pid_lang_en)-en_count}")

    rows = []
    for p in args.paths:
        if not os.path.exists(p):
            print(f"  skip (missing): {p}")
            continue
        d = json.load(open(p))
        rewards = d.get("rewards", [])
        pids = d.get("prompt_ids", [])
        if len(rewards) != len(pids):
            print(f"  skip (mismatch): {p}")
            continue
        en_r = [r for r, pid in zip(rewards, pids) if pid_lang_en.get(pid, False)]
        ne_r = [r for r, pid in zip(rewards, pids) if not pid_lang_en.get(pid, False)]
        full_mean = float(np.mean(rewards)) if rewards else 0.0
        en_mean = float(np.mean(en_r)) if en_r else 0.0
        ne_mean = float(np.mean(ne_r)) if ne_r else 0.0
        # Paper-faithful HB clip to [0, 1] applies to means only.
        full_mean_c = max(0.0, min(1.0, full_mean))
        en_mean_c = max(0.0, min(1.0, en_mean))
        ne_mean_c = max(0.0, min(1.0, ne_mean))
        full_lo, full_hi = bootstrap_ci(rewards)
        en_lo, en_hi = bootstrap_ci(en_r)
        ne_lo, ne_hi = bootstrap_ci(ne_r)
        rows.append({
            "file": os.path.basename(p),
            "n": len(rewards),
            "full": (full_mean_c, full_lo, full_hi),
            "english": (en_mean_c, en_lo, en_hi, len(en_r)),
            "nonenglish": (ne_mean_c, ne_lo, ne_hi, len(ne_r)),
        })

    print()
    print(f"{'file':<70} {'n':>5} {'full':>7} {'english [95% CI]':>24} {'n_en':>5} {'non-en':>7}")
    print("-" * 124)
    for r in rows:
        en_mean, en_lo, en_hi, n_en = r["english"]
        en_s = f"{en_mean:.4f} [{en_lo:.4f}, {en_hi:.4f}]"
        print(f"{r['file']:<70} {r['n']:>5} {r['full'][0]:>7.4f} {en_s:>24} {n_en:>5} {r['nonenglish'][0]:>7.4f}")
        if n_en < en_count:
            print(f"  WARNING: {en_count - n_en} English prompts missing from this file "
                  "(e.g. dropped after grader failures); its English mean is not comparable.")


if __name__ == "__main__":
    main()
