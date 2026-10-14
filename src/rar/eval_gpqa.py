"""Out-of-distribution eval on GPQA Diamond (paper's primary science eval).

GPQA Diamond is the 198-example expert-validated subset of GPQA
(Rein et al. 2023). Each example is a graduate-level multiple-choice
question in biology / chemistry / physics, with one correct answer and
three distractors written by other PhDs.

Unlike HealthBench this is MCQ, so no LLM judge is needed at eval time —
we just check whether the model's emitted letter (A / B / C / D) matches
the correct one. The shuffle of distractors is deterministic per
(prompt_id, seed) so different policies see the same choice ordering.

GPQA Diamond is **gated** on HF: you must run `hf auth login` once,
visit https://huggingface.co/datasets/Idavidrein/gpqa and accept the
data-use terms, then this loader will work.

Usage:
    python -m rar.eval_gpqa \\
        --checkpoint $RAR_CKPT/jvr_sci_mini \\
        --output $RAR_OUT/gpqa_jvr_sci_mini.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np

from .data import SCIENCE_POLICY_SYSTEM_PROMPT


GPQA_HF_REPO = "Idavidrein/gpqa"
GPQA_DIAMOND_CONFIG = "gpqa_diamond"

# GPQA Diamond ships these columns. We rely on the first five; the
# extras are kept for metadata in the output JSON.
GPQA_COL_QUESTION = "Question"
GPQA_COL_CORRECT = "Correct Answer"
GPQA_COL_INCORRECT = [
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
]

MCQ_PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Choices:\n"
    "(A) {a}\n"
    "(B) {b}\n"
    "(C) {c}\n"
    "(D) {d}\n\n"
    "Think step by step, then end your response with a single line of the "
    "form `Answer: X` where X is one of A, B, C, or D."
)

# Corrected parser (default): the LAST explicit answer statement ("Answer: X",
# "answer is X", "final answer: (X)", "\\boxed{X}"), where X is a standalone
# capital A-D.
EXPLICIT_ANSWER_RE = re.compile(
    r"(?:\b(?i:final\s+answer|answer)\b\s*(?i:is\b)?\s*[:=\-]?\s*\(?\s*([ABCD])\b\)?"
    r"|\\boxed\{\s*\(?([ABCD])\)?\s*\})"
)
# Legacy parser (used for the paper's original GPQA numbers): the FIRST match,
# case-insensitive, no word boundary, so e.g. "The answer depends ..." parses
# as D. Kept only to reproduce those numbers (--legacy-answer-parser).
LEGACY_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is)?\s*[:=\-]?\s*\(?([ABCD])\)?",
    re.IGNORECASE,
)
FALLBACK_LAST_LETTER_RE = re.compile(r"\b([ABCD])\b")


def _bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(0)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_resamples)]
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def _load_gpqa_diamond(n: int, seed: int) -> list[dict]:
    """Load GPQA Diamond and return a deterministic n-row subset with shuffled choices.

    Returned rows have:
        question: str       (the question text)
        choices: list[str]  (4 answer strings in display order)
        correct_letter: str (one of "A" / "B" / "C" / "D")
        subdomain: str      (e.g., "Organic Chemistry"); empty if not present
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(GPQA_HF_REPO, GPQA_DIAMOND_CONFIG, split="train")
    except Exception as e:
        raise SystemExit(
            f"[gpqa-eval] failed to load {GPQA_HF_REPO}/{GPQA_DIAMOND_CONFIG}: {e!r}\n"
            "GPQA is gated on the Hub. Run `hf auth login`, then accept the "
            "terms at https://huggingface.co/datasets/Idavidrein/gpqa and retry."
        )

    rows = list(ds)
    if n < len(rows):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))[:n].tolist()
        idx.sort()
        rows = [rows[i] for i in idx]

    out: list[dict] = []
    for i, r in enumerate(rows):
        question = (r.get(GPQA_COL_QUESTION) or "").strip()
        correct = (r.get(GPQA_COL_CORRECT) or "").strip()
        wrongs = [(r.get(c) or "").strip() for c in GPQA_COL_INCORRECT]
        if not question or not correct or any(not w for w in wrongs):
            continue
        # Deterministic per-row shuffle so different policies see identical orderings.
        rng = np.random.default_rng(seed * 7919 + i)
        order = rng.permutation(4).tolist()
        choices_all = [correct, *wrongs]
        choices = [choices_all[j] for j in order]
        correct_idx = order.index(0)
        correct_letter = "ABCD"[correct_idx]
        out.append(
            {
                "question": question,
                "choices": choices,
                "correct_letter": correct_letter,
                "subdomain": (r.get("Subdomain") or r.get("High-level domain") or "").strip(),
            }
        )
    return out


def _extract_letter(text: str) -> str | None:
    """Pull out A / B / C / D from a model completion.

    1. Take the LAST explicit answer statement (`Answer: X`, `final answer
       is X`, `\\boxed{X}`, ...) with X a standalone capital A-D.
    2. Otherwise fall back to the LAST standalone capital A-D token.
    """
    if not text:
        return None
    last = None
    for m in EXPLICIT_ANSWER_RE.finditer(text):
        last = m.group(1) or m.group(2)
    if last:
        return last
    for m in FALLBACK_LAST_LETTER_RE.finditer(text):
        last = m.group(1)
    return last


def _extract_letter_legacy(text: str) -> str | None:
    """Original parser (first case-insensitive match); see LEGACY_ANSWER_RE."""
    if not text:
        return None
    m = LEGACY_ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    last = None
    for m in FALLBACK_LAST_LETTER_RE.finditer(text):
        last = m.group(1).upper()
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=198,
                        help="Number of GPQA Diamond questions to evaluate (max 198).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Generation budget per question. GPQA answers benefit "
                             "from chain-of-thought, so 1024 is the default.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy decoding (matches GPQA's standard eval recipe).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--stratify-by-subdomain",
        action="store_true",
        help="Emit per-subdomain accuracy in the output JSON (chemistry / physics / biology).",
    )
    parser.add_argument(
        "--legacy-answer-parser",
        action="store_true",
        help="Score with the original answer parser (reproduces the paper's original "
             "GPQA numbers). The corrected parser's accuracy is reported either way.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    print(f"[gpqa-eval] loading GPQA Diamond, n={args.n}, seed={args.seed}")
    examples = _load_gpqa_diamond(args.n, args.seed)
    print(f"[gpqa-eval] sampled {len(examples)} valid prompts")

    print(f"[gpqa-eval] loading policy {args.checkpoint}")
    llm = LLM(
        model=args.checkpoint,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens)

    chats = [
        [
            {"role": "system", "content": SCIENCE_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": MCQ_PROMPT_TEMPLATE.format(
                question=ex["question"],
                a=ex["choices"][0],
                b=ex["choices"][1],
                c=ex["choices"][2],
                d=ex["choices"][3],
            )},
        ]
        for ex in examples
    ]

    print(f"[gpqa-eval] generating {len(chats)} completions...")
    outs = llm.chat(chats, sampling_params=sampling, chat_template_kwargs={"enable_thinking": False})
    completions = [o.outputs[0].text for o in outs]

    # Score.
    gold = [ex["correct_letter"] for ex in examples]
    pred_fixed = [_extract_letter(c) for c in completions]
    pred_legacy = [_extract_letter_legacy(c) for c in completions]
    predictions = pred_legacy if args.legacy_answer_parser else pred_fixed
    answer_parser = "legacy" if args.legacy_answer_parser else "corrected"
    correct = [1.0 if p == g else 0.0 for p, g in zip(predictions, gold)]
    n_unparseable = sum(1 for p in predictions if p is None)
    mean_fixed = statistics.fmean(1.0 if p == g else 0.0 for p, g in zip(pred_fixed, gold))
    mean_legacy = statistics.fmean(1.0 if p == g else 0.0 for p, g in zip(pred_legacy, gold))

    mean = statistics.fmean(correct)
    lo, hi = _bootstrap_ci(correct)
    print(f"[gpqa-eval] accuracy ({answer_parser} parser) = {mean:.4f}  95% CI = [{lo:.4f}, {hi:.4f}]  "
          f"n={len(correct)}  unparseable={n_unparseable}")
    print(f"[gpqa-eval] corrected parser = {mean_fixed:.4f}  legacy parser = {mean_legacy:.4f}")

    stratifications: dict = {}
    if args.stratify_by_subdomain:
        from collections import defaultdict
        groups: dict[str, list[float]] = defaultdict(list)
        for c, ex in zip(correct, examples):
            label = ex.get("subdomain") or "unknown"
            groups[label].append(c)
        for label, vals in groups.items():
            m = statistics.fmean(vals)
            blo, bhi = _bootstrap_ci(vals) if len(vals) >= 2 else (m, m)
            stratifications[label] = {
                "n": len(vals),
                "mean": float(m),
                "ci_low": float(blo),
                "ci_high": float(bhi),
            }
        print("[gpqa-eval] by subdomain:")
        for label, info in sorted(stratifications.items(), key=lambda kv: -kv[1]["n"]):
            print(f"  {label:40s} n={info['n']:3d}  acc={info['mean']:.4f}  "
                  f"CI=[{info['ci_low']:.4f}, {info['ci_high']:.4f}]")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "benchmark": "GPQA-Diamond",
                    "scoring": "mcq_exact",
                    "answer_parser": answer_parser,
                    "mean_corrected_parser": mean_fixed,
                    "mean_legacy_parser": mean_legacy,
                    "n": len(correct),
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "rewards": correct,
                    "predictions": predictions,
                    "gold_letters": gold,
                    "completion_chars": [len(c) for c in completions],
                    "subdomains": [ex.get("subdomain", "") for ex in examples],
                    "n_unparseable": n_unparseable,
                    "stratifications": stratifications,
                },
                f,
                indent=2,
            )
        print(f"[gpqa-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
