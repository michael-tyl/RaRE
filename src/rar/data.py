"""Dataset loading utilities for RaR experiments."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset

DEFAULT_HF_DATASET = "anisha2102/RaR-Medicine"
SCIENCE_HF_DATASET = "anisha2102/RaR-Science"


def load_rar_medicine_subset(
    split: str = "train",
    n: int | None = 1000,
    seed: int = 0,
) -> Dataset:
    """Load a deterministic subset of the RaR-Medicine HF dataset.

    Each row contains: question, reference_answer, question_source, rubric,
    rubric_list, rubric_count.
    """
    ds = load_dataset(DEFAULT_HF_DATASET, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def load_rar_science_subset(
    split: str = "train",
    n: int | None = 1000,
    seed: int = 0,
) -> Dataset:
    """Load a deterministic subset of the RaR-Science HF dataset.

    Schema matches RaR-Medicine: question, reference_answer, question_source,
    rubric, rubric_list, rubric_count. Full dataset is ~18k rows; we use a 1k
    subset by default for parity with the medicine training budget.
    """
    ds = load_dataset(SCIENCE_HF_DATASET, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def attach_rubric_column(
    base: Dataset,
    rubric_parquet: str | Path | None,
) -> Dataset:
    """Replace the `rubric` column on `base` with rubrics from a parquet file.

    The parquet file is keyed by `question`. If `rubric_parquet` is None, we
    keep the shipped rubric column unchanged (all paper configs do this).
    """
    if rubric_parquet is None:
        return base

    from datasets import Dataset as HFDataset

    overlay = HFDataset.from_parquet(str(rubric_parquet))
    q_to_rubric = {row["question"]: row["rubric"] for row in overlay}

    def _swap(example: dict) -> dict:
        example["rubric"] = q_to_rubric.get(example["question"], example.get("rubric"))
        return example

    return base.map(_swap, desc="attach_rubric_column")


def to_grpo_dataset(ds: Dataset, system_prompt: str | None = None) -> Dataset:
    """Project a RaR-Medicine row into the (prompt, rubric) shape TRL expects.

    GRPOTrainer accepts a `prompt` column (str OR list[message]) plus any
    extra columns the reward function needs. We pass `rubric` (and `category`
    when present) straight through.
    """
    has_category = "category" in ds.column_names

    def _project(example: dict) -> dict:
        question = example["question"].strip()
        if system_prompt is not None:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        else:
            prompt = [{"role": "user", "content": question}]
        out_row = {"prompt": prompt, "rubric": example["rubric"]}
        if has_category:
            out_row["category"] = example["category"]
        return out_row

    keep = {"prompt", "rubric"}
    if has_category:
        keep.add("category")
    drop = [c for c in ds.column_names if c not in keep and c not in {"question", "reference_answer"}]
    out = ds.map(_project, desc="to_grpo_dataset")
    out = out.remove_columns([c for c in drop if c in out.column_names])
    # Drop the original question/reference_answer columns; the prompt column carries the question.
    for c in ("question", "reference_answer", "question_source", "rubric_list", "rubric_count"):
        if c in out.column_names:
            out = out.remove_columns(c)
    return out


POLICY_SYSTEM_PROMPT = (
    "You are a helpful medical assistant. Answer the question concisely and "
    "accurately, citing relevant medical knowledge. Keep responses focused."
)


SCIENCE_POLICY_SYSTEM_PROMPT = (
    "You are a helpful scientific assistant with expertise in physics, "
    "chemistry, and biology. Answer the question concisely and accurately, "
    "showing relevant reasoning and citing principles or formulas where "
    "appropriate. Keep responses focused."
)
