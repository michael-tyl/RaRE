"""Science domain: RaR-Science training + GPQA Diamond eval.

Mirrors the medicine domain: load the RaR-Science dataset shipped by
Gunjal et al. (same schema as RaR-Medicine), train RaR-GRPO with judge
ensembles, and evaluate on GPQA Diamond (Rein et al. 2023) via MCQ
accuracy. No LLM judge is needed at eval time — GPQA is multiple choice.

All paper experiments use `rubric_path: null` and consume the shipped
rubric column directly.
"""

from __future__ import annotations

from . import DomainSpec, EvalSpec, register
from ..data import SCIENCE_POLICY_SYSTEM_PROMPT, load_rar_science_subset


SCIENCE_RUBRIC_WRITER_SYSTEM = """\
You are an expert rubric writer for evaluating answers to science questions
(physics, chemistry, biology).

Given a science question and a high-quality reference answer, write a
self-contained checklist of criteria ("rubric") that could be used to judge
whether a candidate response is good. The rubric should:

1. Contain between 7 and 15 items.
2. Each item must be one of four categories with the matching integer weight:
   - "Essential" (weight 5):  the response MUST satisfy this
   - "Important" (weight 3 or 4):  strongly expected
   - "Optional"  (weight 1 or 2):  nice to have, small bonus
   - "Pitfall"   (weight -1 or -2):  the response MUST NOT do this
3. Each item must be specific, observable, and judgeable from the response
   alone, without needing the reference answer.
4. Together, the items should capture the scientifically important content
   of the reference answer (correct concepts, formulas, units, reasoning
   steps).
5. Pick the weight integer to reflect how important the item is within its
   category.

Return ONLY valid JSON with this schema (no prose, no code fences):
{
  "rubric": [
    {"title": "<short label>", "category": "Essential|Important|Optional|Pitfall", "description": "<one sentence>", "weight": <integer>},
    ...
  ]
}
"""

SCIENCE_RUBRIC_WRITER_USER_TEMPLATE = """\
Question:
{question}

Reference answer:
{reference_answer}

Write the rubric now."""


SCIENCE = DomainSpec(
    name="science",
    train_loader=load_rar_science_subset,
    policy_system_prompt=SCIENCE_POLICY_SYSTEM_PROMPT,
    rubric_writer_system=SCIENCE_RUBRIC_WRITER_SYSTEM,
    rubric_writer_user_template=SCIENCE_RUBRIC_WRITER_USER_TEMPLATE,
    rubric_parquet_default="rubrics_base_4b_science.parquet",
    eval_specs=[
        EvalSpec(
            name="gpqa_diamond",
            description="GPQA Diamond (198 expert-validated PhD-level science MCQs)",
            default_n=198,
            runner_id="gpqa:diamond",
        ),
    ],
    n_train_default=1000,
    seed_default=0,
)

register(SCIENCE)
