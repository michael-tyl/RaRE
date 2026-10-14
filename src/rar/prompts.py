"""Prompt templates for rubric-based judging (and rubric writing).

- HealthBench grader: OpenAI simple-evals GRADER_TEMPLATE, verbatim
  (per-criterion binary grading for evaluation).
- Hard-Likert judge (1-10): the RaR-Implicit judge prompt of Gunjal et al.
  2025 ("Rubrics as Rewards", arXiv 2507.17746, Appendix A.6). The whole
  rubric is passed as a flat list and the judge returns one rating.
- Soft-Likert judge (1-5): same task on a 1-5 scale with verbal anchors, so
  each rating is a single token whose logprobs can be read.
- Rubric writer: a reconstruction of the RaR rubric-generation prompt. Not
  used by the paper configs, which use the rubrics shipped with
  RaR-Medicine / RaR-Science.

All judge prompts ask for strict JSON; the accompanying JSON Schemas are
passed as `response_format` so the output is constrained to valid JSON.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# JSON Schemas for vLLM structured outputs (xgrammar / guided_json)
# ---------------------------------------------------------------------------


JUDGE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # Paper key (Appendix A.6) is "rating", not "score". reward.py parses
        # this exact field name.
        "rating": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["rating"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Per-criterion binary judge (for paper-faithful HealthBench scoring)
# ---------------------------------------------------------------------------
#
# HealthBench's native scoring is per-criterion binary: for each rubric item,
# ask the judge "does the response satisfy this criterion?" and record a
# yes/no. Aggregate per-prompt as sum(points * met) / sum(points). This is
# what the paper actually uses to report HealthBench numbers.
#
# Different from JUDGE_IMPLICIT which is a single holistic 1-10 rating over
# all rubric items at once. Our training reward uses JUDGE_IMPLICIT because
# it's one judge call per (prompt, response); JUDGE_BINARY needs N_items
# calls per (prompt, response), too expensive for training-time rewards but
# the right choice for eval.
# ---------------------------------------------------------------------------

# VERBATIM from OpenAI simple-evals healthbench_eval.py::GRADER_TEMPLATE.
# Earlier we shipped a simpler binary prompt that scored Qwen2.5-7B-Base at
# 0.3425 via gpt-4o-mini, vs the paper's 0.0818 on the same model+judge.
# Investigation showed the judge prompt was the dominant source of
# divergence: the paper's prompt enforces "if any of the criteria is not
# met → false" which is much stricter than our "does it meet?" phrasing,
# and handles negative-point criteria explicitly. The schema field name
# also changed: `criteria_met` (+ `explanation`), not `met`.
JUDGE_BINARY_GRADER_TEMPLATE = """\
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
"""


JUDGE_BINARY_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
    "additionalProperties": False,
}


def judge_binary_messages(
    question: str,
    response: str,
    criterion: str,
    points: int | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for a per-criterion binary judge call.

    Matches simple-evals' grading: user-only message containing the
    rendered GRADER_TEMPLATE with the conversation and rubric item
    (including point value for negative-criteria handling) substituted in.
    """
    conversation = f"user: {question.strip()}\nassistant: {response.strip()}"
    rubric_item = (
        criterion.strip()
        if points is None
        else f"[{points} points] {criterion.strip()}"
    )
    return [
        {
            "role": "user",
            "content": JUDGE_BINARY_GRADER_TEMPLATE.format(
                conversation=conversation,
                rubric_item=rubric_item,
            ),
        },
    ]

# ---------------------------------------------------------------------------
# Rubric writer (stored on the domain specs; paper runs use shipped rubrics)
# ---------------------------------------------------------------------------

RUBRIC_WRITER_SYSTEM = """\
You are an expert rubric writer for evaluating answers to medical questions.

Given a medical question and a high-quality reference answer, write a
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
4. Together, the items should capture the medically important content of
   the reference answer.
5. Pick the weight integer to reflect how important the item is within its
   category (e.g. an "Important" item that is almost essential should be 4;
   a borderline-optional one should be 2).

Return ONLY valid JSON with this schema (no prose, no code fences):
{
  "rubric": [
    {"title": "<short label>", "category": "Essential|Important|Optional|Pitfall", "description": "<one sentence>", "weight": <integer>},
    ...
  ]
}
"""

RUBRIC_WRITER_USER_TEMPLATE = """\
Question:
{question}

Reference answer:
{reference_answer}

Write the rubric now."""


# ---------------------------------------------------------------------------
# Judge (RaR-Implicit aggregation) — VERBATIM from paper Appendix A.6
# ---------------------------------------------------------------------------
#
# RaR-Implicit: pass *all* rubric items to the judge as a flat list and ask
# it to produce a single 1-10 holistic rating. The paper finds this beats
# explicit weighted-sum aggregation. Output schema is `{"rating": int}`,
# the judge sees the rubrics as a single string with no category labels.
#
# ---------------------------------------------------------------------------

JUDGE_IMPLICIT_SYSTEM = (
    "You are an expert evaluator. Given a user prompt, a generated response, "
    "and a list of quality rubrics, please rate the overall quality of the "
    "response on a scale of 1 to 10 based on how well it satisfies the rubrics. "
    "Respond with ONLY a valid JSON object of the form {\"rating\": <integer 1-10>} "
    "and nothing else (no code fences, no rationale, no additional fields)."
)

JUDGE_IMPLICIT_USER_TEMPLATE = (
    "Given the following prompt, response, and rubrics, please rate the overall "
    "quality of the response on a scale of 1 to 10 based on how well it satisfies "
    "the rubrics. <prompt>{question}</prompt> <response>{response}</response> "
    "<rubrics>{rubric_text}</rubrics> "
    "Your JSON Evaluation (must be exactly {{\"rating\": <integer 1-10>}}):"
)


# ---------------------------------------------------------------------------
# Soft-Likert 1-5 judge (logprob-extracted confidence)
# ---------------------------------------------------------------------------
# Identical task to the 1-10 implicit judge, but on a 1-5 scale. The narrower
# scale is more reliable per-rating for LLMs, and the rating digit is always a
# single token regardless of tokenizer. Pairs with the soft-Likert reward fn
# which reads top_logprobs at the rating position to recover a continuous
# expected score (instead of the hard integer).


JUDGE_LIKERT5_SYSTEM = (
    "You are an expert evaluator. Given a user prompt, a generated response, "
    "and a list of quality rubrics, rate the overall quality of the response "
    "on a scale of 1 to 5 based on how well it satisfies the rubrics, where "
    "1=very poor, 2=poor, 3=neutral, 4=good, 5=excellent. "
    "Respond with ONLY a valid JSON object of the form {\"rating\": <integer 1-5>} "
    "and nothing else (no code fences, no rationale, no additional fields)."
)

JUDGE_LIKERT5_USER_TEMPLATE = (
    "Given the following prompt, response, and rubrics, rate the overall "
    "quality of the response on a scale of 1 to 5 (1=very poor, 5=excellent). "
    "<prompt>{question}</prompt> <response>{response}</response> "
    "<rubrics>{rubric_text}</rubrics> "
    "Your JSON Evaluation (must be exactly {{\"rating\": <integer 1-5>}}):"
)


def judge_likert5_messages(question: str, response: str, rubric: Any) -> list[dict[str, str]]:
    rubric_text = _format_rubric_items(rubric)
    return [
        {"role": "system", "content": JUDGE_LIKERT5_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_LIKERT5_USER_TEMPLATE.format(
                question=question.strip(),
                response=response.strip(),
                rubric_text=rubric_text,
            ),
        },
    ]


_LIKERT_ANCHORS = {
    5: "1=very poor, 2=poor, 3=neutral, 4=good, 5=excellent",
    7: "1=very poor, 2=poor, 3=below average, 4=neutral, 5=above average, 6=good, 7=excellent",
    9: "1-2=very poor, 3-4=poor, 5=neutral, 6-7=good, 8-9=excellent",
    10: "1-2=very poor, 3-4=poor, 5-6=neutral, 7-8=good, 9-10=excellent",
}


def judge_likert_schema(scale_max: int) -> dict:
    return {
        "type": "object",
        "properties": {
            "rating": {"type": "integer", "minimum": 1, "maximum": int(scale_max)},
        },
        "required": ["rating"],
        "additionalProperties": False,
    }


def judge_likert_messages(question: str, response: str, rubric: Any, scale_max: int = 5) -> list[dict[str, str]]:
    if int(scale_max) == 5:
        return judge_likert5_messages(question, response, rubric)
    anchors = _LIKERT_ANCHORS.get(int(scale_max), f"1=very poor, {scale_max}=excellent")
    system = (
        "You are an expert evaluator. Given a user prompt, a generated response, "
        "and a list of quality rubrics, rate the overall quality of the response "
        f"on a scale of 1 to {scale_max} based on how well it satisfies the rubrics, where "
        f"{anchors}. "
        f"Respond with ONLY a valid JSON object of the form {{\"rating\": <integer 1-{scale_max}>}} "
        "and nothing else (no code fences, no rationale, no additional fields)."
    )
    user = (
        f"Given the following prompt, response, and rubrics, rate the overall "
        f"quality of the response on a scale of 1 to {scale_max} ({anchors}). "
        f"<prompt>{question.strip()}</prompt> <response>{response.strip()}</response> "
        f"<rubrics>{_format_rubric_items(rubric)}</rubrics> "
        f"Your JSON Evaluation (must be exactly {{\"rating\": <integer 1-{scale_max}>}}):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_rubric_items(rubric: Any) -> str:
    """Render a rubric as a flat list of description strings.

    Paper Appendix A.6 passes rubrics to the judge as a list/string with no
    category labels — the judge weighs them holistically. We keep our writers'
    weights in the parquet but strip them for the judge prompt.
    """
    if isinstance(rubric, str):
        try:
            rubric = json.loads(rubric)
        except json.JSONDecodeError:
            return rubric
    if isinstance(rubric, dict) and "rubric" in rubric:
        rubric = rubric["rubric"]
    if not isinstance(rubric, list):
        return str(rubric)

    lines: list[str] = []
    for i, item in enumerate(rubric, 1):
        if isinstance(item, str):
            lines.append(f"{i}. {item}")
            continue
        desc = item.get("description", "")
        if not desc:
            desc = item.get("title", "")
        lines.append(f"{i}. {desc}".strip())
    return "\n".join(lines)


def judge_messages(question: str, response: str, rubric: Any) -> list[dict[str, str]]:
    """Build the chat messages for an Implicit RaR judge call."""
    rubric_text = _format_rubric_items(rubric)
    return [
        {"role": "system", "content": JUDGE_IMPLICIT_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_IMPLICIT_USER_TEMPLATE.format(
                question=question.strip(),
                response=response.strip(),
                rubric_text=rubric_text,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Pairwise judge (appendix baseline): pick the better of two responses,
# with or without the rubric. Output schema is {"winner": "A" | "B"}.
# ---------------------------------------------------------------------------

JUDGE_PAIRWISE_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        # "winner" returns "A" or "B"; we don't allow ties because a forced
        # choice is exactly the saturation-breaking property we want.
        "winner": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["winner"],
    "additionalProperties": False,
}

JUDGE_PAIRWISE_SYSTEM = (
    "You are an expert evaluator comparing two responses to a user prompt. "
    "Given a prompt, a list of quality rubrics, and two candidate responses "
    "(A and B), decide which response better satisfies the rubrics overall. "
    "You MUST pick exactly one, even if both are similar. "
    "Respond with ONLY a valid JSON object of the form {\"winner\": \"A\"} or "
    "{\"winner\": \"B\"} and nothing else (no code fences, no rationale, no "
    "additional fields)."
)

JUDGE_PAIRWISE_USER_TEMPLATE = (
    "Given the following prompt, rubrics, and two responses (A and B), decide "
    "which response better satisfies the rubrics. "
    "<prompt>{question}</prompt> "
    "<rubrics>{rubric_text}</rubrics> "
    "<response_A>{response_a}</response_A> "
    "<response_B>{response_b}</response_B> "
    "Your JSON evaluation (must be exactly {{\"winner\": \"A\"}} or {{\"winner\": \"B\"}}):"
)

def judge_pairwise_messages(
    question: str,
    response_a: str,
    response_b: str,
    rubric: Any,
) -> list[dict[str, str]]:
    """Build the chat messages for a pairwise rubric-judge call."""
    rubric_text = _format_rubric_items(rubric)
    return [
        {"role": "system", "content": JUDGE_PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_PAIRWISE_USER_TEMPLATE.format(
                question=question.strip(),
                response_a=response_a.strip(),
                response_b=response_b.strip(),
                rubric_text=rubric_text,
            ),
        },
    ]

JUDGE_PAIRWISE_NORUBRIC_SYSTEM = (
    "You are an expert evaluator comparing two responses to a user prompt. "
    "Given a prompt and two candidate responses (A and B), decide which "
    "response is better overall in terms of accuracy, helpfulness, depth, "
    "and clarity. You MUST pick exactly one, even if both are similar. "
    "Respond with ONLY a valid JSON object of the form {\"winner\": \"A\"} or "
    "{\"winner\": \"B\"} and nothing else (no code fences, no rationale, no "
    "additional fields)."
)

JUDGE_PAIRWISE_NORUBRIC_USER_TEMPLATE = (
    "Given the following prompt and two responses (A and B), decide which "
    "response is better. "
    "<prompt>{question}</prompt> "
    "<response_A>{response_a}</response_A> "
    "<response_B>{response_b}</response_B> "
    "Your JSON evaluation (must be exactly {{\"winner\": \"A\"}} or {{\"winner\": \"B\"}}):"
)

def judge_pairwise_norubric_messages(
    question: str,
    response_a: str,
    response_b: str,
    rubric: Any = None,
) -> list[dict[str, str]]:
    """Pairwise messages with no rubric supplied to the judge.

    `rubric` is accepted (and ignored) so this can be a drop-in for
    `judge_pairwise_messages` in code paths that always pass a rubric.
    """
    return [
        {"role": "system", "content": JUDGE_PAIRWISE_NORUBRIC_SYSTEM},
        {
            "role": "user",
            "content": JUDGE_PAIRWISE_NORUBRIC_USER_TEMPLATE.format(
                question=question.strip(),
                response_a=response_a.strip(),
                response_b=response_b.strip(),
            ),
        },
    ]
