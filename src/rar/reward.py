"""Async rubric-judge reward functions for TRL GRPOTrainer.

Each `make_*_reward()` returns a callable for `GRPOTrainer(reward_funcs=...)`.
TRL passes the batch's `prompts`, `completions` and the per-row `rubric`
column; judges are called over OpenAI-compatible APIs (remote providers or a
local vLLM server).

- `make_rubric_judge_reward`: single judge, hard Likert 1-10 rating,
  reward (rating - 1) / 9. Used per judge for the naive (mean) ensemble.
- `make_ensemble_drop_disagreement_reward`: hard-Likert mean ensemble with
  the drop-on-disagreement mask.
- `make_ensemble_soft_likert_reward`: soft-Likert ensemble (1-5 rating,
  logprob-derived expected rating weighted by 1 / (1 + Var)).
- `make_ensemble_soft_likert_drop_reward`: RaRE (soft-Likert + drop mask).
- `make_pairwise_rubric_reward`: pairwise preference judge (appendix baseline).
- `make_length_penalty_reward`: additive length penalty (ablation).

A failed judge call or unparseable rating scores 0.0 (hard paths) or gets
zero confidence (soft paths).

Optional judge-call logging: if env var ``RAR_JUDGE_LOG_PATH`` is set, every
hard-Likert call appends a JSONL record (prompt_text, response_text,
score_raw, score_norm, timestamp). Logging has no effect on the reward.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any, Awaitable, Callable

from .prompts import (
    JUDGE_JSON_SCHEMA,
    JUDGE_PAIRWISE_JSON_SCHEMA,
    judge_likert_schema,
    judge_likert_messages,
    judge_messages,
    judge_pairwise_messages,
    judge_pairwise_norubric_messages,
)

JSON_OBJ_RE = re.compile(r"\{[^{}]*\"rating\"[^{}]*\}", re.DOTALL)


def _completion_to_text(completion: Any) -> str:
    """Normalize a TRL completion (list[message] or str) into the assistant text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        return str(last)
    return str(completion)


def _prompt_to_question(prompt: Any) -> str:
    """Pull the user question out of a TRL prompt (str or list[message])."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        # Take the last user-role turn.
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "") or ""
    return str(prompt)


def _parse_score(text: str) -> int | None:
    """Extract an integer rating in [1, 10] from a judge completion.

    Paper schema (Appendix A.6) is `{"rating": int}`. We accept `"score"` too
    for backward compatibility with checkpoints that were trained against our
    earlier prompt.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    score = obj.get("rating", obj.get("score"))
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None
    if not 1 <= score <= 10:
        return None
    return score


def make_rubric_judge_reward(
    judge_model: str,
    judge_base_url: str | None = None,
    judge_api_key: str = "EMPTY",
    max_concurrent: int = 64,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 128,
    judge_reasoning_effort: str | None = None,
    response_format: str | None = None,
    name: str = "rubric_judge_reward",
    strict_scores: bool = False,
    audit_log_dir: str | None = None,
    completion_token_parameter: str = "max_tokens",
    extra_body: dict | None = None,
) -> Callable[..., Awaitable[list[float]]]:
    """Build a TRL-compatible async reward function backed by one judge.

    Parameters
    ----------
    judge_model : str
        Model id at the judge endpoint.
    judge_base_url : str
        OpenAI-compatible endpoint, e.g. "https://api.openai.com/v1" or a
        local vLLM server. Falls back to env var ``RAR_JUDGE_URL``.
    judge_api_key : str
        API key; "EMPTY" for a local vLLM server.
    max_concurrent : int
        Cap on parallel judge requests.
    judge_temperature : float
        Greedy by default for stable rewards.
    name : str
        Name reported in the GRPO logs (`reward/<name>/mean`).
    """
    from openai import AsyncOpenAI  # local import: only loaded at training time

    base_url = judge_base_url or os.environ.get("RAR_JUDGE_URL", "http://127.0.0.1:8000/v1")
    # max_retries=8 (default 2): rate-limited judge calls during eval were
    # silently scored as "criterion not met" because the 2-retry budget got
    # exhausted under bursty concurrency. Same risk exists at train time.
    if completion_token_parameter not in ("max_tokens", "max_completion_tokens"):
        raise ValueError("Unsupported completion token parameter")
    client_options = {}
    if audit_log_dir:
        from openai import DefaultAsyncHttpxClient
        from .cost_audit import make_http_cost_hooks
        client_options["http_client"] = DefaultAsyncHttpxClient(
            event_hooks=make_http_cost_hooks(str(audit_log_dir) + "/http", judge_model)
        )
    client = AsyncOpenAI(base_url=base_url, api_key=judge_api_key, max_retries=8, **client_options)
    semaphore = asyncio.Semaphore(max_concurrent)
    # Remote judges (OpenAI direct, Portkey-routed gateways) don't accept
    # vLLM-only args like `extra_body={"chat_template_kwargs": ...}`. OpenAI
    # explicitly returns 400 "Unrecognized request argument". Detect remote
    # via api_key != "EMPTY".
    is_remote_judge = judge_api_key not in (None, "", "EMPTY")

    log_path = os.environ.get("RAR_JUDGE_LOG_PATH")
    log_lock = asyncio.Lock() if log_path else None

    async def _append_log(record: dict) -> None:
        if log_path is None:
            return
        try:
            # Single in-process lock keeps JSONL records from interleaving even
            # when many _score_one tasks finish concurrently. Wrapped in try/
            # except so a logging failure (disk full, permission error, …)
            # never breaks the reward flow.
            async with log_lock:
                with open(log_path, "a", buffering=1) as f:
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception:
            pass

    async def _score_one(question: str, response: str, rubric: Any) -> float:
        messages = judge_messages(question, response, rubric)
        common_kwargs = dict(
            model=judge_model,
            messages=messages,
            temperature=judge_temperature,
        )
        common_kwargs[completion_token_parameter] = judge_max_tokens
        # response_format: default = strict json_schema (works for vLLM,
        # OpenAI direct, Portkey-routed OpenAI). Set "json_object" for
        # DeepSeek direct (rejects json_schema). Set "none" to omit.
        if response_format == "json_object":
            common_kwargs["response_format"] = {"type": "json_object"}
        elif response_format in (None, "json_schema"):
            common_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rating",
                    "schema": JUDGE_JSON_SCHEMA,
                    "strict": True,
                },
            }
        # else "none" / unknown -> omit response_format entirely.
        # OpenAI-Harmony reasoning models (gpt-oss-120b, o1, o3) accept a
        # `reasoning_effort` kwarg. Only attach it when set, since most
        # providers reject unknown kwargs.
        if judge_reasoning_effort is not None:
            common_kwargs["reasoning_effort"] = judge_reasoning_effort
        try:
            async with semaphore:
                request_body = dict(extra_body or {})
                if not is_remote_judge:
                    request_body.setdefault("chat_template_kwargs", {"enable_thinking": False})
                if request_body:
                    resp = await client.chat.completions.create(**common_kwargs, extra_body=request_body)
                else:
                    resp = await client.chat.completions.create(**common_kwargs)
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            if log_path is not None:
                await _append_log({
                    "ts": time.time(),
                    "judge": judge_model,
                    "score_raw": None,
                    "score_norm": 0.0,
                    "question": question,
                    "response": response,
                    "response_hash": hashlib.sha1(response.encode("utf-8")).hexdigest(),
                    "rubric_n_criteria": len(rubric) if isinstance(rubric, list) else None,
                    "error": "judge_call_failed",
                })
            if audit_log_dir:
                from .audit import write_audit
                write_audit(audit_log_dir, {"event": "judge_error", "judge": judge_model,
                            "error_type": type(exc).__name__})
            if strict_scores:
                raise
            return 0.0
        score = _parse_score(text)
        if strict_scores:
            from .audit import validate_hard_response
            score = validate_hard_response(resp, scale_max=10)
        score_norm = 0.0 if score is None else (score - 1) / 9.0
        if audit_log_dir:
            from .audit import write_audit
            write_audit(audit_log_dir, {
                "event": "absolute_reward", "judge": judge_model,
                "score_raw": score, "score_norm": score_norm,
                "question": question, "response": response, "rubric": rubric,
                "request_messages_sha256": hashlib.sha256(
                    json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "judge_response": resp.model_dump(mode="json"),
                "normalization": "(rating-1)/9", "scale_max": 10,
            })
        if log_path is not None:
            await _append_log({
                "ts": time.time(),
                "judge": judge_model,
                "score_raw": score,
                "score_norm": score_norm,
                "question": question,
                "response": response,
                "response_hash": hashlib.sha1(response.encode("utf-8")).hexdigest(),
                "rubric_n_criteria": len(rubric) if isinstance(rubric, list) else None,
            })
        return score_norm

    async def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        rubric: list[Any],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        questions = [_prompt_to_question(p) for p in prompts]
        responses = [_completion_to_text(c) for c in completions]
        rewards = await asyncio.gather(
            *[_score_one(q, r, rb) for q, r, rb in zip(questions, responses, rubric)]
        )
        if log_metric is not None:
            log_metric(f"{name}/mean", sum(rewards) / max(1, len(rewards)))
            log_metric(f"{name}/zero_frac", sum(1 for x in rewards if x == 0.0) / max(1, len(rewards)))
        if log_extra is not None:
            log_extra("judge_score_norm", [float(r) for r in rewards])
        return rewards

    reward_fn.__name__ = name
    return reward_fn


def make_ensemble_drop_disagreement_reward(
    judges: list[dict],
    disagreement_threshold: float,
    num_generations: int,
    default_api_key_env: str | None = None,
    name: str = "ensemble_drop_reward",
    strict_scores: bool = False,
    audit_log_dir: str | None = None,
) -> Callable[..., Awaitable[list[float]]]:
    """Multi-judge ensemble with high-disagreement rollout masking.

    For each rollout: call every judge in `judges`, compute mean score, and
    compute disagreement = max(scores) - min(scores). If disagreement >
    `disagreement_threshold`, the rollout is "masked" — its returned reward
    is set to the *unmasked group mean* (where group = num_generations
    consecutive rollouts of the same prompt). This trick makes the
    GRPO advantage exactly zero for masked rollouts (group_mean of returned
    values equals the unmasked mean, so masked rollouts contribute no
    gradient) while leaving unmasked rollouts' advantages intact.

    Logs per step: <name>/mean (final reward), <name>/drop_frac (% masked),
    <name>/mean_disagreement, and <name>/<tag>/mean for each judge.

    Args
    ----
    judges : list of dict
        Each: {tag, model, url, api_key_env, max_concurrent?, max_tokens?,
               reasoning_effort?}
    disagreement_threshold : float
        Drop rollouts where max(judge_scores) - min(judge_scores) > this.
    num_generations : int
        Must match GRPOConfig.num_generations — used to chunk the flat batch
        into groups for the masking step.
    """
    from openai import AsyncOpenAI

    if not judges or num_generations <= 0 or not 0 <= disagreement_threshold <= 1:
        raise ValueError("Invalid hard-ensemble configuration")
    if strict_scores and not audit_log_dir:
        raise ValueError("Strict hard ensemble requires audit_log_dir")
    from .audit import validate_hard_response, write_audit
    reward_call = 0
    judge_specs = []
    for j in judges:
        key_env = j.get("api_key_env") or default_api_key_env
        api_key = os.environ.get(key_env, "") if key_env else "EMPTY"
        if key_env and not api_key:
            raise SystemExit(f"[ensemble_drop] judge[{j.get('tag','?')}].api_key_env={key_env} empty")
        client_options = {}
        if audit_log_dir:
            from openai import DefaultAsyncHttpxClient
            from .cost_audit import make_http_cost_hooks
            client_options["http_client"] = DefaultAsyncHttpxClient(
                event_hooks=make_http_cost_hooks(str(audit_log_dir) + "/http", j["model"]))
        client = AsyncOpenAI(base_url=j["url"], api_key=api_key, max_retries=8, **client_options)
        sem = asyncio.Semaphore(j.get("max_concurrent", 8))
        is_remote = api_key not in (None, "", "EMPTY")
        judge_specs.append({
            "tag": j.get("tag", j["model"][:8]),
            "model": j["model"],
            "client": client,
            "sem": sem,
            "max_tokens": j.get("max_tokens", 128),
            "reasoning_effort": j.get("reasoning_effort"),
            "response_format": j.get("response_format"),
            "is_remote": is_remote,
            "extra_body": j.get("extra_body"),
        })

    async def _score_one_judge(spec, question, response, rubric):
        messages = judge_messages(question, response, rubric)
        common_kwargs = dict(
            model=spec["model"],
            messages=messages,
            temperature=0.0,
            max_tokens=spec["max_tokens"],
        )
        # response_format: per-judge override. Default = json_schema (works for
        # vLLM, OpenAI direct, Portkey-routed OpenAI). Set "json_object" for
        # DeepSeek direct (rejects json_schema). Set "none" to omit.
        rf = spec.get("response_format")
        if rf == "json_object":
            common_kwargs["response_format"] = {"type": "json_object"}
        elif rf in (None, "json_schema"):
            common_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rating", "schema": JUDGE_JSON_SCHEMA, "strict": True},
            }
        # else "none" / unknown -> omit response_format entirely.
        if spec["reasoning_effort"]:
            common_kwargs["reasoning_effort"] = spec["reasoning_effort"]
        if spec["extra_body"]:
            common_kwargs["extra_body"] = spec["extra_body"]
        try:
            async with spec["sem"]:
                if spec["is_remote"]:
                    resp = await spec["client"].chat.completions.create(**common_kwargs)
                else:
                    resp = await spec["client"].chat.completions.create(
                        **common_kwargs,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            if audit_log_dir:
                write_audit(audit_log_dir, {"event": "judge_error", "judge": spec["tag"], "error_type": type(exc).__name__})
            if strict_scores:
                raise
            return 0.0, {"error_type": type(exc).__name__}
        score = validate_hard_response(resp, scale_max=10) if strict_scores else _parse_score(text)
        metadata = {"rating": score, "logprobs_requested": False}
        if audit_log_dir:
            metadata.update(response=resp.model_dump(mode="json"),
                messages_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
        return (0.0 if score is None else (score - 1) / 9.0), metadata

    async def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        rubric: list[Any],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        nonlocal reward_call
        reward_call += 1
        started = time.perf_counter()
        questions = [_prompt_to_question(p) for p in prompts]
        responses = [_completion_to_text(c) for c in completions]
        N = len(responses)
        # The mask substitutes the group mean, so each reward call must hold
        # whole GRPO groups (per_device_train_batch_size a multiple of
        # num_generations); otherwise the mask would silently do nothing.
        if N % num_generations:
            raise ValueError(
                f"Drop-on-disagreement needs whole GRPO groups per reward call: got {N} "
                f"rollouts with num_generations={num_generations}. Set "
                "per_device_train_batch_size to a multiple of num_generations.")
        if strict_scores:
            if not (N == len(questions) == len(rubric)):
                raise ValueError("Hard reward batch must contain complete GRPO groups")
            for first in range(0, N, num_generations):
                if len(set(questions[first:first + num_generations])) != 1:
                    raise ValueError("Hard reward group contains different questions")
        # Judge batches are sequential; requests within each batch are concurrent.
        per_judge_scores: list[list[float]] = []
        per_judge_metadata = []
        for spec in judge_specs:
            scores = await asyncio.gather(*[
                _score_one_judge(spec, q, r, rb) for q, r, rb in zip(questions, responses, rubric)
            ])
            per_judge_scores.append([s[0] for s in scores])
            per_judge_metadata.append([s[1] for s in scores])

        # Per-rollout: mean and disagreement (max - min across judges).
        means = [
            sum(per_judge_scores[j][i] for j in range(len(judge_specs))) / len(judge_specs)
            for i in range(N)
        ]
        disagreements = [
            max(per_judge_scores[j][i] for j in range(len(judge_specs)))
            - min(per_judge_scores[j][i] for j in range(len(judge_specs)))
            for i in range(N)
        ]
        masks = [d > disagreement_threshold for d in disagreements]

        # Group-wise: replace masked rewards with unmasked group mean (gives
        # advantage = 0 for masked rollouts in TRL's GRPO update).
        rewards = list(means)
        n_groups = N // num_generations
        for g in range(n_groups):
            s = g * num_generations
            e = s + num_generations
            unmasked = [means[i] for i in range(s, e) if not masks[i]]
            grp_unmasked_mean = sum(unmasked) / len(unmasked) if unmasked else 0.0
            for i in range(s, e):
                if masks[i]:
                    rewards[i] = grp_unmasked_mean

        if audit_log_dir:
            state = kwargs.get("trainer_state")
            for i in range(N):
                write_audit(audit_log_dir, {
                    "event": "hard_ensemble_rollout", "run": name, "reward_call": reward_call,
                    "optimizer_step": getattr(state, "global_step", None),
                    "group": i // num_generations, "within_group": i % num_generations,
                    "question": questions[i], "completion": responses[i], "rubric": rubric[i],
                    "judges": {spec["tag"]: {**per_judge_metadata[j][i], "normalized_score": per_judge_scores[j][i]}
                               for j, spec in enumerate(judge_specs)},
                    "scale_max": 10, "normalization": "(rating-1)/9", "aggregation": "mean",
                    "raw_reward": means[i], "reward": rewards[i], "disagreement": disagreements[i],
                    "threshold": disagreement_threshold, "masked": masks[i],
                    "batch_seconds": time.perf_counter() - started,
                })

        if log_metric is not None:
            log_metric(f"{name}/mean", sum(rewards) / max(1, N))
            log_metric(f"{name}/drop_frac", sum(masks) / max(1, N))
            log_metric(f"{name}/mean_disagreement", sum(disagreements) / max(1, N))
            for j_idx, spec in enumerate(judge_specs):
                log_metric(
                    f"{name}/{spec['tag']}_mean",
                    sum(per_judge_scores[j_idx]) / max(1, N),
                )
        return rewards

    reward_fn.__name__ = name
    return reward_fn


# ---------------------------------------------------------------------------
# Soft-Likert 1-5 ensemble with confidence-weighted combination
# ---------------------------------------------------------------------------
#
# Each judge is asked for {"rating": 1..5}. We then read top_logprobs at the
# rating digit position to recover the judge's full distribution over digits
# 1-5. From that we compute:
#   - expected_rating ∈ [1, 5]    = Σ k · p_k   (continuous, replaces hard int)
#   - confidence    ∈ (0, 1]      = 1 / (1 + Var)   (default, "inverse_var")
#     or max_k p_k ("top_logit", confidence-ablation variant)
#
# Ensemble combination across judges: confidence-weighted average of
# expected ratings, so the more certain a judge is, the more its rating
# contributes. If every judge has zero confidence (all calls failed), the
# reward is 0.
#
# Reward = ensemble_expected / 5  ∈ [0.2, 1.0]
#
# Fallback path: if a judge's response has no logprobs (some providers
# strip them) or no recognizable rating digit, we hard-parse the integer
# and synthesize a one-hot distribution at that integer (confidence=1.0
# under the hard parse). The "[soft_likert] fallback" log line surfaces
# the per-judge fallback rate so we know if a provider is silently
# downgrading us.
# ---------------------------------------------------------------------------


def _extract_soft_likert(
    resp,
    scale_max: int = 5,
    confidence_mode: str = "inverse_var",
) -> tuple[float, float, bool]:
    """Read top_logprobs at the rating-digit position; return
    (expected_rating, confidence, used_fallback).

    Confidence is one of:
      - "inverse_var" (default): 1 / (1 + Var[rating]) — penalizes both
        local spread and far disagreement.
      - "top_logit": max_k p_k — peak mass on the modal rating. More
        permissive; doesn't penalize spread shape.

    The rating-digit token is the LAST single-digit token in the response
    (strict JSON schema guarantees only one such token: the integer inside
    ``{"rating": N}``).

    If logprobs are missing or no digit token is found, we hard-parse the
    integer from the response text and return a one-hot distribution at
    that integer (Var=0 → confidence=1.0). Returns (0.0, 0.0, True) only
    if even the hard parse fails.

    For scale_max > 9 the single-digit-token assumption no longer holds
    (multi-digit numbers tokenize differently); supported range is 2..9.
    """
    import math

    # Soft logprob path requires single-digit tokens (assumes 2..9).
    # For scale_max>=10 the single-digit assumption no longer holds, so the
    # logprob walk would mis-parse "10" as "0" (last single digit). Callers
    # using scale_max>=10 must NOT rely on the soft path — Portkey/no-logprob
    # paths automatically skip soft and use the hard JSON-int parser, which
    # supports any scale_max.
    assert 2 <= scale_max, f"_extract_soft_likert: scale_max must be >=2, got {scale_max}"
    use_soft = (scale_max <= 9)
    digits = {str(k) for k in range(1, min(scale_max, 9) + 1)}

    text = (resp.choices[0].message.content or "").strip()
    logprobs_obj = getattr(resp.choices[0], "logprobs", None)
    token_entries = getattr(logprobs_obj, "content", None) if logprobs_obj else None

    # 1) Try logprob path (only if scale fits single-digit tokens).
    if use_soft and token_entries:
        # Harmony-format models (e.g. gpt-oss-120b) split
        # their output into channels: <|channel|>analysis<|message|>...reason...
        # <|channel|>final<|message|>{"rating": N}. The analysis channel may
        # contain incidental digit tokens (e.g. quoting "<1-5>" from the
        # prompt), so restrict the digit search to tokens AFTER the final
        # channel marker. We detect harmony by the presence of any
        # `<|channel|>` token; if absent, fall through to scanning all tokens.
        scan_entries = token_entries
        last_final_idx = -1
        for i, entry in enumerate(token_entries):
            tok = entry.token or ""
            if tok == "<|channel|>" and i + 1 < len(token_entries):
                ch = (token_entries[i + 1].token or "").strip()
                if ch == "final":
                    last_final_idx = i
        if last_final_idx >= 0:
            scan_entries = token_entries[last_final_idx:]

        rating_entry = None
        for entry in scan_entries:
            tok = (entry.token or "").strip()
            if tok in digits:
                rating_entry = entry
        if rating_entry is not None:
            mass = {k: 0.0 for k in range(1, scale_max + 1)}
            for alt in (rating_entry.top_logprobs or []):
                tok = (alt.token or "").strip()
                if tok in digits:
                    mass[int(tok)] += math.exp(alt.logprob)
            total = sum(mass.values())
            if total > 1e-9:
                p = {k: mass[k] / total for k in mass}
                expected = sum(k * p[k] for k in p)
                if confidence_mode == "top_logit":
                    confidence = max(p.values())
                else:
                    variance = sum((k - expected) ** 2 * p[k] for k in p)
                    confidence = 1.0 / (1.0 + variance)
                return expected, confidence, False

    # 2) Hard-parse fallback: read the integer from the response JSON.
    text_stripped = text.strip()
    if text_stripped.startswith("```"):
        text_stripped = re.sub(r"^```[a-zA-Z]*\s*", "", text_stripped)
        text_stripped = re.sub(r"\s*```$", "", text_stripped)
    try:
        obj = json.loads(text_stripped)
    except json.JSONDecodeError:
        m = JSON_OBJ_RE.search(text_stripped)
        if not m:
            return 0.0, 0.0, True
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return 0.0, 0.0, True
    if not isinstance(obj, dict):
        return 0.0, 0.0, True
    raw = obj.get("rating", obj.get("score"))
    try:
        rating = int(raw)
    except (TypeError, ValueError):
        return 0.0, 0.0, True
    if not 1 <= rating <= scale_max:
        return 0.0, 0.0, True
    return float(rating), 1.0, True


def make_ensemble_soft_likert_reward(
    judges: list[dict],
    default_api_key_env: str | None = None,
    top_logprobs: int = 10,
    name: str = "ensemble_soft_likert",
    scale_max: int = 5,
) -> Callable[..., Awaitable[list[float]]]:
    """Multi-judge soft-Likert ensemble with confidence-weighted average.

    Each judge call requests `logprobs=True, top_logprobs=N` and the reward
    fn extracts (expected_rating, confidence) per judge from the rating-
    digit's distribution. The ensemble reward is the confidence-weighted
    mean of expected ratings, normalized to [0, 1] by dividing by scale_max.

    Args
    ----
    judges : list of dict
        Each: {tag, model, url, api_key_env, max_concurrent?, max_tokens?,
               reasoning_effort?}
    default_api_key_env : str | None
        Fallback env-var name for judges without an explicit ``api_key_env``.
    top_logprobs : int
        How many alternatives to request per token; >=scale_max recommended.
    name : str
        Used as the metric-name prefix in GRPO logs.
    scale_max : int
        Likert scale max (default 5; supported 2..9). Single-digit token
        assumption: scale_max>9 is not supported.
    """
    assert scale_max >= 2, f"scale_max must be >=2, got {scale_max}"
    likert_schema = judge_likert_schema(scale_max)
    from openai import AsyncOpenAI

    judge_specs = []
    for j in judges:
        key_env = j.get("api_key_env") or default_api_key_env
        api_key = os.environ.get(key_env, "") if key_env else "EMPTY"
        if key_env and not api_key:
            raise SystemExit(f"[soft_likert] judge[{j.get('tag','?')}].api_key_env={key_env} empty")
        client = AsyncOpenAI(base_url=j["url"], api_key=api_key, max_retries=8)
        sem = asyncio.Semaphore(j.get("max_concurrent", 8))
        is_remote = api_key not in (None, "", "EMPTY")
        judge_specs.append({
            "tag": j.get("tag", j["model"][:8]),
            "model": j["model"],
            "client": client,
            "sem": sem,
            "max_tokens": j.get("max_tokens", 32),
            "reasoning_effort": j.get("reasoning_effort"),
            "is_remote": is_remote,
            # extra_body: pass through arbitrary provider-specific options
            # (e.g. OpenRouter's provider.only pinning).
            "extra_body": j.get("extra_body") or {},
            # response_format: per-judge override. Default = json_schema
            # (works for OpenAI, vLLM). Set to "json_object" for DeepSeek
            # direct (rejects json_schema). Set to "none" to omit entirely.
            "response_format": j.get("response_format"),
        })

    async def _score_one_judge(spec, question, response, rubric):
        messages = judge_likert_messages(question, response, rubric, scale_max=scale_max)
        common_kwargs = dict(
            model=spec["model"],
            messages=messages,
            temperature=0.0,
            max_tokens=spec["max_tokens"],
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        # response_format: default = strict json_schema. Per-judge override
        # via spec["response_format"]: "json_object" (DS-compatible), "none".
        rf_override = spec.get("response_format")
        if rf_override == "json_object":
            common_kwargs["response_format"] = {"type": "json_object"}
        elif rf_override in (None, "json_schema"):
            common_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rating", "schema": likert_schema, "strict": True},
            }
        # else "none" or unknown → omit response_format entirely.
        if spec["reasoning_effort"]:
            common_kwargs["reasoning_effort"] = spec["reasoning_effort"]
        # Build extra_body: judge-config-specified options + vllm-only opts
        # for local judges. Remote judges only get judge-config-specified ones.
        eb = dict(spec["extra_body"])
        if not spec["is_remote"]:
            eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        try:
            async with spec["sem"]:
                if eb:
                    resp = await spec["client"].chat.completions.create(
                        **common_kwargs, extra_body=eb)
                else:
                    resp = await spec["client"].chat.completions.create(**common_kwargs)
        except Exception:
            return 0.0, 0.0, True  # call failed; confidence=0 → contributes nothing
        return _extract_soft_likert(resp, scale_max=scale_max)

    async def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        rubric: list[Any],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        questions = [_prompt_to_question(p) for p in prompts]
        responses = [_completion_to_text(c) for c in completions]
        N = len(responses)

        # Score with all judges in parallel.
        per_judge_ratings: list[list[float]] = []
        per_judge_confs: list[list[float]] = []
        per_judge_fallback: list[int] = []
        for spec in judge_specs:
            results = await asyncio.gather(*[
                _score_one_judge(spec, q, r, rb) for q, r, rb in zip(questions, responses, rubric)
            ])
            ratings = [t[0] for t in results]
            confs = [t[1] for t in results]
            fallback_n = sum(1 for t in results if t[2])
            per_judge_ratings.append(ratings)
            per_judge_confs.append(confs)
            per_judge_fallback.append(fallback_n)

        # Confidence-weighted ensemble per rollout.
        rewards = [0.0] * N
        for i in range(N):
            num = 0.0
            den = 0.0
            for jx in range(len(judge_specs)):
                r_ij = per_judge_ratings[jx][i]
                c_ij = per_judge_confs[jx][i]
                num += r_ij * c_ij
                den += c_ij
            if den > 1e-9:
                rewards[i] = (num / den) / float(scale_max)
            else:
                # All judges failed for this rollout; reward = 0 (no signal).
                rewards[i] = 0.0

        if log_metric is not None:
            log_metric(f"{name}/mean", sum(rewards) / max(1, N))
            log_metric(f"{name}/zero_frac", sum(1 for x in rewards if x == 0.0) / max(1, N))
            for jx, spec in enumerate(judge_specs):
                tag = spec["tag"]
                ratings = per_judge_ratings[jx]
                confs = per_judge_confs[jx]
                log_metric(f"{name}/{tag}_rating_mean", sum(ratings) / max(1, N))
                log_metric(f"{name}/{tag}_conf_mean", sum(confs) / max(1, N))
                log_metric(f"{name}/{tag}_fallback_frac", per_judge_fallback[jx] / max(1, N))
        return rewards

    reward_fn.__name__ = name
    return reward_fn


# ---------------------------------------------------------------------------
# Soft-Likert ensemble with disagreement-DROP (drop030-style)
# ---------------------------------------------------------------------------
#
# Combines two orthogonal mechanisms:
#   1. Soft-Likert + inverse-variance confidence weighting (per judge):
#        reward_continuous_i = Σ_j (E[rating_j] · conf_j) / Σ_j conf_j
#      where conf_j = 1 / (1 + Var_j[rating]) ∈ (0, 1].
#   2. Drop-disagreement masking (drop030-style):
#        If |E[rating_gem] - E[rating_ds]| / 4 > disagreement_threshold,
#        mask the rollout (reward = group unmasked mean → advantage = 0).
#
# The two signals are independent — confidence weighting handles each
# judge's own uncertainty, drop-disagreement handles inter-judge
# disagreement. drop030 alone achieved 0.330 with hard binary scoring;
# this composition hopes to additively lift via the soft-Likert smoothing
# on the 95% non-masked rollouts.
#
# Normalization: judges output E[rating] ∈ [1, 5]. We compute disagreement
# as |E_gem - E_ds| / 4 ∈ [0, 1], directly comparable to drop030's
# normalized scale where 0.30 was the proven threshold.
# ---------------------------------------------------------------------------


# Buffer read by the retro-GRPO-aux trainer (scripts/likert_retro_train.py):
# when `populate_retro_pending=True`, masked rollouts are recorded here keyed by
# sha1(response text) so the trainer can match them in its batch. The trainer
# pops entries after harvesting them.
PENDING_LIKERT_DISAGREEMENTS: dict[str, dict[str, Any]] = {}


def make_ensemble_soft_likert_drop_reward(
    judges: list[dict],
    disagreement_threshold: float,
    num_generations: int,
    default_api_key_env: str | None = None,
    top_logprobs: int = 5,
    name: str = "ensemble_soft_likert_drop",
    populate_retro_pending: bool = False,
    scale_max: int = 5,
    confidence_mode: str = "inverse_var",
    aggregation: str = "weighted_avg",
    normalize_disagreement: bool = False,
    reward_judge_tag: str | None = None,
    strict_scores: bool = False,
    audit_log_dir: str | None = None,
    rating_mode: str = "soft",
    disagreement_rating_mode: str = "soft",
) -> Callable[..., Awaitable[list[float]]]:
    """Soft-Likert + inverse-variance confidence + drop-disagreement mask.

    See module-level docstring for the composition rationale.

    Args
    ----
    judges : list of dict
        Same shape as ``make_ensemble_soft_likert_reward``.
    disagreement_threshold : float
        Disagreement = |E_judge_max - E_judge_min| / 4 on [0, 1] scale.
        Rollouts above this get masked (advantage = 0). 0.30 matches drop030.
    num_generations : int
        Must equal GRPOConfig.num_generations for correct group chunking.
    default_api_key_env : str | None
        Fallback env-var name for judges without an explicit ``api_key_env``.
    top_logprobs : int
        ≥5 covers Likert digits 1-5; Together caps at 5.
    name : str
        Metric-name prefix in GRPO logs.
    """
    if rating_mode not in {"soft", "hard"}:
        raise ValueError(f"Unknown rating mode: {rating_mode}")
    if rating_mode == "hard" and (not strict_scores or (len(judges) != 1 and reward_judge_tag is None)):
        raise ValueError("Hard-rating control requires strict scoring and an explicit reward judge for a panel")
    if disagreement_rating_mode not in {"soft", "hard"}:
        raise ValueError("Unknown disagreement rating mode")
    needs_soft = rating_mode == "soft" or disagreement_rating_mode == "soft"
    assert scale_max >= 2, f"scale_max must be >=2, got {scale_max}"
    if aggregation not in {"weighted_avg", "mean", "winner_takes_all"}:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    tags = [j.get("tag", j["model"][:8]) for j in judges]
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("Judge tags must be nonempty and unique")
    if reward_judge_tag is not None and reward_judge_tag not in tags:
        raise ValueError(f"Unknown reward judge: {reward_judge_tag}")
    if strict_scores and not audit_log_dir:
        raise ValueError("Strict scoring requires an audit_log_dir")
    # Per-judge hard ratings inside a soft ensemble: the judge's integer
    # rating is used as a one-hot distribution (expected = rating,
    # confidence = 1), which is how a temperature-0 DeepSeek-direct judge
    # behaved in the original science runs. Only valid for the
    # confidence-weighted/mean ensemble reward.
    hard_judge_tags = {j.get("tag", j["model"][:8]) for j in judges if j.get("rating_mode") == "hard"}
    if any(j.get("rating_mode") not in (None, "soft", "hard") for j in judges):
        raise ValueError("Per-judge rating_mode must be 'soft' or 'hard'")
    if hard_judge_tags and not (strict_scores and rating_mode == "soft" and reward_judge_tag is None
                                and aggregation in {"weighted_avg", "mean"}):
        raise ValueError("Per-judge hard ratings require strict scoring and a soft weighted/mean ensemble")
    from .audit import validate_hard_response, validate_soft_response, write_audit
    reward_call = 0
    likert_schema = judge_likert_schema(scale_max)
    from openai import AsyncOpenAI

    judge_specs = []
    for j in judges:
        key_env = j.get("api_key_env") or default_api_key_env
        api_key = os.environ.get(key_env, "") if key_env else "EMPTY"
        if key_env and not api_key:
            raise SystemExit(
                f"[soft_likert_drop] judge[{j.get('tag','?')}].api_key_env={key_env} empty"
            )
        http_kwargs = {}
        if audit_log_dir:
            import httpx
            from .cost_audit import make_http_cost_hooks
            http_kwargs["http_client"] = httpx.AsyncClient(event_hooks=make_http_cost_hooks(
                str(audit_log_dir) + "/http_costs", j.get("tag", j["model"][:8])))
        sdk_max_retries = int(j.get("sdk_max_retries", 2 if strict_scores else 8))
        if not 0 <= sdk_max_retries <= 20:
            raise ValueError("sdk_max_retries must be between 0 and 20")
        client = AsyncOpenAI(base_url=j["url"], api_key=api_key,
                            max_retries=sdk_max_retries, timeout=90.0, **http_kwargs)
        sem = asyncio.Semaphore(j.get("max_concurrent", 8))
        is_remote = api_key not in (None, "", "EMPTY")
        judge_specs.append({
            "tag": j.get("tag", j["model"][:8]),
            "sdk_max_retries": sdk_max_retries,
            # A judge used only for a hard disagreement mask needs no logprobs.
            "needs_soft": j.get("rating_mode") != "hard" and (disagreement_rating_mode == "soft" or (
                rating_mode == "soft" and (reward_judge_tag is None or
                j.get("tag", j["model"][:8]) == reward_judge_tag))),
            "judge_hard": j.get("rating_mode") == "hard",
            "model": j["model"],
            "client": client,
            "sem": sem,
            "max_tokens": j.get("max_tokens", 32),
            "reasoning_effort": j.get("reasoning_effort"),
            "is_remote": is_remote,
            # extra_body: pass through provider-specific options (e.g.
            # OpenRouter's provider.only pinning).
            "extra_body": j.get("extra_body") or {},
            # response_format: per-judge override. Default = json_schema
            # (works for OpenAI, vLLM). Set "json_object" for DeepSeek
            # direct (rejects json_schema). Set "none" to omit entirely.
            "response_format": j.get("response_format"),
            # Per-judge temperature override. Default 0.0 (greedy). Set
            # > 0 for DeepSeek-direct to bypass the -9999 logprob sentinel
            # (DS-direct only returns real logprob distributions at
            # temperature >= ~0.7).
            "temperature": j.get("temperature", 0.0),
        })

    async def _score_one_judge(spec, question, response, rubric):
        needs_soft = spec["needs_soft"]
        messages = judge_likert_messages(question, response, rubric, scale_max=scale_max)
        common_kwargs = dict(
            model=spec["model"],
            messages=messages,
            temperature=spec["temperature"],
            max_tokens=spec["max_tokens"],
        )
        if needs_soft:
            common_kwargs.update(logprobs=True, top_logprobs=top_logprobs)
        rf_override = spec.get("response_format")
        if rf_override == "json_object":
            common_kwargs["response_format"] = {"type": "json_object"}
        elif rf_override in (None, "json_schema"):
            common_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rating", "schema": likert_schema, "strict": True},
            }
        # else "none" / unknown -> omit response_format entirely.
        if spec["reasoning_effort"]:
            common_kwargs["reasoning_effort"] = spec["reasoning_effort"]
        eb = dict(spec["extra_body"])
        if not spec["is_remote"]:
            eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        started = time.perf_counter()
        resp = None
        try:
            async with spec["sem"]:
                if eb:
                    resp = await spec["client"].chat.completions.create(
                        **common_kwargs, extra_body=eb)
                else:
                    resp = await spec["client"].chat.completions.create(**common_kwargs)
            if needs_soft:
                distribution_metadata = validate_soft_response(resp, scale_max) if strict_scores else {}
                expected, confidence, fallback = _extract_soft_likert(
                    resp, scale_max=scale_max, confidence_mode=confidence_mode)
                if strict_scores and fallback:
                    raise ValueError("Soft score used a hard-score fallback")
            else:
                validate_hard_response(resp, scale_max)
                distribution_metadata = {}
                expected, confidence, fallback = None, None, False
            metadata = {**distribution_metadata, "requested_model": spec["model"], "response": resp.model_dump(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "sdk_max_retries": spec["sdk_max_retries"]}
            # Identical API requests isolate the reward transformation.
            hard_rating = json.loads(resp.choices[0].message.content)["rating"] if strict_scores else None
            if spec["judge_hard"]:
                # One-hot distribution at the emitted rating.
                expected, confidence = float(hard_rating), 1.0
            metadata.update(hard_rating=hard_rating, soft_expected=expected, rating_mode=rating_mode,
                            judge_rating_mode="hard" if spec["judge_hard"] else "soft",
                            logprobs_requested=needs_soft)
            score = expected if spec["judge_hard"] else (
                float(hard_rating) if rating_mode == "hard" or not needs_soft else expected)
            return score, confidence, fallback, metadata
        except Exception as exc:
            # Persist failures before aborting; never include authentication headers.
            error = {"error_type": type(exc).__name__,
                     "status_code": getattr(exc, "status_code", None),
                     "requested_model": spec["model"],
                     "elapsed_seconds": time.perf_counter() - started}
            if resp is not None:
                error["response"] = resp.model_dump()  # Preserve paid usage even when rating validation fails.
            if isinstance(exc, ValueError):
                error["validation_error"] = str(exc)
            return 0.0, 0.0, True, error

    async def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        rubric: list[Any],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        questions = [_prompt_to_question(p) for p in prompts]
        responses = [_completion_to_text(c) for c in completions]
        N = len(responses)
        nonlocal reward_call
        reward_call += 1
        batch_started = time.perf_counter()
        if num_generations <= 0 or N % num_generations or not (N == len(questions) == len(rubric)):
            raise ValueError("Reward batch lengths/group size mismatch")
        for first in range(0, N, num_generations):
            if len(set(questions[first:first + num_generations])) != 1:
                raise ValueError("A reward group contains different questions")
        per_judge_metadata = []

        # Score judge batches sequentially; requests within each batch are concurrent.
        per_judge_ratings: list[list[float]] = []
        per_judge_confs: list[list[float]] = []
        per_judge_fallback: list[int] = []
        for spec in judge_specs:
            results = await asyncio.gather(*[
                _score_one_judge(spec, q, r, rb) for q, r, rb in zip(questions, responses, rubric)
            ])
            per_judge_metadata.append([t[3] for t in results])
            if strict_scores and any(t[2] for t in results):
                write_audit(audit_log_dir, {"event": "invalid_batch", "run": name,
                    "reward_call": reward_call, "judge": spec["tag"],
                    "fallback_fraction": sum(t[2] for t in results) / max(1, N),
                    "results": [t[3] for t in results]})
                raise RuntimeError(f"{spec['tag']}: invalid soft scores; audit saved, training stopped")
            ratings = [t[0] for t in results]
            confs = [t[1] for t in results]
            fallback_n = sum(1 for t in results if t[2])
            per_judge_ratings.append(ratings)
            per_judge_confs.append(confs)
            per_judge_fallback.append(fallback_n)

        # Per-rollout: confidence-weighted soft-Likert reward (in [1,5]),
        # AND normalized inter-judge disagreement (in [0,1] for threshold comparability).
        n_judges = len(judge_specs)
        soft_rewards = [0.0] * N    # confidence-weighted E[rating], in [1, 5]
        disagreements = [0.0] * N   # |max E - min E| / 4, in [0, 1]
        # Preserve the soft-score filter when changing only the selected reward to hard.
        # A failed judge call has no rating metadata; fall back to its returned
        # score (0.0, with confidence 0) as the pre-audit code did.
        disagreement_ratings = per_judge_ratings if disagreement_rating_mode == "hard" and rating_mode == "hard" else [
            [metadata.get("hard_rating" if disagreement_rating_mode == "hard" else "soft_expected", rating)
             for metadata, rating in zip(records, ratings)]
            for records, ratings in zip(per_judge_metadata, per_judge_ratings)
        ]
        # Per-judge batch means (used only if normalize_disagreement=True).
        # Centers each judge's rating distribution before computing spread,
        # so a judge that grades systematically low (e.g. gpt-oss-120b) no
        # longer inflates disagreement on every rollout.
        if normalize_disagreement:
            judge_batch_means = [
                (sum(disagreement_ratings[jx]) / N) if N > 0 else 0.0
                for jx in range(n_judges)
            ]
        else:
            judge_batch_means = [0.0] * n_judges
        for i in range(N):
            ratings_i = [per_judge_ratings[jx][i] for jx in range(n_judges)]
            confs_i = [per_judge_confs[jx][i] for jx in range(n_judges)]
            if reward_judge_tag is not None:
                soft_rewards[i] = ratings_i[tags.index(reward_judge_tag)]
            elif not needs_soft:
                assert n_judges == 1
                soft_rewards[i] = ratings_i[0]
            elif aggregation == "mean":
                soft_rewards[i] = sum(ratings_i) / n_judges
            elif aggregation == "winner_takes_all":
                # Pick the rating from the more-confident judge; ignore the rest.
                if max(confs_i) > 1e-9:
                    soft_rewards[i] = ratings_i[max(range(n_judges), key=lambda jx: confs_i[jx])]
                else:
                    soft_rewards[i] = 0.0
            else:
                # weighted_avg (default): confidence-weighted average of E[rating].
                num = sum(r * c for r, c in zip(ratings_i, confs_i))
                den = sum(confs_i)
                if den > 1e-9:
                    soft_rewards[i] = num / den
                else:
                    soft_rewards[i] = 0.0
            # Disagreement = spread of E[rating] across judges, normalized.
            # /(scale_max-1) because E[rating] ∈ [1, scale_max] → spread ∈ [0, scale_max-1].
            centered = [disagreement_ratings[jx][i] - judge_batch_means[jx] for jx in range(n_judges)]
            disagreements[i] = (max(centered) - min(centered)) / float(scale_max - 1)

        # Normalize to [0, 1] reward space (E/scale_max; soft-Likert convention).
        rewards = [r / float(scale_max) for r in soft_rewards]

        # Apply drop-disagreement mask per GRPO group: high-disagreement
        # rollouts get their reward replaced with the unmasked group mean,
        # giving advantage = 0 for those rollouts in TRL's GRPO update.
        masks = [d > disagreement_threshold for d in disagreements]
        n_groups = N // num_generations if num_generations > 0 else 0
        for g in range(n_groups):
            s = g * num_generations
            e = s + num_generations
            unmasked = [rewards[i] for i in range(s, e) if not masks[i]]
            grp_unmasked_mean = sum(unmasked) / len(unmasked) if unmasked else 0.0
            for i in range(s, e):
                if masks[i]:
                    rewards[i] = grp_unmasked_mean

        if audit_log_dir:
            state = kwargs.get("trainer_state")
            for i in range(N):
                write_audit(audit_log_dir, {
                    "event": "rollout", "run": name, "reward_call": reward_call,
                    "optimizer_step": getattr(state, "global_step", None),
                    "group": i // num_generations, "within_group": i % num_generations,
                    "example_id": hashlib.sha256(questions[i].encode()).hexdigest(),
                    "question": questions[i], "rubric": rubric[i], "completion": responses[i],
                    "judges": {spec["tag"]: {"expected": per_judge_metadata[j][i].get("soft_expected", per_judge_ratings[j][i]),
                        "score": per_judge_ratings[j][i],
                        "confidence": per_judge_confs[j][i],
                        "fallback": False if strict_scores else None,
                        **per_judge_metadata[j][i]} for j, spec in enumerate(judge_specs)},
                    "reward_judge_tag": reward_judge_tag, "aggregation": aggregation, "rating_mode": rating_mode,
                    "disagreement_rating_mode": disagreement_rating_mode,
                    "raw_reward": soft_rewards[i] / scale_max, "reward": rewards[i],
                    "disagreement": disagreements[i], "masked": masks[i],
                    "batch_elapsed_seconds": time.perf_counter() - batch_started})

        # Optionally record masked rollouts for the retro-GRPO-aux trainer.
        # Keys are sha1(response_text) so the trainer can match completion ids
        # in its batch view.
        if populate_retro_pending and n_groups > 0:
            for g in range(n_groups):
                s = g * num_generations
                e = s + num_generations
                sibling_hashes = [hashlib.sha1(responses[i].encode("utf-8")).hexdigest()
                                  for i in range(s, e)]
                for i in range(s, e):
                    if not masks[i]:
                        continue
                    PENDING_LIKERT_DISAGREEMENTS[sibling_hashes[i - s]] = {
                        "question": questions[i],
                        "response": responses[i],
                        "per_judge_E": [per_judge_ratings[jx][i] for jx in range(n_judges)],
                        "per_judge_conf": [per_judge_confs[jx][i] for jx in range(n_judges)],
                        "judge_tags": [spec["tag"] for spec in judge_specs],
                        "disagreement": disagreements[i],
                        "sibling_hashes": sibling_hashes,
                        "own_idx_in_group": i - s,
                    }

        if log_metric is not None:
            log_metric(f"{name}/mean", sum(rewards) / max(1, N))
            log_metric(f"{name}/drop_frac", sum(masks) / max(1, N))
            log_metric(f"{name}/mean_disagreement", sum(disagreements) / max(1, N))
            reward_metric = "mean_soft_reward_unmasked" if rating_mode == "soft" else "mean_hard_reward_unmasked"
            log_metric(f"{name}/{reward_metric}",
                       sum(r for r, m in zip(rewards, masks) if not m) / max(1, sum(1 for m in masks if not m)))
            for jx, spec in enumerate(judge_specs):
                tag = spec["tag"]
                log_metric(f"{name}/{tag}_rating_mean",
                           sum(per_judge_ratings[jx]) / max(1, N))
                if spec["needs_soft"]:
                    log_metric(f"{name}/{tag}_conf_mean",
                               sum(per_judge_confs[jx]) / max(1, N))
                log_metric(f"{name}/{tag}_fallback_frac",
                           per_judge_fallback[jx] / max(1, N))
        return rewards

    reward_fn.__name__ = name
    return reward_fn


# ---------------------------------------------------------------------------
# Pairwise rubric-judge reward (appendix baseline)
#
# Within each GRPO group of K rollouts, ask the judge which response of each
# pair better satisfies the rubric. Each rollout's reward is its win rate in
# [0, 1].
#
# How TRL hands the batch in:
#   prompts:     [p1]*K + [p2]*K + ... (groups of K identical prompts)
#   completions: K rollouts for p1, then K for p2, etc.
#   rubric:      [r1]*K + [r2]*K + ... (per-prompt rubric, repeated K times)
# We chunk by group, run pairwise within each group, then flatten back.
# ---------------------------------------------------------------------------

def make_pairwise_rubric_reward(
    judge_model: str,
    judge_base_url: str | None = None,
    judge_api_key: str = "EMPTY",
    max_concurrent: int = 64,
    judge_temperature: float = 0.0,
    judge_max_tokens: int = 64,
    num_generations: int = 8,
    pairs_per_group: int | None = None,
    pair_strategy: str = "all_pairs",
    use_rubric: bool = True,
    length_penalty: float = 0.0,
    length_penalty_chars: int = 4000,
    name: str = "pairwise_rubric_judge",
) -> Callable[..., Awaitable[list[float]]]:
    """TRL-compatible async pairwise rubric-judge reward.

    Parameters
    ----------
    num_generations : int
        K — must match `GRPOConfig.num_generations`. Used to chunk the
        batch into per-prompt groups of K.
    pairs_per_group : int or None
        M — number of pairs to compare per group for the `random`
        strategy (default K). Ignored by `all_pairs` (K choose 2 pairs).
    pair_strategy : {"all_pairs", "random", "tournament"}
        How to pick pairs from the K rollouts. The paper uses `all_pairs`.
    use_rubric : bool
        If False, the judge sees only the prompt and the two responses
        (the no-rubric pairwise baseline).
    length_penalty : float
        Optional subtractive penalty per rollout:
        length_penalty * min(1, len(response_chars) / length_penalty_chars),
        clipped so the reward stays >= 0. Default 0.0 (off).
    length_penalty_chars : int
        Character count at which the length penalty saturates.

    Position bias mitigation: each pair is judged twice with positions
    swapped, and the result is averaged.
    """
    from openai import AsyncOpenAI

    base_url = judge_base_url or os.environ.get("RAR_JUDGE_URL", "http://127.0.0.1:8000/v1")
    client = AsyncOpenAI(base_url=base_url, api_key=judge_api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    # Local vLLM judges (api_key "EMPTY") additionally get thinking disabled
    # through chat_template_kwargs, which remote APIs reject.
    is_remote_judge = judge_api_key not in (None, "", "EMPTY")

    if pair_strategy not in {"all_pairs", "random", "tournament"}:
        raise ValueError(f"unknown pair_strategy: {pair_strategy}")

    pair_messages_fn = (
        judge_pairwise_messages if use_rubric else judge_pairwise_norubric_messages
    )

    async def _call_judge(messages: list) -> int | None:
        """Single judge call → 0 (A wins), 1 (B wins), or None on failure.

        Uses structured outputs (response_format=json_schema, strict) so the
        judge cannot answer with free-form text the parser rejects.
        """
        json_schema_kwargs = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "winner",
                    "schema": JUDGE_PAIRWISE_JSON_SCHEMA,
                    "strict": True,
                },
            },
        }
        if not is_remote_judge:
            json_schema_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=judge_model,
                    messages=messages,
                    temperature=judge_temperature,
                    max_tokens=judge_max_tokens,
                    **json_schema_kwargs,
                )
            text = (resp.choices[0].message.content or "").strip()
        except Exception:
            return None
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        try:
            obj = json.loads(text)
            w = obj.get("winner", "").upper()
            if w == "A":
                return 0
            if w == "B":
                return 1
        except json.JSONDecodeError:
            pass
        return None

    async def _judge_pair(question: str, resp_a: str, resp_b: str, rubric: Any) -> float:
        """Position-bias-mitigated P(A wins): judge (A, B) and (B, A), average.

        Returns 0.5 if both calls fail.
        """
        ab, ba = await asyncio.gather(
            _call_judge(pair_messages_fn(question, resp_a, resp_b, rubric)),
            _call_judge(pair_messages_fn(question, resp_b, resp_a, rubric)),
        )
        a_wins = 0.0
        n = 0
        if ab is not None:
            a_wins += (1.0 if ab == 0 else 0.0); n += 1
        if ba is not None:
            a_wins += (1.0 if ba == 1 else 0.0); n += 1
        return (a_wins / n) if n > 0 else 0.5

    def _select_pairs(K: int) -> list[tuple[int, int]]:
        if pair_strategy == "all_pairs":
            return [(i, j) for i in range(K) for j in range(i + 1, K)]
        if pair_strategy == "tournament":
            # Fixed index pairing (K-1 pairs).
            return [(i, i + 1) for i in range(0, K - 1, 2)] + [(0, i) for i in range(2, K, 2)]
        # Random: M = pairs_per_group or K distinct pairs, unseeded.
        import random as _random
        rng = _random.Random()
        M = pairs_per_group or K
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < min(M, K * (K - 1) // 2):
            i, j = rng.sample(range(K), 2)
            if i > j: i, j = j, i
            pairs.add((i, j))
        return list(pairs)

    async def reward_fn(
        prompts: list[Any],
        completions: list[Any],
        rubric: list[Any],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        K = num_generations
        N = len(completions)
        if N % K != 0:
            raise ValueError(
                f"Pairwise reward call holds {N} rollouts, not complete groups of {K}; "
                "set per_device_train_batch_size * steps_per_generation to a multiple of num_generations")

        questions = [_prompt_to_question(p) for p in prompts]
        responses = [_completion_to_text(c) for c in completions]
        rewards = [0.0] * N

        # Judge all pairs across all groups concurrently.
        group_starts = list(range(0, N, K))
        tasks: list[tuple[int, int, int, asyncio.Future]] = []
        for g_idx, start in enumerate(group_starts):
            q = questions[start]
            rb = rubric[start]
            for (i, j) in _select_pairs(K):
                fut = asyncio.ensure_future(_judge_pair(q, responses[start + i], responses[start + j], rb))
                tasks.append((g_idx, i, j, fut))
        results = await asyncio.gather(*(t[3] for t in tasks))

        # Win rate per rollout, then flatten back.
        wins = [[0.0] * K for _ in group_starts]
        counts = [[0] * K for _ in group_starts]
        for (g_idx, i, j, _), p_a_wins in zip(tasks, results):
            wins[g_idx][i] += p_a_wins
            wins[g_idx][j] += 1.0 - p_a_wins
            counts[g_idx][i] += 1
            counts[g_idx][j] += 1
        for g_idx, start in enumerate(group_starts):
            for i in range(K):
                c = counts[g_idx][i]
                rewards[start + i] = wins[g_idx][i] / c if c else 0.5

        # Length penalty (subtractive on the win rate, clipped at 0).
        unpenalized = list(rewards)
        if length_penalty > 0.0:
            for idx in range(N):
                ratio = min(1.0, len(responses[idx]) / max(1, length_penalty_chars))
                rewards[idx] = max(0.0, rewards[idx] - length_penalty * ratio)

        if log_metric is not None:
            log_metric(f"{name}/mean", sum(rewards) / max(1, len(rewards)))
            log_metric(f"{name}/zero_frac", sum(1 for x in rewards if x == 0.0) / max(1, len(rewards)))
            # Within-group reward spread (saturation signal).
            group_stds = []
            for start in group_starts:
                grp = rewards[start:start + K]
                m = sum(grp) / K
                v = sum((r - m) ** 2 for r in grp) / K
                group_stds.append(v ** 0.5)
            log_metric(f"{name}/group_std_mean", sum(group_stds) / max(1, len(group_stds)))
            log_metric(f"{name}/group_std_zero_frac",
                       sum(1 for s in group_stds if s < 1e-6) / max(1, len(group_stds)))
            log_metric(f"{name}/mean_response_chars", sum(len(r) for r in responses) / max(1, N))
            if length_penalty > 0.0:
                log_metric(f"{name}/mean_unpenalized", sum(unpenalized) / max(1, N))
                log_metric(f"{name}/length_penalty_applied",
                           sum(unpenalized[i] - rewards[i] for i in range(N)) / max(1, N))
        return rewards

    reward_fn.__name__ = name
    return reward_fn


# ---------------------------------------------------------------------------
# Length penalty (additive). Combats GRPO length-hacking where the policy
# learns to write longer answers because longer == more chances to hit a
# rubric item by accident.
# ---------------------------------------------------------------------------

def make_length_penalty_reward(
    target_tokens: int = 300,
    max_tokens: int = 800,
    max_penalty: float = 0.5,
    name: str = "length_penalty",
) -> Callable[..., list[float]]:
    """Build a sync reward function that applies a soft length penalty.

    Returns a value in [-max_penalty, 0]:
      - 0 if completion length <= target_tokens (free zone)
      - linear ramp between target_tokens and max_tokens
      - -max_penalty for completions >= max_tokens (clamped)

    Designed to be added to a TRL `reward_funcs=[...]` list alongside the
    rubric-judge reward. TRL sums the rewards (or weights them via
    `reward_weights` in GRPOConfig). With a [0,1] rubric reward and a
    [-0.5, 0] length penalty, total reward is in [-0.5, 1].
    """
    assert max_tokens > target_tokens, "max_tokens must exceed target_tokens"
    ramp = max_tokens - target_tokens

    def reward_fn(
        completion_ids: list[list[int]],
        log_metric=None,
        log_extra=None,
        **kwargs,
    ) -> list[float]:
        rewards: list[float] = []
        lens: list[int] = []
        for ids in completion_ids:
            n = len(ids)
            lens.append(n)
            excess = max(0, n - target_tokens)
            pen = -max_penalty * min(1.0, excess / ramp)
            rewards.append(pen)
        if log_metric is not None and lens:
            log_metric(f"{name}/mean", sum(rewards) / len(rewards))
            log_metric(f"{name}/min", min(rewards))
            log_metric(f"{name}/mean_len_tokens", sum(lens) / len(lens))
            log_metric(f"{name}/max_len_tokens", float(max(lens)))
        return rewards

    reward_fn.__name__ = name
    return reward_fn
