"""HealthBench evaluation (the paper's medicine benchmark).

Loads `openai/healthbench` (the OSS eval JSONL: 5,000 physician-authored
prompts with criterion-level rubrics), generates one greedy response per
prompt with the trained policy via vLLM, and grades it with an LLM grader.

Scoring modes:

- `per_criterion` (default; used in the paper): for each rubric item the
  grader answers `{"criteria_met": bool}` using the simple-evals
  GRADER_TEMPLATE. Per-prompt score = sum(points of met criteria) / sum of
  positive points, and the dataset mean is clipped to [0, 1], as in
  HealthBench's native scoring.
- `likert` (legacy): one holistic 1-10 rubric rating per response, as used
  by the training reward. Not used for any reported number.

The paper grades with gpt-4o-mini (OpenAI API) at up to 8,192 completion
tokens on all 5,000 OSS prompts and reports the English-only subset
(scripts/hb5k_split_by_language.py). Usage:
    python -m rar.eval_healthbench \\
        --checkpoint $RAR_STAGE/checkpoints/jvr_4o_dsmini_softlikert_drop030 \\
        --judge-url https://api.openai.com/v1 \\
        --judge-model gpt-4o-mini \\
        --judge-api-key-env OPENAI_API_KEY \\
        --subset oss --n 5000 --max-tokens 8192 \\
        --output $RAR_OUT/hb_pc_jgpt-4o-mini_mt8192_jvr_4o_dsmini_softlikert_drop030.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import inspect
import os
import statistics
import time
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

from .data import POLICY_SYSTEM_PROMPT
from .prompts import JUDGE_BINARY_JSON_SCHEMA, judge_binary_messages
from .reward import make_rubric_judge_reward

# HealthBench splits published by OpenAI in simple-evals. Our paper uses
# `oss` (and its English-only subset). Its "HealthBench 1K subset" is a
# theme-stratified sample of English OSS prompts built by
# scripts/build_hb_balanced1k.py and passed via --prompt-ids-file; it is
# not the official `hard` split.
HEALTHBENCH_FILES = {
    "oss":       "2025-05-07-06-14-12_oss_eval.jsonl",
    "hard":      "hard_2025-05-08-21-00-10.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
}


def _parse_met(text: str) -> bool | None:
    """Extract a yes/no signal from a per-criterion judge completion.

    Preferred schema (simple-evals GRADER_TEMPLATE):
        {"explanation": "...", "criteria_met": bool}
    Also accepts legacy / drift variants:
      - {"met": bool}, {"satisfies": "yes"}, {"answer": "no"}, {"result": 1}
      - Bare strings: "yes" / "no" / "true" / "false"
      - Code-fence wrappers (```json ... ```)
    Returns None only if no signal could be extracted.
    """
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()

    # Preferred path: valid JSON.
    try:
        obj = json.loads(t)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        # Accept any of our known keys; cast the value through the same
        # yes/no/true/false logic.
        for k in ("criteria_met", "met", "satisfies", "satisfied", "answer", "result", "verdict"):
            if k in obj:
                v = obj[k]
                if isinstance(v, bool):
                    return v
                if isinstance(v, (int, float)):
                    return v > 0
                if isinstance(v, str):
                    vl = v.strip().lower()
                    if vl in ("yes", "true", "y", "1", "met", "pass"):
                        return True
                    if vl in ("no", "false", "n", "0", "not met", "fail"):
                        return False
    # Fallback: substring match on the raw text.
    tl = t.lower()
    if '"criteria_met": true' in tl or '"criteria_met":true' in tl:
        return True
    if '"criteria_met": false' in tl or '"criteria_met":false' in tl:
        return False
    if '"met": true' in tl or '"met":true' in tl or '"satisfies": "yes"' in tl or '"satisfies":"yes"' in tl:
        return True
    if '"met": false' in tl or '"met":false' in tl or '"satisfies": "no"' in tl or '"satisfies":"no"' in tl:
        return False
    # Last resort: bare yes/no.
    if tl in ("yes", "true"):
        return True
    if tl in ("no", "false"):
        return False
    return None


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


def _load_healthbench(n: int, seed: int, subset: str = "oss", data_path: str | None = None) -> list[dict]:
    """Download a HealthBench split and return a deterministic n-prompt subset.

    `subset` picks which of the three OpenAI HealthBench JSONL releases to
    load from the `openai/healthbench` HF dataset repo:
      - "oss"       : full 5k eval (default; used in the paper)
      - "hard"      : HealthBench Hard, 1k examples hard for frontier models
      - "consensus" : physician-consensus subset (~3.7k)
    If `n` is larger than the loaded split, all rows are used.

    Each returned dict has:
        question: str         (the user prompt content)
        rubric: list[dict]    (description/weight, our format)
        rubric_raw: list[dict] (criterion/points/tags, paper format)
    """
    if subset not in HEALTHBENCH_FILES:
        raise ValueError(f"unknown HealthBench subset {subset!r}; pick one of {list(HEALTHBENCH_FILES)}")
    fp = data_path or hf_hub_download(
        repo_id="openai/healthbench",
        filename=HEALTHBENCH_FILES[subset],
        repo_type="dataset",
    )
    rows: list[dict] = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if n >= len(rows):
        selected = rows
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(rows))[:n].tolist()
        indices.sort()
        selected = [rows[i] for i in indices]

    out: list[dict] = []
    for r in selected:
        # HealthBench prompts are multi-turn but almost always one user turn.
        # Concatenate user turns into a single question string for our pipeline.
        prompt_parts: list[str] = []
        for turn in r["prompt"]:
            if turn.get("role") == "user":
                prompt_parts.append(turn.get("content", ""))
        question = "\n\n".join(p for p in prompt_parts if p)

        # Convert HealthBench rubric items. We keep BOTH:
        #   - `rubric`      : our format ({description, weight}) for the
        #                      legacy single-Likert scoring path
        #   - `rubric_raw`  : the original items ({criterion, points, tags})
        #                      for the per-criterion (paper-faithful) path
        rubric: list[dict] = []
        rubric_raw: list[dict] = []
        for item in r.get("rubrics", []):
            criterion = item.get("criterion", "")
            points = int(item.get("points", 1))
            # Bucket HealthBench points into our (-2, 5) weight scale.
            # Negative points in HealthBench mean "pitfall" (the response
            # should NOT do this) — we map those to our negative weights.
            if points <= -2:
                weight = -2
            elif points <= -1:
                weight = -1
            elif points >= 9:
                weight = 5
            elif points >= 7:
                weight = 4
            elif points >= 5:
                weight = 3
            elif points >= 3:
                weight = 2
            else:
                weight = 1
            rubric.append({"description": criterion, "weight": weight})
            rubric_raw.append({"criterion": criterion, "points": points, "tags": item.get("tags", [])})

        out.append(
            {
                "question": question,
                "rubric": rubric,
                "rubric_raw": rubric_raw,
                "prompt_id": r.get("prompt_id", ""),
                # Carry HB-Hard prompt-level tags through so the eval can
                # stratify by theme / physician_agreed_category and (where
                # we add tags post-hoc) by detected language.
                "example_tags": list(r.get("example_tags", [])),
            }
        )
    return out


# English detection. HealthBench ships no language tag, so we use langdetect
# on the prompt's user turns with a probability threshold. This is the same
# rule as scripts/hb5k_split_by_language.py and selects 3,931 of the 5,000
# OSS prompts (the paper's English-only subset).
_LANGDETECT_MIN_PROB = 0.90


def _likely_english(text: str, min_prob: float = _LANGDETECT_MIN_PROB) -> bool:
    """Return True iff `text` is detected as English with prob >= min_prob.

    Uses `langdetect` (deterministic via seed=0 from DetectorFactory).
    Empty / whitespace-only texts return False.
    """
    if not text or not text.strip():
        return False
    # Local import + deterministic seed so per-process state is repeatable.
    from langdetect import detect_langs, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    try:
        cands = detect_langs(text)
    except LangDetectException:
        return False
    if not cands:
        return False
    top = cands[0]
    return top.lang == "en" and top.prob >= min_prob


def _stratified_mean_ci(
    rewards: list[float], group_labels: list[str | None]
) -> dict:
    """Return {label: {n, mean, ci_low, ci_high}} for each non-None group label."""
    from collections import defaultdict
    groups: dict[str, list[float]] = defaultdict(list)
    for r, label in zip(rewards, group_labels):
        if label is None:
            continue
        groups[label].append(r)
    out: dict = {}
    for label, vals in groups.items():
        if not vals:
            continue
        m = statistics.fmean(vals)
        lo, hi = _bootstrap_ci(vals) if len(vals) >= 2 else (m, m)
        out[label] = {
            "n": len(vals),
            "mean": float(m),
            "ci_low": float(lo),
            "ci_high": float(hi),
        }
    return out


async def _score_per_criterion(
    questions: list[str],
    responses: list[str],
    rubric_raws: list[list[dict]],
    judge_model: str,
    judge_base_url: str,
    max_concurrent: int = 64,
    judge_api_key: str = "EMPTY",
    is_remote: bool = False,
    criterion_cache: str | None = None,
    judge_extra_body: dict | None = None,
    judge_max_tokens: int = 512,
    judge_timeout: float = 30.0,
) -> list[float]:
    """Paper-faithful per-criterion scoring for HealthBench.

    Exactly matches OpenAI simple-evals `healthbench_eval.calculate_score`:
        total_possible_points = sum(p for p in points if p > 0)
        achieved_points       = sum(p for p, met in zip(points, mets) if met)
        overall_score         = achieved_points / total_possible_points
    and the dataset-level mean is clipped to [0, 1] *once* at report time,
    NOT per-example. Per-example scores can be negative if pitfalls
    outweigh positives.

    We previously used `sum(|points|)` in the denominator and clipped each
    example to [0, 1]. Both were bugs. The denominator bug shrinks all
    scores (extra mass from pitfall magnitudes); the per-example clip hides
    negative contributions from very weak responses, which inflates the
    mean. Sanity-checked by re-running Qwen2.5-7B against paper Table 1.
    """
    from openai import AsyncOpenAI

    # max_retries=8 (default 2) so transient 429s under remote-judge rate
    # limits get smoothed by SDK-side backoff rather than landing in the
    # failure path below. At high concurrency, rate-limited calls exhausted
    # the default 2-retry budget.
    # Explicit httpx timeouts. The earlier `asyncio.wait_for` wrap didn't
    # actually cancel pending requests for AsyncOpenAI/httpx — observed runs
    # where calls hung indefinitely on dead sockets with no TimeoutError
    # firing. Setting `timeout=` on the SDK propagates to httpx which natively
    # supports per-phase timeouts (connect/read/write/pool). 20s read is well
    # above p99 for gpt-4o-mini judge calls.
    import httpx as _httpx
    http_kwargs = {}
    if criterion_cache:
        from .cost_audit import make_http_cost_hooks
        http_kwargs["http_client"] = _httpx.AsyncClient(event_hooks=make_http_cost_hooks(
            str(criterion_cache) + ".http_costs", "healthbench_grader"))
    client = AsyncOpenAI(
        **http_kwargs,
        base_url=judge_base_url,
        api_key=judge_api_key,
        max_retries=8,
        timeout=_httpx.Timeout(judge_timeout, connect=10.0, read=max(20.0, judge_timeout - 10.0),
                               write=10.0, pool=10.0),
    )
    semaphore = asyncio.Semaphore(max_concurrent)
    fail_counter = {"n": 0}

    # extra_body only gets sent when the judge is vLLM; for remote APIs
    # (OpenRouter, OpenAI direct), `chat_template_kwargs` would be a 400.
    vllm_extras = {"chat_template_kwargs": {"enable_thinking": False}} if not is_remote else {}

    vllm_extras.update(judge_extra_body or {})
    protocol = {"model": judge_model, "url": judge_base_url, "temperature": 0.0,
                "max_tokens": judge_max_tokens, "schema": JUDGE_BINARY_JSON_SCHEMA,
                "extra_body": vllm_extras, "prompt_source": inspect.getsource(judge_binary_messages)}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    cached = {}
    if criterion_cache:
        cache_path = Path(criterion_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        protocol_path = cache_path.with_suffix(cache_path.suffix + ".protocol.json")
        if protocol_path.exists():
            if json.loads(protocol_path.read_text())["hash"] != protocol_hash:
                raise ValueError("Criterion cache protocol mismatch")
        else:
            if cache_path.exists() and cache_path.stat().st_size:
                raise ValueError("Nonempty criterion cache lacks protocol metadata")
            protocol_path.write_text(json.dumps({"hash": protocol_hash, "protocol": protocol}, indent=2))
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                if row.get("protocol_hash") != protocol_hash:
                    raise ValueError("Conflicting cached criterion protocol")
                if isinstance(row.get("met"), bool):
                    if row["key"] in cached and cached[row["key"]] != row["met"]:
                        raise ValueError("Conflicting cached criterion result")
                    cached[row["key"]] = row["met"]

    async def _met(question: str, response: str, criterion: str, points: int) -> bool | None:
        # Per-call wall timeout — guards against silent OpenAI connection
        # drops where the underlying TCP socket sits idle indefinitely (the
        # SDK's `max_retries=8` only catches HTTP-level failures, not dead
        # sockets). 30s is well above p99 latency for gpt-4o-mini judge calls.
        # Retry up to 3 times on TimeoutError before giving up.
        messages = judge_binary_messages(question, response, criterion, points=points)
        task_key = hashlib.sha256(json.dumps([protocol_hash, messages], sort_keys=True).encode()).hexdigest()
        if task_key in cached:
            return cached[task_key]
        started = time.monotonic()
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(
                            model=judge_model,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=judge_max_tokens,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "grader",
                                    "schema": JUDGE_BINARY_JSON_SCHEMA,
                                    "strict": True,
                                },
                            },
                            extra_body=vllm_extras,
                        ),
                        timeout=judge_timeout,
                    )
                message = resp.choices[0].message
                text = (message.content or "").strip()
                if not text:
                    # Some reasoning models (e.g. GLM-5.3-Flash on DeepInfra) put
                    # the JSON verdict in `reasoning_content` and leave content empty.
                    text = (getattr(message, "reasoning_content", None) or "").strip()
                met = _parse_met(text)
                if resp.choices[0].finish_reason != "stop" or met is None:
                    continue
                if criterion_cache:
                    record = {"key": task_key, "protocol_hash": protocol_hash, "met": met,
                              "response_id": resp.id, "model": resp.model,
                              "usage": resp.usage.model_dump() if resp.usage else None,
                              "elapsed_seconds": time.monotonic() - started, "attempt": attempt + 1}
                    with cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    cached[task_key] = met
                return met
            except asyncio.TimeoutError:
                continue
            except Exception:
                break
        fail_counter["n"] += 1
        return None

    # Progress tracking. asyncio.gather is silent until done; without this
    # an in-flight judge phase looks identical to a stalled one for ~30 min.
    progress = {"done": 0, "total": len(questions), "t0": time.monotonic()}

    async def _score_prompt(q: str, r: str, items: list[dict]) -> float | None:
        # Launch one judge call per criterion for this prompt in parallel.
        tasks = [_met(q, r, item["criterion"], int(item["points"])) for item in items]
        mets = await asyncio.gather(*tasks)
        progress["done"] += 1
        if progress["done"] % 100 == 0 or progress["done"] == progress["total"]:
            elapsed = time.monotonic() - progress["t0"]
            rate = progress["done"] / max(elapsed, 1e-6)
            eta = (progress["total"] - progress["done"]) / max(rate, 1e-6)
            print(f"[hb-eval] judged {progress['done']}/{progress['total']} prompts "
                  f"({rate:.1f}/s, eta {eta:.0f}s, fails={fail_counter['n']})", flush=True)
        # If ANY criterion's judge call failed irrecoverably, the prompt's
        # score is unreliable (failed criteria silently contribute 0 to
        # `achieved`, which biases the mean down). Return None and let the
        # caller drop the prompt from the reported mean.
        if any(m is None for m in mets):
            return None
        total_possible = 0.0
        achieved = 0.0
        for item, met in zip(items, mets):
            p = item["points"]
            if p > 0:
                total_possible += p
            if met is True:
                achieved += p
        if total_possible == 0:
            return 0.0
        return achieved / total_possible  # NOT clipped per-example

    results = await asyncio.gather(
        *[_score_prompt(q, r, items) for q, r, items in zip(questions, responses, rubric_raws)]
    )
    if fail_counter["n"]:
        n_failed_prompts = sum(1 for r in results if r is None)
        print(f"[hb-eval] WARNING: {fail_counter['n']} judge calls failed after "
              f"retries; {n_failed_prompts}/{len(results)} prompts will be excluded "
              f"from the mean. Lower --judge-max-concurrent if this is large.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Greedy decoding (matches paper eval).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument(
        "--scoring",
        choices=["likert", "per_criterion"],
        default="per_criterion",
        help="Scoring method. `per_criterion` matches the paper's HealthBench eval.",
    )
    parser.add_argument(
        "--subset",
        choices=list(HEALTHBENCH_FILES.keys()),
        default="oss",
        help="HealthBench split to load. The paper uses 'oss'.",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default=None,
        help="Environment variable name holding the judge API key. When set, "
             "the judge is treated as remote (OpenRouter, OpenAI direct, etc.): "
             "no vLLM-specific extras are sent, and the api_key from $ENV is "
             "used instead of 'EMPTY'. Example: --judge-api-key-env OPENAI_API_KEY",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--judge-max-concurrent",
        type=int,
        default=None,
        help="Override the judge call concurrency. Default 64 (per-criterion "
             "remote-judge path). Drop if you hit 429s; raise for higher-tier "
             "OpenAI keys.",
    )
    parser.add_argument(
        "--prompt-ids-file",
        default=None,
        help="Path to a JSON manifest with a 'prompt_ids' list. Filters the "
             "loaded HealthBench split down to exactly those ids. Use this "
             "for reproducible balanced subsets (see scripts/build_hb_balanced1k.py).",
    )
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Keep only English prompts (langdetect, p >= 0.90; 3,931 of the "
             "5,000 OSS prompts). The paper instead evaluates all prompts and "
             "splits afterwards with scripts/hb5k_split_by_language.py.",
    )
    parser.add_argument(
        "--stratify-by",
        action="append",
        default=None,
        help="Tag prefix to stratify per-prompt scores by in the output JSON. "
             "Repeatable. Common values for HB-Hard: 'theme', "
             "'physician_agreed_category'. Each prompt is grouped by the "
             "FIRST tag matching the prefix.",
    )
    parser.add_argument(
        "--completions-cache",
        default=None,
        help="Path to a JSON file where policy completions are saved after "
             "generation. If --resume-from-completions is also set and this "
             "file exists, the policy generation phase is skipped and "
             "completions are loaded from the cache. Lets you re-run just "
             "the judge phase after a rate-limit / crash.",
    )
    parser.add_argument(
        "--resume-from-completions",
        action="store_true",
        help="If --completions-cache exists, skip vLLM init and policy "
             "generation; load completions from the cache and proceed to "
             "the judge phase directly.",
    )
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--data-path", help="Verified local HealthBench JSONL")
    parser.add_argument("--criterion-cache", help="Resumable per-criterion JSONL, with usage")
    parser.add_argument("--judge-extra-body", default="{}",
                        help='JSON merged into each grader request, e.g. \'{"reasoning_effort": "low"}\'')
    parser.add_argument("--judge-max-tokens", type=int, default=512,
                        help="Grader max_tokens (raise for reasoning graders, e.g. 4096 for GLM-5.3-Flash)")
    parser.add_argument("--judge-timeout", type=float, default=30.0,
                        help="Per-request grader timeout in seconds (raise for reasoning graders)")
    parser.add_argument("--strict-coverage", action="store_true")
    args = parser.parse_args()
    if args.generation_only and not args.completions_cache:
        parser.error("--generation-only requires --completions-cache")
    if args.strict_coverage and args.english_only:
        parser.error("Use the frozen ID manifest, not --english-only, with strict coverage")
    if args.resume_from_completions and (not args.completions_cache or not Path(args.completions_cache).exists()):
        parser.error("Requested completion cache does not exist")

    # Resolve remote-judge parameters from env.
    judge_api_key = "EMPTY"
    is_remote = False
    if args.judge_api_key_env and not args.generation_only:
        judge_api_key = os.environ.get(args.judge_api_key_env, "")
        if not judge_api_key:
            raise SystemExit(
                f"[hb-eval] --judge-api-key-env {args.judge_api_key_env} is set "
                f"but the environment variable is empty"
            )
        is_remote = True
        print(f"[hb-eval] remote judge mode: key from ${args.judge_api_key_env}")

    if args.prompt_ids_file:
        # Manifest-driven: load the FULL split, filter by prompt_ids, ignore --n.
        with open(args.prompt_ids_file) as f:
            manifest = json.load(f)
        wanted = set(manifest["prompt_ids"])
        if len(wanted) != len(manifest["prompt_ids"]):
            raise ValueError("Duplicate IDs in manifest")
        print(f"[hb-eval] manifest: {manifest.get('name', args.prompt_ids_file)}  "
              f"(prompt_ids={len(wanted)})")
        all_rows = _load_healthbench(10**9, args.seed, subset=args.subset, data_path=args.data_path)
        examples = [ex for ex in all_rows if ex["prompt_id"] in wanted]
        print(f"[hb-eval] manifest filter: matched {len(examples)} / {len(wanted)} "
              f"requested prompt_ids in subset={args.subset}")
        if len({e["prompt_id"] for e in examples}) != len(examples):
            raise ValueError("Duplicate IDs in HealthBench data")
        if args.strict_coverage and {e["prompt_id"] for e in examples} != wanted:
            raise ValueError("HealthBench data does not cover the frozen manifest")
        if len(examples) != len(wanted):
            missing = wanted - {ex["prompt_id"] for ex in examples}
            print(f"[hb-eval] WARNING: {len(missing)} prompt_ids in manifest not "
                  f"present in subset (first 5: {sorted(missing)[:5]})")
    else:
        print(f"[hb-eval] loading HealthBench subset={args.subset}, n={args.n}")
        examples = _load_healthbench(args.n, args.seed, subset=args.subset, data_path=args.data_path)
        print(f"[hb-eval] sampled {len(examples)} prompts from {args.subset}")
    if args.english_only:
        before = len(examples)
        examples = [ex for ex in examples if _likely_english(ex["question"])]
        print(
            f"[hb-eval] english-only filter: kept {len(examples)} / {before} "
            f"({100*len(examples)/max(1,before):.1f}%)"
        )

    # Resume from cached completions if available — skips vLLM init entirely.
    have_completions = False
    if args.resume_from_completions and args.completions_cache and os.path.exists(args.completions_cache):
        print(f"[hb-eval] loading cached completions from {args.completions_cache}")
        with open(args.completions_cache) as f:
            cache = json.load(f)
        if len(cache["prompt_ids"]) != len(cache["completions"]) or len(set(cache["prompt_ids"])) != len(cache["prompt_ids"]):
            raise ValueError("Invalid completion cache lengths/duplicate IDs")
        pid_to_compl = dict(zip(cache["prompt_ids"], cache["completions"]))
        if args.strict_coverage and not {e["prompt_id"] for e in examples}.issubset(pid_to_compl):
            raise ValueError("Completion cache is missing requested IDs")
        # Filter examples to those we have completions for (in original order)
        examples = [ex for ex in examples if ex["prompt_id"] in pid_to_compl]
        completions = [pid_to_compl[ex["prompt_id"]] for ex in examples]
        print(f"[hb-eval] loaded {len(completions)} cached completions; skipping vLLM init + generation")
        have_completions = True

    if not have_completions:
        from vllm import LLM, SamplingParams
        print(f"[hb-eval] loading policy {args.checkpoint}")
        llm = LLM(
            model=args.checkpoint,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            seed=args.seed,
        )
        sampling = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=args.max_tokens, seed=args.seed)

        chats = [
            [
                {"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ]
            for ex in examples
        ]

        print(f"[hb-eval] generating {len(chats)} completions...")
        outs = llm.chat(chats, sampling_params=sampling, chat_template_kwargs={"enable_thinking": False})
        completions = [o.outputs[0].text for o in outs]

        # Save completions before judging so any judge-phase crash can be resumed.
        if args.completions_cache:
            os.makedirs(os.path.dirname(args.completions_cache) or ".", exist_ok=True)
            with open(args.completions_cache, "w") as f:
                json.dump(
                    {
                        "checkpoint": args.checkpoint,
                        "subset": args.subset,
                        "n": len(completions),
                        "max_tokens": args.max_tokens,
                        "temperature": args.temperature, "top_p": 0.9, "seed": args.seed,
                        "enable_thinking": False, "system_prompt": POLICY_SYSTEM_PROMPT,
                        "finish_reasons": [o.outputs[0].finish_reason for o in outs],
                        "output_token_counts": [len(o.outputs[0].token_ids) for o in outs],
                        "prompt_ids": [ex["prompt_id"] for ex in examples],
                        "completions": completions,
                    },
                    f,
                )
            print(f"[hb-eval] saved {len(completions)} completions to {args.completions_cache}")

    if args.generation_only:
        return

    print(f"[hb-eval] judging mode={args.scoring} with {args.judge_model} at {args.judge_url}")

    if args.scoring == "per_criterion":
        questions = [ex["question"] for ex in examples]
        rubric_raws = [ex["rubric_raw"] for ex in examples]
        # Remote APIs throttle far below local vLLM; high concurrency against
        # a remote grader causes rate-limit failures. Default 8 for remote,
        # 64 for a local vLLM grader.
        max_conc = (args.judge_max_concurrent if args.judge_max_concurrent
                    else (8 if is_remote else 64))
        rewards = asyncio.run(
            _score_per_criterion(
                questions=questions,
                responses=completions,
                rubric_raws=rubric_raws,
                judge_model=args.judge_model,
                judge_base_url=args.judge_url,
                judge_api_key=judge_api_key,
                is_remote=is_remote,
                max_concurrent=max_conc,
                criterion_cache=args.criterion_cache,
                judge_extra_body=json.loads(args.judge_extra_body),
                judge_max_tokens=args.judge_max_tokens,
                judge_timeout=args.judge_timeout,
            )
        )
    else:
        # Legacy single-Likert path (our training-time reward fn).
        reward_fn = make_rubric_judge_reward(
            judge_model=args.judge_model,
            judge_base_url=args.judge_url,
            name="hb_eval_judge",
        )
        prompts = [[{"role": "user", "content": ex["question"]}] for ex in examples]
        rubrics = [ex["rubric"] for ex in examples]
        rewards = asyncio.run(
            reward_fn(prompts=prompts, completions=completions, rubric=rubrics)
        )

    # HealthBench native scoring: clip the *mean* at [0, 1], not per-example.
    # For per_criterion mode the raw `rewards` list can contain negatives
    # (pitfalls outweighed positives on a given prompt). Report both the
    # unclipped mean (for diagnostic visibility) and the clipped mean (the
    # headline number that matches the paper).
    # Filter out prompts whose judge calls failed (None) — see
    # _score_per_criterion. Track alignment with `examples`/`completions`
    # so per-prompt artifacts (prompt_ids, completion_chars, ...) match.
    valid_mask = [r is not None for r in rewards]
    n_dropped = sum(1 for v in valid_mask if not v)
    if n_dropped and args.strict_coverage:
        raise RuntimeError(f"{n_dropped} prompts have unresolved grades; resume the criterion cache before reporting")
    if n_dropped:
        print(f"[hb-eval] dropping {n_dropped}/{len(rewards)} prompts with judge-call failures")
    rewards = [r for r, ok in zip(rewards, valid_mask) if ok]
    examples = [e for e, ok in zip(examples, valid_mask) if ok]
    completions = [c for c, ok in zip(completions, valid_mask) if ok]
    raw_mean = statistics.fmean(rewards)
    mean = max(0.0, min(1.0, raw_mean)) if args.scoring == "per_criterion" else raw_mean
    lo, hi = _bootstrap_ci(rewards)
    if args.scoring == "per_criterion":
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
    print(f"[hb-eval] scoring={args.scoring}  mean = {mean:.4f} (raw={raw_mean:.4f})  "
          f"95% CI = [{lo:.4f}, {hi:.4f}]  n={len(rewards)}")

    # Build stratification breakdowns (theme / physician_agreed_category / ...).
    stratifications: dict = {}
    if args.stratify_by:
        for prefix in args.stratify_by:
            labels: list[str | None] = []
            for ex in examples:
                first = next(
                    (t for t in ex.get("example_tags", []) if t.startswith(f"{prefix}:")),
                    None,
                )
                labels.append(first)
            stratifications[prefix] = _stratified_mean_ci(rewards, labels)
            print(f"[hb-eval] stratified by {prefix}:")
            for label, info in sorted(stratifications[prefix].items(), key=lambda kv: -kv[1]["n"]):
                print(
                    f"  {label:60s} n={info['n']:4d}  mean={info['mean']:.4f}  "
                    f"CI=[{info['ci_low']:.4f}, {info['ci_high']:.4f}]"
                )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "judge_model": args.judge_model,
                    "benchmark": "HealthBench-OSS",
                    "scoring": args.scoring,
                    "english_only": args.english_only,
                    "n": len(rewards),
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "rewards": rewards,
                    "prompt_ids": [ex["prompt_id"] for ex in examples],
                    # Per-prompt response lengths in chars; needed for
                    # length-residualized scoring (AlpacaEval-LC style).
                    "completion_chars": [len(c) for c in completions],
                    # Per-prompt rubric size (n criteria). Lets downstream
                    # analysis stratify by rubric length without re-loading
                    # the original HB JSONL.
                    "rubric_n_items": [len(ex["rubric_raw"]) for ex in examples],
                    "example_tags": [ex.get("example_tags", []) for ex in examples],
                    "stratifications": stratifications,
                },
                f,
                indent=2,
            )
        print(f"[hb-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
