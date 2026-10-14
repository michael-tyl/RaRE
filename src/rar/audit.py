"""Strict judge-response validation and per-rank JSONL audit output."""
import json
import math
import os
from pathlib import Path
import time


def validate_hard_response(resp, scale_max=5):
    choice = resp.choices[0]
    obj = json.loads(choice.message.content or "")
    if not isinstance(obj, dict) or set(obj) != {"rating"} or type(obj["rating"]) is not int or not 1 <= obj["rating"] <= scale_max:
        raise ValueError("Invalid rating JSON")
    if choice.finish_reason != "stop":
        raise ValueError("Incomplete judge response")
    return obj["rating"]


def validate_soft_response(resp, scale_max=5):
    rating = validate_hard_response(resp, scale_max)
    choice = resp.choices[0]
    obj = {"rating": rating}
    entries = getattr(getattr(choice, "logprobs", None), "content", None) or []
    digits = {str(i) for i in range(1, scale_max + 1)}
    rating_entries = [e for e in entries if (e.token or "").strip() in digits]
    if len(rating_entries) != 1 or rating_entries[0].token.strip() != str(obj["rating"]):
        raise ValueError("Missing or ambiguous rating token logprobs")
    alts = rating_entries[0].top_logprobs or []
    if not alts or any(not math.isfinite(a.logprob) or a.logprob > 1e-6 for a in alts):
        raise ValueError("Invalid rating alternatives")
    if all(a.logprob == 0 or a.logprob <= -9990 for a in alts) and any(a.logprob <= -9990 for a in alts):
        raise ValueError("Provider returned sentinel logprobs")
    mass = sum(math.exp(a.logprob) for a in alts if a.token.strip() in digits)
    if mass <= 1e-9:
        raise ValueError("No usable rating probability mass")
    probabilities = {}
    for alt in alts:
        token = alt.token.strip()
        if token in digits:
            probabilities[int(token)] = probabilities.get(int(token), 0.0) + math.exp(alt.logprob) / mass
    expected = sum(k * v for k, v in probabilities.items())
    return {"retained_digit_mass": mass, "rating_probabilities": probabilities,
            "variance": sum((k - expected) ** 2 * v for k, v in probabilities.items())}


def write_audit(directory, record):
    path = Path(os.path.expandvars(str(directory)))
    path.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    record = {"timestamp": time.time(), "rank": int(rank), "pid": os.getpid(), **record}
    with (path / f"rank{rank}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
