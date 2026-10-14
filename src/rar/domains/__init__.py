"""Domain registry for multi-domain RaR training.

A `DomainSpec` describes everything needed to train + eval on one domain:
- where to load the training prompts from
- the policy/rubric-writer system prompts
- which evaluation benchmarks belong to this domain

Configs select a domain with `domain: medicine` (RaR-Medicine → HealthBench)
or `domain: science` (RaR-Science → GPQA Diamond).

Adding a new domain = creating one file under this directory that builds a
`DomainSpec` and calls `register(spec)` at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from datasets import Dataset

# A train_loader returns a HF Dataset with at least these columns:
#   - question: str (the user prompt)
#   - reference_answer: str (gold response, used by the rubric writer)
# Optional columns (passed through if present):
#   - rubric: list[dict] | str (used by training reward when no rubric_path)
TrainLoader = Callable[..., Dataset]


@dataclass
class EvalSpec:
    """One named eval benchmark within a domain.

    Descriptive only: evals are run via `rar.eval_healthbench` / `rar.eval_gpqa`.
    """

    name: str  # e.g. "healthbench_oss"
    description: str
    default_n: int = 200
    runner_id: str = ""


@dataclass
class DomainSpec:
    """Everything `train_grpo --domain <name>` needs to dispatch.

    `train_loader` is the only required field; the rest have sensible
    defaults that match the existing medicine pipeline.
    """

    name: str
    train_loader: TrainLoader
    policy_system_prompt: str
    rubric_writer_system: str
    rubric_writer_user_template: str
    rubric_parquet_default: str  # filename under $RAR_STAGE/data/
    eval_specs: list[EvalSpec] = field(default_factory=list)
    n_train_default: int = 1000
    seed_default: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


DOMAIN_REGISTRY: dict[str, DomainSpec] = {}


def register(spec: DomainSpec) -> None:
    if spec.name in DOMAIN_REGISTRY:
        # Re-registration is fine; lets us reload during dev.
        pass
    DOMAIN_REGISTRY[spec.name] = spec


def get_domain(name: str) -> DomainSpec:
    if name not in DOMAIN_REGISTRY:
        raise KeyError(
            f"Unknown domain {name!r}. Registered: {sorted(DOMAIN_REGISTRY)}"
        )
    return DOMAIN_REGISTRY[name]


# Auto-import all domain modules so registration happens on `import rar.domains`.
# Keep this list manually maintained (vs `pkgutil.iter_modules`) so we know
# exactly what's loaded.
from . import medicine  # noqa: E402, F401  (registers "medicine")
from . import science  # noqa: E402, F401  (registers "science")
