"""Medicine domain: RaR-Medicine training + HealthBench eval.

Registration shim: the underlying logic lives in `rar.data`, `rar.prompts`
and `rar.eval_healthbench`. Paper configs use the rubrics shipped with
RaR-Medicine (`rubric_path: null`).
"""

from __future__ import annotations

from . import DomainSpec, EvalSpec, register
from ..data import POLICY_SYSTEM_PROMPT, load_rar_medicine_subset
from ..prompts import RUBRIC_WRITER_SYSTEM, RUBRIC_WRITER_USER_TEMPLATE


MEDICINE = DomainSpec(
    name="medicine",
    train_loader=load_rar_medicine_subset,
    policy_system_prompt=POLICY_SYSTEM_PROMPT,
    rubric_writer_system=RUBRIC_WRITER_SYSTEM,
    rubric_writer_user_template=RUBRIC_WRITER_USER_TEMPLATE,
    rubric_parquet_default="rubrics_base_4b.parquet",  # unused by paper configs
    eval_specs=[
        EvalSpec(
            name="healthbench_oss",
            description="HealthBench OSS (5k prompts; the paper reports the 3,931 English-only prompts)",
            default_n=200,
            runner_id="healthbench:oss",
        ),
        EvalSpec(
            name="healthbench_hard",
            description="HealthBench Hard (1k difficult prompts curated by OpenAI)",
            default_n=200,
            runner_id="healthbench:hard",
        ),
    ],
    n_train_default=1000,
    seed_default=0,
)

register(MEDICINE)
