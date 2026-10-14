"""GRPO training entry point for one rubric-as-reward run.

Usage:
    python -m rar.train_grpo --config configs/run_jvr_4o_dsmini_softlikert_drop030.yaml [--seed N]

The YAML config selects:
  - the domain (medicine / science) and rubric source (rubric_path, null = shipped rubrics)
  - the judge(s) and reward type (single judge, mean/drop ensemble, soft-Likert, RaRE)
  - run name, output dir, and a few hyperparam overrides
Most hyperparameters are fixed in code (paper defaults).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from .data import attach_rubric_column, to_grpo_dataset
from .domains import get_domain
from .reward import make_length_penalty_reward, make_rubric_judge_reward


def _local(env_var: str, default: str) -> str:
    return os.environ.get(env_var, default)


def build_dataset(cfg: dict) -> Dataset:
    # Domain dispatch. Default "medicine" preserves Phase 1 behavior so
    # existing configs without a `domain:` field still load RaR-Medicine.
    domain_name = cfg.get("domain", "medicine")
    spec = get_domain(domain_name)
    # data_seed (optional) decouples dataset-subset selection from
    # training-side randomness (GRPOConfig.seed). When unset,
    # falls back to `seed` for backward compat with existing configs.
    _data_seed = cfg.get("data_seed", cfg.get("seed", spec.seed_default))
    if cfg.get("train_data_path"):
        import hashlib
        data_path = Path(os.path.expandvars(cfg["train_data_path"]))
        expected_hash = cfg.get("train_data_sha256")
        if not expected_hash or hashlib.sha256(data_path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Frozen training parquet SHA256 mismatch or missing hash")
        base = Dataset.from_parquet(str(data_path))
        if len(base) != cfg.get("n_train", 1000) or not {"question", "rubric"}.issubset(base.column_names):
            raise ValueError("Frozen training parquet count/schema mismatch")
        if any(not row["question"] or not row["rubric"] for row in base):
            raise ValueError("Frozen data contains empty question/rubric")
    else:
        base = spec.train_loader(
            split=cfg.get("split", "train"),
            n=cfg.get("n_train", spec.n_train_default),
            seed=_data_seed,
        )
    rubric_path = cfg.get("rubric_path")  # None => keep shipped rubrics
    if rubric_path is not None:
        rubric_path = os.path.expandvars(rubric_path)
        if "$" in rubric_path:
            raise RuntimeError(
                f"rubric_path contains unexpanded env var: {rubric_path!r}. "
                "Did you forget to `source scripts/env.sh`?"
            )
    base = attach_rubric_column(base, rubric_path)
    return to_grpo_dataset(base, system_prompt=spec.policy_system_prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None, help="Override config output_dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config max_steps")
    parser.add_argument("--debug", action="store_true", help="Tiny config for smoke test")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed (and data_seed unless the config sets it); "
                             "appends _s<seed> to run_name for seed != 0")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed
        if args.seed != 0:
            cfg["run_name"] = f"{cfg['run_name']}_s{args.seed}"

    run_name = cfg["run_name"]
    output_dir = args.output_dir or cfg.get("output_dir") or f"{_local('RAR_CKPT', 'checkpoints')}/{run_name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(cfg)
    print(f"[train_grpo] dataset size = {len(dataset)}")

    # RAR_JUDGE_URL (set by local-judge launchers from the dynamic JUDGE_PORT)
    # MUST take precedence over the config's `judge_url`. Configs were
    # written when JUDGE_PORT was hardcoded to 8000; after the per-job
    # port-collision fix the actual port differs, and silently hitting
    # port 8000 returns reward 0 for every rollout (judge unreachable
    # → all calls fail → training is wasted).
    judge_url = os.environ.get("RAR_JUDGE_URL") or cfg.get("judge_url")
    # Remote judge via OpenRouter / OpenAI: read API key from env var
    # named in the config (e.g., judge_api_key_env: OPENAI_API_KEY).
    judge_api_key = "EMPTY"
    judge_api_key_env = cfg.get("judge_api_key_env")
    if judge_api_key_env:
        judge_api_key = os.environ.get(judge_api_key_env, "")
        if not judge_api_key:
            raise SystemExit(
                f"[train_grpo] judge_api_key_env={judge_api_key_env} configured "
                f"but env var is empty. Did you `--export=ALL` in sbatch?"
            )
        judge_desc = cfg.get('judge_model') or f"ensemble of {len(cfg.get('judges', []))} judges"
        print(f"[train_grpo] remote judge: {judge_desc} via {judge_url}  (api_key from ${judge_api_key_env})")
    reward_type = cfg.get("reward_type", "absolute")
    if reward_type == "pairwise":
        # Pairwise preference judge (appendix baseline): within each GRPO group,
        # the judge picks the better of each pair; reward = win rate.
        from .reward import make_pairwise_rubric_reward
        pw_args = cfg.get("pairwise_args", {}) or {}
        reward_fn = make_pairwise_rubric_reward(
            judge_model=cfg["judge_model"],
            judge_base_url=judge_url,
            judge_api_key=judge_api_key,
            max_concurrent=cfg.get("judge_max_concurrent", 64),
            num_generations=cfg.get("num_generations", 8),
            pairs_per_group=pw_args.get("pairs_per_group", None),
            pair_strategy=pw_args.get("pair_strategy", "all_pairs"),
            use_rubric=pw_args.get("use_rubric", True),
            length_penalty=pw_args.get("length_penalty", 0.0),
            length_penalty_chars=pw_args.get("length_penalty_chars", 4000),
            name=f"pairwise_judge_{run_name}",
        )
        judge_reward_fns = None
        ensemble_judge_weights = None
        print(
            f"[train_grpo] reward_type=pairwise  strategy={pw_args.get('pair_strategy', 'all_pairs')}  "
            f"use_rubric={pw_args.get('use_rubric', True)}"
        )
    elif reward_type == "absolute":
        # `judges:` list → multi-judge ensemble (mean of per-judge scores).
        # If `disagreement_threshold` is ALSO set, rollouts where the per-judge
        # max-min exceeds the threshold are masked (advantage=0). Otherwise
        # all rollouts contribute normally to the per-judge mean.
        # Otherwise single-judge from cfg["judge_model"].
        judges_list = cfg.get("judges")
        disagreement_threshold = cfg.get("disagreement_threshold")
        if judges_list and disagreement_threshold is not None:
            from .reward import make_ensemble_drop_disagreement_reward
            reward_fn = make_ensemble_drop_disagreement_reward(
                judges=judges_list,
                disagreement_threshold=float(disagreement_threshold),
                num_generations=cfg.get("num_generations", 8),
                default_api_key_env=cfg.get("judge_api_key_env"),
                name=f"ensemble_drop_{run_name}",
                strict_scores=cfg.get("strict_scores", False),
                audit_log_dir=cfg.get("audit_log_dir"),
            )
            judge_reward_fns = None
            ensemble_judge_weights = None
            print(
                f"[train_grpo] reward_type=absolute  multi-judge DROP-disagreement "
                f"(N={len(judges_list)}, threshold={disagreement_threshold})"
            )
        elif judges_list:
            judge_reward_fns = []
            for i, j in enumerate(judges_list):
                jkey_env = j.get("api_key_env", cfg.get("judge_api_key_env"))
                jkey = os.environ.get(jkey_env, "") if jkey_env else "EMPTY"
                if jkey_env and not jkey:
                    raise SystemExit(
                        f"[train_grpo] judges[{i}].api_key_env={jkey_env} empty. --export=ALL?"
                    )
                tag = j.get("tag", f"j{i}")
                judge_reward_fns.append(make_rubric_judge_reward(
                    judge_model=j["model"],
                    judge_base_url=j.get("url", judge_url),
                    judge_api_key=jkey,
                    max_concurrent=j.get("max_concurrent", cfg.get("judge_max_concurrent", 16)),
                    judge_max_tokens=j.get("max_tokens", cfg.get("judge_max_tokens", 128)),
                    judge_reasoning_effort=j.get("reasoning_effort"),
                    response_format=j.get("response_format", cfg.get("response_format")),
                    name=f"rubric_judge_{run_name}_{tag}",
                ))
                print(f"[train_grpo] judge[{tag}]: {j['model']} via {j.get('url', judge_url)}")
            reward_fn = None  # signal multi-judge mode to the assembly below
            n_judges = len(judge_reward_fns)
            ensemble_judge_weights = [1.0 / n_judges] * n_judges
            print(f"[train_grpo] reward_type=absolute  multi-judge ensemble (N={n_judges}, weights={ensemble_judge_weights})")
        else:
            reward_fn = make_rubric_judge_reward(
                judge_model=cfg["judge_model"],
                judge_base_url=judge_url,
                judge_api_key=judge_api_key,
                max_concurrent=cfg.get("judge_max_concurrent", 64),
                judge_max_tokens=cfg.get("judge_max_tokens", 128),
                judge_reasoning_effort=cfg.get("judge_reasoning_effort"),
                response_format=cfg.get("response_format"),
                name=f"rubric_judge_{run_name}",
                strict_scores=cfg.get("strict_scores", False),
                audit_log_dir=cfg.get("audit_log_dir"),
                completion_token_parameter=cfg.get("judge_completion_token_parameter", "max_tokens"),
                extra_body=cfg.get("judge_extra_body"),
            )
            judge_reward_fns = None
            ensemble_judge_weights = None
    elif reward_type == "ensemble_soft_likert":
        # Multi-judge Likert 1-5 with logprob-extracted confidence weights.
        # Reward = Σ (E[rating_j] · confidence_j) / Σ confidence_j , /5 → [0, 1].
        from .reward import make_ensemble_soft_likert_reward
        judges_list = cfg.get("judges")
        if not judges_list:
            raise SystemExit(
                "[train_grpo] reward_type=ensemble_soft_likert requires `judges:` list"
            )
        sl_args = cfg.get("soft_likert_args", {}) or {}
        reward_fn = make_ensemble_soft_likert_reward(
            judges=judges_list,
            default_api_key_env=cfg.get("judge_api_key_env"),
            top_logprobs=sl_args.get("top_logprobs", 10),
            name=f"ensemble_soft_likert_{run_name}",
            scale_max=int(sl_args.get("scale_max", 5)),
        )
        judge_reward_fns = None
        ensemble_judge_weights = None
        print(
            f"[train_grpo] reward_type=ensemble_soft_likert  N_judges={len(judges_list)}  "
            f"top_logprobs={sl_args.get('top_logprobs', 10)}  "
            f"scale_max={sl_args.get('scale_max', 5)}"
        )
    elif reward_type == "ensemble_soft_likert_drop":
        # Soft-Likert + inverse-variance confidence + drop-disagreement mask.
        # Composes drop030's filtering (handles inter-judge disagreement) on top
        # of soft-Likert's confidence weighting (handles intra-judge uncertainty).
        from .reward import make_ensemble_soft_likert_drop_reward
        judges_list = cfg.get("judges")
        if not judges_list:
            raise SystemExit(
                "[train_grpo] reward_type=ensemble_soft_likert_drop requires `judges:` list"
            )
        disagreement_threshold = cfg.get("disagreement_threshold")
        if disagreement_threshold is None:
            raise SystemExit(
                "[train_grpo] reward_type=ensemble_soft_likert_drop requires `disagreement_threshold:`"
            )
        sl_args = cfg.get("soft_likert_args", {}) or {}
        strict = bool(sl_args.get("strict_scores", cfg.get("strict_scores", False)))
        audit_dir = cfg.get("audit_log_dir", cfg.get("judge_log_dir"))
        if strict and not audit_dir:
            # Strict scoring writes per-rollout audit logs; keep them per run
            # (run_name includes the --seed suffix).
            audit_dir = f"{_local('RAR_OUT', 'outputs')}/judge_logs/{run_name}"
        reward_fn = make_ensemble_soft_likert_drop_reward(
            judges=judges_list,
            disagreement_threshold=float(disagreement_threshold),
            num_generations=cfg.get("num_generations", 8),
            default_api_key_env=cfg.get("judge_api_key_env"),
            top_logprobs=sl_args.get("top_logprobs", 5),
            name=f"ensemble_soft_likert_drop_{run_name}",
            scale_max=int(sl_args.get("scale_max", 5)),
            confidence_mode=sl_args.get("confidence_mode", "inverse_var"),
            aggregation=sl_args.get("aggregation", "weighted_avg"),
            normalize_disagreement=bool(sl_args.get("normalize_disagreement", False)),
            reward_judge_tag=sl_args.get("reward_judge_tag"),
            strict_scores=strict,
            rating_mode=sl_args.get("rating_mode", "soft"),
            disagreement_rating_mode=sl_args.get("disagreement_rating_mode", "soft"),
            audit_log_dir=audit_dir,
        )
        judge_reward_fns = None
        ensemble_judge_weights = None
        print(
            f"[train_grpo] reward_type=ensemble_soft_likert_drop  N_judges={len(judges_list)}  "
            f"threshold={disagreement_threshold}  top_logprobs={sl_args.get('top_logprobs', 5)}  "
            f"scale_max={sl_args.get('scale_max', 5)}"
        )
    else:
        raise ValueError(f"unknown reward_type: {reward_type!r} (use 'absolute', 'ensemble_soft_likert', 'ensemble_soft_likert_drop', or 'pairwise')")
    # Assemble reward functions and per-function weights.
    # Multi-judge ensemble: reward_funcs starts as the list of per-judge fns
    # with equal weights summing to 1 (sum-of-weighted == mean). Single-judge:
    # one fn at default weight 1.0 — no reward_weights override needed.
    if 'judge_reward_fns' in locals() and judge_reward_fns:
        reward_funcs = list(judge_reward_fns)
        reward_weights = list(ensemble_judge_weights)
    else:
        reward_funcs = [reward_fn]
        reward_weights = None
    # Optional additive length penalty to combat GRPO length-hacking.
    # Length penalty is in [-max_penalty, 0]; always weight 1.0 so it's
    # added (not averaged) on top of the rubric reward(s).
    if cfg.get("length_penalty", False):
        lp_args = cfg.get("length_penalty_args", {}) or {}
        reward_funcs.append(
            make_length_penalty_reward(
                target_tokens=lp_args.get("target_tokens", 300),
                max_tokens=lp_args.get("max_tokens", 800),
                max_penalty=lp_args.get("max_penalty", 0.5),
                name=f"length_penalty_{run_name}",
            )
        )
        if reward_weights is not None:
            reward_weights.append(1.0)

    max_steps = args.max_steps if args.max_steps is not None else cfg.get("max_steps", 100)

    grpo_args_kwargs = {}
    for key in ("generation_batch_size", "num_iterations", "loss_type", "save_strategy"):
        if key in cfg:
            grpo_args_kwargs[key] = cfg[key]
    if reward_weights is not None:
        grpo_args_kwargs["reward_weights"] = reward_weights
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=cfg.get("learning_rate", 5e-6),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        num_generations=cfg.get("num_generations", 8),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        max_completion_length=cfg.get("max_completion_length", 1024),
        max_steps=max_steps,
        save_steps=cfg.get("save_steps", 50),
        logging_steps=cfg.get("logging_steps", 1),
        bf16=True,
        beta=cfg.get("beta", 0.0),
        scale_rewards=cfg.get("scale_rewards", False),
        use_vllm=cfg.get("use_vllm", True),
        vllm_mode=cfg.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.55),
        vllm_max_model_length=cfg.get(
            "vllm_max_model_length",
            cfg.get("max_completion_length", 1024) * 2 + 256,
        ),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        optim=cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        temperature=cfg.get("temperature", 0.9),
        chat_template_kwargs={"enable_thinking": False},
        report_to=cfg.get("report_to", "wandb"),
        seed=cfg.get("seed", 0),
        data_seed=cfg.get("data_seed", cfg.get("seed", 0)),
        **grpo_args_kwargs,
    )

    if args.debug:
        # GRPO requires per_device_train_batch_size to be a multiple of
        # num_generations (the per-device batch is interpreted as generations,
        # not prompts). With 1 GPU + 4 gens we use 1 prompt per step.
        grpo_args.max_steps = 5
        grpo_args.num_generations = 4
        grpo_args.per_device_train_batch_size = 4
        grpo_args.save_steps = 1000  # don't save during debug

    # Optional LoRA adapter (length-hacking ablation).
    peft_config = None
    if cfg.get("lora", False):
        from peft import LoraConfig
        lcfg = cfg.get("lora_args", {}) or {}
        peft_config = LoraConfig(
            r=lcfg.get("r", 32),
            lora_alpha=lcfg.get("alpha", 64),
            lora_dropout=lcfg.get("dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lcfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
            ),
        )
        print(f"[train_grpo] using LoRA: r={peft_config.r} alpha={peft_config.lora_alpha} "
              f"targets={peft_config.target_modules}")

    trainer = GRPOTrainer(
        model=cfg.get("policy_model", "Qwen/Qwen3-4B"),
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[train_grpo] saved final checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
