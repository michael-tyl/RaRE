# Judging the Judges: Confidence and Disagreement in Rubrics as Rewards

Code for **RaRE** (Rubrics as Rewards with Ensembles). Project page:
https://anneharrington.github.io/judging-the-judges/

[Rubrics as Rewards](https://arxiv.org/abs/2507.17746) (RaR) trains on
non-verifiable domains by having an LLM judge score each rollout against a
rubric. We train Qwen3-4B-Base with GRPO on RaR-Medicine and RaR-Science
under five judges. Judge identity alone moves the final score substantially.
A naive mean of two judges does not beat the best single judge. RaRE does,
by combining two mechanisms:

- **Soft-Likert reward.** Read each judge's top-5 logprobs at the rating
  digit and renormalize over 1–5. This gives an expected rating `E_j` and a
  confidence `c_j = 1 / (1 + Var_j)`. The reward is `Σ c_j E_j / Σ c_j`,
  divided by 5. Judges without logprobs fall back to their integer rating
  (*hard-Likert*).
- **Drop-on-disagreement.** If a rollout's normalized rating spread
  `(max_j r_j − min_j r_j) / (L − 1)` exceeds τ, its reward is replaced by
  the mean of the unmasked rollouts in its GRPO group. Its advantage is then
  zero, so it contributes no gradient.

## Setup

The launchers are Slurm scripts. Training uses one node with 8×80 GB GPUs;
evaluation uses one GPU. Edit the `#SBATCH` lines (partition, time,
resources) in `scripts/slurm/*.sbatch` for your cluster.

```bash
cp .env.example .env         # API keys, HF_TOKEN, WANDB_API_KEY; optional RAR_LOCAL / RAR_STAGE
source scripts/env.sh        # caches and paths (source in every fresh shell)
uv sync --python 3.12
source scripts/env.sh        # again, to activate the new venv
```

`RAR_LOCAL` (default `/tmp/$USER/deep_rl`) holds node-local caches and the
venv. `RAR_STAGE` (default `/workspace/home/$USER/deep_rl_stage`) must be
storage shared by all nodes; final checkpoints and eval JSONs go there.
GPQA Diamond is gated: accept its terms at
huggingface.co/datasets/Idavidrein/gpqa and set `HF_TOKEN` in `.env`.
Submit all jobs from the repo root.

Judges are called over OpenAI-compatible APIs. Each config names its key in
`api_key_env:`:

| Key | Judges |
|---|---|
| `OPENAI_API_KEY` | gpt-4o-mini (training judge and HealthBench grader); GPT-4o (`run_jvr_4o_gpt4o`) |
| `PORTKEY_API_KEY` | Gemini-2.5-flash-lite, gpt-oss-120b, Qwen3-235B-Instruct; DeepSeek-V3.1 in the medicine hard-Likert, Gem+DS and pairwise configs |
| `TOGETHER_API_KEY` | DeepSeek-V3.1 in the medicine soft-Likert and DS+Mini retro configs |
| `DEEPSEEK_API_KEY` | DeepSeek-V3.1 (DeepSeek API) in `run_jvr_4o_dsmini_softlikert_drop035` |
| `DEEPINFRA_API_KEY` | DeepSeek-V3.1 in the science single-judge and Gem+DS configs |
| `OPENROUTER_API_KEY` | DeepSeek-V3.1 and gpt-4o-mini in the science DS+Mini soft-Likert configs and `run_jvr_4o_mini_reward_dsmini_drop030` |

Portkey model names in the configs (`@google/...`, `@togetherai/...`) are
provider slugs from a Portkey workspace; set up matching providers in yours.

## Train

All paper runs use one launcher. The policy uses all 8 GPUs and the judges
are remote.

```bash
sbatch --export=ALL scripts/slurm/run_one_remote_judge.sbatch \
    --config configs/run_jvr_4o_dsmini_softlikert_drop030.yaml --train-gpus 8 [--seed N]
```

`--seed N` overrides the config's seed, which also selects the 1,000-prompt
training subset, and appends `_sN` to the run name for N ≠ 0. Checkpoints are
copied to `$RAR_STAGE/checkpoints/<run_name>/`. LoRA runs are also merged
into a full model at `<run_name>_merged`; evaluate that directory.

Shared recipe: 1,000 training prompts, G = 8 rollouts per prompt, 100 steps,
LR 5e-6 (constant with 10% warmup), β = 0, AdamW-8bit, full fine-tuning, max
completion 1,024 tokens, vLLM colocate. Full hyperparameters are in the
paper's appendix.

| Experiment | Medicine (`configs/run_jvr_4o_*`) | Science (`configs/run_jvr_sci_*`) |
|---|---|---|
| Single judge | `{mini,ds,gem,oss,qw}` | `{mini,ds,gem,oss,qw}` |
| Gem+DS mean / + drop | `gemds_avg`, `gemds_drop030` | `gemds`, `gemds_drop030` |
| DS+Mini soft-Likert | `dsmini_softlikert` | `dsmini_softlikert` |
| **RaRE** (soft + drop) | `dsmini_softlikert_drop030` | `dsmini_softlikert_drop040` |
| τ sweep | Gem+DS: `gemds_avg` (none), `gemds_drop{015,025,030,035}` | DS+Mini: `dsmini_softlikert` (none), `dsmini_softlikert_drop{030,040,045,050}` |
| Soft vs. hard Likert | `dsmini_drop030` (hard) | — |
| Confidence ablation | `dsmini_softlikert_drop030_{toplogit,wta}` | — |
| Length hacking | `mini_lora`, `mini_lenpen`, `gemds_drop030_lora` | — |

Science results are averaged over five seeds: run each science config with
`--seed 0` through `--seed 4`. The science DS+Mini soft-Likert configs and
`gemds_drop030` use 8 rollouts per GPU, so each GRPO group is scored in one
reward call (which the drop mask needs); launch them with `--train-gpus 4`,
giving the same 32 rollouts per step as the other science configs. In the
science DS+Mini configs DeepSeek is a hard judge (one-hot rating, confidence 1)
and gpt-4o-mini is soft.

The reward is selected by `reward_type:` (implemented in `src/rar/reward.py`):

- `absolute` (default): a single judge. With a `judges:` list, it averages
  the judges; adding `disagreement_threshold:` also applies the drop mask.
- `ensemble_soft_likert`: soft-Likert ensemble.
- `ensemble_soft_likert_drop`: RaRE. `soft_likert_args.confidence_mode` and
  `.aggregation` select the ablation variants.
- `pairwise`: pairwise preference judge (appendix baseline).

## Evaluate

**HealthBench (medicine).** Graded per criterion by gpt-4o-mini with the
simple-evals template. The paper reports the English-only subset (n = 3,931).

```bash
sbatch --export=ALL scripts/slurm/healthbench_remote_judge.sbatch \
    --subset oss --n 5000 --max-tokens 8192 <checkpoints...>
python scripts/hb5k_split_by_language.py $RAR_STAGE/eval/hb_pc_jgpt-4o-mini_mt8192_*.json
```

The grader defaults to gpt-4o-mini via the OpenAI API (`--judge-url`,
`--judge-model` and `--judge-api-key-env` override it). The second command
prints each checkpoint's English-only score with a 95% bootstrap CI, and
warns if any English prompts are missing (e.g. after grader failures).

Ablations use the HealthBench 1K subset (1,000 English prompts, stratified by
theme):

```bash
python scripts/build_hb_balanced1k.py   # -> $RAR_STAGE/data/healthbench_oss_en_balanced1000_seed42.json
sbatch --export=ALL scripts/slurm/healthbench_remote_judge.sbatch \
    --prompt-ids-file $RAR_STAGE/data/healthbench_oss_en_balanced1000_seed42.json --max-tokens 1024 <checkpoints...>
```

**GPQA Diamond (science).** Multiple-choice accuracy, so no judge is needed.

```bash
sbatch --export=ALL scripts/slurm/gpqa_eval.sbatch --max-tokens 8192 <checkpoints...>
python scripts/holdout_tau_select.py
```

`holdout_tau_select.py` splits GPQA Diamond 50/50 (split seed 0). It picks τ
by the 5-seed mean on the validation half and reports per-seed scores on the
test half. `eval_gpqa.py` scores answers with a corrected parser (the last
explicit `Answer: X`); `--legacy-answer-parser` reproduces the original
parser, and both accuracies are saved in the output JSON.

## Appendix experiments

All medicine; train with `run_one_remote_judge.sbatch` unless noted.
"1K" means evaluated on the HealthBench 1K subset.

| Appendix | Configs (`configs/run_jvr_4o_*`) |
|---|---|
| Single-judge reward with two-judge drop | `mini_reward_dsmini_drop030` |
| Training cost (GPT-4o single judge) | `gpt4o` |
| Soft-Likert scale, L = 7 vs. 5 (1K) | `dsmini_softlikert7` vs. `dsmini_softlikert` |
| DS+Mini soft-Likert τ sweep | `dsmini_softlikert` (none), `dsmini_softlikert_drop{025,030,035}` |
| LoRA vs. full fine-tuning (1K) | `{mini,ds}_lora`, `gemds_drop030_lora` vs. `mini`, `ds`, `gemds_drop030` |
| Pairwise baselines (1K) | `{mini,ds,gem}_pairwise`, `gem_pairwise_norubric` |
| Retro-GRPO-aux (1K; LoRA) | `dsmini_likert_retro_k{10,20,50}`, `gemds_likert_retro_k{10,20,50}_lora` |

Retro-GRPO-aux runs use their own trainer:

```bash
sbatch --export=ALL scripts/slurm/run_likert_retro.sbatch \
    --config configs/run_jvr_4o_dsmini_likert_retro_k10.yaml --train-gpus 8
```

**Per-theme and per-category breakdowns.** Add `--stratify-by theme` and/or
`--stratify-by physician_agreed_category` to the HealthBench eval.

**Independent evaluator (GLM-5.3-Flash).** Regrade the saved responses
(`--resume-from-completions` reuses the completions cache from the gpt-4o-mini
run) with the same per-criterion prompt:

```bash
sbatch --export=ALL scripts/slurm/healthbench_remote_judge.sbatch \
    --subset oss --n 5000 --max-tokens 8192 --resume-from-completions \
    --judge-url https://api.deepinfra.com/v1/openai --judge-model zai-org/GLM-5.3-Flash \
    --judge-api-key-env DEEPINFRA_API_KEY \
    --judge-extra-body '{"reasoning_effort": "low"}' --judge-max-tokens 4096 --judge-timeout 120 \
    <checkpoints...>
```

**GPQA answer-order permutations.** Evaluate each policy on ten answer
orderings (`--seed 0` is the default order) and aggregate (mean ± 2 SE):

```bash
for k in 0 1 2 3 4 5 6 7 8 9; do
    sbatch --export=ALL scripts/slurm/gpqa_eval.sbatch --max-tokens 8192 --seed $k <checkpoints...>
done
python scripts/gpqa_permutations.py <checkpoint basenames...>
```

## Results (HealthBench, English-only)

| Training signal | Score | 95% CI |
|---|---:|---|
| Qwen3-4B-Base (no RL) | 0.257 | [0.246, 0.267] |
| Qwen3-4B (tuned reference) | 0.308 | [0.297, 0.318] |
| gemini-2.5-flash-lite | 0.296 | [0.286, 0.307] |
| qwen3-235B-Instruct | 0.309 | [0.299, 0.320] |
| gpt-oss-120b | 0.322 | [0.312, 0.332] |
| deepseek-v3.1 | 0.338 | [0.328, 0.349] |
| gpt-4o-mini | 0.339 | [0.329, 0.349] |
| Gem+DS mean | 0.307 | [0.297, 0.317] |
| Gem+DS mean + drop | 0.347 | [0.337, 0.357] |
| DS+Mini soft-Likert | 0.330 | [0.320, 0.339] |
| **RaRE (DS+Mini soft + drop)** | **0.364** | [0.353, 0.374] |

See the paper for the GPQA Diamond results.

## Layout

```
configs/                 run configs (table above)
scripts/slurm/           training and eval launchers
scripts/                 eval post-processing and analysis
src/rar/train_grpo.py    GRPO entry point (TRL + vLLM)
scripts/likert_retro_train.py  retro-GRPO-aux trainer (appendix)
src/rar/reward.py        judge rewards: single, mean/drop, soft-Likert, RaRE
src/rar/prompts.py       judge prompts and HealthBench grader template
src/rar/data.py          RaR-Medicine / RaR-Science loaders
src/rar/domains/         medicine and science domain registry
src/rar/eval_*.py        HealthBench and GPQA evaluation
src/rar/audit.py         opt-in strict judge validation and audit logs
src/rar/cost_audit.py    (`strict_scores`, `audit_log_dir` in a config)
```

## Notes

- Soft-Likert needs judge logprobs. DeepSeek-V3.1 served by Together returns them, but this endpoint is no longer supported.
