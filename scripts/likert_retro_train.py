"""Likert retro-GRPO-aux trainer (paper appendix "Retro-GRPO-aux"; null result).

Launch with scripts/slurm/run_likert_retro.sbatch --config configs/run_jvr_4o_*_retro_*.yaml.

Composes:
  1. drop030-style Likert disagreement masking (the main loss path)
  2. Soft-Likert + inverse-variance confidence weighting
  3. Retrospective GRPO-aux loss on the masked rollouts (option A)

The disagreement rollouts that drop030 masks (advantage=0 in main loss) are
NOT discarded — they're harvested with cached lp_old of the rollout itself
AND its K-1 group siblings, then at step T+k:

  1. Forward the cached completions through current policy → lp_new for the
     disagreement rollout and all K-1 sibling rollouts.
  2. Compute Δlp = lp_new − lp_old per rollout.
  3. Rank the disagreement rollout's Δlp within the K-rollout group.
  4. Retro-label: if Δlp rank is high (policy "promoted" this rollout),
     use the HIGHER judge's E[rating]; if low, use the LOWER one.
     Smooth version: α = rank/(K-1) ∈ [0,1]; retro_E = α·max_E + (1-α)·min_E.
  5. Compute GRPO-aux loss for this rollout: standard PPO-clip objective with
     ratio = exp(lp_new − lp_old) and advantage = (retro_E − mean_E)/5, where
     mean_E is the mean of the judges' E[rating] for this rollout.
     This is "option A" — same numerical safeguards as the main GRPO loss.
  6. Total = main_grpo_loss + retro_lambda * retro_aux_loss.

End-of-training flush: process still-pending buckets oldest first, skipping
those with drift < min_drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure src/ is importable when running standalone.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Module-level deferred buffer: keyed by step_idx → list[pending_entries]
# (each entry has prompt_ids/comp_ids/lp_old/sibling_lp_olds + per-judge scores)
DISAGREEMENT_BUFFER: dict[int, list[dict[str, Any]]] = {}

# All retro-labeled rollouts for end-of-training output.
RETRO_LABELED: list[dict[str, Any]] = []


def _pad_concat(prompt_ids: torch.Tensor, comp_ids: torch.Tensor) -> torch.Tensor:
    return torch.cat([prompt_ids, comp_ids], dim=0).unsqueeze(0)


def build_trainer_subclass(
    retro_k: int = 25,
    retro_lambda: float = 0.5,
    retro_clip_ratio: float = 0.2,
    smooth_retro_label: bool = True,
    retro_loss_type: str = "ppo_clip",
    retro_beta_dpo: float = 0.1,
    retro_max_comp_length: int = 0,
):
    """Subclass GRPOTrainer to add retro-GRPO-aux loss on drop030-masked rollouts.

    Returns the class; instantiate with the same args as GRPOTrainer.
    """
    from trl import GRPOTrainer

    class LikertRetroTrainer(GRPOTrainer):
        _retro_k = retro_k
        _retro_lambda = retro_lambda
        _retro_clip = retro_clip_ratio
        _smooth_label = smooth_retro_label
        _retro_loss_type = retro_loss_type
        _retro_beta_dpo = retro_beta_dpo
        _retro_max_comp_len = retro_max_comp_length

        def _compute_logp_single(self, model, prompt_ids: torch.Tensor, comp_ids: torch.Tensor) -> torch.Tensor:
            """Forward (prompt+comp) under current model state, return sum_logp of comp.

            Bypasses DDP wrapping (uses model.module): the number of retro
            forwards differs across ranks, which DDP's collective hooks can't
            handle.
            """
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            device = next(model.parameters()).device
            inner = model.module if hasattr(model, "module") else model
            # Optional truncation: score only the last N tokens of the completion.
            # Applied identically at harvest (lp_old) and retro-fire (lp_new) so
            # Δlogπ remains apples-to-apples. Cuts activation memory ~1 - N/L on
            # all K forwards per pair, enabling higher num_generations.
            if self._retro_max_comp_len > 0 and comp_ids.size(0) > self._retro_max_comp_len:
                comp_ids = comp_ids[-self._retro_max_comp_len:]
            full = _pad_concat(prompt_ids, comp_ids).to(device)
            attn = (full != pad_id).long()
            # NOTE: explicit torch.utils.checkpoint.checkpoint wrap (both
            # reentrant=True AND reentrant=False) hits a SavedTensorHooks
            # INTERNAL ASSERT under HF's gradient_checkpointing. Memory must
            # be controlled at the per_device + num_generations level instead.
            out = inner(input_ids=full, attention_mask=attn).logits
            logp = torch.log_softmax(out.float(), dim=-1)
            n_p = prompt_ids.size(0)
            tgt = comp_ids.to(device)
            sel = logp[0, n_p - 1 : n_p - 1 + tgt.size(0)].gather(1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = (tgt != pad_id).float()
            return (sel * mask).sum()

        def _maybe_harvest_pending(self, model, inputs: dict[str, torch.Tensor], step_idx: int) -> None:
            """For each rollout in this batch, check if it was masked by drop030
            (lives in PENDING_LIKERT_DISAGREEMENTS). If so, cache its lp_old AND
            the lp_old of all K-1 sibling rollouts in the same group, then move
            into DISAGREEMENT_BUFFER[step_idx] for retro processing at T+k.
            """
            from rar.reward import PENDING_LIKERT_DISAGREEMENTS

            K = self.num_generations
            comp_ids = inputs["completion_ids"]
            comp_mask = inputs["completion_mask"]
            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            N = comp_ids.size(0)
            if N % K != 0 or not PENDING_LIKERT_DISAGREEMENTS:
                return

            # Build hash -> batch index mapping for this batch.
            tok = self.processing_class if hasattr(self, "processing_class") else self.tokenizer
            batch_hashes: list[str] = []
            for n in range(N):
                clen = int(comp_mask[n].sum().item())
                text = tok.decode(comp_ids[n, :clen].tolist(), skip_special_tokens=True)
                batch_hashes.append(hashlib.sha1(text.encode("utf-8")).hexdigest())
            hash_to_idx = {h: i for i, h in enumerate(batch_hashes)}

            harvested: list[dict[str, Any]] = []
            for h in list(PENDING_LIKERT_DISAGREEMENTS.keys()):
                if h not in hash_to_idx:
                    continue
                # Peek first — verify all siblings are in this batch BEFORE
                # popping. If a sibling is missing (rare; DDP batches groups
                # together), leave the entry in PENDING so it can be retried
                # when a future batch contains the full group.
                entry = PENDING_LIKERT_DISAGREEMENTS[h]
                if any(sh not in hash_to_idx for sh in entry["sibling_hashes"]):
                    continue
                # All siblings present — commit the pop.
                PENDING_LIKERT_DISAGREEMENTS.pop(h)
                idx = hash_to_idx[h]
                grp_start = (idx // K) * K
                # Cache lp_old for ALL K rollouts in this group (no_grad).
                pair_data = {
                    "question": entry["question"],
                    "response": entry["response"],
                    "per_judge_E": entry["per_judge_E"],
                    "per_judge_conf": entry["per_judge_conf"],
                    "judge_tags": entry["judge_tags"],
                    "disagreement": entry["disagreement"],
                    "judged_at_step": step_idx,
                    "own_idx_in_group": entry["own_idx_in_group"],
                    "prompt_ids": None,
                    "group_comp_ids": [],
                    "group_lp_old": [],
                }
                p_len = int(prompt_mask[grp_start].sum().item())
                pair_data["prompt_ids"] = prompt_ids[grp_start, :p_len].detach().cpu().clone()
                with torch.no_grad():
                    for k in range(K):
                        nb = grp_start + k
                        clen = int(comp_mask[nb].sum().item())
                        cids = comp_ids[nb, :clen].detach().cpu().clone()
                        pair_data["group_comp_ids"].append(cids)
                        lp = self._compute_logp_single(model, pair_data["prompt_ids"], cids)
                        pair_data["group_lp_old"].append(float(lp.item()))
                harvested.append(pair_data)
            if harvested:
                DISAGREEMENT_BUFFER.setdefault(step_idx, []).extend(harvested)

        def _retro_label(self, pair: dict, group_delta_lp: list[float]) -> float:
            """Given Δlp for all K rollouts in the group, return the retro-
            labeled E[rating] for this disagreement rollout.

            Mechanism: rank the disagreement rollout's Δlp within the group.
            High rank → policy "promoted" this rollout → trust the HIGHER judge.
            """
            own = pair["own_idx_in_group"]
            order = sorted(range(len(group_delta_lp)), key=lambda i: group_delta_lp[i])
            rank = order.index(own)  # 0 = lowest Δlp, K-1 = highest
            K = len(group_delta_lp)
            max_E = max(pair["per_judge_E"])
            min_E = min(pair["per_judge_E"])
            if self._smooth_label:
                alpha = rank / max(1, K - 1)
                return alpha * max_E + (1.0 - alpha) * min_E
            else:
                return max_E if rank >= K // 2 else min_E

        def _retro_grpo_aux_loss(self, model, step_idx: int) -> torch.Tensor:
            """For each unprocessed bucket aged exactly retro_k steps, compute
            the GRPO-aux loss using PPO-clipped importance correction.

            Returns scalar; 0 if nothing to do.
            """
            device = next(model.parameters()).device
            if step_idx < self._retro_k:
                return torch.tensor(0.0, device=device)
            target_step = step_idx - self._retro_k
            pairs = DISAGREEMENT_BUFFER.pop(target_step, None)
            if not pairs:
                return torch.tensor(0.0, device=device)
            return self._retro_loss_from_pairs(model, pairs, step_idx, target_step)

        def _retro_loss_from_pairs(
            self,
            model,
            pairs: list[dict],
            step_idx: int,
            queued_at_step: int,
        ) -> torch.Tensor:
            """Inner loop reused by both in-training retro and end-of-training flush."""
            device = next(model.parameters()).device
            total = torch.tensor(0.0, device=device)
            n_used = 0
            for pair in pairs:
                K = len(pair["group_comp_ids"])
                # Compute lp_new for all K rollouts. Only the disagreement
                # rollout gets grad; siblings are no_grad (we just need Δlp for ranking).
                lp_new_all: list[float] = []
                lp_new_self: torch.Tensor | None = None
                own = pair["own_idx_in_group"]
                for k in range(K):
                    cids = pair["group_comp_ids"][k]
                    if k == own:
                        lp_new_self = self._compute_logp_single(model, pair["prompt_ids"], cids)
                        lp_new_all.append(float(lp_new_self.item()))
                    else:
                        with torch.no_grad():
                            lp_k = self._compute_logp_single(model, pair["prompt_ids"], cids)
                        lp_new_all.append(float(lp_k.item()))
                if lp_new_self is None:
                    continue
                # Δlp per rollout.
                group_delta = [lp_new_all[k] - pair["group_lp_old"][k] for k in range(K)]
                retro_E = self._retro_label(pair, group_delta)
                # Group baseline = mean of per-judge mean E across all K rollouts.
                # For non-masked siblings we don't have per-rollout E here, so we
                # use this rollout's per-judge mean as the group baseline proxy.
                # (Tighter baseline would require caching all K rollouts' scores
                # too.)
                rollout_E_proxy = sum(pair["per_judge_E"]) / len(pair["per_judge_E"])
                advantage = (retro_E - rollout_E_proxy) / 5.0  # normalize to [-1, 1] scale
                lp_diff = lp_new_self - pair["group_lp_old"][own]
                if self._retro_loss_type == "dpo":
                    # DPO-style logsigmoid: bounded above by 0, bounded grad by
                    # retro_beta_dpo * |advantage|. Avoids the explosive PPO
                    # ratio that NaN'd the unbounded clip path.
                    # advantage > 0 → push lp_new up; advantage < 0 → push down.
                    pref_logit = advantage * lp_diff
                    pair_loss = -torch.nn.functional.logsigmoid(self._retro_beta_dpo * pref_logit)
                    ratio_for_log = torch.exp(lp_diff.detach())
                else:
                    # PPO clip ratio: ratio = exp(lp_new - lp_old) on token-sum-logp.
                    ratio = torch.exp(lp_diff)
                    ratio_clipped = torch.clamp(ratio, 1.0 - self._retro_clip, 1.0 + self._retro_clip)
                    surr1 = ratio * advantage
                    surr2 = ratio_clipped * advantage
                    pair_loss = -torch.min(surr1, surr2)
                    ratio_for_log = ratio
                total = total + pair_loss
                n_used += 1
                RETRO_LABELED.append({
                    "step": step_idx,
                    "queued_at_step": queued_at_step,
                    "question": pair["question"],
                    "response": pair["response"],
                    "per_judge_E": pair["per_judge_E"],
                    "judge_tags": pair["judge_tags"],
                    "group_delta_lp": group_delta,
                    "retro_E": retro_E,
                    "advantage": advantage,
                    "ratio": float(ratio_for_log.item()),
                })
            if n_used == 0:
                return torch.tensor(0.0, device=device)
            return total / n_used

        def flush_remaining_retro(self, n_flush_steps: int = 10, min_drift: int = 10) -> None:
            """End-of-training flush: process remaining buckets, oldest first."""
            cur = int(self.state.global_step)
            candidates = sorted(
                k for k in DISAGREEMENT_BUFFER if (cur - k) >= min_drift
            )[:n_flush_steps]
            if not candidates:
                print(f"[likert-retro-flush] no buckets to process (drift>={min_drift})", flush=True)
                return
            model = self.model_wrapped
            model.train()
            print(f"[likert-retro-flush] processing {len(candidates)} buckets, drifts={[cur-k for k in candidates]}", flush=True)
            for k in candidates:
                pairs = DISAGREEMENT_BUFFER.pop(k)
                self.optimizer.zero_grad()
                loss = self._retro_loss_from_pairs(model, pairs, step_idx=cur, queued_at_step=k)
                if not loss.requires_grad:
                    continue
                self.accelerator.backward(loss)
                self.optimizer.step()
                print(f"[likert-retro-flush] bucket@{k} drift={cur-k} n_pairs={len(pairs)} loss={float(loss.item()):.4f}", flush=True)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            step_idx = int(self.state.global_step)
            # 1. Harvest masked rollouts that landed in this batch.
            self._maybe_harvest_pending(model, inputs, step_idx)
            # 2. Standard GRPO main loss (the reward fn already masked
            #    disagreement rollouts to group mean → their main-loss
            #    advantage is ~0).
            grpo_loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
            # 3. Retro GRPO-aux on rollouts harvested k steps ago.
            aux = self._retro_grpo_aux_loss(model, step_idx)
            total = grpo_loss + self._retro_lambda * aux
            # 4. Log.
            mode = "train" if model.training else "eval"
            try:
                self._metrics[mode].setdefault("retro/aux_loss", []).append(float(aux.detach().item()))
                from rar.reward import PENDING_LIKERT_DISAGREEMENTS
                self._metrics[mode].setdefault("retro/pending_buffer_size", []).append(float(len(PENDING_LIKERT_DISAGREEMENTS)))
                self._metrics[mode].setdefault("retro/buffer_queued_steps", []).append(float(len(DISAGREEMENT_BUFFER)))
                self._metrics[mode].setdefault("retro/labels_so_far", []).append(float(len(RETRO_LABELED)))
            except Exception:
                pass
            return total

    return LikertRetroTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_name = cfg["run_name"]
    output_dir = os.path.join(os.environ.get("RAR_CKPT", "/tmp/ckpt"), run_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Reward + dataset setup uses our existing train_grpo path for parity.
    from rar.reward import make_ensemble_soft_likert_drop_reward
    from rar.train_grpo import build_dataset  # type: ignore

    # Hyperparameters.
    retro_k = int(cfg.get("retro_k", 25))
    retro_lambda = float(cfg.get("retro_lambda", 0.5))
    retro_clip = float(cfg.get("retro_clip_ratio", 0.2))
    smooth_label = bool(cfg.get("smooth_retro_label", True))
    retro_loss_type = str(cfg.get("retro_loss_type", "ppo_clip"))
    retro_beta_dpo = float(cfg.get("retro_beta_dpo", 0.1))
    retro_max_comp_length = int(cfg.get("retro_max_comp_length", 0))
    print(f"[likert-retro] retro_k={retro_k}  λ={retro_lambda}  clip={retro_clip}  smooth={smooth_label}  loss_type={retro_loss_type}  β_dpo={retro_beta_dpo}  max_comp_len={retro_max_comp_length}", flush=True)

    # Dataset.
    dataset = build_dataset(cfg)
    print(f"[likert-retro] dataset size = {len(dataset)}", flush=True)

    # Reward fn — populate_retro_pending=True so masked rollouts get
    # written to PENDING_LIKERT_DISAGREEMENTS for the trainer.
    judges = cfg.get("judges")
    assert judges, "config must specify `judges:` list"
    threshold = float(cfg["disagreement_threshold"])
    sl_args = cfg.get("soft_likert_args", {}) or {}
    reward_fn = make_ensemble_soft_likert_drop_reward(
        judges=judges,
        disagreement_threshold=threshold,
        num_generations=cfg.get("num_generations", 8),
        default_api_key_env=cfg.get("judge_api_key_env"),
        top_logprobs=sl_args.get("top_logprobs", 5),
        name=f"likert_retro_{run_name}",
        populate_retro_pending=True,
    )

    # GRPO config.
    from trl import GRPOConfig
    max_steps = args.max_steps or cfg.get("max_steps", 100)
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
        vllm_gpu_memory_utilization=cfg.get("vllm_gpu_memory_utilization", 0.12),
        vllm_max_model_length=cfg.get("vllm_max_model_length", 2304),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "adamw_bnb_8bit"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant_with_warmup"),
        temperature=cfg.get("temperature", 1.0),
        chat_template_kwargs={"enable_thinking": False},
        report_to=cfg.get("report_to", "wandb"),
        seed=cfg.get("seed", 0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        # Rank-asymmetric retro_aux is handled via a zero-grad anchor
        # in _retro_grpo_aux_loss (every rank touches model params each
        # step, even when no buckets are due). This avoids the memory cost
        # of ddp_find_unused_parameters=True, which OOM'd at the retro
        # boundary step.
    )

    # LoRA (off unless `lora: true`).
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
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        print(f"[likert-retro] LoRA: r={peft_config.r} α={peft_config.lora_alpha}", flush=True)

    TrainerCls = build_trainer_subclass(
        retro_k=retro_k,
        retro_lambda=retro_lambda,
        retro_clip_ratio=retro_clip,
        smooth_retro_label=smooth_label,
        retro_loss_type=retro_loss_type,
        retro_beta_dpo=retro_beta_dpo,
        retro_max_comp_length=retro_max_comp_length,
    )
    trainer = TrainerCls(
        model=cfg.get("policy_model", "Qwen/Qwen3-4B"),
        args=grpo_args,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    print(f"[likert-retro] starting training: {max_steps} steps; k={retro_k}", flush=True)
    trainer.train()

    n_flush = int(cfg.get("n_flush_steps", 0) or 0)
    if n_flush > 0:
        min_drift = int(cfg.get("flush_min_drift", 10))
        trainer.flush_remaining_retro(n_flush_steps=n_flush, min_drift=min_drift)

    trainer.save_model(output_dir)
    print(f"[likert-retro] saved adapter to {output_dir}", flush=True)

    out_jsonl = Path(output_dir) / "retro_labeled.jsonl"
    with open(out_jsonl, "w") as f:
        for row in RETRO_LABELED:
            f.write(json.dumps(row) + "\n")
    print(f"[likert-retro] wrote {len(RETRO_LABELED)} retro-labeled rollouts", flush=True)


if __name__ == "__main__":
    main()
