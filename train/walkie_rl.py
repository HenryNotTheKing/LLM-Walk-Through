"""Walkie GRPO/DAPO entrypoint with colocated rollout abstraction."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from core.model.walkie import WalkieConfig, WalkieForCausalLM
from core.utils.config import load_config
from core.utils.device import amp_enabled, select_device, select_dtype
from core.utils.distributed import cleanup_distributed, setup_distributed
from core.utils.walkie_checkpoint import apply_walkie_checkpoint, load_walkie_checkpoint, resolve_resume_path, save_walkie_checkpoint, unwrap_model
from core.utils.walkie_optim import build_walkie_optimizers
from posttrain.data.chat_template import ChatTemplate, normalize_messages
from posttrain.rewards.code_execution import SandboxCodeRewardRunner
from posttrain.rewards.registry import RewardConfig, RewardInput, RewardScore, build_reward_fn
from posttrain.rl.algorithms.dapo import apply_overlong_penalty, dapo_group_filter, dapo_policy_loss
from posttrain.rl.algorithms.grpo import compute_group_advantages, grpo_policy_loss
from posttrain.rl.logprobs import causal_lm_logprobs
from posttrain.rollout.base import SamplingConfig
from posttrain.rollout.fake import FakeRolloutEngine
from posttrain.rollout.torch_engine import TorchRolloutEngine
from posttrain.rollout.vllm_engine import RemoteVLLMRolloutEngine, VLLMRolloutEngine
from posttrain.sandbox.jupyter_client import JupyterSandboxClient
from posttrain.utils.hf_export import export_walkie_to_hf
from posttrain.utils.schedule import WarmupDecaySchedule, apply_lrs


class TokenizerAdapter:
    def __init__(self, tokenizer: Any, *, eos_token: str, pad_token: str) -> None:
        self.tokenizer = tokenizer
        self.eos_token_id = int(tokenizer.token_to_id(eos_token))
        self.pad_token_id = int(tokenizer.token_to_id(pad_token))

    def encode(self, text: str) -> list[int]:
        return [int(item) for item in self.tokenizer.encode(text).ids]

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode([int(item) for item in token_ids]))


@dataclass(frozen=True)
class PromptExample:
    text: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--init-from", default=None, type=str)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    dist_info = setup_distributed(str(getattr(cfg.distributed, "backend", "none")))
    try:
        train(cfg, resume=args.resume, init_from=args.init_from, dist_info=dist_info)
    finally:
        cleanup_distributed()


def train(cfg, *, resume: str | None, init_from: str | None, dist_info) -> None:
    started_at = time.time()
    train_cfg = cfg.train
    rl_cfg = cfg.rl
    seed = int(getattr(train_cfg, "seed", 42)) + int(dist_info.rank)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = select_device(str(train_cfg.device))
    dtype = select_dtype(str(train_cfg.dtype), device)
    use_amp = amp_enabled(device, dtype, bool(train_cfg.amp))
    if device.type == "cuda":
        torch.cuda.set_device(dist_info.local_rank)
        torch.set_float32_matmul_precision("high")

    tokenizer = _load_tokenizer(cfg.data.tokenizer_path, cfg.data.eos_token, cfg.data.pad_token)
    template = _resolve_template(tokenizer, cfg.data)
    prompts = _load_prompts(cfg.data, template)
    if not prompts:
        raise ValueError("RL data must provide at least one prompt")
    prompt_cursor = 0

    model_cfg_dict = _plain_cfg_dict(cfg.model)
    model_cfg_dict.setdefault("block_size", int(train_cfg.block_size))
    model_cfg_dict["gradient_checkpointing"] = bool(train_cfg.gradient_checkpointing)
    model_cfg = WalkieConfig.from_dict(model_cfg_dict)
    actor = WalkieForCausalLM(model_cfg).to(device)
    needs_ref_model = str(rl_cfg.algorithm) != "dapo"
    ref_device = torch.device("cpu") if str(cfg.ref.get("offload", "none")) == "cpu" else device
    ref_model = WalkieForCausalLM(model_cfg).to(ref_device) if needs_ref_model else None
    if ref_model is not None:
        ref_model.eval().requires_grad_(False)

    if init_from is not None:
        payload = load_walkie_checkpoint(resolve_resume_path(init_from), map_location="cpu", expected_model_cfg=model_cfg.to_dict())
        apply_walkie_checkpoint(payload, model=actor, optimizers=None, scaler=None, restore_rng=False)
        if ref_model is not None:
            apply_walkie_checkpoint(payload, model=ref_model, optimizers=None, scaler=None, restore_rng=False)
    if cfg.ref.get("checkpoint") is not None and ref_model is not None:
        ref_payload = load_walkie_checkpoint(resolve_resume_path(cfg.ref.checkpoint), map_location="cpu", expected_model_cfg=model_cfg.to_dict())
        apply_walkie_checkpoint(ref_payload, model=ref_model, optimizers=None, scaler=None, restore_rng=False)

    raw_actor = unwrap_model(actor)
    optimizers = build_walkie_optimizers(raw_actor, adamw_lr=float(train_cfg.adamw.peak_lr), muon_lr=float(train_cfg.muon.peak_lr))
    schedule = WarmupDecaySchedule.from_config(
        total_steps=int(train_cfg.total_steps),
        warmup_steps=int(train_cfg.warmup_steps),
        decay_shape=str(train_cfg.decay_shape),
        tracks={
            "adamw": {"peak_lr": float(train_cfg.adamw.peak_lr), "final_lr": float(train_cfg.adamw.final_lr)},
            "muon": {"peak_lr": float(train_cfg.muon.peak_lr), "final_lr": float(train_cfg.muon.final_lr)},
        },
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))
    start_step = 0
    resume_extra: dict[str, Any] | None = None
    if resume is not None:
        payload = load_walkie_checkpoint(resolve_resume_path(resume), map_location="cpu", expected_model_cfg=model_cfg.to_dict())
        info = apply_walkie_checkpoint(payload, model=actor, optimizers=optimizers, scaler=scaler, restore_rng=True)
        start_step = int(info.get("step", 0))
        if payload.get("schedule"):
            schedule.load_state_dict(payload["schedule"])
        resume_extra = payload.get("extra", {})
        prompt_cursor = int(resume_extra.get("data_state", {}).get("prompt_cursor", 0))

    reward_fn = build_reward_fn([RewardConfig(**dict(item)) for item in cfg.rewards])
    sandbox_runner = _build_sandbox_runner(cfg)
    rollout_backend = str(cfg.rollout.backend)
    rollout_engine = None
    if rollout_backend != "vllm" and not (rollout_backend == "remote_vllm" and dist_info.enabled and not dist_info.is_main):
        rollout_engine = _build_rollout_engine(cfg, raw_actor, model_cfg, tokenizer, device=device, dtype=dtype, use_amp=use_amp)
    sampling = SamplingConfig(
        num_generations=int(rl_cfg.num_generations),
        temperature=float(rl_cfg.temperature),
        top_p=float(rl_cfg.top_p),
        max_tokens=int(rl_cfg.max_completion_length),
        stop=list(rl_cfg.get("stop", [])),
        seed=int(train_cfg.seed),
    )
    logprob_micro_batch_size = int(rl_cfg.get("logprob_micro_batch_size", 1))
    out_dir = Path(train_cfg.out_dir)
    if dist_info.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    swanlab_run = None
    resume_swanlab_run_id = _swanlab_resume_run_id(resume_extra)
    if dist_info.is_main:
        swanlab_run = _init_swanlab_run(
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            out_dir=out_dir,
            resume_path=resume,
            resume_run_id=resume_swanlab_run_id,
        )
    latest_stats: dict[str, Any] = {}
    total_steps = int(train_cfg.total_steps)
    stop_step = _resolve_stop_step(train_cfg, total_steps=total_steps)
    if stop_step < start_step:
        raise RuntimeError(f"train.stop_step={stop_step} is smaller than resumed step={start_step}")

    def checkpoint_extra() -> dict[str, Any]:
        return _build_checkpoint_extra(
            rl_cfg=rl_cfg,
            prompt_cursor=prompt_cursor,
            latest_stats=latest_stats,
            swanlab_run=swanlab_run,
            resume_swanlab_run_id=resume_swanlab_run_id,
        )

    async_rollout_prefetch = bool(cfg.rollout.get("async_prefetch", False)) and rollout_backend == "remote_vllm" and not dist_info.enabled
    prefetch_executor = ThreadPoolExecutor(max_workers=1) if async_rollout_prefetch else None
    prefetch_future: Future[list[Any]] | None = None
    prefetch_batch_examples: list[PromptExample] | None = None
    prefetch_next_cursor: int | None = None
    prefetch_step: int | None = None

    def next_prompt_batch_from(cursor: int) -> tuple[list[PromptExample], int]:
        global_batch, next_cursor = _next_prompt_batch(prompts, cursor, int(rl_cfg.prompt_batch_size))
        return _shard_prompt_batch(global_batch, dist_info), next_cursor

    def run_rollout(batch_examples: list[PromptExample]) -> list[Any]:
        assert rollout_engine is not None
        batch_prompts = [example.text for example in batch_examples]
        return rollout_engine.generate(batch_prompts, sampling)

    def score_rollout(outputs: list[Any], batch_examples: list[PromptExample]) -> list[RewardScore]:
        reward_inputs = _build_reward_inputs(outputs, batch_examples)
        if sandbox_runner is not None:
            reward_inputs = asyncio.run(sandbox_runner.evaluate(reward_inputs))
        return reward_fn(reward_inputs)

    def submit_rollout_prefetch(target_step: int, cursor: int) -> None:
        nonlocal prefetch_future, prefetch_batch_examples, prefetch_next_cursor, prefetch_step
        assert prefetch_executor is not None
        batch_examples, next_cursor = next_prompt_batch_from(cursor)
        prefetch_batch_examples = batch_examples
        prefetch_next_cursor = next_cursor
        prefetch_step = int(target_step)
        prefetch_future = prefetch_executor.submit(run_rollout, batch_examples)

    for step in range(start_step, stop_step):
        lrs = schedule.step_to(step)
        apply_lrs(optimizers, lrs)
        if async_rollout_prefetch:
            if prefetch_future is not None:
                if prefetch_step != step:
                    raise RuntimeError(f"rollout prefetch step mismatch: expected {step}, got {prefetch_step}")
                assert prefetch_batch_examples is not None and prefetch_next_cursor is not None
                batch_examples = prefetch_batch_examples
                rollout_outputs = prefetch_future.result()
                prompt_cursor = prefetch_next_cursor
                prefetch_future = None
                prefetch_batch_examples = None
                prefetch_next_cursor = None
                prefetch_step = None
            else:
                batch_examples, prompt_cursor = next_prompt_batch_from(prompt_cursor)
                rollout_outputs = run_rollout(batch_examples)
            if step + 1 < stop_step and not _remote_vllm_sync_due(cfg, step + 1):
                submit_rollout_prefetch(step + 1, prompt_cursor)
            reward_scores = score_rollout(rollout_outputs, batch_examples)
        elif rollout_backend in {"vllm", "remote_vllm"} and dist_info.enabled:
            batch_examples, prompt_cursor = next_prompt_batch_from(prompt_cursor)
            batch_prompts = [example.text for example in batch_examples]
            payload: list[Any] = [None]
            if dist_info.is_main:
                if rollout_backend == "vllm":
                    rollout_engine = _build_rollout_engine(cfg, raw_actor, model_cfg, tokenizer, device=device, dtype=dtype, use_amp=use_amp)
                assert rollout_engine is not None
                rollout_outputs = rollout_engine.generate(batch_prompts, sampling)
                reward_inputs = _build_reward_inputs(rollout_outputs, batch_examples)
                if sandbox_runner is not None:
                    reward_inputs = asyncio.run(sandbox_runner.evaluate(reward_inputs))
                reward_scores = reward_fn(reward_inputs)
                payload[0] = (rollout_outputs, reward_scores)
            dist.broadcast_object_list(payload, src=0)
            rollout_outputs, reward_scores = payload[0]
        else:
            batch_examples, prompt_cursor = next_prompt_batch_from(prompt_cursor)
            if rollout_backend == "vllm":
                rollout_engine = _build_rollout_engine(cfg, raw_actor, model_cfg, tokenizer, device=device, dtype=dtype, use_amp=use_amp)
            rollout_outputs = run_rollout(batch_examples)
            reward_scores = score_rollout(rollout_outputs, batch_examples)
        rewards = torch.tensor([score.score for score in reward_scores], device=device, dtype=torch.float32)
        prompt_ids = torch.tensor([output.prompt_index for output in rollout_outputs], device=device, dtype=torch.long)
        dapo_filter_component = str(rl_cfg.dapo.get("filter_reward_component", "") or "") if str(rl_cfg.algorithm) == "dapo" else ""
        filter_rewards = _reward_component_tensor(reward_scores, dapo_filter_component, device=device) if dapo_filter_component else rewards
        if str(rl_cfg.algorithm) == "dapo" and bool(rl_cfg.dapo.get("overlong_filtering", False)):
            completion_lengths = torch.tensor(
                [len(tokenizer.encode(output.response)) for output in rollout_outputs],
                device=device,
                dtype=torch.long,
            )
            rewards = apply_overlong_penalty(
                rewards,
                completion_lengths,
                max_completion_length=int(rl_cfg.max_completion_length),
                penalty=float(rl_cfg.dapo.get("overlong_penalty", 0.0)),
            )
        advantages = compute_group_advantages(rewards, prompt_ids)
        keep_mask = torch.ones_like(rewards, dtype=torch.bool)
        if str(rl_cfg.algorithm) == "dapo" and bool(rl_cfg.dapo.dynamic_filtering):
            keep_mask = dapo_group_filter(filter_rewards, prompt_ids)
        completed_step = step + 1
        rollout_stats = _rollout_log_stats(
            rewards,
            keep_mask,
            prompt_ids,
            filter_rewards=filter_rewards,
            filter_component=dapo_filter_component,
            device=device,
            dist_info=dist_info,
        )
        keep_fraction = float(rollout_stats["keep_fraction"])
        global_keep_count = int(rollout_stats["kept_rollouts"])
        if global_keep_count == 0:
            latest_stats = {**rollout_stats, "skipped_all": 1.0}
            if dist_info.is_main and completed_step % int(train_cfg.log_interval) == 0:
                print(f"[walkie/rl] step={completed_step} algorithm={rl_cfg.algorithm} reward={latest_stats['reward_mean']:.3f} keep={latest_stats['keep_fraction']:.3f} skipped_all=1 stats={_format_rollout_log_stats(latest_stats)}")
                _log_swanlab(swanlab_run, latest_stats, completed_step, total_steps=total_steps, lrs=lrs, elapsed=time.time() - started_at)
            if completed_step % int(train_cfg.ckpt_interval) == 0:
                _save_checkpoint(out_dir, actor, optimizers, scaler, schedule, completed_step, model_cfg, train_cfg, str(rl_cfg.algorithm), checkpoint_extra(), dist_info)
            continue
        if bool(keep_mask.any().item()):
            sequences, completion_mask = _encode_rollouts(rollout_outputs, tokenizer, device=device, max_length=int(train_cfg.block_size))
            sequences = sequences[keep_mask]
            completion_mask = completion_mask[keep_mask]
            advantages = advantages[keep_mask]
            with torch.no_grad(), _temporary_eval(raw_actor), torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
                old_logprobs = causal_lm_logprobs(raw_actor, sequences, micro_batch_size=logprob_micro_batch_size).detach()
            ref_logprobs = None
            if ref_model is not None:
                ref_sequences = sequences.to(ref_device) if ref_device != device else sequences
                ref_use_amp = use_amp and ref_device.type == device.type
                with torch.no_grad(), torch.autocast(device_type=ref_device.type, dtype=dtype, enabled=ref_use_amp):
                    ref_logprobs = causal_lm_logprobs(ref_model, ref_sequences, micro_batch_size=logprob_micro_batch_size).detach().to(device)
        else:
            sequences = torch.empty((0, 0), dtype=torch.long, device=device)
            completion_mask = torch.empty((0, 0), dtype=torch.float32, device=device)
            advantages = torch.empty((0,), dtype=torch.float32, device=device)
            old_logprobs = torch.empty((0, 0), dtype=torch.float32, device=device)
            ref_logprobs = torch.empty((0, 0), dtype=torch.float32, device=device) if ref_model is not None else None

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        stats = _backward_policy_loss_microbatched(
            actor,
            sequences,
            old_logprobs,
            ref_logprobs,
            advantages,
            completion_mask,
            rl_cfg,
            device=device,
            dtype=dtype,
            use_amp=use_amp,
            scaler=scaler,
            micro_batch_size=logprob_micro_batch_size,
            dist_info=dist_info,
        )
        _sync_gradients(raw_actor, dist_info)
        if float(train_cfg.grad_clip) > 0:
            if use_amp and dtype == torch.float16:
                for optimizer in optimizers.values():
                    scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_actor.parameters(), float(train_cfg.grad_clip))
        if use_amp and dtype == torch.float16:
            for optimizer in optimizers.values():
                scaler.step(optimizer)
            scaler.update()
        else:
            for optimizer in optimizers.values():
                optimizer.step()

        if rollout_backend == "remote_vllm" and (not dist_info.enabled or dist_info.is_main):
            assert isinstance(rollout_engine, RemoteVLLMRolloutEngine)
            _sync_remote_vllm_if_needed(cfg, rollout_engine, raw_actor, model_cfg, completed_step)

        if dist_info.is_main and completed_step % int(train_cfg.log_interval) == 0:
            latest_stats = {**rollout_stats, **stats}
            print(f"[walkie/rl] step={completed_step} algorithm={rl_cfg.algorithm} reward={latest_stats['reward_mean']:.3f} keep={keep_fraction:.3f} stats={_format_rollout_log_stats(latest_stats, policy_stats=stats)}")
            _log_swanlab(swanlab_run, latest_stats, completed_step, total_steps=total_steps, lrs=lrs, elapsed=time.time() - started_at)
        if completed_step % int(train_cfg.ckpt_interval) == 0:
            _save_checkpoint(out_dir, actor, optimizers, scaler, schedule, completed_step, model_cfg, train_cfg, str(rl_cfg.algorithm), checkpoint_extra(), dist_info)
    _save_checkpoint(out_dir, actor, optimizers, scaler, schedule, stop_step, model_cfg, train_cfg, str(rl_cfg.algorithm), checkpoint_extra(), dist_info)
    if prefetch_executor is not None:
        prefetch_executor.shutdown(wait=True)
    if dist_info.is_main and swanlab_run is not None:
        swanlab_run.log({"train/step": int(stop_step), "train/segment_stop_step": int(stop_step), "train/total_steps": int(total_steps)}, step=int(stop_step))
        swanlab_run.finish()


def _encode_rollouts(outputs, tokenizer: TokenizerAdapter, *, device: torch.device, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[list[int]] = []
    masks: list[list[float]] = []
    for output in outputs:
        prompt_ids = tokenizer.encode(output.prompt)
        completion_ids = tokenizer.encode(output.response) + [tokenizer.eos_token_id]
        ids = (prompt_ids + completion_ids)[-max_length:]
        prompt_len = min(len(prompt_ids), len(ids))
        logprob_len = max(0, len(ids) - 1)
        mask = [0.0] * logprob_len
        start = max(0, prompt_len - 1)
        for index in range(start, logprob_len):
            mask[index] = 1.0
        rows.append(ids)
        masks.append(mask)
    max_len = max(len(row) for row in rows)
    padded_rows: list[list[int]] = []
    padded_masks: list[list[float]] = []
    for row, mask in zip(rows, masks):
        padded_rows.append(row + [tokenizer.pad_token_id] * (max_len - len(row)))
        padded_masks.append(mask + [0.0] * (max_len - 1 - len(mask)))
    return torch.tensor(padded_rows, dtype=torch.long, device=device), torch.tensor(padded_masks, dtype=torch.float32, device=device)


def _build_reward_inputs(outputs, batch_examples: list[PromptExample]) -> list[RewardInput]:
    return [
        RewardInput(
            output.prompt,
            output.response,
            {**batch_examples[output.prompt_index].metadata, **output.metadata},
        )
        for output in outputs
    ]


def _reward_component_tensor(reward_scores: list[RewardScore], component_name: str, *, device: torch.device) -> torch.Tensor:
    values: list[float] = []
    for score in reward_scores:
        if component_name not in score.components:
            raise KeyError(f"reward component {component_name!r} is missing from reward score")
        values.append(float(score.components[component_name]))
    return torch.tensor(values, device=device, dtype=torch.float32)


def _next_prompt_batch(prompts: list[PromptExample], cursor: int, batch_size: int) -> tuple[list[PromptExample], int]:
    batch = [prompts[(cursor + offset) % len(prompts)] for offset in range(batch_size)]
    return batch, cursor + batch_size


def _shard_prompt_batch(batch: list[PromptExample], dist_info) -> list[PromptExample]:
    if not getattr(dist_info, "enabled", False):
        return batch
    shard = batch[int(dist_info.rank) :: int(dist_info.world_size)]
    if not shard:
        raise ValueError("rl.prompt_batch_size must be at least distributed.world_size")
    return shard


def _global_sum_int(value: int, *, device: torch.device, dist_info) -> int:
    if not getattr(dist_info, "enabled", False):
        return int(value)
    tensor = torch.tensor(int(value), dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _rollout_log_stats(
    rewards: torch.Tensor,
    keep_mask: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    filter_rewards: torch.Tensor | None,
    filter_component: str,
    device: torch.device,
    dist_info,
) -> dict[str, float]:
    if rewards.ndim != 1 or keep_mask.ndim != 1 or prompt_ids.ndim != 1:
        raise ValueError("rewards, keep_mask and prompt_ids must be 1D tensors")
    if rewards.numel() != keep_mask.numel() or rewards.numel() != prompt_ids.numel():
        raise ValueError("rewards, keep_mask and prompt_ids must have the same length")
    if filter_rewards is not None and (filter_rewards.ndim != 1 or filter_rewards.numel() != rewards.numel()):
        raise ValueError("filter_rewards must be a 1D tensor with the same length as rewards")

    local_count = float(rewards.numel())
    local_keep_count = float(keep_mask.sum().item())
    filter_values = filter_rewards if filter_rewards is not None else rewards
    group_total = 0.0
    group_kept = 0.0
    group_all_positive = 0.0
    group_all_non_positive = 0.0
    for prompt_id in torch.unique(prompt_ids):
        group_total += 1.0
        group = filter_values[prompt_ids == prompt_id]
        has_positive = bool((group > 0).any().item())
        has_non_positive = bool((group <= 0).any().item())
        if has_positive and has_non_positive:
            group_kept += 1.0
        elif has_positive:
            group_all_positive += 1.0
        else:
            group_all_non_positive += 1.0

    stats_tensor = torch.tensor(
        [
            float(rewards.sum().item()),
            local_count,
            local_keep_count,
            float(filter_values.sum().item()),
            float((filter_values > 0).sum().item()),
            group_total,
            group_kept,
            group_all_positive,
            group_all_non_positive,
        ],
        device=device,
        dtype=torch.float32,
    )
    if getattr(dist_info, "enabled", False):
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    rollout_count = max(float(stats_tensor[1].item()), 1.0)
    group_count = max(float(stats_tensor[5].item()), 1.0)
    stats: dict[str, float] = {
        "reward_mean": float(stats_tensor[0].item()) / rollout_count,
        "keep_fraction": float(stats_tensor[2].item()) / rollout_count,
        "kept_rollouts": float(stats_tensor[2].item()),
        "total_rollouts": float(stats_tensor[1].item()),
        "local_keep_fraction": local_keep_count / max(local_count, 1.0),
        "dapo_kept_groups": float(stats_tensor[6].item()),
        "dapo_total_groups": float(stats_tensor[5].item()),
        "dapo_group_keep_fraction": float(stats_tensor[6].item()) / group_count,
        "dapo_all_positive_groups": float(stats_tensor[7].item()),
        "dapo_all_non_positive_groups": float(stats_tensor[8].item()),
    }
    if filter_component:
        stats[f"{filter_component}_mean"] = float(stats_tensor[3].item()) / rollout_count
        stats[f"{filter_component}_positive_fraction"] = float(stats_tensor[4].item()) / rollout_count
    return stats


def _format_rollout_log_stats(stats: dict[str, Any], *, policy_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    keys = [
        "loss",
        "clip_fraction",
        "loss_normalization",
        "code_execution_mean",
        "code_execution_positive_fraction",
        "dapo_kept_groups",
        "dapo_total_groups",
        "dapo_all_positive_groups",
        "dapo_all_non_positive_groups",
        "local_keep_fraction",
    ]
    formatted = {key: stats[key] for key in keys if key in stats}
    if policy_stats:
        for key, value in policy_stats.items():
            formatted.setdefault(key, value)
    return formatted


def _backward_policy_loss_microbatched(
    actor,
    sequences: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    rl_cfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    micro_batch_size: int,
    dist_info,
) -> dict[str, float | str]:
    if old_logprobs.shape != completion_mask.shape or advantages.shape != (sequences.shape[0],):
        raise ValueError("logprob, mask and advantage tensors have incompatible shapes")
    chunk_size = max(1, int(micro_batch_size))
    local_token_denom = completion_mask.sum().to(torch.float32)
    local_batch_denom = torch.tensor(float(sequences.shape[0]), device=device, dtype=torch.float32)
    global_token_denom = local_token_denom.detach().clone()
    global_batch_denom = local_batch_denom.detach().clone()
    if getattr(dist_info, "enabled", False):
        dist.all_reduce(global_token_denom, op=dist.ReduceOp.SUM)
        dist.all_reduce(global_batch_denom, op=dist.ReduceOp.SUM)
    global_token_denom = global_token_denom.clamp_min(1.0)
    global_batch_denom = global_batch_denom.clamp_min(1.0)
    loss_num_total = 0.0
    clip_num_total = 0.0
    kl_num_total = 0.0
    algorithm = str(rl_cfg.algorithm)
    if algorithm != "dapo":
        if ref_logprobs is None or ref_logprobs.shape != old_logprobs.shape:
            raise ValueError("ref_logprobs and old_logprobs must have matching shapes")

    for start in range(0, sequences.shape[0], chunk_size):
        stop = min(start + chunk_size, sequences.shape[0])
        seq_chunk = sequences[start:stop]
        old_chunk = old_logprobs[start:stop]
        ref_chunk = ref_logprobs[start:stop] if ref_logprobs is not None else None
        adv_chunk = advantages[start:stop]
        mask_chunk = completion_mask[start:stop]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            new_chunk = causal_lm_logprobs(actor, seq_chunk)
            mask = mask_chunk.to(new_chunk.dtype)
            adv = adv_chunk.to(new_chunk.dtype).unsqueeze(-1)
            ratio = torch.exp(new_chunk - old_chunk.to(new_chunk.dtype))
            if algorithm == "dapo":
                clip_low = float(rl_cfg.dapo.clip_low)
                clip_high = float(rl_cfg.dapo.clip_high)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
                loss_tokens = -torch.minimum(ratio * adv, clipped_ratio * adv)
                loss_normalization = str(rl_cfg.dapo.get("loss_normalization", "token"))
                if loss_normalization == "token":
                    loss_num = (loss_tokens * mask).sum()
                    chunk_loss = loss_num / global_token_denom.to(loss_tokens.dtype)
                elif loss_normalization == "sequence":
                    per_sequence = (loss_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    loss_num = per_sequence.sum()
                    chunk_loss = loss_num / global_batch_denom.to(loss_tokens.dtype)
                else:
                    raise ValueError("loss_normalization must be 'token' or 'sequence'")
                clip_num = (((ratio - clipped_ratio).abs() > 1e-8).to(mask.dtype) * mask).sum()
                kl_num = None
            else:
                assert ref_chunk is not None
                clip_range = float(rl_cfg.grpo.clip_range)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
                log_ratio_ref = ref_chunk.to(new_chunk.dtype) - new_chunk
                token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                loss_num = ((pg_loss + float(rl_cfg.grpo.kl_coef) * token_kl) * mask).sum()
                chunk_loss = loss_num / global_token_denom.to(new_chunk.dtype)
                clip_num = (((ratio - clipped_ratio).abs() > 1e-8).to(mask.dtype) * mask).sum()
                kl_num = (token_kl * mask).sum()

        if use_amp and dtype == torch.float16:
            scaler.scale(chunk_loss).backward()
        else:
            chunk_loss.backward()
        loss_num_total += float(loss_num.detach().cpu())
        clip_num_total += float(clip_num.detach().cpu())
        if kl_num is not None:
            kl_num_total += float(kl_num.detach().cpu())

    stats_tensor = torch.tensor(
        [loss_num_total, clip_num_total, kl_num_total, float(local_token_denom.detach().cpu()), float(local_batch_denom.detach().cpu())],
        device=device,
        dtype=torch.float32,
    )
    if getattr(dist_info, "enabled", False):
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
    loss_normalization = str(rl_cfg.dapo.get("loss_normalization", "token")) if algorithm == "dapo" else "token"
    loss_denom = max(float(stats_tensor[4].item()), 1.0) if loss_normalization == "sequence" else max(float(stats_tensor[3].item()), 1.0)
    clip_denom = max(float(stats_tensor[3].item()), 1.0)
    stats: dict[str, float | str] = {
        "loss": float(stats_tensor[0].item()) / loss_denom,
        "clip_fraction": float(stats_tensor[1].item()) / clip_denom,
    }
    if algorithm == "dapo":
        stats["loss_normalization"] = loss_normalization
    else:
        stats["approx_kl"] = float(stats_tensor[2].item()) / clip_denom
    return stats


def _sync_gradients(model: torch.nn.Module, dist_info) -> None:
    if not getattr(dist_info, "enabled", False):
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)


@contextmanager
def _temporary_eval(model):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


def _build_rollout_engine(cfg, actor: WalkieForCausalLM, model_cfg: WalkieConfig, tokenizer: TokenizerAdapter, *, device: torch.device, dtype: torch.dtype, use_amp: bool):
    if str(cfg.rollout.backend) == "fake":
        return FakeRolloutEngine(str(cfg.rollout.get("fake_response", "```python\nprint(1)\n```")))
    if str(cfg.rollout.backend) in {"torch", "hf"}:
        return TorchRolloutEngine(actor, tokenizer, device=device, dtype=dtype, use_amp=use_amp)
    if str(cfg.rollout.backend) == "vllm":
        export_dir = str(cfg.rollout.export_dir)
        export_walkie_to_hf(actor, export_dir, tokenizer_path=cfg.data.tokenizer_path, model_cfg=model_cfg.to_dict())
        return VLLMRolloutEngine(export_dir, tensor_parallel_size=int(cfg.rollout.tensor_parallel_size), dtype=str(cfg.rollout.dtype))
    if str(cfg.rollout.backend) == "remote_vllm":
        server_url = cfg.rollout.get("server_url")
        if not server_url:
            raise ValueError("rollout.server_url is required for remote_vllm")
        server_urls = cfg.rollout.get("server_urls")
        server_target = [str(item) for item in server_urls] if server_urls else str(server_url)
        engine = RemoteVLLMRolloutEngine(
            server_target,
            request_timeout=float(cfg.rollout.get("request_timeout", 120.0)),
            reload_timeout=float(cfg.rollout.get("reload_timeout", 300.0)),
            max_retries=int(cfg.rollout.get("max_retries", 2)),
            request_shards=int(cfg.rollout.get("request_shards", 1)),
            max_concurrent_requests=int(cfg.rollout.get("max_concurrent_requests", 0)) or None,
        )
        if bool(cfg.rollout.get("export_before_rollout", True)):
            export_dir = _remote_vllm_export_path(cfg, 0)
            _export_walkie_to_hf_atomic(actor, export_dir, tokenizer_path=cfg.data.tokenizer_path, model_cfg=model_cfg)
            engine.reload(str(export_dir))
        return engine
    raise ValueError(f"unknown rollout backend: {cfg.rollout.backend}")


def _sync_remote_vllm_if_needed(cfg, engine: RemoteVLLMRolloutEngine, actor: WalkieForCausalLM, model_cfg: WalkieConfig, step: int) -> None:
    if not _remote_vllm_sync_due(cfg, step):
        return
    export_dir = _remote_vllm_export_path(cfg, step)
    _export_walkie_to_hf_atomic(actor, export_dir, tokenizer_path=cfg.data.tokenizer_path, model_cfg=model_cfg)
    engine.reload(str(export_dir))


def _remote_vllm_sync_due(cfg, step: int) -> bool:
    interval = int(cfg.rollout.get("sync_interval", 1))
    return bool(interval > 0 and int(step) % interval == 0)


def _remote_vllm_export_path(cfg, step: int) -> Path:
    return Path(str(cfg.rollout.export_dir)) / f"step_{int(step):08d}"


def _export_walkie_to_hf_atomic(actor: WalkieForCausalLM, export_dir: Path, *, tokenizer_path: str | Path | None, model_cfg: WalkieConfig) -> None:
    export_dir = Path(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = export_dir.with_name(f"{export_dir.name}.tmp-{uuid4().hex}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    export_walkie_to_hf(actor, tmp_dir, tokenizer_path=tokenizer_path, model_cfg=model_cfg.to_dict())
    if export_dir.exists():
        shutil.rmtree(export_dir)
    tmp_dir.rename(export_dir)


def _build_sandbox_runner(cfg) -> SandboxCodeRewardRunner | None:
    if not bool(cfg.get("sandbox", {}).get("enabled", False)):
        return None
    sandbox_cfg = cfg.sandbox
    client = JupyterSandboxClient(
        list(sandbox_cfg.base_urls),
        timeout=float(sandbox_cfg.timeout),
        retries=int(sandbox_cfg.retries),
        clear_session=bool(sandbox_cfg.clear_session),
    )
    return SandboxCodeRewardRunner(
        client,
        tests_field=str(sandbox_cfg.tests_field),
        test_program_template_field=str(sandbox_cfg.get("test_program_template_field", "test_program_template")),
        max_concurrency=int(sandbox_cfg.get("max_concurrency", 8)),
    )


def _load_prompts(data_cfg, template: ChatTemplate) -> list[PromptExample]:
    paths = [str(item) for item in data_cfg.get("paths", [])]
    if not paths and data_cfg.get("path") is not None:
        paths = [str(data_cfg.path)]
    prompts: list[PromptExample] = []
    for path in _expand_prompt_paths(paths):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata_keys = (
                    "tests",
                    "test_program_template",
                    "entry_point",
                    "task_id",
                    "source",
                    "task_type",
                    "num_tests",
                )
                metadata = {key: row[key] for key in metadata_keys if key in row}
                if isinstance(row.get("prompt"), str):
                    prompts.append(PromptExample(text=row["prompt"], metadata=metadata))
                else:
                    turns = [turn for turn in normalize_messages(row) if turn.role != "assistant"]
                    prompts.append(PromptExample(text=template.render_prompt(turns), metadata=metadata))
    return prompts


def _expand_prompt_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            matches = sorted(path.glob("*.jsonl"))
            if not matches:
                raise FileNotFoundError(f"no JSONL prompt files found in {path}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    return expanded


def _load_tokenizer(path: str | None, eos_token: str, pad_token: str) -> TokenizerAdapter:
    if path is None:
        raise ValueError("data.tokenizer_path is required for RL")
    from tokenizers import Tokenizer

    tokenizer_path = Path(path)
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    return TokenizerAdapter(Tokenizer.from_file(str(tokenizer_path)), eos_token=eos_token, pad_token=pad_token)


def _resolve_template(tokenizer: TokenizerAdapter, data_cfg) -> ChatTemplate:
    requested = str(data_cfg.template)
    if requested == "chatml_lowfreq_alias":
        has_chatml = tokenizer.tokenizer.token_to_id("<|im_start|>") is not None and tokenizer.tokenizer.token_to_id("<|im_end|>") is not None
        if not has_chatml:
            print("[walkie/rl] tokenizer lacks ChatML alias tokens; falling back to plain_eot template")
            requested = "plain_eot"
    return ChatTemplate(kind=requested, eos_token=str(data_cfg.eos_token))


def _save_checkpoint(out_dir: Path, model, optimizers, scaler, schedule, step: int, model_cfg: WalkieConfig, train_cfg, stage: str, extra: dict[str, Any], dist_info) -> None:
    if dist_info.enabled:
        dist.barrier()
    if dist_info.is_main:
        save_walkie_checkpoint(
            out_dir,
            model=model,
            optimizers=optimizers,
            scaler=scaler,
            schedule_state=schedule.state_dict(),
            step=step,
            stage=stage,
            best_metric=None,
            model_cfg=model_cfg.to_dict(),
            train_cfg=_plain_cfg_dict(train_cfg),
            extra=extra,
            format="latest",
        )
    if dist_info.enabled:
        dist.barrier()


def _resolve_stop_step(train_cfg, *, total_steps: int) -> int:
    stop_step = getattr(train_cfg, "stop_step", None)
    if stop_step is None:
        return int(total_steps)
    return min(int(total_steps), int(stop_step))


def _build_checkpoint_extra(
    *,
    rl_cfg,
    prompt_cursor: int,
    latest_stats: dict[str, Any],
    swanlab_run: Any | None,
    resume_swanlab_run_id: str | None,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "rl": _plain_cfg_dict(rl_cfg),
        "data_state": {"prompt_cursor": int(prompt_cursor)},
        "rollout_stats": dict(latest_stats),
    }
    if swanlab_run is not None:
        run_id = getattr(swanlab_run, "id", None) or getattr(swanlab_run, "run_id", None) or getattr(swanlab_run, "_walkie_run_id", None)
        if run_id is not None:
            extra["swanlab_run_id"] = str(run_id)
    elif resume_swanlab_run_id is not None:
        extra["swanlab_run_id"] = str(resume_swanlab_run_id)
    return extra


def _swanlab_resume_run_id(extra: dict[str, Any] | None) -> str | None:
    if not extra:
        return None
    run_id = extra.get("swanlab_run_id")
    return str(run_id) if run_id is not None else None


def _init_swanlab_run(*, train_cfg, model_cfg: WalkieConfig, out_dir: Path, resume_path: str | None, resume_run_id: str | None) -> Any | None:
    swanlab_cfg = train_cfg.get("swanlab")
    if not swanlab_cfg or not bool(swanlab_cfg.get("enabled", False)):
        return None
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError("train.swanlab.enabled=true but swanlab is not installed") from exc
    mode = str(swanlab_cfg.get("mode", "online"))
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError(f"train.swanlab.mode must be online/offline/disabled, got {mode}")
    if mode == "disabled":
        return None
    api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key=api_key)
    config_payload = _plain_cfg_dict(train_cfg)
    if isinstance(config_payload.get("swanlab"), dict):
        config_payload["swanlab"].pop("api_key", None)
    run_id = str(swanlab_cfg.get("run_id") or resume_run_id or uuid4().hex[:8])
    run = swanlab.init(
        project=str(swanlab_cfg.get("project", "walkie")),
        workspace=swanlab_cfg.get("workspace") or swanlab_cfg.get("entity"),
        experiment_name=swanlab_cfg.get("experiment_name") or swanlab_cfg.get("name"),
        description=swanlab_cfg.get("description") or swanlab_cfg.get("notes"),
        tags=list(swanlab_cfg.get("tags", [])),
        log_dir=str(out_dir),
        mode=mode,
        id=run_id,
        resume=str(swanlab_cfg.get("resume", "allow")),
        config={"model": model_cfg.to_dict(), "train": config_payload, "runtime": {"out_dir": str(out_dir), "resume_from": resume_path}},
    )
    if run is None:
        raise RuntimeError("swanlab.init returned None")
    try:
        setattr(run, "_walkie_run_id", run_id)
    except Exception:
        pass
    print(f"[walkie/swanlab] project={swanlab_cfg.get('project', 'walkie')} experiment={swanlab_cfg.get('experiment_name') or swanlab_cfg.get('name') or run_id} mode={mode} run_id={run_id}")
    return run


def _log_swanlab(swanlab_run: Any | None, stats: dict[str, Any], step: int, *, total_steps: int, lrs: dict[str, float], elapsed: float) -> None:
    if swanlab_run is None:
        return
    payload: dict[str, float] = {
        "train/step": float(step),
        "train/progress": float(step / max(1, total_steps)),
        "lr/adamw": float(lrs["adamw"]),
        "lr/muon": float(lrs["muon"]),
        "perf/elapsed_sec": float(elapsed),
    }
    for key, value in stats.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        payload[f"rl/{key}"] = float(value)
    swanlab_run.log(payload, step=int(step))


def _plain_cfg_dict(cfg) -> dict[str, Any]:
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


if __name__ == "__main__":
    main()
