"""GRPO math primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GRPOConfig:
    clip_range: float = 0.2
    kl_coef: float = 0.02
    advantage_eps: float = 1e-8


def compute_group_advantages(
    rewards: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if rewards.ndim != 1 or prompt_ids.ndim != 1 or rewards.numel() != prompt_ids.numel():
        raise ValueError("rewards and prompt_ids must be 1D tensors with the same length")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for prompt_id in torch.unique(prompt_ids):
        mask = prompt_ids == prompt_id
        group = rewards[mask].float()
        mean = group.mean()
        std = group.std(unbiased=False)
        if float(std.item()) <= eps:
            advantages[mask] = 0.0
        else:
            advantages[mask] = (group - mean) / (std + eps)
    return advantages


def grpo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_range: float = 0.2,
    kl_coef: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    _validate_shapes(new_logprobs, old_logprobs, ref_logprobs, completion_mask, advantages)
    mask = completion_mask.to(new_logprobs.dtype)
    adv = advantages.to(new_logprobs.dtype).unsqueeze(-1)
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    pg_loss = -torch.minimum(ratio * adv, clipped_ratio * adv)
    log_ratio_ref = ref_logprobs - new_logprobs
    token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
    denom = mask.sum().clamp_min(1.0)
    loss = ((pg_loss + kl_coef * token_kl) * mask).sum() / denom
    clip_fraction = (((ratio - clipped_ratio).abs() > 1e-8).to(mask.dtype) * mask).sum() / denom
    approx_kl = (token_kl * mask).sum() / denom
    return loss, {
        "loss": float(loss.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "approx_kl": float(approx_kl.detach().cpu()),
    }


def _validate_shapes(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
) -> None:
    shape = new_logprobs.shape
    if old_logprobs.shape != shape or ref_logprobs.shape != shape or mask.shape != shape:
        raise ValueError("logprob and mask tensors must have matching shapes")
    if advantages.shape != (shape[0],):
        raise ValueError("advantages must have shape [batch]")
