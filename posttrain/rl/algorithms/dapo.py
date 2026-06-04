"""DAPO math primitives built on GRPO-style policy gradients."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DAPOConfig:
    clip_low: float = 0.2
    clip_high: float = 0.28
    overlong_penalty: float = 0.0


def apply_overlong_penalty(
    rewards: torch.Tensor,
    completion_lengths: torch.Tensor,
    *,
    max_completion_length: int,
    penalty: float,
) -> torch.Tensor:
    if penalty <= 0:
        return rewards
    overlong = completion_lengths >= int(max_completion_length)
    return rewards - overlong.to(rewards.dtype) * float(penalty)


def dapo_group_filter(rewards: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
    if rewards.ndim != 1 or prompt_ids.ndim != 1 or rewards.numel() != prompt_ids.numel():
        raise ValueError("rewards and prompt_ids must be 1D tensors with the same length")
    keep = torch.zeros_like(rewards, dtype=torch.bool)
    for prompt_id in torch.unique(prompt_ids):
        mask = prompt_ids == prompt_id
        group = rewards[mask]
        has_positive = bool((group > 0).any().item())
        has_non_positive = bool((group <= 0).any().item())
        keep[mask] = has_positive and has_non_positive
    return keep


def dapo_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    loss_normalization: str = "token",
) -> tuple[torch.Tensor, dict[str, float | str]]:
    if old_logprobs.shape != new_logprobs.shape or completion_mask.shape != new_logprobs.shape:
        raise ValueError("new_logprobs, old_logprobs and completion_mask must have matching shapes")
    if advantages.shape != (new_logprobs.shape[0],):
        raise ValueError("advantages must have shape [batch]")
    mask = completion_mask.to(new_logprobs.dtype)
    adv = advantages.to(new_logprobs.dtype).unsqueeze(-1)
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    loss_tokens = -torch.minimum(ratio * adv, clipped_ratio * adv)
    if loss_normalization == "token":
        denom = mask.sum().clamp_min(1.0)
        loss = (loss_tokens * mask).sum() / denom
    elif loss_normalization == "sequence":
        per_sequence = (loss_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        loss = per_sequence.mean()
        denom = mask.sum().clamp_min(1.0)
    else:
        raise ValueError("loss_normalization must be 'token' or 'sequence'")
    clip_fraction = (((ratio - clipped_ratio).abs() > 1e-8).to(mask.dtype) * mask).sum() / denom
    return loss, {
        "loss": float(loss.detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "loss_normalization": loss_normalization,
    }
