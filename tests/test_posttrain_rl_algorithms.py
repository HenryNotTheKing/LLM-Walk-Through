"""Post-training GRPO/DAPO math primitives."""

from __future__ import annotations

import torch

from posttrain.rl.algorithms.dapo import apply_overlong_penalty, dapo_group_filter, dapo_policy_loss
from posttrain.rl.algorithms.grpo import compute_group_advantages, grpo_policy_loss
from posttrain.rl.logprobs import causal_lm_logprobs, gather_token_logprobs


def test_group_advantages_are_normalized_per_prompt() -> None:
    rewards = torch.tensor([1.0, 3.0, 2.0, 2.0])
    prompt_ids = torch.tensor([0, 0, 1, 1])

    advantages = compute_group_advantages(rewards, prompt_ids)

    assert torch.allclose(advantages[:2], torch.tensor([-1.0, 1.0]), atol=1e-6)
    assert torch.allclose(advantages[2:], torch.zeros(2), atol=1e-6)


def test_grpo_policy_loss_applies_ratio_clip_and_kl() -> None:
    new_logprobs = torch.log(torch.tensor([[0.8, 0.4]]))
    old_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
    ref_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 1.0]])

    loss, stats = grpo_policy_loss(
        new_logprobs,
        old_logprobs,
        ref_logprobs,
        advantages,
        mask,
        clip_range=0.2,
        kl_coef=0.1,
    )

    assert loss.dim() == 0
    assert stats["clip_fraction"] > 0.0
    assert stats["approx_kl"] > 0.0


def test_dapo_filter_drops_all_correct_and_all_wrong_groups() -> None:
    rewards = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    prompt_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    keep = dapo_group_filter(rewards, prompt_ids)

    assert keep.tolist() == [False, False, True, True, False, False]


def test_dapo_policy_loss_supports_clip_higher() -> None:
    new_logprobs = torch.log(torch.tensor([[0.9, 0.9]]))
    old_logprobs = torch.log(torch.tensor([[0.5, 0.5]]))
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[1.0, 1.0]])

    loss, stats = dapo_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        clip_low=0.2,
        clip_high=0.4,
    )

    assert loss.dim() == 0
    assert stats["clip_fraction"] == 1.0


def test_dapo_overlong_penalty_subtracts_only_long_completions() -> None:
    rewards = torch.tensor([1.0, 1.0, 1.0])
    lengths = torch.tensor([8, 9, 16])

    adjusted = apply_overlong_penalty(rewards, lengths, max_completion_length=9, penalty=0.25)

    assert torch.allclose(adjusted, torch.tensor([1.0, 0.75, 0.75]))


def test_dapo_sequence_normalization_averages_per_sequence_first() -> None:
    new_logprobs = torch.log(torch.tensor([[0.6, 0.2], [0.9, 0.9]]))
    old_logprobs = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    advantages = torch.tensor([1.0, 1.0])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    token_loss, _ = dapo_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        loss_normalization="token",
    )
    sequence_loss, stats = dapo_policy_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        mask,
        loss_normalization="sequence",
    )

    assert sequence_loss.dim() == 0
    assert stats["loss_normalization"] == "sequence"
    assert not torch.allclose(token_loss, sequence_loss)


def test_gather_token_logprobs_aligns_next_token_targets() -> None:
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    target_ids = torch.tensor([[0, 1]])

    gathered = gather_token_logprobs(logits, target_ids)

    expected = torch.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(gathered, expected)


def test_causal_lm_logprobs_micro_batch_matches_full_batch() -> None:
    class ToyLM:
        def __call__(self, input_ids, targets, *, return_logits: bool):
            del targets, return_logits
            vocab = 7
            base = torch.arange(vocab, dtype=torch.float32, device=input_ids.device)
            logits = base.view(1, 1, vocab).expand(input_ids.shape[0], input_ids.shape[1], vocab).clone()
            logits = logits + input_ids.to(torch.float32).unsqueeze(-1) * 0.01
            return logits, None

    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

    full = causal_lm_logprobs(ToyLM(), input_ids)
    chunked = causal_lm_logprobs(ToyLM(), input_ids, micro_batch_size=1)

    assert torch.allclose(chunked, full)