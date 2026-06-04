"""Log-probability helpers shared by GRPO and DAPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} and target shape {tuple(target_ids.shape)} mismatch")
    return F.log_softmax(logits, dim=-1).gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def causal_lm_logprobs(model, input_ids: torch.Tensor, *, micro_batch_size: int | None = None) -> torch.Tensor:
    if input_ids.size(1) < 2:
        raise ValueError("input_ids must contain at least two tokens")
    if micro_batch_size is not None and 0 < int(micro_batch_size) < input_ids.size(0):
        chunks = [
            causal_lm_logprobs(model, input_ids[start : start + int(micro_batch_size)])
            for start in range(0, input_ids.size(0), int(micro_batch_size))
        ]
        return torch.cat(chunks, dim=0)
    logits, _ = model(input_ids[:, :-1], input_ids[:, 1:], return_logits=True)
    assert logits is not None
    return gather_token_logprobs(logits, input_ids[:, 1:])
