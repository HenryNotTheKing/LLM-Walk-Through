"""基础 top-k MoE (Mixture of Experts)。

Fedus et al. (2021) *Switch Transformers: Scaling to Trillion Parameter
Models with Simple and Efficient Sparsity* 与 Lepikhin et al. (2020)
*GShard: Scaling Giant Models with Conditional Computation* 提出 MoE 架构：

- 一个**路由网络 (router)** 将每个 token 分配给 k 个专家；
- 多个**专家 FFN (experts)** 并行处理不同 token 子集；
- **容量限制 (capacity factor)** 防止单个专家过载；
- **负载均衡损失 (aux loss)** 鼓励均匀分配。

本实现支持通用专家类（SwiGLU / GEGLU / GELU 等）与 top-k 路由，
训练时返回辅助损失，推理时跳过 aux loss 计算。
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Compute the Switch/GShard auxiliary load-balancing loss.

    Args:
        router_probs: Router softmax probabilities, shape ``(N, num_experts)``.
        expert_indices: Long tensor of selected expert ids, shape ``(N, top_k)``.
        num_experts: Total number of experts.

    Returns:
        Scalar tensor.
    """
    # f_e = fraction of tokens dispatched to expert e
    # Use one-hot over selected experts and average
    # expert_indices: (N, top_k)
    mask = F.one_hot(expert_indices, num_classes=num_experts).float()  # (N, top_k, num_experts)
    # Average over top_k dimension
    mask = mask.mean(dim=1)  # (N, num_experts)
    f = mask.mean(dim=0)  # (num_experts,)

    # P_e = average routing probability to expert e
    P = router_probs.mean(dim=0)  # (num_experts,)

    loss = num_experts * (f * P).sum()
    return loss


class TopKMoE(nn.Module):
    """Top-k Mixture of Experts.

    Args:
        n_embd: Input/output dimension.
        num_experts: Number of parallel expert FFNs.
        top_k: Number of experts each token is routed to. ``k=1`` is Switch Transformer.
        d_ffn: Hidden dimension of each expert.
        expert_factory: Callable that returns an ``nn.Module`` accepting
            ``(n_embd, d_ffn)`` and optional ``dropout`` / ``bias`` kwargs.
            Default builds a SwiGLU-style expert.
        capacity_factor: Capacity multiplier per expert.
            Capacity = ``ceil(seq_len * capacity_factor / num_experts)``.
            ``None`` disables capacity capping (useful for inference).
        aux_loss_coef: Coefficient for load-balancing auxiliary loss.
        dropout: Expert output dropout.
        bias: Whether experts use bias.
    """

    def __init__(
        self,
        n_embd: int,
        num_experts: int,
        top_k: int = 2,
        d_ffn: int = 256,
        expert_factory: Callable[..., nn.Module] | None = None,
        capacity_factor: float | None = 1.25,
        aux_loss_coef: float = 0.01,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        # Router
        self.router = nn.Linear(n_embd, num_experts, bias=False)

        # Experts
        if expert_factory is None:
            from core.ffn.swiglu import SwiGLUMLP
            expert_factory = lambda n, d: SwiGLUMLP(n, d, dropout=dropout, bias=bias)

        self.experts = nn.ModuleList(
            expert_factory(n_embd, d_ffn) for _ in range(num_experts)
        )

    def _compute_capacity(self, num_tokens: int) -> int | None:
        if self.capacity_factor is None:
            return None
        return math.ceil(num_tokens * self.capacity_factor / self.num_experts)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Shape ``(batch, seq_len, n_embd)``.

        Returns:
            ``(output, aux_loss)`` where ``output`` has the same shape as ``x``.
            ``aux_loss`` is ``None`` during eval mode.
        """
        batch, seq_len, n_embd = x.shape
        num_tokens = batch * seq_len
        x_flat = x.view(num_tokens, n_embd)  # (N, D)

        # Router
        router_logits = self.router(x_flat)  # (N, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)  # (N, num_experts)

        # Top-k selection
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Normalize weights within selected experts
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Capacity limit
        capacity = self._compute_capacity(num_tokens)

        # Accumulate outputs
        output = torch.zeros_like(x_flat)  # (N, D)
        expert_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        for expert_id in range(self.num_experts):
            # Find tokens that selected this expert (any of top_k positions)
            mask = (top_k_indices == expert_id)  # (N, top_k)
            token_mask = mask.any(dim=-1)  # (N,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            # Capacity capping (only during training)
            if self.training and capacity is not None and token_indices.numel() > capacity:
                token_indices = token_indices[:capacity]

            expert_counts[expert_id] = token_indices.numel()

            # Gather weights for this expert across top_k positions
            expert_mask = mask[token_indices]  # (M, top_k)
            expert_weights = top_k_weights[token_indices]  # (M, top_k)
            # Weight for this specific expert
            w = (expert_mask.float() * expert_weights).sum(dim=-1, keepdim=True)  # (M, 1)

            # Run expert
            expert_in = x_flat[token_indices]  # (M, D)
            expert_out = self.experts[expert_id](expert_in)  # (M, D)

            # Weighted scatter-add
            output[token_indices] += w * expert_out

        # Aux loss (only during training)
        aux_loss: torch.Tensor | None = None
        if self.training:
            aux_loss = load_balancing_loss(router_probs, top_k_indices, self.num_experts)
            aux_loss = self.aux_loss_coef * aux_loss

        return output.view(batch, seq_len, n_embd), aux_loss
