"""DeepSeekMoE — Fine-Grained Expert Segregation with Shared Experts.

Dai et al. (2024) *DeepSeek-V2: A Strong, Economical, and Efficient
Mixture-of-Experts Language Model* 提出将专家分为两类：

1.  **Shared experts**（共享专家，通常 2–4 个）：所有 token 始终激活，
    负责学习通用、高频的语言知识；
2.  **Routed experts**（路由专家，通常 64–256 个）：通过 top-k 路由
    按需激活，负责学习特定、低频的知识模式。

输出公式：

    y = sum(shared_expert_i(x)) + sum(g_j * routed_expert_j(x))

其中共享专家始终求和，路由专家按门控权重 $g_j$ 加权。
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSeekMoE(nn.Module):
    """DeepSeekMoE with shared + routed expert segregation.

    Args:
        n_embd: Input/output dimension.
        num_shared_experts: Number of always-active shared experts.
        num_routed_experts: Number of routed experts.
        top_k: Number of routed experts each token activates.
        d_ffn: Hidden dimension of each expert.
        expert_factory: Callable returning an ``nn.Module`` for a single expert.
            Default builds SwiGLUMLP.
        capacity_factor: Capacity multiplier per routed expert.
            ``None`` disables capping.
        aux_loss_coef: Coefficient for routed-expert load-balancing loss.
        dropout: Expert output dropout.
        bias: Whether experts use bias.
    """

    def __init__(
        self,
        n_embd: int,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        top_k: int = 6,
        d_ffn: int = 256,
        expert_factory: Callable[..., nn.Module] | None = None,
        capacity_factor: float | None = 1.0,
        aux_loss_coef: float = 0.01,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef

        if expert_factory is None:
            from core.ffn.swiglu import SwiGLUMLP
            expert_factory = lambda n, d: SwiGLUMLP(n, d, dropout=dropout, bias=bias)

        # Shared experts: always activated
        self.shared_experts = nn.ModuleList(
            expert_factory(n_embd, d_ffn) for _ in range(num_shared_experts)
        )

        # Routed experts: conditionally activated
        self.routed_experts = nn.ModuleList(
            expert_factory(n_embd, d_ffn) for _ in range(num_routed_experts)
        )

        # Router for routed experts only
        self.router = nn.Linear(n_embd, num_routed_experts, bias=False)

    def _compute_capacity(self, num_tokens: int) -> int | None:
        if self.capacity_factor is None:
            return None
        return math.ceil(num_tokens * self.capacity_factor / self.num_routed_experts)

    def _load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Aux loss for routed experts."""
        mask = F.one_hot(expert_indices, num_classes=self.num_routed_experts).float()
        mask = mask.mean(dim=1)
        f = mask.mean(dim=0)
        P = router_probs.mean(dim=0)
        loss = self.num_routed_experts * (f * P).sum()
        return self.aux_loss_coef * loss

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: Shape ``(batch, seq_len, n_embd)``.

        Returns:
            ``(output, aux_loss)`` where ``output`` has the same shape as ``x``.
        """
        batch, seq_len, n_embd = x.shape
        num_tokens = batch * seq_len
        x_flat = x.view(num_tokens, n_embd)

        # --- Shared experts (always active) ---
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        # --- Routed experts (top-k selected) ---
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / (
            top_k_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        capacity = self._compute_capacity(num_tokens)
        routed_out = torch.zeros_like(x_flat)

        for expert_id in range(self.num_routed_experts):
            mask = (top_k_indices == expert_id)
            token_mask = mask.any(dim=-1)
            token_indices = token_mask.nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            if self.training and capacity is not None and token_indices.numel() > capacity:
                token_indices = token_indices[:capacity]

            expert_mask = mask[token_indices]
            expert_weights = top_k_weights[token_indices]
            w = (expert_mask.float() * expert_weights).sum(dim=-1, keepdim=True)

            expert_in = x_flat[token_indices]
            expert_out = self.routed_experts[expert_id](expert_in)
            routed_out[token_indices] += w * expert_out

        output = shared_out + routed_out
        output = output.view(batch, seq_len, n_embd)

        aux_loss = None
        if self.training:
            aux_loss = self._load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss
