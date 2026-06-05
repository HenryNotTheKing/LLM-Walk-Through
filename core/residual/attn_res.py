"""Kimi Attention Residuals（Block AttnRes 教学实现）。

来源：Moonshot AI, *Attention Residuals* (2026, arXiv:2603.15031)。

标准 PreNorm 将每层子层输出等权累加，深层 hidden state 幅度随深度线性增长。
AttnRes 在深度方向做 softmax 加权聚合，使每层能选择性读取历史子层输出：

    α_{i→l} ∝ exp(w_lᵀ RMSNorm(v_i))
    h_l = Σ_i α_{i→l} v_i

Block AttnRes（生产变体）：将 L 层划分为 N 个块，块内标准残差求和，
块间做 AttnRes，内存从 O(Ld) 降至 O(Nd)。

本模块提供状态管理与聚合函数，以及 ``PreNormBlockWithAttnRes`` 教学包装。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from core.norm.rmsnorm import RMSNorm


@dataclass
class AttnResState:
    """Block AttnRes 跨层状态。"""

    blocks: list[torch.Tensor] = field(default_factory=list)
    partial: torch.Tensor | None = None


def block_attn_res(
    values: list[torch.Tensor],
    partial: torch.Tensor,
    w: torch.Tensor,
    norm: RMSNorm,
) -> torch.Tensor:
    """对块摘要列表 + 当前块 partial sum 做深度方向 softmax 聚合。

    Args:
        values: 已完成块的 partial sum，每项形状 ``(B, T, D)``；首项通常为 embedding。
        partial: 当前块内 running sum，形状 ``(B, T, D)``。
        w: 伪查询向量，形状 ``(D,)`` 或 ``(1, D)``。
        norm: 对 key 做 RMSNorm。

    Returns:
        聚合后的 hidden state，形状 ``(B, T, D)``。
    """
    stacked = torch.stack(values + [partial], dim=0)  # (N+1, B, T, D)
    keys = norm(stacked)
    w_vec = w.squeeze()
    logits = torch.einsum("d, n b t d -> n b t", w_vec, keys)
    alpha = logits.softmax(dim=0)
    return torch.einsum("n b t, n b t d -> b t d", alpha, stacked)


def init_attn_res_state(embedding: torch.Tensor) -> AttnResState:
    """用 embedding 初始化状态（作为 block 0）。"""
    return AttnResState(blocks=[embedding], partial=torch.zeros_like(embedding))


def update_attn_res_partial(
    state: AttnResState,
    sublayer_out: torch.Tensor,
) -> AttnResState:
    """块内标准残差累加。"""
    if state.partial is None:
        partial = sublayer_out
    else:
        partial = state.partial + sublayer_out
    return AttnResState(blocks=state.blocks, partial=partial)


def finalize_attn_res_block(state: AttnResState) -> AttnResState:
    """当前块结束，将 partial 推入 blocks 并重置 partial。"""
    if state.partial is None:
        raise ValueError("partial 为空，无法 finalize block")
    new_blocks = state.blocks + [state.partial]
    return AttnResState(blocks=new_blocks, partial=torch.zeros_like(state.partial))


class AttnResAggregator(nn.Module):
    """单层伪查询投影 + key 归一化，用于 Block AttnRes。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w_proj = nn.Linear(dim, 1, bias=False)
        self.key_norm = RMSNorm(dim, eps=eps)

    def forward(
        self,
        state: AttnResState,
        partial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        partial = state.partial if partial is None else partial
        if partial is None:
            raise ValueError("需要 partial 张量")
        w = self.w_proj.weight.squeeze(0)
        return block_attn_res(state.blocks, partial, w, self.key_norm)


class PreNormBlockWithAttnRes(nn.Module):
    """PreNorm 残差块 + Block AttnRes 聚合（教学包装）。

    每个子层（attention / FFN）前先用 AttnRes 聚合历史输出，
    再送入 RMSNorm → 子层 → 块内残差累加。
    """

    def __init__(
        self,
        dim: int,
        attn_norm: nn.Module,
        attn: nn.Module,
        ffn_norm: nn.Module,
        ffn: nn.Module,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.ffn = ffn
        self.attn_res_agg = AttnResAggregator(dim, eps=eps)
        self.mlp_res_agg = AttnResAggregator(dim, eps=eps)

    def forward(
        self,
        state: AttnResState,
    ) -> tuple[torch.Tensor, AttnResState]:
        """返回 (sublayer_outputs_sum, updated_state)。

        块内对 attention 与 FFN 输出做标准累加；调用方在块边界调用
        ``finalize_attn_res_block``。
        """
        h_attn = self.attn_res_agg(state)
        attn_out = self.attn(self.attn_norm(h_attn))
        state = update_attn_res_partial(state, attn_out)

        h_ffn = self.mlp_res_agg(state)
        ffn_out = self.ffn(self.ffn_norm(h_ffn))
        state = update_attn_res_partial(state, ffn_out)

        total = attn_out + ffn_out
        return total, state
