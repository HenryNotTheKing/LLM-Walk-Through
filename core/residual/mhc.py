"""DeepSeek Manifold-Constrained Hyper-Connections（mHC）。

来源：DeepSeek-AI, *mHC: Manifold-Constrained Hyper-Connections* (2025,
arXiv:2512.24880)；基于 Hyper-Connections (Zhu et al., 2024, arXiv:2409.19606)。

将残差流从单向量扩展为 n 路并行流 ``x ∈ R^{n×C}``，每层学习
读 (H_pre)、写 (H_post)、混合 (H_res) 映射。H_res 经 Sinkhorn-Knopp
投影到双随机矩阵（Birkhoff 多面体），保证谱范数 ≤ 1，避免 HC 的
信号爆炸问题。

当 n=1 时退化为标准 PreNorm 残差连接。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinkhorn_knopp(
    logits: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """将 logits 投影为双随机矩阵。

    Args:
        logits: 形状 ``(..., n, n)``。
    """
    m = logits.exp()
    for _ in range(num_iters):
        m = m / m.sum(dim=-1, keepdim=True).clamp(min=eps)
        m = m / m.sum(dim=-2, keepdim=True).clamp(min=eps)
    return m


class ManifoldHyperConnections(nn.Module):
    """单层 mHC 映射：读/写/混合 + 子层调用。"""

    def __init__(
        self,
        dim: int,
        n_streams: int = 4,
        alpha: float = 0.01,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_streams = n_streams
        self.alpha = alpha
        self.sinkhorn_iters = sinkhorn_iters

        flat = n_streams * dim
        self.phi_pre = nn.Parameter(torch.randn(flat, n_streams) * 0.02)
        self.phi_post = nn.Parameter(torch.randn(flat, n_streams) * 0.02)
        self.phi_res = nn.Parameter(torch.randn(flat, n_streams * n_streams) * 0.02)
        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams * n_streams))

    def compute_maps(self, x_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从 n 路残差流计算 H_pre, H_post, H_res。

        Args:
            x_streams: ``(B, T, n, C)``。

        Returns:
            H_pre: ``(B, T, n)``，H_post: ``(B, T, n)``，H_res: ``(B, T, n, n)``。
        """
        B, T, n, C = x_streams.shape
        x_flat = x_streams.reshape(B, T, n * C)

        h_pre = self.alpha * (x_flat @ self.phi_pre) + self.b_pre
        h_post = self.alpha * (x_flat @ self.phi_post) + self.b_post
        h_res = self.alpha * (x_flat @ self.phi_res) + self.b_res

        h_pre = h_pre.sigmoid()
        h_pre = h_pre / h_pre.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        h_post = 2.0 * h_post.sigmoid()
        h_res = h_res.view(B, T, n, n)
        h_res = sinkhorn_knopp(h_res, num_iters=self.sinkhorn_iters)
        return h_pre, h_post, h_res

    def forward(
        self,
        x_streams: torch.Tensor,
        sublayer: nn.Module,
        norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """应用 mHC 包裹子层。

        Args:
            x_streams: ``(B, T, n, C)``。
            sublayer: Attention 或 FFN，输入 ``(B, T, C)``。
            norm: PreNorm（可选）。

        Returns:
            更新后的 ``(B, T, n, C)``。
        """
        h_pre, h_post, h_res = self.compute_maps(x_streams)

        h_in = torch.einsum("btn,btnc->btc", h_pre, x_streams)

        if norm is not None:
            h_in = norm(h_in)
        f_out = sublayer(h_in)

        f_broadcast = h_post.unsqueeze(-1) * f_out.unsqueeze(-2)
        x_mixed = torch.einsum("btij,btjc->btic", h_res, x_streams)
        return x_mixed + f_broadcast


def expand_to_streams(x: torch.Tensor, n_streams: int) -> torch.Tensor:
    """将 ``(B, T, C)`` 复制/扩展为 ``(B, T, n, C)``。"""
    return x.unsqueeze(-2).expand(-1, -1, n_streams, -1).contiguous()


def collapse_streams(x_streams: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """将 n 路流折叠回 ``(B, T, C)``。"""
    if mode == "mean":
        return x_streams.mean(dim=-2)
    if mode == "sum":
        return x_streams.sum(dim=-2)
    raise ValueError(f"未知 collapse mode: {mode}")


class PreNormBlockWithMHC(nn.Module):
    """PreNorm 残差块 + mHC 多路残差流包装。"""

    def __init__(
        self,
        dim: int,
        n_streams: int,
        attn_norm: nn.Module,
        attn: nn.Module,
        ffn_norm: nn.Module,
        ffn: nn.Module,
        sinkhorn_iters: int = 20,
    ) -> None:
        super().__init__()
        self.mhc_attn = ManifoldHyperConnections(
            dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters
        )
        self.mhc_ffn = ManifoldHyperConnections(
            dim, n_streams=n_streams, sinkhorn_iters=sinkhorn_iters
        )
        self.attn_norm = attn_norm
        self.attn = attn
        self.ffn_norm = ffn_norm
        self.ffn = ffn

    def forward(self, x_streams: torch.Tensor) -> torch.Tensor:
        x_streams = self.mhc_attn(
            x_streams, self.attn, norm=self.attn_norm
        )
        x_streams = self.mhc_ffn(
            x_streams, self.ffn, norm=self.ffn_norm
        )
        return x_streams
