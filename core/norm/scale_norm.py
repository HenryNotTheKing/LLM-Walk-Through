"""ScaleNorm。

来源：Nguyen & Chiang, *Transformers without Tears: Improving the Normalization
of Self-Attention* (2019)。

公式：
    ScaleNorm(x; g) = g · x / ||x||_2

与 LayerNorm 的区别：
    - 不减均值、不除标准差，而是用 ℓ2 范数做整体缩放。
    - 只有一个可学习标量 g（而非逐维的 weight/bias），参数量极少。
    - 计算量约为 LayerNorm 的一半，适合低资源或隐私训练场景。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScaleNorm(nn.Module):
    """基于 ℓ2 范数的缩放归一化。

    Args:
        normalized_shape: 最后一维的特征维度（用于类型检查，实际只产生一个标量参数）。
        eps: 数值稳定项，防止除零。
        g: 若提供则直接作为初始缩放值，否则默认为 1.0。
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, g: float | None = None) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        init_g = g if g is not None else 1.0
        self.g = nn.Parameter(torch.tensor(init_g, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 计算以提高数值稳定性
        orig_dtype = x.dtype
        x_fp32 = x.float()
        # 在最后一维计算 ℓ2 范数
        norm = x_fp32.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        out = x_fp32 / norm * self.g.to(x_fp32.dtype)
        return out.to(orig_dtype)
