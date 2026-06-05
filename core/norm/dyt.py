"""DyT — Dynamic Tanh（无统计归一化的逐元素非线性替代）。

来源：Zhu et al., *Transformers without Normalization* (2025, arXiv:2503.10622)。

核心洞察：LayerNorm 的输入-输出映射在统计上呈现类 tanh 的 S 型曲线。
与其先算 mean/var 再做仿射，不如直接用可学习的 tanh 进行有界非线性变换。

公式：
    DyT(x) = tanh(α · x) · γ

其中 α 是可学习标量（控制饱和区位置），γ 是逐通道可学习缩放。

优点：
    - 完全逐元素，无 reduction 操作，GPU 利用率更高。
    - 可无缝替换 Pre-LN/RMSNorm，无需修改模型其他部分。
    - 在 ViT、LLaMA、DiT、wav2vec 2.0 上已验证与 LayerNorm 相当。

注意：
    - 深层网络中对 α 的初始化敏感，建议从较小值开始（如 0.5）。
    - 与某些优化器（如 Muon）搭配时可能出现耦合问题，需适当调参。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DyT(nn.Module):
    """Dynamic Tanh 归一化替代层。

    Args:
        normalized_shape: 特征维度，用于生成逐通道 γ。
        alpha_init: α 的初始值，默认 0.5；若训练不稳定可尝试 0.3。
        gamma_init: γ 的初始值，默认 1.0。
        use_gamma: 是否使用逐通道 γ；若为 False，则退化为纯 tanh(αx)。
    """

    def __init__(
        self,
        normalized_shape: int,
        alpha_init: float = 0.5,
        gamma_init: float = 1.0,
        use_gamma: bool = True,
    ) -> None:
        super().__init__()
        # α 用对数参数化以保持正值，并允许较小初始梯度
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), dtype=torch.float32))
        if use_gamma:
            self.gamma = nn.Parameter(torch.full((normalized_shape,), gamma_init, dtype=torch.float32))
        else:
            self.register_parameter("gamma", None)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.float()
        alpha = self.alpha.to(x_fp32.dtype)
        out = torch.tanh(alpha * x_fp32)
        if self.gamma is not None:
            out = out * self.gamma.to(x_fp32.dtype)
        return out.to(orig_dtype)
