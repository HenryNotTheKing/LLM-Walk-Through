"""DeepNorm。

来源：Wang et al., *DeepNet: Scaling Transformers to 1,000 Layers* (2022,
arXiv:2203.00555)。

DeepNorm 并不是传统意义上的"归一化函数"，而是一套**深度相关的残差缩放
+ Post-LayerNorm + 权重初始化**的联合方案，用来训练极深的 Transformer。

核心公式（以 Decoder-only 为例）：
    x_{l+1} = LayerNorm( α · x_l + G_l(x_l) )

其中 α > 1 放大残差分支，同时 G_l 内部特定权重矩阵在 Xavier 初始化后
再乘以一个小的深度相关因子 β < 1，使得每层对主信号的贡献保持 O(1)。

深度常数：
    - Encoder-only (N 层):   α = (2N)^{1/4},   β = (8N)^{-1/4}
    - Decoder-only (M 层):   α = (2M)^{1/4},   β = (8M)^{-1/4}
    - Encoder-Decoder (Enc): α = 0.81·(N^4·M)^{1/16}, β = 0.87·(N^4·M)^{-1/16}
    - Encoder-Decoder (Dec): α = (3M)^{1/4},   β = (12M)^{-1/4}
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def get_deepnorm_constants(
    n_layer: int,
    arch_type: str = "decoder",
    n_enc: int | None = None,
) -> tuple[float, float]:
    """计算 DeepNorm 的深度相关常数 (α, β)。

    Args:
        n_layer: 当前侧的层数（Encoder-only 时为 N，Decoder-only 时为 M，
                 Encoder-Decoder 时 Encoder 侧为 N，Decoder 侧为 M）。
        arch_type: "encoder" | "decoder" | "encoder_decoder".
        n_enc: Encoder-Decoder 架构下 Encoder 的层数（仅当 arch_type 为
               "encoder_decoder" 的 Decoder 侧时需要）。

    Returns:
        (α, β) 两个缩放常数。
    """
    if arch_type == "encoder":
        alpha = math.pow(2 * n_layer, 1.0 / 4.0)
        beta = math.pow(8 * n_layer, -1.0 / 4.0)
    elif arch_type == "decoder":
        alpha = math.pow(2 * n_layer, 1.0 / 4.0)
        beta = math.pow(8 * n_layer, -1.0 / 4.0)
    elif arch_type == "encoder_decoder":
        # 这里假设调用时 n_layer 就是 Encoder 侧层数 N
        if n_enc is None:
            raise ValueError("arch_type='encoder_decoder' 时必须提供 n_enc")
        # Decoder 侧层数 M 由调用方通过 n_layer 传入
        # 但公式需要同时知道 N 和 M。为了接口统一，
        # 约定：arch_type='encoder_decoder' 时，n_layer 始终指**当前侧**层数。
        # 若当前侧是 Encoder，则 n_enc 被忽略（直接用 n_layer 作为 N）。
        # 若当前侧是 Decoder，则 n_enc 是 Encoder 层数 N，n_layer 是 Decoder 层数 M。
        # 本函数按 Encoder 侧计算：
        alpha = 0.81 * math.pow((n_layer ** 4) * n_enc, 1.0 / 16.0)
        beta = 0.87 * math.pow((n_layer ** 4) * n_enc, -1.0 / 16.0)
    else:
        raise ValueError(f"不支持的 arch_type: {arch_type}")
    return alpha, beta


class DeepNorm(nn.Module):
    """DeepNorm：带深度相关残差缩放的 Post-LayerNorm。

    本模块封装了 ``α * x + sublayer(x)`` 再经过 LayerNorm 的过程。
    用户仍需在模型初始化阶段，对残差路径上的特定权重矩阵（FFN 中间/输出层、
    Attention 的 V 和 Output 投影）乘上 β 系数。

    Args:
        normalized_shape: 特征维度。
        alpha: 残差分支的放大系数（由 ``get_deepnorm_constants`` 计算）。
        eps: LayerNorm 的数值稳定项。
        bias: LayerNorm 是否使用 bias。
    """

    def __init__(
        self,
        normalized_shape: int,
        alpha: float,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=True, bias=bias)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        """执行 DeepNorm 残差更新。

        Args:
            x: 输入张量 (..., normalized_shape)。
            sublayer_out: 子层（Attention 或 FFN）的输出，与 x 同 shape。

        Returns:
            LayerNorm(α·x + sublayer_out)。
        """
        return self.layer_norm(self.alpha * x + sublayer_out)
