"""归一化模块集合。

包含从经典 LayerNorm 到最新非线性替代方案（DyT、Derf）的多种实现，
并提供 ``build_norm`` 工厂函数，支持通过字符串配置一键切换 norm 类型。
"""

from __future__ import annotations

from core.norm.deep_norm import DeepNorm, get_deepnorm_constants
from core.norm.derf import Derf
from core.norm.dyt import DyT
from core.norm.layer_norm import LayerNorm
from core.norm.rmsnorm import RMSNorm
from core.norm.scale_norm import ScaleNorm

__all__ = [
    "LayerNorm",
    "RMSNorm",
    "ScaleNorm",
    "DeepNorm",
    "DyT",
    "Derf",
    "build_norm",
    "get_deepnorm_constants",
]


# 名称到类的映射表（不区分大小写、支持常见别名）
_NORM_REGISTRY: dict[str, type] = {
    "layernorm": LayerNorm,
    "layer_norm": LayerNorm,
    "rmsnorm": RMSNorm,
    "rms_norm": RMSNorm,
    "scalenorm": ScaleNorm,
    "scale_norm": ScaleNorm,
    "deepnorm": DeepNorm,
    "deep_norm": DeepNorm,
    "dyt": DyT,
    "dynamic_tanh": DyT,
    "derf": Derf,
    "dynamic_erf": Derf,
}


def build_norm(name: str, normalized_shape: int, **kwargs):
    """通过名称字符串构造归一化层。

    Args:
        name: norm 类型名称，不区分大小写。支持：
            layernorm / layer_norm, rmsnorm / rms_norm,
            scalenorm / scale_norm, deepnorm / deep_norm,
            dyt / dynamic_tanh, derf / dynamic_erf。
        normalized_shape: 最后一维的特征维度。
        **kwargs: 额外构造参数，会透传给对应类的 ``__init__``。
            - 对 ``deepnorm``，若未提供 ``alpha``，则自动按 ``n_layer=12``、
              ``arch_type='decoder'`` 计算默认值。

    Returns:
        对应的归一化模块实例。

    Raises:
        ValueError: 若名称不在注册表中。
    """
    cls = _NORM_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"未知的 norm 类型: {name!r}。"
            f"可用选项: {list(_NORM_REGISTRY.keys())}"
        )
    if cls is DeepNorm and "alpha" not in kwargs:
        n_layer = kwargs.pop("n_layer", 12)
        arch_type = kwargs.pop("arch_type", "decoder")
        kwargs["alpha"] = get_deepnorm_constants(n_layer, arch_type=arch_type)[0]
    return cls(normalized_shape, **kwargs)
