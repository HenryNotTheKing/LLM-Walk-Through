"""前馈网络：涵盖密集 MLP、GLU 变体与 MoE 稀疏架构。

- GeluMLP: GPT-2 经典 GELU 感知机
- SwiGLUMLP: LLaMA / Mistral 门控前馈
- GEGLUMLP: PaLM 风格 GELU 门控前馈
- ReGLUMLP: ReLU 门控前馈
- TopKMoE: Switch/GShard 基础 top-k 专家混合
- DeepSeekMoE: 共享专家 + 路由专家的细粒度 MoE
"""

from core.ffn.mlp import GeluMLP
from core.ffn.swiglu import SwiGLUMLP
from core.ffn.geglu import GEGLUMLP
from core.ffn.reglu import ReGLUMLP
from core.ffn.moe_base import TopKMoE
from core.ffn.moe_deepseek import DeepSeekMoE

__all__ = [
    "GeluMLP",
    "SwiGLUMLP",
    "GEGLUMLP",
    "ReGLUMLP",
    "TopKMoE",
    "DeepSeekMoE",
]
