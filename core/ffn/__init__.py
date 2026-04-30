"""前馈网络：首版实现 GPT-2 的 GELU MLP，后续加入 SwiGLU/MoE 等。"""

from core.ffn.mlp import GeluMLP

__all__ = ["GeluMLP"]
