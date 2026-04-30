"""注意力机制：首版实现 Causal MHA，后续加入 MQA/GQA/Flash 等。"""

from core.attention.mha import CausalSelfAttention

__all__ = ["CausalSelfAttention"]
