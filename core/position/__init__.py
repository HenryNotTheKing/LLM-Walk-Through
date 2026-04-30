"""位置编码：首版仅实现 GPT-2 的 learned absolute position embedding。

后续会在此目录加入 sinusoidal/ALiBi/RoPE/NTK/YaRN 等。
"""

from core.position.learned import LearnedPositionalEmbedding

__all__ = ["LearnedPositionalEmbedding"]
