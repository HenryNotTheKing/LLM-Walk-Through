"""位置编码模块：包含绝对、相对与旋转式位置编码实现。

- LearnedPositionalEmbedding：GPT-2 式可学习绝对位置向量
- SinusoidalPositionalEncoding：Vaswani 等正弦固定位置编码
- RotaryPositionalEmbedding：Su 等旋转位置编码（RoPE）
- YarnRotaryPositionalEmbedding：Peng 等 YaRN 长上下文扩展
- ALiBiPositionalBias：Press 等线性注意力偏置
"""

from core.position.learned import LearnedPositionalEmbedding
from core.position.sinusoidal import SinusoidalPositionalEncoding
from core.position.rope import RotaryPositionalEmbedding
from core.position.yarn import YarnRotaryPositionalEmbedding
from core.position.alibi import ALiBiPositionalBias

__all__ = [
    "LearnedPositionalEmbedding",
    "SinusoidalPositionalEncoding",
    "RotaryPositionalEmbedding",
    "YarnRotaryPositionalEmbedding",
    "ALiBiPositionalBias",
]
