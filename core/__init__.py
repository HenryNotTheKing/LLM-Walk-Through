"""LLM Walk-Through 核心包。

各子模块对应开题报告 2.1 节的六大可替换单元：
    - tokenizer / position / norm / attention / ffn / kv_cache
首版只实现 GPT-2 主线所需组件，其余目录留作后续 LLaMA 化改造的占位。
"""

__version__ = "0.1.0"
