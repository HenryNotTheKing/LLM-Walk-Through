"""设备选择与 dtype 解析。

约定：
    - ``device='auto'`` 时按 cuda > mps > cpu 顺序自动选择。
    - ``dtype='auto'`` 时：cuda 上选 bfloat16（若支持，否则 float16），mps/cpu 选 float32。
"""

from __future__ import annotations

import torch


def select_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(preferred: str, device: torch.device) -> torch.dtype:
    if preferred != "auto":
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[preferred]
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def amp_enabled(device: torch.device, dtype: torch.dtype, requested: bool) -> bool:
    """判断是否真正启用 autocast：仅在 CUDA + 半精度 + requested=True 时。"""
    return bool(requested and device.type == "cuda" and dtype != torch.float32)
