"""Walkie-Code-1B 优化器：纯 PyTorch 的 AdamW + Muon。

设计：
    - **AdamW**：直接复用 ``torch.optim.AdamW``，托管所有 1D / embedding / norm / bias 参数。
    - **Muon**：自研实现，托管 2D 矩阵权重（Q/K/V/O 投影、gate/up/down 投影）。
      Muon 对动量后的更新做 Newton-Schulz 迭代，将其投影到正交空间，使更新方向
      在每个奇异方向上具有相近的步长。

参考：
    - Keller Jordan 的 Muon (https://github.com/KellerJordan/Muon) 中的 Newton-Schulz 系数。
    - 这里**不**安装外部包，全部用 ``torch`` 原生算子实现，符合“PyTorch only”要求。
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


# ---------------------------------------------------------------------------
# Newton-Schulz 正交化
# ---------------------------------------------------------------------------
@torch.no_grad()
def zeropower_via_newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """对矩阵 ``G`` 做 5 阶 Newton-Schulz 迭代，近似计算 ``U V^T``（即 G 的极分解中的正交因子）。

    系数沿用 Keller Jordan 的稳健配置 ``(3.4445, -4.7750, 2.0315)``。
    输入是 2D 张量；为了避免复制太大权重，在 ``bf16`` 下计算迭代再 cast 回来。
    """
    assert G.ndim == 2, "Muon 仅作用于 2D 权重"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16) if G.dtype != torch.bfloat16 and G.is_cuda else G.to(torch.float32)
    # 让短边在前以减小中间张量
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t()
    # 归一化到谱范数 ≈ 1，使迭代收敛
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.t()
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# Muon 优化器
# ---------------------------------------------------------------------------
class Muon(Optimizer):
    """对 2D 矩阵权重做正交化更新的优化器。

    Args:
        params: 仅包含 2D 权重的参数迭代器。
        lr: 学习率。
        momentum: 动量系数（heavy-ball / Nesterov）。
        nesterov: 是否使用 Nesterov 动量。
        weight_decay: 解耦权重衰减系数。
        ns_steps: Newton-Schulz 迭代步数，默认 5。
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr 不能为负: {lr}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Muon 不支持稀疏梯度")
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon 期望 2D 权重，但收到 ndim={p.ndim}，请检查参数分组。"
                    )
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if nesterov else buf

                # 把更新方向正交化（同时做形状无关的尺度归一）
                ortho = zeropower_via_newton_schulz5(update, steps=ns_steps)
                # 保留与 Adam 量级相当的更新尺度：max(1, fan_out/fan_in)^0.5
                fan_out, fan_in = p.shape
                scale = max(1.0, (fan_out / fan_in) ** 0.5)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(ortho, alpha=-lr * scale)

        return loss


# ---------------------------------------------------------------------------
# 参数分组与构造
# ---------------------------------------------------------------------------
def _is_muon_param(name: str, param: torch.Tensor) -> bool:
    """规则：只在 2D 矩阵权重上启用 Muon，且排除 embedding / lm_head / norm。"""
    if param.ndim != 2:
        return False
    lname = name.lower()
    if "embed" in lname or "lm_head" in lname:
        return False
    if "norm" in lname:
        return False
    # q/k/v/o + gate/up/down 都是合规目标
    return True


def split_walkie_params(
    model: nn.Module,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], list[str]]:
    """根据规则把模型参数拆成 (muon_params, adamw_params, muon_names, adamw_names)。"""
    muon, adamw, mn, an = [], [], [], []
    seen: set[int] = set()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen:  # 处理 tied weights，避免重复加入两个组
            continue
        seen.add(id(p))
        if _is_muon_param(name, p):
            muon.append(p)
            mn.append(name)
        else:
            adamw.append(p)
            an.append(name)
    return muon, adamw, mn, an


def build_walkie_optimizers(
    model: nn.Module,
    *,
    adamw_lr: float,
    muon_lr: float,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_weight_decay: float = 0.1,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_weight_decay: float = 0.0,
    muon_ns_steps: int = 5,
) -> dict[str, Optimizer]:
    """返回 ``{"adamw": ..., "muon": ...}``。"""
    muon_params, adamw_params, _, _ = split_walkie_params(model)
    optimizers: dict[str, Optimizer] = {}
    if adamw_params:
        optimizers["adamw"] = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        )
    if muon_params:
        optimizers["muon"] = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            weight_decay=muon_weight_decay,
            ns_steps=muon_ns_steps,
        )
    if not optimizers:
        raise RuntimeError("模型里没有任何可训练参数？")
    return optimizers
