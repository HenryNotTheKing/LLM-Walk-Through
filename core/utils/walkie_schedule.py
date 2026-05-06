"""Walkie 学习率调度：连续两阶段 WSD（Warmup-Stable-Decay）。

整个训练只有 **一个** step 计数器，但被分成两段数据流：
    - 主训练阶段（main）：``[0, anneal_start)``，使用大语料。
    - 退火阶段（anneal）：``[anneal_start, total_steps)``，切换到高质量小语料。

学习率轨迹：

    - ``warmup``: ``[0, warmup_steps)``，从 0 线性升到 peak_lr。
    - ``stable``: ``[warmup_steps, anneal_start)``，常量 peak_lr。
    - ``decay``: ``[anneal_start, total_steps]``，按 ``decay_shape``（默认 ``sqrt``）
      光滑下降到 ``final_lr``。在 ``step == anneal_start`` 处恰好等于 peak_lr，
      因此退火切换不会产生 LR 跳变（学习率连续）。

AdamW 与 Muon 共享同一个调度形状，但允许各自有独立的 ``peak_lr`` / ``final_lr``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _LRTrack:
    peak_lr: float
    final_lr: float


@dataclass
class WalkieWSDSchedule:
    total_steps: int
    warmup_steps: int
    anneal_start_ratio: float = 0.8
    decay_shape: str = "sqrt"  # "sqrt" | "linear" | "cosine"
    # 不同优化器分别配置 lr，键名建议与 ``build_walkie_optimizers`` 返回的字典一致
    tracks: dict[str, _LRTrack] = field(default_factory=dict)
    _step: int = 0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正")
        if not (0.0 < self.anneal_start_ratio < 1.0):
            raise ValueError("anneal_start_ratio 必须在 (0, 1) 之间")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps 必须小于 total_steps")
        if self.decay_shape not in ("sqrt", "linear", "cosine"):
            raise ValueError(f"未知 decay_shape={self.decay_shape}")

    # ----- 工厂 -----
    @classmethod
    def from_config(
        cls,
        *,
        total_steps: int,
        warmup_steps: int,
        anneal_start_ratio: float,
        decay_shape: str,
        tracks: dict[str, dict[str, float]],
    ) -> "WalkieWSDSchedule":
        return cls(
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            anneal_start_ratio=anneal_start_ratio,
            decay_shape=decay_shape,
            tracks={
                name: _LRTrack(peak_lr=float(t["peak_lr"]), final_lr=float(t["final_lr"]))
                for name, t in tracks.items()
            },
        )

    # ----- 核心 -----
    @property
    def anneal_start(self) -> int:
        return int(self.total_steps * self.anneal_start_ratio)

    def current_stage(self, step: int) -> str:
        return "anneal" if step >= self.anneal_start else "main"

    def _shape_factor(self, step: int) -> float:
        """返回 [0, 1] 之间的 lr 缩放因子。"""
        if step < self.warmup_steps:
            # 线性 warmup：保证 step=0 时严格为 0
            return step / max(1, self.warmup_steps)
        anneal_start = self.anneal_start
        if step < anneal_start:
            return 1.0
        # decay 段
        if step >= self.total_steps:
            return 0.0  # 之后会用 final_lr 兜底
        progress = (step - anneal_start) / max(1, self.total_steps - anneal_start)
        progress = min(max(progress, 0.0), 1.0)
        if self.decay_shape == "sqrt":
            return math.sqrt(1.0 - progress)
        if self.decay_shape == "linear":
            return 1.0 - progress
        # cosine：从 1 平滑到 0
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def lr_at(self, step: int, name: str) -> float:
        track = self.tracks[name]
        if step < self.warmup_steps:
            return track.peak_lr * (step / max(1, self.warmup_steps))
        if step < self.anneal_start:
            return track.peak_lr
        # decay：保证两端连续——shape=1 时 lr=peak_lr，shape=0 时 lr=final_lr
        f = self._shape_factor(step)
        return track.final_lr + (track.peak_lr - track.final_lr) * f

    def lrs_at(self, step: int) -> dict[str, float]:
        return {name: self.lr_at(step, name) for name in self.tracks}

    def step_to(self, step: int) -> dict[str, float]:
        """显式跳到 ``step``（恢复时常用），返回该步 lr 字典。"""
        self._step = int(step)
        return self.lrs_at(self._step)

    def advance(self) -> dict[str, float]:
        """推进 1 个 step，返回最新 lr 字典。"""
        self._step += 1
        return self.lrs_at(self._step)

    @property
    def step(self) -> int:
        return self._step

    # ----- 序列化 -----
    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "anneal_start_ratio": self.anneal_start_ratio,
            "decay_shape": self.decay_shape,
            "tracks": {
                name: {"peak_lr": t.peak_lr, "final_lr": t.final_lr}
                for name, t in self.tracks.items()
            },
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.anneal_start_ratio = float(state["anneal_start_ratio"])
        self.decay_shape = state["decay_shape"]
        self.tracks = {
            name: _LRTrack(peak_lr=float(t["peak_lr"]), final_lr=float(t["final_lr"]))
            for name, t in state["tracks"].items()
        }
        self._step = int(state.get("step", 0))


def apply_lrs_to_optimizers(
    optimizers: dict[str, Any], lrs: dict[str, float]
) -> None:
    """把 ``lrs[name]`` 写入对应优化器所有 param_group 的 ``lr``。"""
    for name, opt in optimizers.items():
        if name not in lrs:
            continue
        lr = lrs[name]
        for g in opt.param_groups:
            g["lr"] = lr
