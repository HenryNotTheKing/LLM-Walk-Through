"""Warmup + decay schedule for SFT and online RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class WarmupDecaySchedule:
    total_steps: int
    warmup_steps: int
    tracks: dict[str, dict[str, float]]
    decay_shape: str = "cosine"
    _step: int = 0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps)")
        if self.decay_shape not in {"cosine", "linear", "constant"}:
            raise ValueError(f"unknown decay_shape={self.decay_shape}")

    @classmethod
    def from_config(cls, *, total_steps: int, warmup_steps: int, tracks: dict[str, dict[str, float]], decay_shape: str = "cosine") -> "WarmupDecaySchedule":
        return cls(total_steps=total_steps, warmup_steps=warmup_steps, tracks=tracks, decay_shape=decay_shape)

    def lrs_at(self, step: int) -> dict[str, float]:
        return {name: self.lr_at(step, name) for name in self.tracks}

    def lr_at(self, step: int, name: str) -> float:
        track = self.tracks[name]
        peak_lr = float(track["peak_lr"])
        final_lr = float(track.get("final_lr", 0.0))
        if step < self.warmup_steps:
            return peak_lr * step / max(1, self.warmup_steps)
        if self.decay_shape == "constant":
            return peak_lr
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if self.decay_shape == "linear":
            factor = 1.0 - progress
        else:
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr + (peak_lr - final_lr) * factor

    def step_to(self, step: int) -> dict[str, float]:
        self._step = int(step)
        return self.lrs_at(self._step)

    def advance(self) -> dict[str, float]:
        self._step += 1
        return self.lrs_at(self._step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "tracks": self.tracks,
            "decay_shape": self.decay_shape,
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.tracks = dict(state["tracks"])
        self.decay_shape = str(state.get("decay_shape", "cosine"))
        self._step = int(state.get("step", 0))


def apply_lrs(optimizers: dict[str, Any], lrs: dict[str, float]) -> None:
    for name, optimizer in optimizers.items():
        if name not in lrs:
            continue
        for group in optimizer.param_groups:
            group["lr"] = float(lrs[name])
