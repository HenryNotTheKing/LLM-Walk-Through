"""Rollout engine protocol shared by fake and vLLM backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SamplingConfig:
    num_generations: int = 8
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)
    seed: int | None = None


@dataclass(frozen=True)
class RolloutOutput:
    prompt: str
    response: str
    prompt_index: int
    generation_index: int
    metadata: dict


class RolloutEngine(Protocol):
    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[RolloutOutput]:
        ...
