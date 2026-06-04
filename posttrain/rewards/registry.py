"""Composable reward registry used by GRPO/DAPO."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping


@dataclass(frozen=True)
class RewardInput:
    prompt: str
    response: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardScore:
    score: float
    components: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardConfig:
    name: str
    weight: float = 1.0
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    clip_min: float | None = None
    clip_max: float | None = None


RewardImpl = Callable[[RewardInput], float]


def build_reward_fn(configs: Iterable[RewardConfig | Mapping[str, Any]]) -> Callable[[list[RewardInput]], list[RewardScore]]:
    reward_configs = [_coerce_config(config) for config in configs]
    reward_impls = [(config, _build_single_reward(config.name, dict(config.kwargs))) for config in reward_configs]

    def reward_fn(items: list[RewardInput]) -> list[RewardScore]:
        results: list[RewardScore] = []
        for item in items:
            total = 0.0
            components: dict[str, float] = {}
            for config, impl in reward_impls:
                raw = float(impl(item))
                clipped = raw
                if config.clip_min is not None:
                    clipped = max(float(config.clip_min), clipped)
                if config.clip_max is not None:
                    clipped = min(float(config.clip_max), clipped)
                components[config.name] = clipped
                total += float(config.weight) * clipped
            results.append(RewardScore(score=float(total), components=components))
        return results

    return reward_fn


def _coerce_config(config: RewardConfig | Mapping[str, Any]) -> RewardConfig:
    if isinstance(config, RewardConfig):
        return config
    return RewardConfig(
        name=str(config["name"]),
        weight=float(config.get("weight", 1.0)),
        kwargs=dict(config.get("kwargs", {})),
        clip_min=config.get("clip_min"),
        clip_max=config.get("clip_max"),
    )


def _build_single_reward(name: str, kwargs: dict[str, Any]) -> RewardImpl:
    if name == "contains":
        pattern = str(kwargs["pattern"])
        return lambda item: 1.0 if pattern in item.response else 0.0
    if name == "regex":
        pattern = re.compile(str(kwargs["pattern"]), flags=int(kwargs.get("flags", 0)))
        return lambda item: 1.0 if pattern.search(item.response) else 0.0
    if name == "length":
        min_chars = int(kwargs.get("min_chars", 0))
        max_chars = int(kwargs.get("max_chars", 10**12))
        return lambda item: 1.0 if min_chars <= len(item.response) <= max_chars else 0.0
    if name == "code_block":
        language = str(kwargs.get("language", ""))
        needle = f"```{language}" if language else "```"
        return lambda item: 1.0 if needle in item.response and item.response.count("```") >= 2 else 0.0
    if name == "repeat_penalty":
        max_ratio = float(kwargs.get("max_repetition_ratio", 0.3))
        return lambda item: 1.0 - min(1.0, _repetition_ratio(item.response) / max(max_ratio, 1e-8))
    if name == "code_execution":
        pass_markers = [str(marker) for marker in kwargs.get("pass_markers", ["PASS", "passed"])]
        fail_markers = [str(marker) for marker in kwargs.get("fail_markers", ["FAIL", "FAILED", "Traceback"])]
        return lambda item: _execution_score(item, pass_markers=pass_markers, fail_markers=fail_markers)
    raise KeyError(f"unknown reward function: {name}")


def _execution_score(item: RewardInput, *, pass_markers: list[str], fail_markers: list[str]) -> float:
    if item.metadata.get("passed") is True:
        return 1.0
    if item.metadata.get("passed") is False:
        return 0.0
    joined = "\n".join(
        str(item.metadata.get(key, "") or "") for key in ("stdout", "stderr", "result", "status")
    )
    if any(marker in joined for marker in fail_markers):
        return 0.0
    if any(marker in joined for marker in pass_markers):
        return 1.0
    return 0.0


def _repetition_ratio(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))
