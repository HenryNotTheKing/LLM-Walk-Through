"""Deterministic rollout backend used by CPU smoke tests."""

from __future__ import annotations

from .base import RolloutOutput, SamplingConfig


class FakeRolloutEngine:
    def __init__(self, response_template: str = "```python\nprint(1)\n```") -> None:
        self.response_template = response_template

    def generate(self, prompts: list[str], sampling: SamplingConfig) -> list[RolloutOutput]:
        outputs: list[RolloutOutput] = []
        for prompt_index, prompt in enumerate(prompts):
            for generation_index in range(sampling.num_generations):
                outputs.append(
                    RolloutOutput(
                        prompt=prompt,
                        response=self.response_template,
                        prompt_index=prompt_index,
                        generation_index=generation_index,
                        metadata={},
                    )
                )
        return outputs
