"""Execution reward helpers."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import re
from typing import Protocol

from posttrain.rewards.registry import RewardInput
from posttrain.sandbox.jupyter_client import JupyterExecutionResult


class SandboxClientLike(Protocol):
    async def run_code(self, code: str, *, session_id: str | None = None) -> JupyterExecutionResult:
        ...


def extract_python_code(response: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def build_humaneval_code(response: str, tests: str) -> str:
    code = extract_python_code(response)
    return f"{code}\n\n{tests}\n\nprint('ALL TESTS PASSED')\n"


class SandboxCodeRewardRunner:
    """Execute generated code in the Jupyter sandbox and enrich reward metadata."""

    def __init__(
        self,
        client: SandboxClientLike,
        *,
        tests_field: str = "tests",
        test_program_template_field: str = "test_program_template",
        max_concurrency: int = 8,
    ) -> None:
        self.client = client
        self.tests_field = tests_field
        self.test_program_template_field = test_program_template_field
        self.max_concurrency = int(max_concurrency)

    async def evaluate(self, items: list[RewardInput]) -> list[RewardInput]:
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def run_one(item: RewardInput) -> RewardInput:
            template = item.metadata.get(self.test_program_template_field)
            if isinstance(template, str) and template.strip():
                code = build_template_code(item.response, template)
            else:
                tests = item.metadata.get(self.tests_field, "")
                code = build_humaneval_code(item.response, str(tests))
            async with semaphore:
                result = await self.client.run_code(code)
            metadata = dict(item.metadata)
            metadata.update(
                {
                    "status": result.status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "result": result.result,
                    "execution_time": result.execution_time,
                }
            )
            return replace(item, metadata=metadata)

        return list(await asyncio.gather(*(run_one(item) for item in items)))


def build_template_code(response: str, template: str) -> str:
    code = extract_python_code(response)
    return template.replace("{{candidate_code}}", repr(code))
