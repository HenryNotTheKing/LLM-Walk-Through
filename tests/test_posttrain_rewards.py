"""Post-training reward registry and sandbox adapters."""

from __future__ import annotations

import pytest

from posttrain.rewards.code_execution import SandboxCodeRewardRunner
from posttrain.rewards.registry import RewardConfig, RewardInput, build_reward_fn
from posttrain.sandbox.jupyter_client import JupyterExecutionResult, parse_jupyter_response


def test_weighted_reward_registry_combines_rule_rewards() -> None:
    reward_fn = build_reward_fn(
        [
            RewardConfig(name="contains", weight=2.0, kwargs={"pattern": "```python"}),
            RewardConfig(name="length", weight=0.5, kwargs={"min_chars": 10, "max_chars": 80}),
        ]
    )

    scores = reward_fn(
        [
            RewardInput(prompt="", response="```python\nprint(1)\n```", metadata={}),
            RewardInput(prompt="", response="short", metadata={}),
        ]
    )

    assert scores[0].score == 2.5
    assert scores[0].components == {"contains": 1.0, "length": 1.0}
    assert scores[1].score == 0.0


def test_parse_jupyter_response_matches_multimodal_sandbox_schema() -> None:
    payload = {
        "status": "success",
        "execution_time": 0.25,
        "output": {
            "stdout": "ok\n",
            "stderr": "",
            "result": "42",
            "images": ["ignored-base64"],
        },
    }

    result = parse_jupyter_response(payload)

    assert result == JupyterExecutionResult(
        status="success",
        execution_time=0.25,
        stdout="ok\n",
        stderr="",
        result="42",
        images=["ignored-base64"],
    )


def test_code_execution_reward_scores_pass_from_stdout() -> None:
    reward_fn = build_reward_fn(
        [
            RewardConfig(
                name="code_execution",
                weight=1.0,
                kwargs={"pass_markers": ["ALL TESTS PASSED"], "fail_markers": ["FAILED"]},
            )
        ]
    )

    scores = reward_fn(
        [
            RewardInput(prompt="", response="", metadata={"stdout": "ALL TESTS PASSED"}),
            RewardInput(prompt="", response="", metadata={"stdout": "FAILED"}),
        ]
    )

    assert [score.score for score in scores] == [1.0, 0.0]


@pytest.mark.anyio
async def test_sandbox_code_runner_enriches_reward_metadata() -> None:
    class FakeClient:
        async def run_code(self, code: str, *, session_id: str | None = None) -> JupyterExecutionResult:
            assert "assert add(1, 2) == 3" in code
            return JupyterExecutionResult(
                status="success",
                execution_time=0.1,
                stdout="ALL TESTS PASSED\n",
                stderr="",
                result=None,
                images=[],
            )

    runner = SandboxCodeRewardRunner(FakeClient())
    enriched = await runner.evaluate(
        [
            RewardInput(
                prompt="",
                response="```python\ndef add(a, b):\n    return a + b\n```",
                metadata={"tests": "assert add(1, 2) == 3"},
            )
        ]
    )

    assert enriched[0].metadata["status"] == "success"
    assert enriched[0].metadata["stdout"] == "ALL TESTS PASSED\n"
    assert enriched[0].metadata["execution_time"] == 0.1


@pytest.mark.anyio
async def test_sandbox_code_runner_uses_test_program_template() -> None:
    class FakeClient:
        async def run_code(self, code: str, *, session_id: str | None = None) -> JupyterExecutionResult:
            assert code.startswith("candidate_code = ")
            assert "print(1)" in code
            assert "ALL TESTS PASSED" in code
            return JupyterExecutionResult(
                status="success",
                execution_time=0.1,
                stdout="ALL TESTS PASSED\n",
                stderr="",
                result=None,
                images=[],
            )

    runner = SandboxCodeRewardRunner(FakeClient())
    enriched = await runner.evaluate(
        [
            RewardInput(
                prompt="",
                response="```python\nprint(1)\n```",
                metadata={"test_program_template": "candidate_code = {{candidate_code}}\nprint('ALL TESTS PASSED')"},
            )
        ]
    )

    assert enriched[0].metadata["status"] == "success"