"""Walkie RL data loading helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from posttrain.rewards.code_execution import build_template_code
from posttrain.data.chat_template import ChatTemplate
from posttrain.rollout.base import SamplingConfig
from posttrain.rollout.torch_engine import TorchRolloutEngine
from scripts.prepare_kodcode_rl import _build_record, build_pytest_template
from train.walkie_rl import _load_prompts


def test_load_prompts_expands_jsonl_directory(tmp_path) -> None:
    data_dir = tmp_path / "rl_data"
    data_dir.mkdir()
    row = {
        "prompt": "user:\nSolve it.\nassistant:\n",
        "test_program_template": "candidate_code = {{candidate_code}}\nprint('ALL TESTS PASSED')",
        "task_id": "deepcoder/example",
        "source": "deepcoder",
        "task_type": "stdin",
        "num_tests": 5,
    }
    (data_dir / "train-00000.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    prompts = _load_prompts(OmegaConf.create({"path": str(data_dir), "paths": []}), ChatTemplate(kind="plain_lower"))

    assert len(prompts) == 1
    assert prompts[0].text == row["prompt"]
    assert prompts[0].metadata["test_program_template"] == row["test_program_template"]
    assert prompts[0].metadata["task_type"] == "stdin"


def test_kodcode_pytest_template_runs_full_solution() -> None:
    template = build_pytest_template(
        "from solution import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        style="instruct",
        starter_code="",
        entry_points=["add"],
        case_timeout=5.0,
    )
    program = build_template_code("def add(a, b):\n    return a + b\n", template)

    exec(program, {})


def test_kodcode_pytest_template_accepts_complete_missing_body() -> None:
    starter_code = 'def add(a: int, b: int) -> int:\n    """Return the sum."""'
    template = build_pytest_template(
        "from solution import add\n\ndef test_add():\n    assert add(2, 5) == 7\n",
        style="complete",
        starter_code=starter_code,
        entry_points=["add"],
        case_timeout=5.0,
    )
    program = build_template_code("return a + b", template)

    exec(program, {})


def test_kodcode_pytest_template_exposes_solution_symbols_without_import() -> None:
    template = build_pytest_template(
        "def test_add():\n    assert add(4, 6) == 10\n",
        style="instruct",
        starter_code="",
        entry_points=["add"],
        case_timeout=5.0,
    )
    program = build_template_code("def add(a, b):\n    return a + b\n", template)

    exec(program, {})


def test_kodcode_pytest_template_supports_common_pytest_patterns() -> None:
    template = build_pytest_template(
        """
import pytest
from solution import divide, square

@pytest.mark.parametrize("value, expected", [(2, 4), (3, 9)])
def test_square(value, expected):
    assert square(value) == expected

def test_divide_raises():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
""",
        style="instruct",
        starter_code="",
        entry_points=["square", "divide"],
        case_timeout=5.0,
    )
    program = build_template_code(
        "def square(value):\n    return value * value\n\ndef divide(a, b):\n    return a / b\n",
        template,
    )

    exec(program, {})


def test_kodcode_pytest_template_supports_unittest_cases() -> None:
    template = build_pytest_template(
        """
import unittest
from solution import add

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
""",
        style="instruct",
        starter_code="",
        entry_points=["add"],
        case_timeout=5.0,
    )
    program = build_template_code("def add(a, b):\n    return a + b\n", template)

    exec(program, {})


def test_kodcode_record_filters_sandbox_incompatible_tests() -> None:
    args = SimpleNamespace(
        max_question_chars=6000,
        max_test_chars=200000,
        min_test_functions=1,
        min_gpt_pass_percentage=None,
        max_gpt_pass_percentage=None,
        case_timeout=5.0,
        filter_sandbox_incompatible=True,
    )
    row = {
        "style": "instruct",
        "question": "Write add(a, b).",
        "test": "import tempfile\n\ndef test_add():\n    with tempfile.TemporaryDirectory():\n        assert add(1, 2) == 3\n",
        "test_info": {"fn_name": "add"},
        "question_id": "sandbox-risky",
    }

    decision = _build_record(row, source_file="unit.parquet", bench_index={}, allowed_styles={"instruct"}, args=args)

    assert decision.record is None
    assert decision.reason == "sandbox_incompatible_test:tempfile"


def test_kodcode_record_filters_subprocess_variants_in_prompt() -> None:
    args = SimpleNamespace(
        max_question_chars=6000,
        max_test_chars=200000,
        min_test_functions=1,
        min_gpt_pass_percentage=None,
        max_gpt_pass_percentage=None,
        case_timeout=5.0,
        filter_sandbox_incompatible=True,
    )
    row = {
        "style": "instruct",
        "question": "Use asyncio.create_subprocess_shell to run commands concurrently.",
        "test": "def test_dummy():\n    assert solve() is not None\n",
        "test_info": {"fn_name": "solve"},
        "question_id": "subprocess-risky",
    }

    decision = _build_record(row, source_file="unit.parquet", bench_index={}, allowed_styles={"instruct"}, args=args)

    assert decision.record is None
    assert decision.reason == "sandbox_incompatible_question:create_subprocess_shell"


def test_torch_rollout_engine_decodes_completion_and_restores_train_mode() -> None:
    class FakeTokenizer:
        eos_token_id = 0

        def encode(self, text: str) -> list[int]:
            return [2, 3]

        def decode(self, token_ids: list[int]) -> str:
            return "".join(str(item) for item in token_ids)

    class FakeModel:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def generate(self, input_ids, **kwargs):
            return torch.tensor([[2, 3, 4, 5, 0, 9]], dtype=torch.long)

    model = FakeModel()
    engine = TorchRolloutEngine(model, FakeTokenizer(), device=torch.device("cpu"), dtype=torch.float32, use_amp=False)

    outputs = engine.generate(["prompt"], SamplingConfig(num_generations=1, max_tokens=4, stop=[]))

    assert outputs[0].response == "45"
    assert outputs[0].metadata["backend"] == "torch"
    assert model.training is True