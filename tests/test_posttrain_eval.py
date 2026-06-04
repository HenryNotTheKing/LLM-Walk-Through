"""HumanEval-style evaluation helpers."""

from __future__ import annotations

import pytest

from posttrain.eval.code_bench import (
    build_code_bench_candidates,
    build_code_bench_test_program,
    render_code_bench_prompt,
    sample_from_bench_row,
)
from posttrain.eval.humaneval import (
    CodeEvalSample,
    build_humaneval_test_program,
    compute_pass_at_k,
    extract_completion_code,
    summarize_pass_at_k,
)


def test_extract_completion_code_prefers_python_fence() -> None:
    response = "Here is code:\n```python\ndef add(a, b):\n    return a + b\n```"

    assert extract_completion_code(response) == "def add(a, b):\n    return a + b"


def test_build_humaneval_program_calls_check_with_entry_point() -> None:
    sample = CodeEvalSample(
        task_id="HumanEval/0",
        prompt="def add(a, b):\n",
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
    )

    program = build_humaneval_test_program(sample, "    return a + b")

    assert "def add(a, b):\n    return a + b" in program
    assert "check(add)" in program
    assert "ALL TESTS PASSED" in program


def test_code_bench_prompt_uses_lowercase_plain_dialog() -> None:
    sample = sample_from_bench_row(
        "mbpp",
        {"task_id": 1, "prompt": "Write add(a, b).", "test_list": ["assert add(1, 2) == 3"]},
    )

    prompt = render_code_bench_prompt(sample)

    assert prompt.startswith("user:\n")
    assert prompt.endswith("assistant:\n")
    assert "User:" not in prompt
    assert "Assistant:" not in prompt


def test_code_bench_mbpp_prompt_includes_canonical_signature() -> None:
    sample = sample_from_bench_row(
        "mbpp",
        {
            "task_id": 2,
            "prompt": "Write a function to find shared elements from two lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\n    return tuple(set(test_tup1) & set(test_tup2))",
            "test_list": ["assert set(similar_elements((1, 2), (2, 3))) == {2}"],
        },
    )

    prompt = render_code_bench_prompt(sample)

    assert sample.entry_point == "similar_elements"
    assert "Required function signature:" in prompt
    assert "def similar_elements(test_tup1, test_tup2):" in prompt
    assert "do not rename the required function" in prompt


def test_code_bench_humaneval_accepts_full_function_completion() -> None:
    sample = sample_from_bench_row(
        "openai_humaneval",
        {
            "task_id": "HumanEval/0",
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            "entry_point": "add",
        },
    )

    program = build_code_bench_test_program(sample, "def add(a, b):\n    return a + b")

    assert program.count("def add(a, b):") == 1
    assert "check(add)" in program


def test_code_bench_mbpp_program_uses_assert_list_without_entry_point() -> None:
    sample = sample_from_bench_row(
        "mbpp",
        {
            "task_id": 11,
            "prompt": "Write a function to add two numbers.",
            "test_imports": ["import math"],
            "test_list": ["assert add(1, 2) == 3"],
        },
    )

    program = build_code_bench_test_program(sample, "```python\ndef add(a, b):\n    return a + b\n```")

    assert "import math" in program
    assert "def add(a, b):" in program
    assert "assert add(1, 2) == 3" in program
    assert "check(" not in program
    assert "ALL TESTS PASSED" in program


def test_code_bench_mbppplus_prefers_plus_test_field() -> None:
    sample = sample_from_bench_row(
        "mbppplus",
        {
            "task_id": 12,
            "prompt": "Write a function to add two numbers.",
            "test_imports": [],
            "test_list": ["assert add(0, 0) == 0"],
            "test": "assert add(2, 2) == 4",
        },
    )

    program = build_code_bench_test_program(sample, "def add(a, b):\n    return a + b")

    assert "assert add(2, 2) == 4" in program
    assert "assert add(0, 0) == 0" not in program


def test_build_code_bench_candidates_preserves_dataset_metadata() -> None:
    sample = sample_from_bench_row(
        "mbpp",
        {"task_id": 1, "prompt": "Write add(a, b).", "test_list": ["assert add(1, 2) == 3"]},
    )

    candidates = build_code_bench_candidates([sample], [["def add(a, b):\n    return a + b"]], prompts=["user:\n...\nassistant:\n"])

    assert candidates[0].dataset == "mbpp"
    assert candidates[0].prompt == "user:\n...\nassistant:\n"
    assert "assert add(1, 2) == 3" in candidates[0].test_program


@pytest.mark.parametrize(
    ("total", "correct", "k", "expected"),
    [(1, 1, 1, 1.0), (10, 0, 1, 0.0), (10, 10, 5, 1.0), (10, 1, 1, 0.1)],
)
def test_compute_pass_at_k(total: int, correct: int, k: int, expected: float) -> None:
    assert compute_pass_at_k(total, correct, k) == pytest.approx(expected)


def test_summarize_pass_at_k_groups_by_task() -> None:
    rows = [
        {"task_id": "a", "passed": True},
        {"task_id": "a", "passed": False},
        {"task_id": "b", "passed": False},
        {"task_id": "b", "passed": False},
    ]

    summary = summarize_pass_at_k(rows, ks=[1, 2])

    assert summary["num_tasks"] == 2
    assert summary["pass@1"] == pytest.approx(0.25)
    assert summary["pass@2"] == pytest.approx(0.5)


def test_resolve_num_samples_bumps_to_max_pass_at() -> None:
    from scripts.evaluate_code_bench import _resolve_num_samples

    assert _resolve_num_samples(1, pass_at=[1, 4, 8]) == 8
    assert _resolve_num_samples(8, pass_at=[1, 4, 8]) == 8
    assert _resolve_num_samples(4, pass_at=[1, 4]) == 4


def test_macro_pass_at_k_averages_datasets() -> None:
    from scripts.evaluate_code_bench import _macro_pass_at_k

    summaries = {
        "a": {"pass@1": 0.2, "pass@4": 0.4, "pass@8": 0.6},
        "b": {"pass@1": 0.4, "pass@4": 0.6, "pass@8": 0.8},
    }

    macro = _macro_pass_at_k(summaries, pass_at=[1, 4, 8])

    assert macro["pass@1"] == pytest.approx(0.3)
    assert macro["pass@4"] == pytest.approx(0.5)
    assert macro["pass@8"] == pytest.approx(0.7)