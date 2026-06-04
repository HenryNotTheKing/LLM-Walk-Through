"""KodCode SFT cleaning helpers."""

from __future__ import annotations

from posttrain.data.chat_template import normalize_messages
from posttrain.data.kodcode_sft import clean_kodcode_row, detect_bench_leak, normalize_text
from scripts.prepare_kodcode_sft import _shuffle_records, parse_args


def test_clean_kodcode_row_outputs_minimal_messages() -> None:
    row = {
        "style": "instruct",
        "subset": "Leetcode",
        "question_id": "sample_1",
        "question": "Write a function add(a, b).",
        "r1_correctness": "True",
        "r1_solution": "```python\ndef add(a, b):\n    return a + b\n```",
        "gpt_difficulty": "easy",
    }

    decision = clean_kodcode_row(row)

    assert decision.reason == "kept"
    assert decision.record is not None
    assert set(decision.record) == {
        "messages",
        "source_id",
        "style",
        "subset",
        "gpt_difficulty",
        "prompt_kind",
        "question_hash",
        "solution_chars",
    }
    turns = normalize_messages(decision.record)
    assert turns[0].role == "user"
    assert "Return only Python code" in turns[0].content
    assert turns[1].content == "def add(a, b):\n    return a + b"


def test_clean_kodcode_row_drops_bad_assistant_prefix() -> None:
    row = {
        "style": "instruct",
        "question_id": "sample_2",
        "question": "Write add.",
        "r1_correctness": "True",
        "r1_solution": "Write a complete Python solution. Return only Python code.\n\ndef add(a, b):\n    return a + b",
    }

    decision = clean_kodcode_row(row)

    assert decision.reason == "kept"
    assert decision.record is not None
    turns = normalize_messages(decision.record)
    assert turns[1].content == "def add(a, b):\n    return a + b"


def test_clean_kodcode_row_rejects_text_only_solution() -> None:
    row = {
        "style": "instruct",
        "question_id": "sample_3",
        "question": "Write add.",
        "r1_correctness": "True",
        "r1_solution": "Write a complete Python solution. Return only Python code.",
    }

    decision = clean_kodcode_row(row)

    assert decision.record is None
    assert decision.reason in {"assistant_text_prefix", "syntax_error"}


def test_detect_bench_leak_matches_normalized_prompt() -> None:
    prompt = "def has_close_elements(numbers: List[float], threshold: float) -> bool:"
    bench_index = {normalize_text(prompt): "openai_humaneval:HumanEval/0"}
    row = {"question": prompt}

    assert detect_bench_leak(row, bench_index) == "leak:question:openai_humaneval:HumanEval/0"


def test_prepare_kodcode_defaults_to_shuffle() -> None:
    args = parse_args([])

    assert args.shuffle is True
    assert args.seed == 42


def test_shuffle_records_is_deterministic() -> None:
    records_a = [str(index) for index in range(10)]
    records_b = [str(index) for index in range(10)]

    _shuffle_records(records_a, 7)
    _shuffle_records(records_b, 7)

    assert records_a == records_b
    assert records_a != [str(index) for index in range(10)]