"""KodCode SFT cleaning helpers for code benchmark fine-tuning."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


HUMANEVAL_INSTRUCTION = (
    "Complete the following Python function. Return only the missing code; "
    "do not repeat the prompt."
)
MBPP_INSTRUCTION = "Write a complete Python solution. Return only Python code."
DEFAULT_ALLOWED_STYLES = ("instruct", "complete")

CODE_START_RE = re.compile(
    r"^\s*(?:@|def\s+|async\s+def\s+|class\s+|from\s+\S+\s+import\s+|import\s+|if\s+__name__\s*==|[A-Za-z_]\w*\s*=)",
)
FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
BAD_ASSISTANT_PREFIXES = (
    "write a complete python solution",
    "complete the following python function",
    "return only python code",
    "here is",
    "the solution",
    "we need",
)


@dataclass(frozen=True)
class CleanDecision:
    record: dict[str, Any] | None
    reason: str


def clean_kodcode_row(
    row: Mapping[str, Any],
    *,
    bench_index: Mapping[str, str] | None = None,
    allowed_styles: tuple[str, ...] = DEFAULT_ALLOWED_STYLES,
    require_correct: bool = True,
    drop_leaks: bool = True,
) -> CleanDecision:
    style = str(row.get("style") or "").strip().lower()
    if style not in {item.lower() for item in allowed_styles}:
        return CleanDecision(None, f"style:{style or 'missing'}")

    if require_correct and not _is_true(row.get("r1_correctness")):
        return CleanDecision(None, "r1_incorrect")

    question = str(row.get("question") or "").strip()
    if not question:
        return CleanDecision(None, "missing_question")

    if drop_leaks and bench_index:
        leak_reason = detect_bench_leak(row, bench_index)
        if leak_reason is not None:
            return CleanDecision(None, leak_reason)

    solution = clean_python_solution(str(row.get("r1_solution") or ""))
    if not solution:
        return CleanDecision(None, "missing_solution")
    if starts_with_bad_assistant_text(solution):
        return CleanDecision(None, "assistant_text_prefix")
    if not parses_as_python(solution):
        return CleanDecision(None, "syntax_error")

    source_id = str(row.get("question_id") or stable_hash(question))
    prompt_kind = prompt_kind_for_style(style)
    record = {
        "messages": [
            {"role": "user", "content": render_user_content(question, style)},
            {"role": "assistant", "content": solution},
        ],
        "source_id": source_id,
        "style": style,
        "subset": str(row.get("subset") or ""),
        "gpt_difficulty": str(row.get("gpt_difficulty") or ""),
        "prompt_kind": prompt_kind,
        "question_hash": stable_hash(question),
        "solution_chars": len(solution),
    }
    return CleanDecision(record, "kept")


def render_user_content(question: str, style: str) -> str:
    instruction = HUMANEVAL_INSTRUCTION if prompt_kind_for_style(style) == "humaneval" else MBPP_INSTRUCTION
    return f"{instruction}\n\n{question.strip()}"


def prompt_kind_for_style(style: str) -> str:
    return "humaneval" if style.strip().lower() == "complete" else "mbpp"


def clean_python_solution(text: str) -> str:
    text = THINK_RE.sub("", text or "").strip()
    fence_matches = FENCE_RE.findall(text)
    if fence_matches:
        text = max(fence_matches, key=len).strip()
    text = re.sub(r"^\s*(?:assistant|Assistant)\s*:\s*", "", text).strip()
    text = text.replace("```", "").strip()

    lines = text.splitlines()
    first_code_line = _first_code_line_index(lines)
    if first_code_line > 0:
        prefix = " ".join(line.strip().lower() for line in lines[:first_code_line] if line.strip())
        if any(prefix.startswith(item) or item in prefix for item in BAD_ASSISTANT_PREFIXES):
            lines = lines[first_code_line:]
    return "\n".join(lines).strip()


def starts_with_bad_assistant_text(text: str) -> bool:
    first_line = next((line.strip().lower() for line in text.splitlines() if line.strip()), "")
    return any(first_line.startswith(prefix) for prefix in BAD_ASSISTANT_PREFIXES)


def parses_as_python(text: str) -> bool:
    try:
        ast.parse(text)
    except SyntaxError:
        return False
    return True


def build_bench_text_index(bench_root: str | Path) -> dict[str, str]:
    from posttrain.eval.code_bench import SUPPORTED_CODE_BENCH_DATASETS, load_code_bench_samples

    index: dict[str, str] = {}
    for dataset in SUPPORTED_CODE_BENCH_DATASETS:
        for sample in load_code_bench_samples(dataset, bench_root=bench_root):
            normalized = normalize_text(sample.prompt)
            if normalized:
                index[normalized] = f"{dataset}:{sample.task_id}"
    return index


def detect_bench_leak(row: Mapping[str, Any], bench_index: Mapping[str, str]) -> str | None:
    for field_name, text in candidate_leak_texts(row):
        normalized = normalize_text(text)
        if not normalized:
            continue
        match = bench_index.get(normalized)
        if match is not None:
            return f"leak:{field_name}:{match}"
        near_match = _near_bench_match(normalized, bench_index)
        if near_match is not None:
            return f"near_leak:{field_name}:{near_match}"
    return None


def candidate_leak_texts(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    candidates = [("question", str(row.get("question") or ""))]
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        candidates.append(("metadata.original_instruction", str(metadata.get("original_instruction") or "")))
    test_info = row.get("test_info")
    if isinstance(test_info, list):
        for index, item in enumerate(test_info):
            if isinstance(item, Mapping):
                for key in ("docstring", "function_declaration"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append((f"test_info.{index}.{key}", value))
    return candidates


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def _near_bench_match(normalized: str, bench_index: Mapping[str, str]) -> str | None:
    if len(normalized) < 120:
        return None
    for bench_text, label in bench_index.items():
        if len(bench_text) < 120:
            continue
        shorter = min(len(normalized), len(bench_text))
        longer = max(len(normalized), len(bench_text))
        if shorter / longer < 0.85:
            continue
        if normalized in bench_text or bench_text in normalized:
            return label
    return None


def _first_code_line_index(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if CODE_START_RE.search(line):
            return index
    return 0


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"