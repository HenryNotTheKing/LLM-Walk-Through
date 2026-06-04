"""HumanEval-style generation and execution evaluation helpers."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from posttrain.sandbox.jupyter_client import JupyterSandboxClient


@dataclass(frozen=True)
class CodeEvalSample:
    task_id: str
    prompt: str
    test: str
    entry_point: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CodeEvalCandidate:
    task_id: str
    completion_id: int
    prompt: str
    completion: str
    test_program: str


def load_humaneval_jsonl(path: str | Path) -> list[CodeEvalSample]:
    samples: list[CodeEvalSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(sample_from_row(row))
    return samples


def load_custom_jsonl(
    path: str | Path,
    *,
    prompt_field: str = "prompt",
    test_field: str = "test",
    task_id_field: str = "task_id",
    entry_point_field: str = "entry_point",
) -> list[CodeEvalSample]:
    samples: list[CodeEvalSample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            samples.append(
                CodeEvalSample(
                    task_id=str(row.get(task_id_field, f"sample/{index}")),
                    prompt=str(row[prompt_field]),
                    test=str(row[test_field]),
                    entry_point=str(row[entry_point_field]),
                    metadata={key: value for key, value in row.items() if key not in {prompt_field, test_field, task_id_field, entry_point_field}},
                )
            )
    return samples


def sample_from_row(row: dict[str, Any]) -> CodeEvalSample:
    return CodeEvalSample(
        task_id=str(row.get("task_id")),
        prompt=str(row["prompt"]),
        test=str(row["test"]),
        entry_point=str(row["entry_point"]),
        metadata={key: value for key, value in row.items() if key not in {"task_id", "prompt", "test", "entry_point"}},
    )


def extract_completion_code(response: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip("\n")
    return response.strip("\n")


def build_humaneval_test_program(sample: CodeEvalSample, completion: str) -> str:
    code = extract_completion_code(completion)
    return (
        f"{sample.prompt}{code}\n\n"
        f"{sample.test}\n\n"
        f"check({sample.entry_point})\n"
        "print('ALL TESTS PASSED')\n"
    )


def build_candidates(samples: Sequence[CodeEvalSample], completions: Sequence[Sequence[str]]) -> list[CodeEvalCandidate]:
    if len(samples) != len(completions):
        raise ValueError(f"samples/completions length mismatch: {len(samples)} vs {len(completions)}")
    candidates: list[CodeEvalCandidate] = []
    for sample, sample_completions in zip(samples, completions):
        if not sample_completions:
            raise ValueError(f"sample {sample.task_id} has no completions")
        for completion_id, completion in enumerate(sample_completions):
            candidates.append(
                CodeEvalCandidate(
                    task_id=sample.task_id,
                    completion_id=completion_id,
                    prompt=sample.prompt,
                    completion=completion,
                    test_program=build_humaneval_test_program(sample, completion),
                )
            )
    return candidates


def compute_pass_at_k(total: int, correct: int, k: int) -> float:
    if total <= 0 or k <= 0:
        return 0.0
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / value for value in range(total - correct + 1, total + 1))


def summarize_pass_at_k(rows: Iterable[dict[str, Any]], *, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float | int]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(bool(row["passed"]))
    summary: dict[str, float | int] = {"num_tasks": len(grouped)}
    for k in ks:
        if not grouped:
            summary[f"pass@{k}"] = 0.0
            continue
        values = [compute_pass_at_k(len(results), sum(results), k) for results in grouped.values()]
        summary[f"pass@{k}"] = float(sum(values) / len(values))
    return summary


async def evaluate_candidates_async(
    candidates: Sequence[CodeEvalCandidate],
    *,
    sandbox_urls: Sequence[str],
    timeout: float = 10.0,
    max_concurrency: int = 32,
) -> list[dict[str, Any]]:
    client = JupyterSandboxClient(sandbox_urls, timeout=timeout)
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def run_one(candidate: CodeEvalCandidate) -> dict[str, Any]:
        result = None
        for attempt in range(3):
            async with semaphore:
                result = await client.run_code(candidate.test_program)
            if result.status == "success":
                break
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
        assert result is not None
        passed = result.status == "success" and "ALL TESTS PASSED" in result.stdout
        return {
            **asdict(candidate),
            "passed": passed,
            "status": result.status,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "result": result.result,
            "execution_time": result.execution_time,
        }

    return list(await asyncio.gather(*(run_one(candidate) for candidate in candidates)))


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
