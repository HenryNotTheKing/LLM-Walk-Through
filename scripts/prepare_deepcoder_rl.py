"""Prepare DeepCoder Preview data for verifier-driven Walkie RL."""

from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.parquet as pq

from posttrain.data.kodcode_sft import build_bench_text_index, normalize_text, stable_hash


DEFAULT_SOURCES = ("lcbv5", "primeintellect", "taco")
STDIN_INSTRUCTION = "Write a complete Python program that reads from standard input and writes to standard output. Return only Python code."
FUNCTION_INSTRUCTION = "Write a complete Python solution. Return only Python code."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DeepCoder Preview parquet files into RL prompt/test-template JSONL")
    parser.add_argument("--input", default="data/RL/DeepCoder-Preview-Dataset")
    parser.add_argument("--output", default="data/RL/deepcoder_preview_rl_prepared")
    parser.add_argument("--bench-root", default="data/bench")
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="comma-separated source folders")
    parser.add_argument("--include-dataset-test", action="store_true", help="include DeepCoder test split files such as lcbv5/test and codeforces/test")
    parser.add_argument("--skip-leak-filter", action="store_true")
    parser.add_argument("--skip-stdin", action="store_true")
    parser.add_argument("--skip-functional", action="store_true")
    parser.add_argument("--min-test-cases", type=int, default=5)
    parser.add_argument("--max-test-cases", type=int, default=32)
    parser.add_argument("--min-problem-chars", type=int, default=0)
    parser.add_argument("--max-problem-chars", type=int, default=6000)
    parser.add_argument("--max-case-chars", type=int, default=20000)
    parser.add_argument("--min-template-chars", type=int, default=0)
    parser.add_argument("--max-template-chars", type=int, default=200000)
    parser.add_argument("--case-timeout", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input)
    output_root = Path(args.output)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty; pass --overwrite: {output_root}")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    bench_index = {} if args.skip_leak_filter else build_bench_text_index(args.bench_root)
    records: list[str] = []
    stats: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    sources = tuple(item.strip() for item in str(args.sources).split(",") if item.strip())

    for source in sources:
        for path in _source_files(input_root, source, include_dataset_test=bool(args.include_dataset_test)):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=max(1, int(args.batch_size))):
                for row_index, row in enumerate(batch.to_pylist()):
                    stats["seen"] += 1
                    decision = _build_record(
                        row,
                        source=source,
                        source_file=path.name,
                        source_index=int(stats["seen"]),
                        bench_index=bench_index,
                        args=args,
                    )
                    if decision.record is None:
                        stats[decision.reason] += 1
                        continue
                    records.append(json.dumps(decision.record, ensure_ascii=False))
                    stats["kept"] += 1
                    source_counts[source] += 1
                    task_type_counts[str(decision.record["task_type"])] += 1
                    if args.limit is not None and stats["kept"] >= int(args.limit):
                        break
                if args.limit is not None and stats["kept"] >= int(args.limit):
                    break
            if args.limit is not None and stats["kept"] >= int(args.limit):
                break
        if args.limit is not None and stats["kept"] >= int(args.limit):
            break

    if args.shuffle:
        random.Random(int(args.seed)).shuffle(records)
    output_files = _write_shards(output_root, records, int(args.shard_size))
    manifest = {
        "source": str(input_root),
        "output_files": output_files,
        "sources": list(sources),
        "include_dataset_test": bool(args.include_dataset_test),
        "leak_filter": not bool(args.skip_leak_filter),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed) if args.shuffle else None,
        "filters": {
            "min_test_cases": int(args.min_test_cases),
            "max_test_cases": int(args.max_test_cases),
            "min_problem_chars": int(args.min_problem_chars),
            "max_problem_chars": int(args.max_problem_chars),
            "max_case_chars": int(args.max_case_chars),
            "min_template_chars": int(args.min_template_chars),
            "max_template_chars": int(args.max_template_chars),
        },
        "stats": dict(stats),
        "source_counts": dict(source_counts),
        "task_type_counts": dict(task_type_counts),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


class Decision:
    def __init__(self, record: dict[str, Any] | None, reason: str) -> None:
        self.record = record
        self.reason = reason


def _source_files(input_root: Path, source: str, *, include_dataset_test: bool) -> list[Path]:
    source_root = input_root / source
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing source folder: {source_root}")
    patterns = ["train-*.parquet"]
    if include_dataset_test:
        patterns.append("test-*.parquet")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(source_root.glob(pattern)))
    if not paths and include_dataset_test:
        paths = sorted(source_root.glob("*.parquet"))
    return paths


def _build_record(row: dict[str, Any], *, source: str, source_file: str, source_index: int, bench_index: dict[str, str], args: argparse.Namespace) -> Decision:
    problem = str(row.get("problem") or "").strip()
    if not problem:
        return Decision(None, "missing_problem")
    if len(problem) < int(args.min_problem_chars):
        return Decision(None, "easy_problem_too_short")
    if len(problem) > int(args.max_problem_chars):
        return Decision(None, "problem_too_long")
    leak = _bench_leak(problem, bench_index)
    if leak is not None:
        return Decision(None, leak)
    try:
        raw_tests = json.loads(str(row.get("tests") or ""))
    except json.JSONDecodeError:
        return Decision(None, "tests_json_error")

    normalized = _normalize_tests(raw_tests, row=row, source=source, max_case_chars=int(args.max_case_chars))
    if normalized is None:
        return Decision(None, "unsupported_tests")
    task_type, entry_point, uses_solution_class, cases = normalized
    if task_type == "stdin" and bool(args.skip_stdin):
        return Decision(None, "skip_stdin")
    if task_type == "function" and bool(args.skip_functional):
        return Decision(None, "skip_functional")
    if len(cases) < int(args.min_test_cases):
        return Decision(None, "too_few_tests")
    cases = _even_sample(cases, int(args.max_test_cases))

    prompt = _render_prompt(problem, task_type=task_type, entry_point=entry_point, starter_code=str(row.get("starter_code") or ""), uses_solution_class=uses_solution_class)
    template = _build_template(task_type, cases, entry_point=entry_point, uses_solution_class=uses_solution_class, case_timeout=float(args.case_timeout))
    if len(template) < int(args.min_template_chars):
        return Decision(None, "easy_template_too_short")
    if len(template) > int(args.max_template_chars):
        return Decision(None, "template_too_long")
    task_id = f"deepcoder/{source}/{stable_hash(problem)[:16]}"
    return Decision(
        {
            "prompt": prompt,
            "tests": "",
            "test_program_template": template,
            "task_id": task_id,
            "source": source,
            "source_file": source_file,
            "source_index": source_index,
            "task_type": task_type,
            "entry_point": entry_point,
            "num_tests": len(cases),
            "problem_chars": len(problem),
            "template_chars": len(template),
        },
        "kept",
    )


def _normalize_tests(raw_tests: Any, *, row: dict[str, Any], source: str, max_case_chars: int) -> tuple[str, str, bool, list[dict[str, Any]]] | None:
    if isinstance(raw_tests, list):
        if _is_function_list(raw_tests):
            return _normalize_function_list(raw_tests, row=row, source=source, max_case_chars=max_case_chars)
        return _normalize_stdin_list(raw_tests, max_case_chars=max_case_chars)
    if isinstance(raw_tests, dict):
        if "fn_name" in raw_tests:
            return _normalize_function_dict(raw_tests, max_case_chars=max_case_chars)
        if "inputs" in raw_tests and "outputs" in raw_tests:
            return _normalize_stdin_dict(raw_tests, max_case_chars=max_case_chars)
    return None


def _is_function_list(items: list[Any]) -> bool:
    return bool(items) and all(isinstance(item, dict) for item in items) and any(
        item.get("testtype") == "functional" or item.get("type") == "function_call" or "fn_name" in item for item in items
    )


def _normalize_stdin_list(items: list[Any], *, max_case_chars: int) -> tuple[str, str, bool, list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        case_input = str(item.get("input", ""))
        case_output = str(item.get("output", ""))
        if _case_chars(case_input, case_output) <= max_case_chars:
            cases.append({"input": case_input, "output": case_output})
    return "stdin", "", False, cases


def _normalize_stdin_dict(data: dict[str, Any], *, max_case_chars: int) -> tuple[str, str, bool, list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    inputs = data.get("inputs") or []
    outputs = data.get("outputs") or []
    for case_input, case_output in zip(inputs, outputs):
        case_input_text = str(case_input)
        case_output_text = str(case_output)
        if _case_chars(case_input_text, case_output_text) <= max_case_chars:
            cases.append({"input": case_input_text, "output": case_output_text})
    return "stdin", "", False, cases


def _normalize_function_list(items: list[Any], *, row: dict[str, Any], source: str, max_case_chars: int) -> tuple[str, str, bool, list[dict[str, Any]]] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    entry_point = str(metadata.get("func_name") or "")
    uses_solution_class = source == "lcbv5" and bool(entry_point)
    cases: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fn_name = str(item.get("fn_name") or entry_point)
        if not fn_name:
            return None
        entry_point = entry_point or fn_name
        if str(item.get("testtype") or item.get("type") or "") == "functional":
            raw_input = str(item.get("input", ""))
            if len(raw_input) + len(str(item.get("output", ""))) > max_case_chars:
                continue
            args = _parse_lcb_function_args(raw_input)
            expected = _parse_literal(str(item.get("output", "")))
        else:
            args = item.get("input", [])
            if not isinstance(args, list):
                args = [args]
            expected = _unwrap_expected(item.get("output"))
            if _case_chars(json.dumps(args, ensure_ascii=False), json.dumps(expected, ensure_ascii=False)) > max_case_chars:
                continue
        cases.append({"input": args, "output": expected})
    return "function", entry_point, uses_solution_class, cases


def _normalize_function_dict(data: dict[str, Any], *, max_case_chars: int) -> tuple[str, str, bool, list[dict[str, Any]]]:
    entry_point = str(data.get("fn_name") or "")
    cases: list[dict[str, Any]] = []
    for args, expected in zip(data.get("inputs") or [], data.get("outputs") or []):
        if not isinstance(args, list):
            args = [args]
        expected = _unwrap_expected(expected)
        if _case_chars(json.dumps(args, ensure_ascii=False), json.dumps(expected, ensure_ascii=False)) <= max_case_chars:
            cases.append({"input": args, "output": expected})
    return "function", entry_point, False, cases


def _render_prompt(problem: str, *, task_type: str, entry_point: str, starter_code: str, uses_solution_class: bool) -> str:
    if task_type == "stdin":
        content = f"{STDIN_INSTRUCTION}\n\n{problem.strip()}"
    elif uses_solution_class:
        content = f"{FUNCTION_INSTRUCTION} Implement the required method in class Solution.\n\n{problem.strip()}"
    else:
        content = f"{FUNCTION_INSTRUCTION} Define the required function `{entry_point}` exactly.\n\n{problem.strip()}"
    if starter_code.strip():
        content = f"{content}\n\nStarter code:\n```python\n{starter_code.rstrip()}\n```"
    return f"user:\n{content}\nassistant:\n"


def _build_template(task_type: str, cases: list[dict[str, Any]], *, entry_point: str, uses_solution_class: bool, case_timeout: float) -> str:
    if task_type == "stdin":
        return _stdin_template(cases, case_timeout=case_timeout)
    return _function_template(cases, entry_point=entry_point, uses_solution_class=uses_solution_class)


def _stdin_template(cases: list[dict[str, Any]], *, case_timeout: float) -> str:
    cases_json = json.dumps(cases, ensure_ascii=False)
    return f'''import json
import os
import subprocess
import sys
import tempfile

candidate_code = {{{{candidate_code}}}}
TEST_CASES = json.loads({cases_json!r})
CASE_TIMEOUT = {float(case_timeout)!r}


def _normalize_output(value):
    return "\\n".join(line.rstrip() for line in str(value).strip().splitlines()).strip()


handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False)
try:
    handle.write(candidate_code)
    handle.flush()
    handle.close()
    for index, case in enumerate(TEST_CASES):
        completed = subprocess.run(
            [sys.executable, handle.name],
            input=str(case["input"]),
            text=True,
            capture_output=True,
            timeout=CASE_TIMEOUT,
        )
        if completed.returncode != 0:
            raise AssertionError(f"case {{index}} exited with {{completed.returncode}}: {{completed.stderr[:1000]}}")
        actual = _normalize_output(completed.stdout)
        expected = _normalize_output(case["output"])
        if actual != expected:
            raise AssertionError(f"case {{index}} mismatch: expected={{expected!r}} actual={{actual!r}}")
finally:
    try:
        os.unlink(handle.name)
    except OSError:
        pass

print('ALL TESTS PASSED')
'''


def _function_template(cases: list[dict[str, Any]], *, entry_point: str, uses_solution_class: bool) -> str:
    cases_json = json.dumps(cases, ensure_ascii=False)
    return f'''import json
import math
from typing import *

candidate_code = {{{{candidate_code}}}}
TEST_CASES = json.loads({cases_json!r})
ENTRY_POINT = {entry_point!r}
USES_SOLUTION_CLASS = {bool(uses_solution_class)!r}

namespace = {{"List": List, "Dict": Dict, "Tuple": Tuple, "Set": Set, "Optional": Optional, "math": math}}
exec(candidate_code, namespace)

if USES_SOLUTION_CLASS:
    solution_cls = namespace.get("Solution")
    if solution_cls is None:
        raise AssertionError("missing class Solution")
    target = getattr(solution_cls(), ENTRY_POINT)
else:
    target = namespace.get(ENTRY_POINT)
    if target is None and namespace.get("Solution") is not None:
        target = getattr(namespace["Solution"](), ENTRY_POINT)
    if target is None:
        raise AssertionError(f"missing function {{ENTRY_POINT}}")

for index, case in enumerate(TEST_CASES):
    args = case["input"]
    if not isinstance(args, list):
        args = [args]
    actual = target(*args)
    expected = case["output"]
    if actual != expected:
        raise AssertionError(f"case {{index}} mismatch: expected={{expected!r}} actual={{actual!r}}")

print('ALL TESTS PASSED')
'''


def _parse_lcb_function_args(text: str) -> list[Any]:
    return [_parse_literal(line) for line in str(text).splitlines() if line.strip()]


def _parse_literal(text: str) -> Any:
    stripped = str(text).strip()
    try:
        return ast.literal_eval(stripped)
    except Exception:
        try:
            return json.loads(stripped)
        except Exception:
            return stripped


def _unwrap_expected(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _even_sample(items: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    if max_items == 1:
        return [items[0]]
    last = len(items) - 1
    indices = sorted({round(index * last / (max_items - 1)) for index in range(max_items)})
    return [items[index] for index in indices]


def _case_chars(case_input: str, case_output: str) -> int:
    return len(str(case_input)) + len(str(case_output))


def _bench_leak(problem: str, bench_index: dict[str, str]) -> str | None:
    if not bench_index:
        return None
    normalized = normalize_text(problem)
    exact = bench_index.get(normalized)
    if exact is not None:
        return f"leak:{exact}"
    if len(normalized) < 120:
        return None
    for bench_text, label in bench_index.items():
        if len(bench_text) < 120:
            continue
        shorter = min(len(normalized), len(bench_text))
        longer = max(len(normalized), len(bench_text))
        if shorter / longer >= 0.85 and (normalized in bench_text or bench_text in normalized):
            return f"near_leak:{label}"
    return None


def _write_shards(output_root: Path, records: Sequence[str], shard_size: int) -> list[str]:
    output_files: list[str] = []
    shard_size = max(1, shard_size)
    for shard_index, start in enumerate(range(0, len(records), shard_size)):
        path = output_root / f"train-{shard_index:05d}.jsonl"
        chunk = records[start : start + shard_size]
        path.write_text("\n".join(chunk) + ("\n" if chunk else ""), encoding="utf-8")
        output_files.append(str(path))
    return output_files


if __name__ == "__main__":
    raise SystemExit(main())