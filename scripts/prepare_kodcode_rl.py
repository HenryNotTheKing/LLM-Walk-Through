"""Prepare KodCode-V1 for verifier-driven Walkie RL."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq

from posttrain.data.kodcode_sft import build_bench_text_index, detect_bench_leak, render_user_content, stable_hash


READ_COLUMNS = [
    "style",
    "subset",
    "question_id",
    "question",
    "solution",
    "test",
    "test_info",
    "gpt_difficulty",
    "gpt_pass_percentage",
    "benchmark_similarity",
    "benchmark_instruction",
    "benchmark_task_id",
    "filter_reason",
    "metadata",
]
DEFAULT_STYLES = ("instruct", "complete")
TEST_DEF_RE = re.compile(r"\bdef\s+test_\w+\s*\(")
SANDBOX_INCOMPATIBLE_RE = re.compile(
    r"\b(?:tempfile|\w*subprocess\w*|shutil|TemporaryDirectory|NamedTemporaryFile|mkdtemp|mkstemp)\b"
    r"|\b(?:os|Path)\.(?:getcwd|chdir|system|popen|spawn\w*|fork|exec\w*|cwd)\b"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert KodCode-V1 parquet files into RL prompt/test-template JSONL")
    parser.add_argument("--input", default="data/RL/KodCode-V1/data")
    parser.add_argument("--output", default="data/RL/kodcode_v1_rl_bench_aligned")
    parser.add_argument("--bench-root", default="data/bench")
    parser.add_argument("--split", default="train", choices=["train", "use_with_caution"])
    parser.add_argument("--styles", default=",".join(DEFAULT_STYLES), help="comma-separated KodCode styles to keep")
    parser.add_argument("--include-online-judge", action="store_true")
    parser.add_argument("--skip-leak-filter", action="store_true")
    parser.add_argument("--min-gpt-pass-percentage", type=float, default=None)
    parser.add_argument("--max-gpt-pass-percentage", type=float, default=None)
    parser.add_argument("--min-test-functions", type=int, default=1)
    parser.add_argument("--max-question-chars", type=int, default=6000)
    parser.add_argument("--max-test-chars", type=int, default=200000)
    parser.add_argument(
        "--filter-sandbox-incompatible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="drop rows whose prompt or tests require process/tempdir APIs blocked by the training sandbox",
    )
    parser.add_argument("--case-timeout", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None, help="reservoir-sample this many accepted records after scanning the selected split")
    parser.add_argument("--batch-size", type=int, default=2048)
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

    files = sorted(input_root.glob(f"{args.split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found for split {args.split!r} in {input_root}")

    allowed_styles = _allowed_styles(args)
    bench_index = {} if args.skip_leak_filter else build_bench_text_index(args.bench_root)
    records: list[str] = []
    stats: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    subset_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    pass_bucket_counts: Counter[str] = Counter()
    sample_rng = random.Random(int(args.seed))

    leakage_writer = (output_root / "leakage_removed.ndjson").open("w", encoding="utf-8")
    try:
        for path in files:
            parquet = pq.ParquetFile(path)
            available_columns = [column for column in READ_COLUMNS if column in parquet.schema_arrow.names]
            for batch in parquet.iter_batches(batch_size=max(1, int(args.batch_size)), columns=available_columns):
                for row in batch.to_pylist():
                    stats["seen"] += 1
                    decision = _build_record(row, source_file=path.name, bench_index=bench_index, allowed_styles=allowed_styles, args=args)
                    if decision.record is None:
                        stats[decision.reason] += 1
                        if decision.reason.startswith(("leak:", "near_leak:")):
                            leakage_writer.write(json.dumps({"reason": decision.reason, "question_id": row.get("question_id")}, ensure_ascii=False) + "\n")
                        continue
                    record_line = json.dumps(decision.record, ensure_ascii=False)
                    accepted_index = int(stats["kept"])
                    stats["kept"] += 1
                    _append_or_sample(records, record_line, accepted_index=accepted_index, sample_size=args.sample_size, rng=sample_rng)
                    style_counts[str(decision.record["style"])] += 1
                    subset_counts[str(decision.record["subset"])] += 1
                    difficulty_counts[str(decision.record["gpt_difficulty"])] += 1
                    pass_bucket_counts[_pass_bucket(decision.record.get("gpt_pass_percentage"))] += 1
                    if args.limit is not None and stats["kept"] >= int(args.limit):
                        raise StopIteration
    except StopIteration:
        pass
    finally:
        leakage_writer.close()

    if args.shuffle:
        random.Random(int(args.seed)).shuffle(records)
    output_files = _write_shards(output_root, records, int(args.shard_size))
    manifest = {
        "source": str(input_root),
        "split": str(args.split),
        "bench_root": str(args.bench_root),
        "output_files": output_files,
        "output_record_count": len(records),
        "sample_size": int(args.sample_size) if args.sample_size is not None else None,
        "allowed_styles": list(allowed_styles),
        "leak_filter": not bool(args.skip_leak_filter),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed) if args.shuffle else None,
        "filters": {
            "min_gpt_pass_percentage": args.min_gpt_pass_percentage,
            "max_gpt_pass_percentage": args.max_gpt_pass_percentage,
            "min_test_functions": int(args.min_test_functions),
            "max_question_chars": int(args.max_question_chars),
            "max_test_chars": int(args.max_test_chars),
            "filter_sandbox_incompatible": bool(args.filter_sandbox_incompatible),
            "case_timeout": float(args.case_timeout),
        },
        "stats": dict(stats),
        "style_counts": dict(style_counts),
        "subset_counts": dict(subset_counts),
        "difficulty_counts": dict(difficulty_counts),
        "pass_bucket_counts": dict(pass_bucket_counts),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


class Decision:
    def __init__(self, record: dict[str, Any] | None, reason: str) -> None:
        self.record = record
        self.reason = reason


def _build_record(
    row: Mapping[str, Any],
    *,
    source_file: str,
    bench_index: Mapping[str, str],
    allowed_styles: set[str],
    args: argparse.Namespace,
) -> Decision:
    style = str(row.get("style") or "").strip().lower()
    if style not in allowed_styles:
        return Decision(None, f"style:{style or 'missing'}")

    question = str(row.get("question") or "").strip()
    if not question:
        return Decision(None, "missing_question")
    if len(question) > int(args.max_question_chars):
        return Decision(None, "question_too_long")
    if bool(args.filter_sandbox_incompatible):
        sandbox_reason = _sandbox_incompatibility_reason(question)
        if sandbox_reason is not None:
            return Decision(None, f"sandbox_incompatible_question:{sandbox_reason}")

    test_code = str(row.get("test") or "").strip()
    if not test_code:
        return Decision(None, "missing_test")
    if len(test_code) > int(args.max_test_chars):
        return Decision(None, "test_too_long")
    if bool(args.filter_sandbox_incompatible):
        sandbox_reason = _sandbox_incompatibility_reason(test_code)
        if sandbox_reason is not None:
            return Decision(None, f"sandbox_incompatible_test:{sandbox_reason}")
    test_function_count = len(TEST_DEF_RE.findall(test_code))
    if test_function_count < int(args.min_test_functions):
        return Decision(None, "too_few_test_functions")

    pass_percentage = _float_or_none(row.get("gpt_pass_percentage"))
    if args.min_gpt_pass_percentage is not None and (pass_percentage is None or pass_percentage < float(args.min_gpt_pass_percentage)):
        return Decision(None, "gpt_pass_percentage_too_low")
    if args.max_gpt_pass_percentage is not None and (pass_percentage is None or pass_percentage > float(args.max_gpt_pass_percentage)):
        return Decision(None, "gpt_pass_percentage_too_high")

    if bench_index:
        leak_reason = detect_bench_leak(row, bench_index)
        if leak_reason is not None:
            return Decision(None, leak_reason)

    entry_points = _entry_points(row.get("test_info"))
    prompt = f"user:\n{render_user_content(question, style)}\nassistant:\n"
    template = build_pytest_template(
        test_code,
        style=style,
        starter_code=question if style == "complete" else "",
        entry_points=entry_points,
        case_timeout=float(args.case_timeout),
    )
    question_id = str(row.get("question_id") or stable_hash(question))
    record = {
        "prompt": prompt,
        "tests": "",
        "test_program_template": template,
        "task_id": f"kodcode/{question_id}",
        "source": "kodcode_v1",
        "source_file": source_file,
        "task_type": "pytest",
        "style": style,
        "prompt_kind": "humaneval" if style == "complete" else "mbpp",
        "subset": str(row.get("subset") or ""),
        "gpt_difficulty": str(row.get("gpt_difficulty") or ""),
        "gpt_pass_percentage": pass_percentage,
        "benchmark_similarity": _float_or_none(row.get("benchmark_similarity")),
        "benchmark_task_id": str(row.get("benchmark_task_id") or ""),
        "entry_point": entry_points[0] if entry_points else "",
        "entry_points": entry_points,
        "num_tests": test_function_count,
        "question_chars": len(question),
        "test_chars": len(test_code),
        "template_chars": len(template),
    }
    return Decision(record, "kept")


def build_pytest_template(test_code: str, *, style: str, starter_code: str, entry_points: Sequence[str], case_timeout: float) -> str:
    entry_points_json = json.dumps(list(entry_points), ensure_ascii=False)
    template = r'''import ast
import inspect
import json
import math
import re
import sys
import textwrap
import types
import unittest
from typing import *

candidate_code = {{candidate_code}}
TEST_CODE = __TEST_CODE_REPR__
STARTER_CODE = __STARTER_CODE_REPR__
STYLE = __STYLE_REPR__
ENTRY_POINTS = json.loads(__ENTRY_POINTS_JSON_REPR__)
CASE_TIMEOUT = __CASE_TIMEOUT_REPR__
PRELUDE = """from __future__ import annotations
from typing import *
import collections
import functools
import heapq
import itertools
import math
"""


class _SkipTest(Exception):
    pass


class _RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exception}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match is not None and re.search(str(self.match), str(exc)) is None:
            raise AssertionError(f"exception message {exc!r} does not match {self.match!r}")
        self.value = exc
        return True


class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, actual):
        try:
            return math.isclose(actual, self.expected, rel_tol=self.rel, abs_tol=self.abs)
        except TypeError:
            return actual == self.expected


class _MonkeyPatch:
    def setattr(self, target, name=None, value=None, raising=True):
        if isinstance(target, str):
            module_name, attr = target.rsplit(".", 1)
            target = __import__(module_name, fromlist=[attr])
            name = attr
        if name is None:
            raise TypeError("setattr requires a name")
        if raising and not hasattr(target, name):
            raise AttributeError(name)
        setattr(target, name, value)

    def setitem(self, mapping, name, value):
        mapping[name] = value


class _PytestMark:
    def parametrize(self, argnames, argvalues, *args, **kwargs):
        names = [name.strip() for name in str(argnames).split(",")] if isinstance(argnames, str) else list(argnames)

        def decorator(func):
            current = list(getattr(func, "__kodcode_parametrize__", []))
            current.append((names, list(argvalues)))
            setattr(func, "__kodcode_parametrize__", current)
            return func

        return decorator

    def __getattr__(self, name):
        def marker(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def decorator(func):
                return func

            return decorator

        return marker


class _PytestModule(types.ModuleType):
    def __init__(self):
        super().__init__("pytest")
        self.mark = _PytestMark()

    def raises(self, expected_exception, *args, **kwargs):
        return _RaisesContext(expected_exception, match=kwargs.get("match"))

    def approx(self, expected, rel=1e-6, abs=1e-12, **kwargs):
        return _Approx(expected, rel=rel, abs=abs)

    def fixture(self, func=None, *args, **kwargs):
        def decorator(inner):
            setattr(inner, "__kodcode_fixture__", True)
            return inner

        if callable(func):
            return decorator(func)
        return decorator

    def skip(self, reason=""):
        raise _SkipTest(reason)


def _defined_symbols(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    symbols = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
    return symbols


def _candidate_defines_entry(source):
    symbols = _defined_symbols(source)
    if not ENTRY_POINTS:
        return bool(symbols)
    return any(name in symbols for name in ENTRY_POINTS) or "Solution" in symbols


def _indent_missing_body(source):
    stripped = source.strip("\n")
    if not stripped.strip():
        return "    pass\n"
    lines = stripped.splitlines()
    if any(line[:1].isspace() for line in lines if line.strip()):
        return stripped.rstrip() + "\n"
    return textwrap.indent(stripped.rstrip(), "    ") + "\n"


def _solution_source():
    source = str(candidate_code or "").strip()
    if STYLE == "complete" and STARTER_CODE.strip() and not _candidate_defines_entry(source):
        return PRELUDE + "\n" + STARTER_CODE.rstrip() + "\n" + _indent_missing_body(source)
    return PRELUDE + "\n" + source.rstrip() + "\n"


def _base_namespace(name):
    namespace = {"__name__": name, "List": List, "Dict": Dict, "Tuple": Tuple, "Set": Set, "Optional": Optional, "Any": Any, "math": math}
    return namespace


def _install_solution_module():
    solution_namespace = _base_namespace("solution")
    exec(_solution_source(), solution_namespace)
    module = types.ModuleType("solution")
    module.__dict__.update(solution_namespace)
    sys.modules["solution"] = module
    return module


def _parameter_cases(func):
    parametrized = list(getattr(func, "__kodcode_parametrize__", []))
    cases = [{}]
    for names, values in parametrized:
        next_cases = []
        for raw_value in values:
            if len(names) == 1:
                value_tuple = (raw_value,)
            else:
                value_tuple = tuple(raw_value)
            for case in cases:
                merged = dict(case)
                merged.update(dict(zip(names, value_tuple)))
                next_cases.append(merged)
        cases = next_cases
    return cases


def _fixture_value(name, namespace):
    value = namespace.get(name)
    if callable(value) and getattr(value, "__kodcode_fixture__", False):
        return _call_with_fixtures(value, namespace, {})
    if name == "monkeypatch":
        return _MonkeyPatch()
    raise AssertionError(f"unsupported pytest fixture: {name}")


def _call_with_fixtures(func, namespace, provided):
    signature = inspect.signature(func)
    kwargs = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in provided:
            kwargs[name] = provided[name]
        else:
            kwargs[name] = _fixture_value(name, namespace)
    return func(**kwargs)


def _run_test_functions(namespace):
    count = 0
    for name, value in sorted(namespace.items()):
        if not name.startswith("test_") or not callable(value):
            continue
        for case in _parameter_cases(value):
            try:
                _call_with_fixtures(value, namespace, case)
                count += 1
            except _SkipTest:
                count += 1
    return count


def _run_unittest_cases(namespace):
    count = 0
    for value in list(namespace.values()):
        if not isinstance(value, type) or not issubclass(value, unittest.TestCase) or value is unittest.TestCase:
            continue
        set_up_class = getattr(value, "setUpClass", None)
        tear_down_class = getattr(value, "tearDownClass", None)
        if callable(set_up_class):
            set_up_class()
        try:
            for method_name in sorted(name for name in dir(value) if name.startswith("test")):
                case = value(method_name)
                case.setUp()
                try:
                    getattr(case, method_name)()
                    count += 1
                finally:
                    case.tearDown()
        finally:
            if callable(tear_down_class):
                tear_down_class()
    return count


pytest_module = _PytestModule()
sys.modules["pytest"] = pytest_module
solution_module = _install_solution_module()
test_namespace = _base_namespace("kodcode_test")
test_namespace.update({name: value for name, value in solution_module.__dict__.items() if not name.startswith("__")})
test_namespace.update({"pytest": pytest_module})
exec(TEST_CODE, test_namespace)
executed = _run_test_functions(test_namespace) + _run_unittest_cases(test_namespace)
if executed <= 0:
    raise AssertionError("no test functions were executed")

print('ALL TESTS PASSED')
'''
    return (
        template.replace("__TEST_CODE_REPR__", repr(test_code))
        .replace("__STARTER_CODE_REPR__", repr(starter_code))
        .replace("__STYLE_REPR__", repr(style))
        .replace("__ENTRY_POINTS_JSON_REPR__", repr(entry_points_json))
        .replace("__CASE_TIMEOUT_REPR__", repr(float(case_timeout)))
    )


def _sandbox_incompatibility_reason(source: str) -> str | None:
    match = SANDBOX_INCOMPATIBLE_RE.search(source)
    return match.group(0) if match is not None else None


def _allowed_styles(args: argparse.Namespace) -> set[str]:
    styles = {item.strip().lower() for item in str(args.styles).split(",") if item.strip()}
    if args.include_online_judge:
        styles.add("online_judge")
    return styles


def _entry_points(test_info: Any) -> list[str]:
    names: list[str] = []
    if isinstance(test_info, list):
        for item in test_info:
            if isinstance(item, Mapping):
                name = str(item.get("function_name") or "").strip()
                if name and name not in names:
                    names.append(name)
    return names


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pass_bucket(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "none"
    if number <= 0.0:
        return "0"
    if number < 0.25:
        return "(0,.25)"
    if number < 0.75:
        return "[.25,.75)"
    if number < 1.0:
        return "[.75,1)"
    return "1"


def _append_or_sample(records: list[str], record: str, *, accepted_index: int, sample_size: int | None, rng: random.Random) -> None:
    if sample_size is None:
        records.append(record)
        return
    sample_size = max(0, int(sample_size))
    if sample_size == 0:
        return
    if len(records) < sample_size:
        records.append(record)
        return
    replacement_index = rng.randrange(accepted_index + 1)
    if replacement_index < sample_size:
        records[replacement_index] = record


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