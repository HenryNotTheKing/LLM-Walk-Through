"""Code benchmark loaders and test-program builders."""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .humaneval import extract_completion_code


SUPPORTED_CODE_BENCH_DATASETS = (
    "humanevalplus",
    "openai_humaneval",
    "mbpp",
    "mbppplus",
)

HUMANEVAL_STYLE_DATASETS = {"humanevalplus", "openai_humaneval"}
MBPP_STYLE_DATASETS = {"mbpp", "mbppplus"}

DEFAULT_DATASET_PATTERNS: dict[str, tuple[str, ...]] = {
    "humanevalplus": (
        "humanevalplus/test.jsonl",
        "humanevalplus/data/test-*.parquet",
    ),
    "openai_humaneval": ("openai_humaneval/openai_humaneval/test-*.parquet",),
    "mbpp": ("mbpp/sanitized/test-*.parquet",),
    "mbppplus": ("mbppplus/data/test-*.parquet",),
}


@dataclass(frozen=True)
class CodeBenchSample:
    dataset: str
    task_id: str
    prompt: str
    kind: str
    test: str = ""
    entry_point: str = ""
    canonical_signature: str = ""
    test_imports: tuple[str, ...] = ()
    test_list: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CodeBenchCandidate:
    dataset: str
    task_id: str
    completion_id: int
    prompt: str
    completion: str
    extracted_code: str
    test_program: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_code_bench_samples(
    dataset: str,
    *,
    bench_root: str | Path = "data/bench",
    data: str | Path | None = None,
    limit: int | None = None,
) -> list[CodeBenchSample]:
    """Load one supported benchmark split into normalized samples."""
    dataset = normalize_dataset_name(dataset)
    rows: list[dict[str, Any]] = []
    for path in resolve_dataset_files(dataset, bench_root=bench_root, data=data):
        if path.suffix == ".jsonl":
            rows.extend(_load_jsonl_rows(path))
        elif path.suffix == ".parquet":
            rows.extend(_load_parquet_rows(path))
        else:
            raise ValueError(f"unsupported data file suffix: {path}")

    samples = [sample_from_bench_row(dataset, row, index) for index, row in enumerate(rows)]
    if limit is not None:
        samples = samples[: max(0, int(limit))]
    return samples


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower().replace("-", "_")
    aliases = {
        "human_eval_plus": "humanevalplus",
        "humaneval_plus": "humanevalplus",
        "human_eval": "openai_humaneval",
        "humaneval": "openai_humaneval",
        "openai_human_eval": "openai_humaneval",
        "mbpp_plus": "mbppplus",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_CODE_BENCH_DATASETS:
        supported = ", ".join(SUPPORTED_CODE_BENCH_DATASETS)
        raise ValueError(f"unsupported dataset {dataset!r}; expected one of: {supported}")
    return normalized


def resolve_dataset_files(
    dataset: str,
    *,
    bench_root: str | Path = "data/bench",
    data: str | Path | None = None,
) -> list[Path]:
    if data is not None:
        paths = _expand_data_path(data)
        if not paths:
            raise FileNotFoundError(f"no benchmark data files matched {data}")
        return paths

    root = Path(bench_root)
    for pattern in DEFAULT_DATASET_PATTERNS[dataset]:
        matches = sorted(Path(match) for match in glob.glob(str(root / pattern)))
        if matches:
            return matches
    patterns = ", ".join(str(root / pattern) for pattern in DEFAULT_DATASET_PATTERNS[dataset])
    raise FileNotFoundError(f"no data files found for {dataset}; tried: {patterns}")


def sample_from_bench_row(dataset: str, row: dict[str, Any], index: int = 0) -> CodeBenchSample:
    dataset = normalize_dataset_name(dataset)
    if dataset in HUMANEVAL_STYLE_DATASETS:
        excluded = {"task_id", "prompt", "test", "entry_point"}
        return CodeBenchSample(
            dataset=dataset,
            task_id=str(row.get("task_id", f"{dataset}/{index}")),
            prompt=str(row["prompt"]),
            kind="humaneval",
            test=str(row["test"]),
            entry_point=str(row["entry_point"]),
            metadata={key: value for key, value in row.items() if key not in excluded},
        )

    if dataset in MBPP_STYLE_DATASETS:
        prompt = row.get("prompt", row.get("text"))
        if prompt is None:
            raise KeyError(f"{dataset} row is missing prompt/text")
        test_imports = list(_as_string_tuple(row.get("test_imports")))
        setup_code = row.get("test_setup_code")
        if isinstance(setup_code, str) and setup_code.strip():
            test_imports.insert(0, setup_code.strip())
        test_list = _as_string_tuple(row.get("test_list"))
        plus_test = str(row.get("test") or "")
        canonical_signature = _extract_first_function_signature(str(row.get("code") or ""))
        excluded = {
            "task_id",
            "prompt",
            "text",
            "code",
            "test_imports",
            "test_setup_code",
            "test_list",
            "test",
        }
        return CodeBenchSample(
            dataset=dataset,
            task_id=str(row.get("task_id", f"{dataset}/{index}")),
            prompt=str(prompt),
            kind="mbpp",
            entry_point=_function_name_from_signature(canonical_signature),
            canonical_signature=canonical_signature,
            test=plus_test,
            test_imports=tuple(test_imports),
            test_list=test_list,
            metadata={key: value for key, value in row.items() if key not in excluded},
        )

    raise ValueError(f"unsupported dataset: {dataset}")


def render_code_bench_prompt(sample: CodeBenchSample, *, prompt_style: str = "plain_dialog") -> str:
    """Render the model-facing prompt using lowercase user/assistant labels by default."""
    style = prompt_style.strip().lower()
    if style == "raw":
        return sample.prompt
    if style not in {"plain_dialog", "dialog"}:
        raise ValueError("prompt_style must be 'plain_dialog' or 'raw'")

    if sample.kind == "humaneval":
        instruction = (
            "Complete the following Python function. Return only the missing code; "
            "do not repeat the prompt."
        )
        task_prompt = sample.prompt.rstrip()
    elif sample.kind == "mbpp":
        instruction = (
            "Write a complete Python solution. Return only Python code. "
            "Define the required function exactly as named in the signature below; "
            "helper functions are allowed, but do not rename the required function."
        )
        task_prompt = sample.prompt.rstrip()
        if sample.canonical_signature:
            task_prompt = (
                f"{task_prompt}\n\n"
                "Required function signature:\n"
                f"{sample.canonical_signature}"
            )
    else:
        raise ValueError(f"unsupported sample kind: {sample.kind}")
    return f"user:\n{instruction}\n\n{task_prompt}\nassistant:\n"


def build_code_bench_test_program(sample: CodeBenchSample, completion: str) -> str:
    code = extract_code_for_execution(completion)
    if sample.kind == "humaneval":
        if _contains_function_definition(code, sample.entry_point):
            solution = code.rstrip() + "\n"
        else:
            solution = f"{sample.prompt}{code.rstrip()}\n"
        return (
            f"{solution}\n"
            f"{sample.test}\n\n"
            f"check({sample.entry_point})\n"
            "print('ALL TESTS PASSED')\n"
        )

    if sample.kind == "mbpp":
        tests = sample.test.strip() if sample.test.strip() else "\n".join(sample.test_list)
        parts = ["\n".join(sample.test_imports).strip(), code.rstrip(), tests.strip(), "print('ALL TESTS PASSED')"]
        return "\n\n".join(part for part in parts if part) + "\n"

    raise ValueError(f"unsupported sample kind: {sample.kind}")


def build_code_bench_candidates(
    samples: Sequence[CodeBenchSample],
    completions: Sequence[Sequence[str]],
    *,
    prompts: Sequence[str] | None = None,
) -> list[CodeBenchCandidate]:
    if len(samples) != len(completions):
        raise ValueError(f"samples/completions length mismatch: {len(samples)} vs {len(completions)}")
    rendered_prompts = list(prompts) if prompts is not None else [sample.prompt for sample in samples]
    if len(rendered_prompts) != len(samples):
        raise ValueError(f"prompts/samples length mismatch: {len(rendered_prompts)} vs {len(samples)}")

    candidates: list[CodeBenchCandidate] = []
    for sample, sample_prompt, sample_completions in zip(samples, rendered_prompts, completions):
        if not sample_completions:
            raise ValueError(f"sample {sample.task_id} has no completions")
        for completion_id, completion in enumerate(sample_completions):
            candidates.append(
                CodeBenchCandidate(
                    dataset=sample.dataset,
                    task_id=sample.task_id,
                    completion_id=completion_id,
                    prompt=sample_prompt,
                    completion=completion,
                    extracted_code=extract_code_for_execution(completion),
                    test_program=build_code_bench_test_program(sample, completion),
                    metadata={"sample": sample.metadata},
                )
            )
    return candidates


def extract_code_for_execution(response: str) -> str:
    code = extract_completion_code(response)
    code = re.sub(r"^\s*(?:assistant|Assistant)\s*:\s*", "", code)
    return code.strip("\n")


def _contains_function_definition(code: str, entry_point: str) -> bool:
    pattern = rf"^\s*def\s+{re.escape(entry_point)}\s*\("
    return re.search(pattern, code, flags=re.MULTILINE) is not None


def _extract_first_function_signature(code: str) -> str:
    match = re.search(
        r"(?m)^\s*def\s+[A-Za-z_]\w*\s*\([^\n)]*\)\s*(?:->\s*[^:]+\s*)?:",
        code,
    )
    return match.group(0).strip() if match else ""


def _function_name_from_signature(signature: str) -> str:
    match = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", signature)
    return match.group(1) if match else ""


def _expand_data_path(path: str | Path) -> list[Path]:
    text = str(path)
    if any(char in text for char in "*?[]"):
        return sorted(Path(match) for match in glob.glob(text))
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if candidate.is_dir():
        jsonl = sorted(candidate.rglob("*.jsonl"))
        parquet = sorted(candidate.rglob("*.parquet"))
        return jsonl + parquet
    return []


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return list(table.to_pylist())


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    return (str(value),)