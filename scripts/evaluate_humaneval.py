"""Compatibility wrapper for HumanEval-style vLLM evaluation."""

from __future__ import annotations

from scripts.evaluate_code_bench import main


if __name__ == "__main__":
    raise SystemExit(main(default_dataset="openai_humaneval"))