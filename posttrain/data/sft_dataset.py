"""SFT dataset helpers for JSONL/JSON/Parquet conversation data."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .chat_template import ChatTemplate, EncodedChatExample, normalize_messages


@dataclass(frozen=True)
class SFTDatasetState:
    global_index: int = 0


def iter_sft_rows(paths: Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            json_files = [file_path for file_path in sorted(path.rglob("*.json")) if file_path.name != "manifest.json"]
            files = sorted(path.rglob("*.jsonl")) + json_files + sorted(path.rglob("*.parquet"))
            for file_path in files:
                yield from iter_sft_rows([file_path])
            continue
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        if isinstance(row, dict):
                            yield row
        elif suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for row in payload:
                    if isinstance(row, dict):
                        yield row
            elif isinstance(payload, dict):
                rows = payload.get("data")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            yield row
                else:
                    yield payload
        elif suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    if isinstance(row, dict):
                        yield row
        else:
            raise ValueError(f"unsupported SFT data file: {path}")


class SFTIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        tokenizer: Any,
        template: ChatTemplate,
        max_length: int,
        rank: int = 0,
        world_size: int = 1,
        start_index: int = 0,
    ) -> None:
        super().__init__()
        self.paths = [str(path) for path in paths]
        self.tokenizer = tokenizer
        self.template = template
        self.max_length = int(max_length)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[EncodedChatExample]:
        for global_index, row in enumerate(iter_sft_rows(self.paths)):
            if global_index < self.start_index:
                continue
            if global_index % self.world_size != self.rank:
                continue
            turns = normalize_messages(row)
            yield self.template.encode(turns, tokenizer=self.tokenizer, max_length=self.max_length)


def collate_sft_batch(
    examples: Sequence[EncodedChatExample],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    max_len = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for example in examples:
        pad_len = max_len - len(example.input_ids)
        input_rows.append(example.input_ids + [pad_token_id] * pad_len)
        label_rows.append(example.labels + [-1] * pad_len)
        mask_rows.append(example.attention_mask + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long),
        "labels": torch.tensor(label_rows, dtype=torch.long),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
    }
