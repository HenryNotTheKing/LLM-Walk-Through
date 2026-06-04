"""Fixed-size tokenizer alias tools.

This rewrites selected token strings in a tokenizer JSON while preserving their
integer ids. It is meant for the 65,536-token Walkie tokenizer where adding rows
would break checkpoint compatibility and uint16 data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .chat_template import TokenizerAliasPlan, build_tokenizer_alias_plan


DEFAULT_CHATML_TOKENS = ["<|im_start|>", "<|im_end|>"]
DEFAULT_PROTECTED_TOKENS = {"<|endoftext|>", "<|pad|>"}


def write_tokenizer_aliases(
    tokenizer_json: str | Path,
    output_json: str | Path,
    *,
    required_tokens: Iterable[str] = DEFAULT_CHATML_TOKENS,
    reserved_patterns: Iterable[str] = ("unused_", "<unused", "[unused"),
    expected_vocab_size: int | None = None,
) -> TokenizerAliasPlan:
    src = Path(tokenizer_json)
    dst = Path(output_json)
    payload = json.loads(src.read_text(encoding="utf-8"))
    vocab = _extract_vocab(payload)
    plan = build_tokenizer_alias_plan(
        vocab,
        required_tokens=required_tokens,
        reserved_patterns=reserved_patterns,
        protected_tokens=DEFAULT_PROTECTED_TOKENS,
        expected_vocab_size=expected_vocab_size,
    )
    if plan.requires_fallback:
        return plan

    id_to_old = {idx: token for token, idx in vocab.items()}
    for alias, token_id in plan.alias_to_id.items():
        old_token = id_to_old.get(token_id)
        if old_token is not None and old_token != alias:
            del vocab[old_token]
        vocab[alias] = token_id

    _write_vocab(payload, vocab)
    added_tokens = [item for item in payload.get("added_tokens", []) if item.get("content") not in plan.alias_to_id]
    existing_ids = {int(item.get("id", -1)) for item in added_tokens if isinstance(item, dict)}
    for alias, token_id in plan.alias_to_id.items():
        if token_id in existing_ids:
            continue
        added_tokens.append(
            {
                "id": token_id,
                "content": alias,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    payload["added_tokens"] = added_tokens
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def _extract_vocab(payload: dict) -> dict[str, int]:
    model = payload.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("vocab"), dict):
        raise ValueError("tokenizer JSON must contain model.vocab")
    return {str(token): int(idx) for token, idx in model["vocab"].items()}


def _write_vocab(payload: dict, vocab: dict[str, int]) -> None:
    payload["model"]["vocab"] = {token: int(idx) for token, idx in sorted(vocab.items(), key=lambda item: item[1])}
