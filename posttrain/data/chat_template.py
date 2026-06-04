"""Chat template rendering and assistant-only label masks.

The current Walkie tokenizer is fixed at 65,536 entries so SFT/RL code must not
require adding new vocabulary rows. This module therefore supports both a
ChatML-like alias mode and a pure-text fallback that only depends on the
existing end-of-text token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol


class TokenizerLike(Protocol):
    eos_token_id: int


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str


@dataclass(frozen=True)
class EncodedChatExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]
    text: str


@dataclass(frozen=True)
class TokenizerAliasPlan:
    alias_to_id: dict[str, int]
    vocab_size: int
    requires_fallback: bool
    reason: str = ""


@dataclass
class ChatTemplate:
    kind: str = "chatml_lowfreq_alias"
    eos_token: str = "<|endoftext|>"
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    role_names: Mapping[str, str] = field(
        default_factory=lambda: {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
        }
    )
    train_on_assistant_only: bool = True

    def encode(
        self,
        turns: Iterable[ChatTurn | Mapping[str, Any]],
        *,
        tokenizer: Any,
        max_length: int,
        add_eos: bool = True,
        truncate_side: str = "left",
    ) -> EncodedChatExample:
        normalized = [turn if isinstance(turn, ChatTurn) else _turn_from_mapping(turn) for turn in turns]
        token_ids: list[int] = []
        supervised_targets: list[bool] = []
        text_parts: list[str] = []

        for turn in normalized:
            prefix, content, suffix = self._render_turn_parts(turn)
            prefix_ids = _encode(tokenizer, prefix)
            content_ids = _encode(tokenizer, content)
            suffix_ids = _encode(tokenizer, suffix)
            text_parts.extend([prefix, content, suffix])

            supervise = turn.role == "assistant" or not self.train_on_assistant_only
            token_ids.extend(prefix_ids)
            supervised_targets.extend([False] * len(prefix_ids))
            token_ids.extend(content_ids)
            supervised_targets.extend([supervise] * len(content_ids))
            token_ids.extend(suffix_ids)
            supervised_targets.extend([supervise and self.kind == "chatml_lowfreq_alias"] * len(suffix_ids))

        if add_eos:
            eos_ids = _eos_ids(tokenizer, self.eos_token)
            token_ids.extend(eos_ids)
            last_is_assistant = bool(normalized and normalized[-1].role == "assistant")
            supervised_targets.extend([last_is_assistant or not self.train_on_assistant_only] * len(eos_ids))
            text_parts.append(self.eos_token)

        if max_length <= 0:
            raise ValueError("max_length must be positive")
        # WalkieForCausalLM expects caller-provided next-token targets, matching
        # pretraining's (x[t] -> y[t+1]) contract.
        if len(token_ids) > max_length + 1:
            if truncate_side == "left":
                token_ids = token_ids[-(max_length + 1):]
                supervised_targets = supervised_targets[-(max_length + 1):]
            elif truncate_side == "right":
                token_ids = token_ids[: max_length + 1]
                supervised_targets = supervised_targets[: max_length + 1]
            else:
                raise ValueError("truncate_side must be 'left' or 'right'")

        if len(token_ids) < 2:
            input_ids: list[int] = []
            labels: list[int] = []
        else:
            input_ids = token_ids[:-1]
            labels = [token_ids[index + 1] if supervised_targets[index + 1] else -1 for index in range(len(input_ids))]

        return EncodedChatExample(
            input_ids=input_ids,
            labels=labels,
            attention_mask=[1] * len(input_ids),
            text="".join(text_parts),
        )

    def render_prompt(self, turns: Iterable[ChatTurn | Mapping[str, Any]]) -> str:
        pieces: list[str] = []
        for raw_turn in turns:
            turn = raw_turn if isinstance(raw_turn, ChatTurn) else _turn_from_mapping(raw_turn)
            prefix, content, suffix = self._render_turn_parts(turn)
            pieces.extend([prefix, content, suffix])
        if self.kind == "chatml_lowfreq_alias":
            pieces.append(f"{self.im_start}assistant\n")
        else:
            pieces.append("Assistant:\n")
        return "".join(pieces)

    def _render_turn_parts(self, turn: ChatTurn) -> tuple[str, str, str]:
        role = turn.role.lower().strip()
        content = turn.content.strip()
        if self.kind == "chatml_lowfreq_alias":
            return f"{self.im_start}{role}\n", content, f"{self.im_end}\n"
        if self.kind == "plain_eot":
            role_name = self.role_names.get(role, role.title())
            return f"{role_name}:\n", content, "\n"
        if self.kind == "plain_lower":
            return f"{role}:\n", content, "\n"
        raise ValueError(f"unknown chat template kind: {self.kind}")


def normalize_messages(row: Mapping[str, Any]) -> list[ChatTurn]:
    """Normalize common SFT row schemas into role/content turns."""
    if isinstance(row.get("messages"), list):
        return [_turn_from_mapping(item) for item in row["messages"] if isinstance(item, Mapping)]
    for key in ("conversations", "conversation"):
        if isinstance(row.get(key), list):
            return [_turn_from_mapping(item) for item in row[key] if isinstance(item, Mapping)]

    prompt = row.get("prompt")
    response = row.get("response", row.get("completion"))
    if isinstance(prompt, str) and isinstance(response, str):
        return [ChatTurn("user", prompt.strip()), ChatTurn("assistant", response.strip())]

    instruction = row.get("instruction")
    output = row.get("output")
    if isinstance(instruction, str) and isinstance(output, str):
        user_parts = [instruction.strip()]
        input_text = row.get("input")
        if isinstance(input_text, str) and input_text.strip():
            user_parts.append(input_text.strip())
        return [ChatTurn("user", "\n".join(user_parts)), ChatTurn("assistant", output.strip())]

    raise ValueError("row must contain messages, prompt/response, or instruction/output fields")


def build_tokenizer_alias_plan(
    vocab: Mapping[str, int],
    *,
    required_tokens: Iterable[str],
    reserved_patterns: Iterable[str] = (),
    protected_tokens: Iterable[str] = ("<|endoftext|>", "<|pad|>"),
    expected_vocab_size: int | None = None,
    max_token_id: int | None = 65535,
) -> TokenizerAliasPlan:
    """Find stable token ids that can be aliased without changing vocab size.

    This function does not edit the tokenizer. It only returns a plan. It uses
    explicit reserved-looking slots such as ``unused_*``; if none exist, callers
    should fall back to the plain text template instead of guessing.
    """
    required = list(required_tokens)
    protected = set(protected_tokens)
    vocab_size = (max(vocab.values()) + 1) if vocab else 0
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        return TokenizerAliasPlan({}, vocab_size, True, f"vocab_size={vocab_size} != expected {expected_vocab_size}")
    if max_token_id is not None and any(int(idx) > max_token_id for idx in vocab.values()):
        return TokenizerAliasPlan({}, vocab_size, True, f"token id exceeds {max_token_id}")
    existing = {token: int(vocab[token]) for token in required if token in vocab}
    missing = [token for token in required if token not in existing]
    patterns = tuple(reserved_patterns)
    candidates = [
        (token, int(idx))
        for token, idx in vocab.items()
        if token not in required and token not in protected and any(pattern in token for pattern in patterns)
    ]
    candidates.sort(key=lambda item: item[1])

    if len(candidates) < len(missing):
        return TokenizerAliasPlan(
            alias_to_id=existing,
            vocab_size=vocab_size,
            requires_fallback=True,
            reason="not enough explicit reserved tokenizer slots",
        )

    alias_to_id = dict(existing)
    for token, (_, idx) in zip(missing, candidates):
        alias_to_id[token] = idx
    return TokenizerAliasPlan(
        alias_to_id=alias_to_id,
        vocab_size=vocab_size,
        requires_fallback=False,
    )


def _turn_from_mapping(item: Mapping[str, Any]) -> ChatTurn:
    role = item.get("role", item.get("from", item.get("speaker", "user")))
    content = item.get("content", item.get("value", item.get("text", "")))
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        content = "\n".join(parts)
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError(f"invalid chat turn: {item!r}")
    role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role.lower(), role.lower())
    return ChatTurn(role=role, content=content)


def _encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    encoded = tokenizer.encode(text)
    if isinstance(encoded, list):
        return [int(item) for item in encoded]
    if hasattr(encoded, "ids"):
        return [int(item) for item in encoded.ids]
    raise TypeError("tokenizer.encode must return list[int] or an object with .ids")


def _eos_ids(tokenizer: Any, eos_token: str) -> list[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        return [int(eos_id)]
    ids = _encode(tokenizer, eos_token)
    if not ids:
        raise ValueError("could not resolve eos token id")
    return ids
