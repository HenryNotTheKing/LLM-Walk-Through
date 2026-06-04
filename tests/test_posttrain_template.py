"""Post-training chat templates and supervised masks."""

from __future__ import annotations

from posttrain.data.chat_template import (
    ChatTemplate,
    ChatTurn,
    build_tokenizer_alias_plan,
    normalize_messages,
)
from posttrain.data.tokenizer_alias import write_tokenizer_aliases


class ToyTokenizer:
    eos_token_id = 0
    pad_token_id = 1

    def __init__(self) -> None:
        self.vocab = {
            "<|endoftext|>": 0,
            "<|pad|>": 1,
            "System": 2,
            "User": 3,
            "Assistant": 4,
            "hello": 5,
            "world": 6,
            "answer": 7,
            "ok": 8,
        }

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in text.replace("\n", " ").split():
            ids.append(self.vocab.get(token.strip(":"), 9))
        return ids


def test_plain_eot_masks_only_assistant_tokens() -> None:
    tokenizer = ToyTokenizer()
    template = ChatTemplate(kind="plain_eot", eos_token="<|endoftext|>")
    example = template.encode(
        [
            ChatTurn(role="system", content="hello"),
            ChatTurn(role="user", content="world"),
            ChatTurn(role="assistant", content="answer ok"),
        ],
        tokenizer=tokenizer,
        max_length=64,
    )

    supervised_ids = [label for label in example.labels if label != -1]
    assert supervised_ids == tokenizer.encode("answer ok") + [tokenizer.eos_token_id]
    assert example.attention_mask == [1] * len(example.input_ids)
    first_answer_label_index = example.labels.index(tokenizer.vocab["answer"])
    assert example.input_ids[first_answer_label_index] != tokenizer.vocab["answer"]


def test_plain_lower_renders_lowercase_dialog_labels() -> None:
    tokenizer = ToyTokenizer()
    template = ChatTemplate(kind="plain_lower", eos_token="<|endoftext|>")
    example = template.encode(
        [
            ChatTurn(role="user", content="hello"),
            ChatTurn(role="assistant", content="answer"),
        ],
        tokenizer=tokenizer,
        max_length=64,
    )

    assert example.text.startswith("user:\nhello\nassistant:\nanswer")


def test_normalize_messages_accepts_prompt_response_and_instruction_rows() -> None:
    prompt_row = {"prompt": "hello", "response": "world"}
    instruction_row = {"instruction": "hello", "input": "world", "output": "answer"}

    assert normalize_messages(prompt_row) == [
        ChatTurn(role="user", content="hello"),
        ChatTurn(role="assistant", content="world"),
    ]
    assert normalize_messages(instruction_row) == [
        ChatTurn(role="user", content="hello\nworld"),
        ChatTurn(role="assistant", content="answer"),
    ]


def test_alias_plan_uses_reserved_or_tail_slots_without_changing_vocab_size() -> None:
    vocab = {"<|endoftext|>": 0, "<|pad|>": 1, "unused_1": 65534, "unused_2": 65535}
    plan = build_tokenizer_alias_plan(
        vocab,
        required_tokens=["<|im_start|>", "<|im_end|>"],
        reserved_patterns=["unused_"],
    )

    assert plan.vocab_size == 65536
    assert plan.alias_to_id == {"<|im_start|>": 65534, "<|im_end|>": 65535}
    assert plan.requires_fallback is False


def test_alias_plan_falls_back_when_no_safe_slots_exist() -> None:
    vocab = {"<|endoftext|>": 0, "<|pad|>": 1, "token": 2}
    plan = build_tokenizer_alias_plan(vocab, required_tokens=["<|im_start|>"])

    assert plan.alias_to_id == {}
    assert plan.requires_fallback is True


def test_write_tokenizer_aliases_preserves_vocab_size(tmp_path) -> None:
    tokenizer_json = {
        "model": {"type": "BPE", "vocab": {"<|endoftext|>": 0, "<|pad|>": 1, "unused_1": 2}},
        "added_tokens": [],
    }
    src = tmp_path / "tokenizer.json"
    dst = tmp_path / "tokenizer.chatml.json"
    src.write_text(__import__("json").dumps(tokenizer_json), encoding="utf-8")

    plan = write_tokenizer_aliases(
        src,
        dst,
        required_tokens=["<|im_start|>"],
        reserved_patterns=["unused_"],
    )
    payload = __import__("json").loads(dst.read_text(encoding="utf-8"))

    assert plan.requires_fallback is False
    assert len(payload["model"]["vocab"]) == 3
    assert payload["model"]["vocab"]["<|im_start|>"] == 2
    assert "unused_1" not in payload["model"]["vocab"]