"""BaseTokenizer + factory + load_tokenizer 单测。"""

from __future__ import annotations

import pytest

from core.tokenizer import (
    BPETokenizer,
    ByteBPETokenizer,
    UnigramTokenizer,
    WordPieceTokenizer,
    build_tokenizer,
    load_tokenizer,
)
from core.tokenizer.base import BaseTokenizer

SAMPLE = (
    "the quick brown fox jumps over the lazy dog\n"
    "she sells sea shells by the sea shore\n"
    "to be or not to be that is the question\n"
) * 30


@pytest.mark.parametrize("kind,cls", [
    ("bpe", BPETokenizer),
    ("byte_bpe", ByteBPETokenizer),
    ("wordpiece", WordPieceTokenizer),
    ("unigram", UnigramTokenizer),
])
def test_build_tokenizer_returns_correct_class(kind, cls):
    tok = build_tokenizer(kind)
    assert isinstance(tok, cls)
    assert isinstance(tok, BaseTokenizer)


def test_build_tokenizer_unknown_kind():
    with pytest.raises(ValueError):
        build_tokenizer("nope")


@pytest.mark.parametrize("cls", [BPETokenizer, ByteBPETokenizer, WordPieceTokenizer, UnigramTokenizer])
def test_save_writes_kind_field(cls, tmp_path):
    tok = cls.train(SAMPLE, vocab_size=200)
    p = tmp_path / "tok.json"
    tok.save(p)
    import json
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["kind"] == cls.KIND


@pytest.mark.parametrize("cls", [BPETokenizer, ByteBPETokenizer, WordPieceTokenizer, UnigramTokenizer])
def test_load_tokenizer_dispatches_by_kind(cls, tmp_path):
    tok = cls.train(SAMPLE, vocab_size=200)
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = load_tokenizer(p)
    assert isinstance(loaded, cls)
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")
