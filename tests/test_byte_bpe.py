"""Byte-level BPE 单测：训练、字节往返、保存/加载、UTF-8 鲁棒性。"""

from __future__ import annotations

from core.tokenizer.byte_bpe import ByteBPETokenizer, bytes_to_unicode

SAMPLE = (
    "the quick brown fox jumps over the lazy dog\n"
    "she sells sea shells by the sea shore\n"
    "to be or not to be that is the question\n"
    "你好，世界。GPT-2 uses byte-level BPE!\n"
) * 30


def test_bytes_to_unicode_is_a_bijection():
    b2u = bytes_to_unicode()
    assert len(b2u) == 256
    assert len(set(b2u.values())) == 256


def test_train_and_special_tokens():
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=400)
    assert tok.eos_token_id == tok.token_to_id["<|endoftext|>"]
    # 256 字节 + 1 个特殊 token 一定都在
    assert tok.vocab_size >= 257
    assert tok.vocab_size <= 400
    assert len(tok.merges) > 0


def test_ascii_roundtrip():
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=500)
    text = "the quick brown fox"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_unicode_roundtrip_no_oov():
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=500)
    # 训练时见过的 utf-8 字节都能 round-trip
    text = "你好，世界。"
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_unseen_unicode_still_decodes():
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=500)
    # 训练数据没见过的 emoji，但字节级 BPE 仍然不应当 OOV
    text = "🚀✨"
    ids = tok.encode(text)
    assert len(ids) > 0
    # 解码不抛异常即可（此处不强制 == text，因 emoji 字节可能没合并机会）
    assert tok.decode(ids) == text


def test_eos_appended():
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=400)
    ids = tok.encode("hello", add_eos=True)
    assert ids[-1] == tok.eos_token_id


def test_save_and_load(tmp_path):
    tok = ByteBPETokenizer.train(SAMPLE, vocab_size=400)
    p = tmp_path / "byte_bpe.json"
    tok.save(p)
    loaded = ByteBPETokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")
