"""教学版 BPE 单测：训练、roundtrip、保存/加载。"""

from __future__ import annotations

from core.tokenizer.bpe import BPETokenizer

SAMPLE = (
    "the quick brown fox jumps over the lazy dog\n"
    "she sells sea shells by the sea shore\n"
    "to be or not to be that is the question\n"
) * 20


def test_train_and_vocab_size_grows():
    tok = BPETokenizer.train(SAMPLE, vocab_size=200)
    # 至少包含特殊 token
    assert tok.eos_token_id is not None
    assert tok.unk_token_id is not None
    # 学到了一些合并
    assert len(tok.merges) > 0
    # 词表不超过目标
    assert tok.vocab_size <= 200


def test_encode_decode_roundtrip():
    tok = BPETokenizer.train(SAMPLE, vocab_size=300)
    text = "the quick brown fox"
    ids = tok.encode(text)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
    decoded = tok.decode(ids)
    # BPE 在词与词之间会用空格保留语义；这里只要求字母层面恢复
    assert decoded.replace(" ", "") == text.replace(" ", "")


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(SAMPLE, vocab_size=200)
    p = tmp_path / "tok.json"
    tok.save(p)
    loaded = BPETokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")


def test_eos_token():
    tok = BPETokenizer.train(SAMPLE, vocab_size=200)
    ids = tok.encode("hello", add_eos=True)
    assert ids[-1] == tok.eos_token_id
