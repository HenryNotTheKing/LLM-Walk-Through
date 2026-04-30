"""Unigram LM 单测。"""

from __future__ import annotations

from core.tokenizer.unigram import UnigramTokenizer

SAMPLE = (
    "the quick brown fox jumps over the lazy dog\n"
    "she sells sea shells by the sea shore\n"
    "playing played plays player\n"
    "running ran runs runner\n"
) * 30


def test_train_basic():
    tok = UnigramTokenizer.train(SAMPLE, vocab_size=300, seed_size=2000, n_iter=3)
    for s in UnigramTokenizer.SPECIAL_TOKENS:
        assert s in tok.token_to_id
    assert len(tok.log_probs) > len(tok.SPECIAL_TOKENS)


def test_viterbi_returns_pieces_that_concat_back():
    from core.tokenizer.unigram import _normalize

    tok = UnigramTokenizer.train(SAMPLE, vocab_size=300, seed_size=2000, n_iter=3)
    text = "the quick brown fox"
    pieces = tok._viterbi(_normalize(text))
    assert "".join(pieces) == _normalize(text)


def test_encode_decode_roundtrip():
    tok = UnigramTokenizer.train(SAMPLE, vocab_size=300, seed_size=2000, n_iter=3)
    text = "the quick brown fox"
    ids = tok.encode(text)
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert decoded == text


def test_eos_appended():
    tok = UnigramTokenizer.train(SAMPLE, vocab_size=300, seed_size=2000, n_iter=3)
    ids = tok.encode("hello", add_eos=True)
    assert ids[-1] == tok.eos_token_id


def test_save_and_load(tmp_path):
    tok = UnigramTokenizer.train(SAMPLE, vocab_size=300, seed_size=2000, n_iter=3)
    p = tmp_path / "uni.json"
    tok.save(p)
    loaded = UnigramTokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")
