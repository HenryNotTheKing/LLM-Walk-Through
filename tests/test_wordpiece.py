"""WordPiece 单测。"""

from __future__ import annotations

from core.tokenizer.wordpiece import WordPieceTokenizer

SAMPLE = (
    "the quick brown fox jumps over the lazy dog\n"
    "she sells sea shells by the sea shore\n"
    "playing played plays player\n"
    "running ran runs runner\n"
) * 30


def test_train_basic():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    # 特殊 token 都在
    for s in WordPieceTokenizer.SPECIAL_TOKENS:
        assert s in tok.token_to_id
    assert tok.vocab_size <= 200


def test_continuation_prefix_present():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    # 至少应学到一些 ## 续接子词（来自非首字符）
    assert any(t.startswith("##") for t in tok.token_to_id)


def test_encode_known_word_no_unk():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    ids = tok.encode("the")
    assert tok.unk_token_id not in ids


def test_encode_unknown_char_falls_back_to_unk():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    # 训练语料里没有的字符，应触发 UNK
    ids = tok.encode("ξ")  # 训练里没有希腊字母
    assert tok.unk_token_id in ids


def test_eos_uses_sep():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    ids = tok.encode("the", add_eos=True)
    assert ids[-1] == tok.token_to_id["[SEP]"]


def test_decode_strips_continuation_prefix():
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    ids = tok.encode("playing")
    decoded = tok.decode(ids)
    # 解码出来应当是连贯的词，不包含 ##
    assert "##" not in decoded
    assert "playing" in decoded.replace(" ", "") or decoded.replace(" ", "") == "playing"


def test_save_and_load(tmp_path):
    tok = WordPieceTokenizer.train(SAMPLE, vocab_size=200)
    p = tmp_path / "wp.json"
    tok.save(p)
    loaded = WordPieceTokenizer.load(p)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the quick fox") == tok.encode("the quick fox")
