"""HuggingFace GPT-2 logits 对齐测试。

默认标记为 ``slow + network``，CI 默认跳过；本地需要时通过
    ``pytest -m "slow and network" tests/test_hf_gpt2_parity.py``
触发。
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.slow
@pytest.mark.network
def test_gpt2_small_logits_match_hf():
    from core.model import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained_hf("gpt2")
    model.eval()

    from transformers import GPT2LMHeadModel as HFGPT2
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    hf = HFGPT2.from_pretrained("gpt2").eval()

    text = "Hello, my name is"
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)

    with torch.no_grad():
        ours, _ = model(ids, ids)  # 用 ids 当 targets 触发返回完整 logits
        hf_out = hf(ids).logits

    # bfloat16/不同 kernel 路径，容差给宽
    max_diff = (ours - hf_out).abs().max().item()
    assert max_diff < 1e-3, f"max logits diff = {max_diff}"
