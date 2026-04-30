"""单步训练测试：验证 forward/backward/optimizer step 与 checkpoint 读写均能正常完成。"""

from __future__ import annotations

import torch

from core.model import GPT2Config, GPT2LMHeadModel


def test_single_optimizer_step_decreases_loss_on_overfit():
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=32, n_layer=2, n_head=2, n_embd=32,
                     block_size=16, dropout=0.0)
    model = GPT2LMHeadModel(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randint(0, 32, (4, 8))
    y = torch.randint(0, 32, (4, 8))

    _, loss0 = model(x, y)
    for _ in range(20):
        optim.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optim.step()
    assert loss.item() < loss0.item()


def test_checkpoint_save_load(tmp_path):
    cfg = GPT2Config(vocab_size=32, n_layer=2, n_head=2, n_embd=32, block_size=16)
    m1 = GPT2LMHeadModel(cfg)
    p = tmp_path / "ckpt.pt"
    torch.save({"model": m1.state_dict(), "model_cfg": cfg.__dict__}, p)

    payload = torch.load(p, map_location="cpu", weights_only=False)
    m2 = GPT2LMHeadModel(GPT2Config(**payload["model_cfg"]))
    m2.load_state_dict(payload["model"])

    x = torch.randint(0, 32, (1, 4))
    m1.eval(); m2.eval()
    with torch.no_grad():
        l1, _ = m1(x)
        l2, _ = m2(x)
    assert torch.allclose(l1, l2)
