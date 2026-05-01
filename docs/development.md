# 开发指南

## 环境

```powershell
uv sync                              # 基础
uv sync --extra dev --extra hf       # 开发 + HF
# CUDA 上需要时：uv sync --extra flash
```

## 测试

```powershell
uv run pytest                                         # 默认（不含 slow / network / cuda / ddp）
uv run pytest tests/test_gpt2_model.py -k generate    # 单文件 / 关键字
uv run pytest -m "slow and network"                   # 触发 HF 对齐测试
```

`pytest` markers：
- `slow`：耗时较长
- `network`：需要联网（HF 模型下载等）
- `cuda`：仅在有 GPU 时运行
- `ddp`：需要 torchrun 多进程

## 训练 / 生成（最小闭环）

```powershell
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml
uv run python -m scripts.generate `
    --checkpoint runs/tiny_shakespeare/ckpt.pt `
    --tokenizer data/cache/tiny_shakespeare/tokenizer.json `
    --prompt "ROMEO:" --max-new-tokens 200
```

## DDP 调试

```powershell
uv run torchrun --nproc_per_node=2 -m train.pretrain `
    --config configs/train/pretrain_tiny.yaml distributed.backend=ddp
```

如果只有单卡或 CPU，也可以用 `--nproc_per_node=2` 跑 gloo backend 调通流程。

## 新增模块流程

1. 在 `core/<module_dir>/` 下加入新文件，并在该目录 `__init__.py` 暴露主类。
2. 在 `tests/` 下加对应 `test_*.py`，至少覆盖：shape、数值正确性、梯度可反传、边界条件。
3. 在 `docs/` 下按 [`_template_module_report.ipynb`](_template_module_report.ipynb) 编写模块交互式文档。
   - 所有文档均使用 Jupyter Notebook (.ipynb) 格式。
   - **源码导航**: 在开头添加对源码的软链接（例如 `> 源码对应：[core/module/xxx.py](../../core/module/xxx.py)`）。
   - **交互式代码块**: 编写 Python Cell 进行基础的数据维度展示与模型打印。
   - **内容结构**: 必须包含核心原理（使用 LaTeX 公式）、实现解析、与前代模型的差异对比，以及参考资源。
4. 在 `configs/model/*.yaml` 暴露开关，必要时加新尺寸配置。

## 提交前检查

- [ ] `uv run pytest` 通过（不含 slow / network 也算通过）
- [ ] 新增 / 修改的模块有对应测试
- [ ] 新增 / 修改的模块有文档（或更新已有文档）
- [ ] 文档的相对链接在 VS Code 中可点击打开
- [ ] `git status` 不包含调试残留 / 大文件 / 数据集

## 仓库提交

第一阶段建议流程（待用户授权后再 push）：

```powershell
git init
git add .
git commit -m "V0: GPT-2 baseline + module skeleton + docs"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```
