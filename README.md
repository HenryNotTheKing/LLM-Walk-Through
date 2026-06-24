# LLM Walk-Through

**从 GPT-2 到现代代码大模型：亲手拆解每一处架构改动，并把它训出来。**

本项目不只是教学实现——它同时是一处可运行的实验场：

- **教学侧**：把 Decoder-only Transformer 拆成可独立替换的模块，每个模块配一篇交互式NoteBook。
- **工程侧**：包含一个完整的现代代码模型 **Walkie-Code**（0.5B / 1B）的预训练、SFT、RL（GRPO / DAPO）与代码评测流水线。

> 目标不是“跑通 demo”，而是回答：**这一处改动为什么出现、数学上怎么写、工程上什么时候才真正见效、切换前后代码差异在哪。**

---

## 项目现在能做什么

| 能力 | 状态 | 入口 |
|------|------|------|
| GPT-2 教学 baseline（124M / tiny） | ✅ | [`train/pretrain.py`](train/pretrain.py) |
| 可替换模块：Tokenizer / Position / Norm / Attention / FFN / Residual | ✅ | [`core/`](core/) |
| Walkie-Code 现代架构（RMSNorm + RoPE + GQA + QK-Norm + SwiGLU） | ✅ | [`core/model/walkie.py`](core/model/walkie.py) |
| Walkie-Code 0.5B / 1B 预训练（WSD + AdamW + Muon） | ✅ | [`train/walkie_pretrain.py`](train/walkie_pretrain.py) |
| 监督微调（SFT） | ✅ | [`train/walkie_sft.py`](train/walkie_sft.py) |
| 强化学习：GRPO / DAPO | ✅ | [`train/walkie_rl.py`](train/walkie_rl.py) |
| 代码评测：HumanEval / KodCode / 自定义 code bench | ✅ | [`posttrain/eval/`](posttrain/eval/) |
| HF 权重导出与 vLLM 推理 | ✅ | [`scripts/export_walkie_to_hf.py`](scripts/export_walkie_to_hf.py) |

---

## 目录结构

```
core/                 # 可替换的核心积木
├── attention/        # MHA / MQA / GQA / MLA / Sliding Window / Linear Attention
├── ffn/              # GELU MLP / SwiGLU / GEGLU / ReGLU / MoE
├── norm/             # LayerNorm / RMSNorm / ScaleNorm / DeepNorm / DyT / DeRF
├── position/         # Sinusoidal / RoPE / YaRN / ALiBi
├── residual/         # 标准残差 / AttnRes / mHC
├── tokenizer/        # BPE / Byte-BPE / WordPiece / Unigram
├── model/            # GPT-2 / WalkieForCausalLM 组装入口
├── kv_cache/         # KV Cache 相关实现预留
└── utils/            # 配置、checkpoint、WSD 调度、Muon 优化器
train/                # 训练入口
├── pretrain.py       # GPT-2 教学预训练
├── gpt2_pretrain.py  # GPT-2 完整预训练
├── walkie_pretrain.py
├── walkie_sft.py
└── walkie_rl.py
posttrain/            # 后训练与评测
├── data/             # SFT / RL 数据集与 chat template
├── eval/             # 代码评测（HumanEval / KodCode / vLLM / Ray sandbox）
├── rl/               # GRPO / DAPO 算法实现
├── rollout/          # torch / vLLM 双 rollout 引擎
├── rewards/          # 代码执行奖励
├── sandbox/          # Jupyter sandbox 客户端
└── utils/            # HF 导出、schedule 辅助
configs/              # 模型与训练配置
├── model/            # gpt2_tiny / gpt2_124m / walkie_tiny / walkie_code_0.5b / walkie_code_1b
├── train/            # pretrain / sft / rl (grpo / dapo) 配置
└── eval/             # 评测配置
docs/                 # 技术底稿与实验 Notebook
├── modules/          # 按模块讲解（tokenizer / position / norm / attention / ffn / residual / model / utils）
├── experiments/      # 预训练实验（最小闭环、轻量预训练、Walkie 数据、归一化对比）
├── roadmap.ipynb     # 项目路线图与进度
├── walkie_pretrain_playbook.md
├── walkie_code_learning_guide.md
└── folder_guide.md
scripts/              # 辅助脚本（generate / encode / export / evaluate / install_torch）
tests/                # 单元测试与回归测试
runs/                 # 训练输出（checkpoint）
data/                 # 数据缓存与 tokenizer
```

---

## 快速开始

### 1. 环境安装

```powershell
# 克隆仓库
git clone https://github.com/your-org/llm-walk-through
cd llm-walk-through

# 创建虚拟环境（推荐 Python 3.10）
uv venv --python 3.10

# 安装基础依赖
uv sync

# 训练 Walkie 需要额外依赖
uv sync --extra walkie

# CUDA 环境可安装 flash-attn（Linux / 非 Windows）
uv sync --extra flash
```

有 NVIDIA GPU 时，建议运行自动检测脚本安装匹配 CUDA 的 PyTorch：

```powershell
python scripts/install_torch.py --run
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 2. 最小闭环：5 分钟跑通 GPT-2 教学预训练

```powershell
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml

uv run python -m scripts.generate `
    --checkpoint runs/smoltalk_chinese_small/ckpt.pt `
    --tokenizer data/cache/smoltalk_chinese_small/tokenizer.json `
    --prompt "你好，请介绍一下你自己：" --max-new-tokens 200
```

### 3. Walkie-Code 冒烟训练（无需真实数据）

```powershell
uv run python -m train.walkie_pretrain --config configs/train/pretrain_walkie_tiny.yaml
```

### 4. 跑测试

```powershell
uv run pytest                     # 默认（跳过 slow / network / cuda / ddp）
uv run pytest -m "slow and network"  # HF 权重对齐测试
```

---

## 模块演进全景

| 模块 | 已实现 | 说明 |
|------|--------|------|
| **Tokenizer** | BPE / Byte-BPE / WordPiece / Unigram | [`docs/modules/tokenizer/`](docs/modules/tokenizer/) |
| **位置编码** | Sinusoidal / RoPE / YaRN / ALiBi | RoPE 为 Walkie 默认；YaRN/ALiBi 用于对比 |
| **归一化** | LayerNorm / RMSNorm / ScaleNorm / DeepNorm / DyT / DeRF | Walkie 使用 RMSNorm + QK-Norm |
| **注意力** | MHA / MQA / GQA / MLA / Sliding Window / Linear Attention | Walkie 使用 GQA + RoPE + QK-Norm，可选 flash/sdpa/eager |
| **前馈网络** | GELU MLP / SwiGLU / GEGLU / ReGLU / MoE base / DeepSeek MoE | Walkie 使用 SwiGLU |
| **残差拓扑** | 标准残差 / AttnRes / mHC | 用于深层稳定性对比 |
| **KV Cache** | 预留 | generate 当前全序列重算，KV Cache 为后续扩展 |

完整进度见 [`docs/roadmap.ipynb`](docs/roadmap.ipynb)。

---

## Walkie-Code 模型

Walkie-Code 是一个面向代码任务的现代 Decoder-only Transformer，主要设计选择：

- **Pre-norm + RMSNorm**：无 bias，仅可学习 scale
- **GQA + QK-Norm + RoPE**：共享全局 RoPE 表，24Q/8KV（1B）或 20Q/5KV（0.5B）
- **SwiGLU FFN**：替代 GELU MLP
- **Weight Tying**：`lm_head` 与 `tok_embeddings` 共享
- **两阶段 WSD 预训练**：`main` 阶段稳定学习率 + `anneal` 阶段高质量数据 + 平方根衰减
- **AdamW + Muon 双优化器**：Muon 处理二维权重矩阵，AdamW 处理 Embedding / Norm / lm_head
- **工程优化**：分块交叉熵、gradient checkpointing、`torch.compile`、DDP 无锁步梯度累积

| 变体 | 参数量 | 上下文 | 配置 |
|------|--------|--------|------|
| Walkie-Tiny | ~0.33M | 128 | [`configs/model/walkie_tiny.yaml`](configs/model/walkie_tiny.yaml) |
| Walkie-Code-0.5B | ~501M | 4096 | [`configs/model/walkie_code_0.5b.yaml`](configs/model/walkie_code_0.5b.yaml) |
| Walkie-Code-1B | ~964M | 2048 | [`configs/model/walkie_code_1b.yaml`](configs/model/walkie_code_1b.yaml) |

🤗 已发布检查点：[Henry665/Walkie-Code-0.5B](https://huggingface.co/Henry665/Walkie-Code-0.5B)

详细 playbook：[`docs/walkie_pretrain_playbook.md`](docs/walkie_pretrain_playbook.md)  
逐行源码讲读：[`docs/walkie_code_learning_guide.md`](docs/walkie_code_learning_guide.md)

### Walkie 预训练命令示例

```powershell
# 单卡 tiny 冒烟
uv run python -m train.walkie_pretrain --config configs/train/pretrain_walkie_tiny.yaml

# 多卡正式训练
uv run torchrun --nproc_per_node=2 -m train.walkie_pretrain `
    --config configs/train/pretrain_walkie.yaml

# 恢复训练
uv run torchrun --nproc_per_node=2 -m train.walkie_pretrain `
    --config configs/train/pretrain_walkie.yaml `
    --resume runs/walkie_code_0.5b

# 仅加载权重继续
uv run python -m train.walkie_pretrain --config configs/train/pretrain_walkie.yaml `
    --init-from path/to/ckpt.pt
```

### Walkie SFT / RL 命令示例

```powershell
# SFT
uv run python -m train.walkie_sft --config configs/train/sft_walkie.yaml

# GRPO
uv run python -m train.walkie_rl --config configs/train/rl_walkie_grpo.yaml

# DAPO
uv run python -m train.walkie_rl --config configs/train/rl_walkie_dapo.yaml
```

### 导出到 Hugging Face 并用 vLLM 推理

```powershell
uv run python -m scripts.export_walkie_to_hf `
    --checkpoint runs/walkie_code_0.5b/best.pt `
    --config configs/model/walkie_code_0.5b.yaml `
    --output-dir checkpoints/walkie-code-0.5b-hf
```

---

## 常用命令速查

```powershell
# 安装
uv sync                              # 基础依赖
uv sync --extra dev                  # + 测试与 ruff
uv sync --extra walkie               # + Walkie 训练（tensorboard / swanlab / safetensors / rich）
uv sync --extra posttrain            # + 后训练评测（Linux/CUDA：vLLM / Ray / transformers）
uv sync --extra flash                # + flash-attn（Linux/CUDA）

# GPT-2 教学
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml

# Walkie 预训练
uv run python -m train.walkie_pretrain --config configs/train/pretrain_walkie_tiny.yaml

# 文本生成
uv run python -m scripts.generate `
    --checkpoint runs/smoltalk_chinese_small/ckpt.pt `
    --tokenizer data/cache/smoltalk_chinese_small/tokenizer.json `
    --prompt "你好" --max-new-tokens 200

# 测试
uv run pytest
uv run pytest tests/test_walkie_training_smoke.py -q
uv run pytest -m "slow and network"

# 多卡 DDP（GPT-2）
uv run torchrun --nproc_per_node=2 -m train.pretrain `
    --config configs/train/pretrain_tiny.yaml distributed.backend=ddp

# 代码评测
uv run python -m scripts.evaluate_humaneval --config configs/eval/humaneval_vllm.yaml
uv run python -m scripts.evaluate_code_bench --config configs/eval/walkie_code_bench.yaml
```

---

## 与同类项目的区别

| | nanoGPT | MiniMind | **本项目** |
|---|---------|----------|-----------|
| 核心目标 | 极简跑通 GPT-2 | 小参数全链路 SFT/DPO | 拆解每处架构改动的动机，并训出 Walkie-Code |
| 代码组织 | 单文件 | 单一架构 | 模块解耦，可换积木做消融 |
| 文档 | 注释 | README + 博客 | 每模块一篇交互式 Notebook 底稿 |
| 架构覆盖 | GPT-2 | GPT-2 派生 | GPT-2 → GQA / MLA / MoE / RoPE / RMSNorm 逐项替换 |
| 训练阶段 | 预训练 | 预训练 + SFT/DPO | 预训练 + SFT + RL（GRPO/DAPO）+ 代码评测 |

---

## 学习路径建议

**完全新手**：
1. 跑通 [`configs/train/pretrain_tiny.yaml`](configs/train/pretrain_tiny.yaml)
2. 读 [`docs/modules/model/02_gpt2_baseline.ipynb`](docs/modules/model/02_gpt2_baseline.ipynb)
3. 依次读 [`docs/modules/attention/03_attention_mha.ipynb`](docs/modules/attention/03_attention_mha.ipynb)、[`docs/modules/norm/01_rmsnorm.ipynb`](docs/modules/norm/01_rmsnorm.ipynb)、[`docs/modules/position/01_rope.ipynb`](docs/modules/position/01_rope.ipynb)

**想训 Walkie**：
1. 读 [`docs/walkie_code_learning_guide.md`](docs/walkie_code_learning_guide.md)
2. 读 [`docs/walkie_pretrain_playbook.md`](docs/walkie_pretrain_playbook.md)
3. 准备数据 → 跑 [`scripts/encode_walkie.py`](scripts/encode_walkie.py) → 改 [`configs/train/pretrain_walkie.yaml`](configs/train/pretrain_walkie.yaml) → 训练

**想加模块**：
1. 在 `core/<module>/` 下实现并暴露主类
2. 在 `tests/` 加测试
3. 在 `docs/modules/` 按模板写 Notebook
4. 更新 `configs/model/*.yaml` 开关

详见 [`docs/development.md`](docs/development.md)。

---

## License

MIT.
