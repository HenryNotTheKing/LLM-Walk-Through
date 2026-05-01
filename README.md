# LLM Walk-Through

**从原理到实现，亲手解剖现代大语言模型的演进轨迹。**

以 GPT-2 为起点，一处一处地"换积木"，看清每一步设计取舍背后的真问题——
不只是"怎么跑通"，而是**"为什么要这么改"**。

---

## 项目是什么

nanoGPT 解决了"跑通一个模型"的问题。本项目回答的是：**GPT-2 之后，每一处架构改动究竟解决了什么**。

现代 LLM 不是一次跳跃出来的，而是一连串**有动机、有取舍**的改造：
位置编码从绝对换到旋转、归一化和激活函数被重新设计、KV cache 反复重写、稠密 FFN 被稀疏专家替换；
后训练侧从 SFT 到偏好对齐再到推理强化，每一代都在重塑模型的行为。

本项目把模型解剖成 **6 大类可独立替换的模块**，每一处改动配一篇技术底稿，回答：
> 改了什么 / 为什么改 / 数学形式是什么 / 工程上什么时候真正见效 / 切换前后代码差异在哪

---

## 从哪里开始

### 第一步：5 分钟跑通最小闭环

```powershell
# 1. 克隆 & 安装
git clone https://github.com/your-org/llm-walk-through
cd llm-walk-through
uv sync                          # 基础依赖（CPU / MPS 即可，无需 GPU）

# 2. 端到端预训练（tiny shakespeare，CPU 下约 3 分钟）
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml

# 3. 用训练好的模型生成文本
uv run python -m scripts.generate `
    --checkpoint runs/tiny_shakespeare/ckpt.pt `
    --tokenizer data/cache/tiny_shakespeare/tokenizer.json `
    --prompt "ROMEO:" --max-new-tokens 200
```

跑完这三步，你就有了一个在莎士比亚文本上预训练的微型 GPT-2。

---

### 第二步：读懂代码结构

```
core/                # 所有可替换的积木模块
├── tokenizer/       # BPE / Byte-BPE / WordPiece / Unigram / GPT-2 官方兼容
├── position/        # Learned PE → 后续 RoPE / NTK / YaRN
├── norm/            # LayerNorm → 后续 RMSNorm
├── attention/       # Causal MHA → 后续 GQA / MLA / FlashAttention
├── ffn/             # GELU MLP → 后续 SwiGLU / 稀疏 MoE
├── kv_cache/        # 占位 → 后续 naive / Paged / Streaming
└── model/           # 模型组装入口
configs/             # YAML 配置（model/ 与 train/ 分目录）
train/               # 训练脚本（pretrain.py，后续 sft / 偏好对齐 / 推理强化）
scripts/             # 工具脚本（generate.py 等）
docs/                # 每个模块一篇技术底稿
```

**关键入口文件：**

| 文件 | 作用 |
|------|------|
| [train/pretrain.py](train/pretrain.py) | 预训练主循环 |
| [core/model/gpt2.py](core/model/gpt2.py) | GPT-2 模型组装 |
| [configs/train/pretrain_tiny.yaml](configs/train/pretrain_tiny.yaml) | 最小训练配置（从这里调参） |
| [docs/00_roadmap.ipynb](docs/00_roadmap.ipynb) | 项目路线图，了解已落地与待办 |

---

### 第三步：选一个感兴趣的方向深入

**想理解分词器？** → [docs/01_tokenizer/README.ipynb](docs/01_tokenizer/README.ipynb)，4 种实现逐一对照。

**想理解注意力机制？** → [docs/03_attention_mha.ipynb](docs/03_attention_mha.ipynb)，从 MHA 到 GQA/MLA 的演进。

**想跑通完整预训练？** → [docs/04_pretrain_minimal.ipynb](docs/04_pretrain_minimal.ipynb)，含梯度累积、混合精度、DDP 说明。

**想看整体演进计划？** → [docs/00_roadmap.ipynb](docs/00_roadmap.ipynb)，V0 已落地，V1/V2/V3 路线。

---

## 常用命令速查

```powershell
# 安装（不同选项）
uv sync                               # 基础依赖
uv sync --extra dev --extra hf        # 加上测试与 HF 权重对齐
uv sync --extra monitor               # 加上 SwanLab 实验跟踪

# 切换分词器（bpe / byte_bpe / wordpiece / unigram / gpt2）
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml `
    data.tokenizer.kind=byte_bpe

# 开启 SwanLab 实验跟踪（需先 uv sync --extra monitor）
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml `
    train.swanlab=true train.swanlab_project=my-project

# 多卡训练（DDP）
uv run torchrun --nproc_per_node=2 -m train.pretrain `
    --config configs/train/pretrain_tiny.yaml distributed.backend=ddp

# 运行测试
uv run pytest                         # 默认（跳过 slow / network 标记）
uv run pytest -m "slow and network"   # 触发 HF 权重对齐测试
```

---

## 模块演进全景

| 模块 | 当前（V0 GPT-2） | 后续替换计划 |
|------|-----------------|-------------|
| 分词器 | BPE / Byte-BPE / WordPiece / Unigram | 按需扩展 |
| 位置编码 | Learned absolute PE | Sinusoidal → ALiBi → **RoPE** → YaRN |
| 归一化 | LayerNorm (Pre-LN) | **RMSNorm** |
| 注意力 | Causal MHA (SDPA) | MQA → **GQA** → **MLA** + FlashAttention-2 |
| 前馈 | GELU MLP (4× hidden) | **SwiGLU** → 稀疏 **MoE** |
| KV Cache | 未启用 | Naive → **PagedAttention** → StreamingLLM |

完整进度见 [docs/00_roadmap.ipynb](docs/00_roadmap.ipynb)。

---

## 与同类项目的区别

| | nanoGPT | MiniMind | **本项目** |
|---|---------|----------|-----------|
| 核心目标 | 极简跑通 GPT-2 | 26M 全链路 SFT/DPO | 理清每处架构改动的动机与代价 |
| 代码组织 | 单文件 | 单一架构 | 模块解耦，可换积木做消融 |
| 文档 | 注释 | README + 博客 | 每模块一篇技术底稿 |
| 架构覆盖 | GPT-2 | GPT-2 派生 | GPT-2 起步 → 现代主流逐项替换 |

---

## License

MIT.
