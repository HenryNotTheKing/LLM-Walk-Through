# LLM Walk-Through

> 从原理到实现，亲手解剖**现代大语言模型的演进轨迹**——以 GPT-2 为起点，
> 一处一处地"换积木"，看清每一步设计取舍背后的真问题。

[模块文档](docs/README.md) · [路线图](docs/00_roadmap.md) · [开发约定](docs/development.md)

---

## 这是个什么项目？

nanoGPT 与 MiniMind 解决了"跑通一个模型"的问题。本项目旨在回答**"为什么要这么改"。**

GPT-2 之后的现代 LLM 不是一次跳跃出来的，而是一连串**有动机、有取舍**的改造：
位置编码从绝对换到相对、再到旋转式、再到长度外推；
归一化、激活、注意力结构都被重新设计；
KV 经济学被反复重写；稠密 FFN 被替换为稀疏专家；
而**后训练侧**——从 SFT 到偏好对齐再到推理强化——每一代都在重塑模型的"行为方式"。

**LLM Walk-Through 的目标，是把这些"为什么"讲清楚，并让读者亲手跑出每一处改动的代价与收益。**
GPT-2 是这条 walk-through 的起点，而非终点；项目会沿着现代 LLM 实际发生的演进逐项实现，
并不预先把自己绑定到某一条具体的"哪个模型 → 哪个模型"路径上。

为此，本项目把模型解剖成 **6 大类可独立替换的模块**（开题报告 2.1）+ **3 条后训练路线**（开题报告 2.2），
每一处架构与训练范式的演进都配一篇专门的技术底稿，
回答："改了什么 / 为什么改 / 数学形式是什么 / 工程上什么时候真正见效 / 切换前后的代码差异在哪"。

## 它和 nanoGPT / MiniMind 有什么不同？

| 维度 | nanoGPT | MiniMind | **LLM Walk-Through** |
| --- | --- | --- | --- |
| 主要目标 | 极简跑通 GPT-2 | 26M 全链路（含 SFT / DPO） | 理清演进路径与设计动机 |
| 代码组织 | 单文件 | 单一架构 | 按模块解耦，可换积木 |
| 文档 | 注释 | README + 博客 | 每个模块一篇技术底稿 + 论文/博客/代码三类索引 |
| 架构覆盖 | GPT-2 | GPT-2 派生 | GPT-2 起步 → 现代主流架构创新逐项替换 |
| 后训练 | — | SFT + DPO | SFT → 偏好对齐 → 推理强化 |

并不是说前两者不好——它们各自的定位都很清楚；只是当你想问"现代 LLM 为什么不再长得像 GPT-2、各处具体改动到底解决了什么问题"时，本项目希望比它们更称手。

## 模块全景

每个模块在 V0 阶段先实现 GPT-2 同款的"经典版"，再沿着现代 LLM 的实际演进引入替换实现。
所有替换之间通过同一套 `core/` 接口可互换，便于做消融对照。

| 模块大类 | V0：GPT-2 经典版 | 后续替换计划 | 关键问题 |
| --- | --- | --- | --- |
| 分词器 | BPE / Byte-BPE / WordPiece / Unigram | （Unigram 即现代主流；后续按需扩展） | 词表对中英文 / 代码 / 数学切割效率的影响 |
| 位置编码 | Learned absolute PE | Sinusoidal → ALiBi → **RoPE** → NTK / **YaRN**（长上下文外推） | 长度外推、训练-推理长度不一致 |
| 归一化 | LayerNorm + Pre-LN | **RMSNorm** | 深层稳定性 / 计算开销 |
| 注意力 | Causal MHA（SDPA / eager） | MQA → **GQA** → **MLA** + FlashAttention-2 | KV 显存 / 推理吞吐 |
| 前馈 | GELU MLP（4× hidden） | **SwiGLU** → 稀疏 **MoE** | 收敛速度 / 容量利用 / 路由稳定性 |
| KV cache | 暂未启用 | Naive → **PagedAttention** → **StreamingLLM** | 长生成 OOM |

具体进度（已实现 / 待办）见 [docs/00_roadmap.md](docs/00_roadmap.md)。

## 后训练全景

| 阶段 | 方法 | 关键数据 | 评测 |
| --- | --- | --- | --- |
| 预训练 | PyTorch + DeepSpeed ZeRO | FineWeb-Edu / OpenCSG / The Stack v2 / MathPile | MMLU / HellaSwag / WikiText PPL |
| SFT | Full FT / LoRA / QLoRA | OpenHermes-2.5 / MetaMathQA / Magicoder-OSS | MT-Bench / IFEval |
| 偏好对齐 | **SimPO** | UltraFeedback-Binarized | AlpacaEval 2.0 |
| 推理强化 | **GRPO** | 在线采样 + 编译/答案校验 | GSM8K / MATH / HumanEval / MBPP |

评测统一接入 HuggingFace [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。

## 设计原则

- **模块化优先**：从 day 1 起就拆进 `core/` 子目录，便于"换积木"做消融。
- **文档即入口**：每个模块都有专门 `.md`，含问题背景、数学形式、工程取舍、源码索引、References。
- **跨设备友好**：默认实现走 CPU/MPS/CUDA 三平台，FlashAttention 是可选优化而非硬依赖。
- **测试与代码同步**：新增模块必须配单元测试，没有测试的 PR 不视为完成。
- **尊重规模效应**：玩具尺度跑通流程，关键效果在合适规模（≥ 60M）才下结论；不强求所有结论在玩具上立刻兑现。

## 快速上手

```powershell
# 安装
uv sync                          # 基础依赖（CPU/MPS 即可）
uv sync --extra dev --extra hf   # 加上测试与 HF 对齐

# 端到端最小闭环（CPU 几分钟）
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml

# 文本生成
uv run python -m scripts.generate `
    --checkpoint runs/tiny_shakespeare/ckpt.pt `
    --tokenizer data/cache/tiny_shakespeare/tokenizer.json `
    --prompt "ROMEO:" --max-new-tokens 200

# 切换分词器（bpe / byte_bpe / wordpiece / unigram / gpt2）
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml data.tokenizer.kind=byte_bpe

# 多卡 / 多进程
uv run torchrun --nproc_per_node=2 -m train.pretrain `
    --config configs/train/pretrain_tiny.yaml distributed.backend=ddp

# 测试
uv run pytest                            # 默认（不含 slow / network）
uv run pytest -m "slow and network"      # 触发 HF 对齐测试
```

## 目录结构

```
core/                # 可替换的积木模块（按开题报告 2.1 节六大类组织）
├── tokenizer/       # BPE / Byte-BPE / WordPiece / Unigram + GPT-2 官方兼容
├── position/        # learned PE（首版）；后续 sinusoidal / RoPE / NTK / YaRN
├── norm/            # LayerNorm（首版）；后续 RMSNorm
├── attention/       # Causal MHA（SDPA + eager）；后续 MQA / GQA / MLA / Flash
├── ffn/             # GELU MLP（首版）；后续 SwiGLU / 稀疏 MoE
├── kv_cache/        # 占位；后续 naive / Paged / Streaming
├── model/           # 组装入口；后续按演进追加新架构配置
└── utils/           # 配置、设备、分布式
configs/             # YAML + OmegaConf 配置（model / train 分目录）
data/                # 数据准备脚本（语料缓存放 data/cache/）
train/               # 训练入口（首版仅 pretrain.py；后续 sft / 偏好对齐 / 推理强化）
scripts/             # 工具脚本（generate.py / 后续 convert_hf 等）
tests/               # pytest 单测
docs/                # 模块文档（父-子双向软链接，按 docs/README.md 规范组织）
experiments/         # 消融脚本与解释（V1 起填充）
```

## 文档

- [docs/README.md](docs/README.md) — 文档规范（层级、双向软链接、Code Map、References 三类资源）
- [docs/00_roadmap.md](docs/00_roadmap.md) — 路线图：V0（GPT-2 已落地） → V1 / V2（架构演进） → V3（完整预训练 + 后训练）
- [docs/01_tokenizer/](docs/01_tokenizer/README.md) — 分词器全景对照与 4 种实现
- [docs/02_gpt2_baseline.md](docs/02_gpt2_baseline.md) — GPT-2 baseline 总览
- [docs/03_attention_mha.md](docs/03_attention_mha.md) — Causal MHA
- [docs/04_pretrain_minimal.md](docs/04_pretrain_minimal.md) — 最小预训练闭环
- [docs/development.md](docs/development.md) — 开发约定（环境 / 测试 / 新增模块流程 / 提交清单）

## License

MIT.

---

> *"The best way to understand an opaque system is to build it from scratch and watch it fail until it doesn't."*
