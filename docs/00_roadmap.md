# 路线图：现代 LLM 演进 walk-through

本项目以 GPT-2 为起点，沿着现代大语言模型在**架构**与**后训练**两个维度上的实际演进，
逐项替换 / 扩展实现。每一处替换都有独立的模块文档与消融实验。
**GPT-2 是 walk-through 的起点而非终点**——后续走到哪一步，由"哪一处改动值得讲清楚"决定，而不预先绑定到某条具体的"模型 → 模型"路径上。

## V0 · 原始 GPT-2（已完成）

| 模块 | 实现 | 文档 |
| --- | --- | --- |
| 分词器 | BPE / Byte-BPE / WordPiece / Unigram + GPT-2 官方 BPE 兼容 | [01_tokenizer/](01_tokenizer/README.md) |
| 位置编码 | Learned absolute PE | （V1 起独立成文） |
| 归一化 | LayerNorm + Pre-LN | （V1 起独立成文） |
| 注意力 | Causal MHA（SDPA / eager） | [03_attention_mha.md](03_attention_mha.md) |
| 前馈 | GELU MLP（4× hidden） | （V1 起独立成文） |
| KV cache | 未实现，generate 走全量重算 | （V2 加入） |

汇总入口：[02_gpt2_baseline.md](02_gpt2_baseline.md)。

## V1 · 现代主干替换（计划）

把 GPT-2 baseline 中的 6 大类模块逐个换成现代 LLM 通用的对应组件，每替换一项都做 A/B 消融。

| 模块 | 替换为 | 期望收益 |
| --- | --- | --- |
| 位置编码 | RoPE | 长度外推 / 无需绝对长度上限 |
| 归一化 | RMSNorm | 更便宜、稳定深层训练 |
| 前馈 | SwiGLU | 收敛速度 / 容量 |
| 注意力 | GQA（+ 可选 Flash） | KV 显存 / 推理速度 |
| KV cache | naive → Paged / Streaming | 长生成不 OOM |

每项替换附带：模块文档（按 [模板](_template_module_report.md)）、单元测试、消融脚本。

## V2 · 进一步的架构创新（计划）

继续把 baseline 推到当代前沿模型实际采用的架构改动：

| 方向 | 拟引入实现 |
| --- | --- |
| 长上下文外推 | NTK-aware / **YaRN** RoPE 缩放 |
| KV 经济学 | **Multi-head Latent Attention（MLA）** |
| 稀疏专家 | 顶 K 路由 + auxiliary loss / 细粒度 + 共享专家 |
| 训练稳定性 | Z-loss、router balance loss、深度缩放残差等 |

## V3 · 完整预训练 + 后训练（计划）

| 阶段 | 方法 | 数据 / 评测 |
| --- | --- | --- |
| 预训练 | DeepSpeed ZeRO 完整训练 1B 量级 | FineWeb-Edu / OpenCSG / The Stack v2 / MathPile |
| SFT | Full FT / LoRA / QLoRA | OpenHermes-2.5 / MetaMathQA / Magicoder-OSS |
| 偏好对齐 | **SimPO** | UltraFeedback-Binarized → AlpacaEval 2.0 |
| 推理强化 | **GRPO** | 编译 / 答案校验在线采样 → GSM8K / MATH / HumanEval / MBPP |

详见开题报告 [LLM Walk-Through.md](../LLM%20Walk-Through.md)。
