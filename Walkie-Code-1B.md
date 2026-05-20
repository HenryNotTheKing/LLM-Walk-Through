# Code-Specialized 1B 英语模型完整计划书 v2
## 目标：在 HumanEval / HumanEval+ / MBPP+ / LiveCodeBench 上超越 Maincoder-1B

> 修订版：基于 Maincoder-1B 架构分析


## 核心差异总览：我们 vs Maincoder-1B

| 维度 | Maincoder-1B | 本方案 | 改动必要性 |
|------|-------------|--------|-----------|
| Vocab | 151,936（Qwen） | **65,536** | 节省267M参数，补偿更多层 |
| head_dim | 96（非标准） | **64** | GPU kernel对齐，FlashAttn效率↑ |
| 层数 | 32 | **48** | deep-thin + 节省vocab参数换来 |
| Q/KV 头 | 16Q/4KV | **24Q/8KV** | head_dim=64，保持4:1 GQA |
| 上下文 | **2,048**（致命弱点） | **16,384** | 实际使用场景关键 |
| FIM | 无（未提及） | **PSM+SPM 50%率** | 代码补全核心能力 |
| 代码比例 | 未知 | **85%** | 与DeepSeek-Coder对齐 |
| RL后训练 | MCPO（黑盒） | **GRPO+执行验证** | 可复现 |

---

## 第一部分：模型架构

### 1.1 完整规格

```
================================
  代号：YourCoder-1B
================================

Tokenizer:
  算法：       BPE (byte-level fallback)
  vocab_size:  65,536
  特殊 tokens: <fim_prefix> <fim_middle> <fim_suffix> <fim_pad>
               <|repo_name|> <|file_sep|> <|endoftext|>
  训练语料：   The Stack v2 Python + top-10 语言 + 代码文档（50GB子集）

架构:
  hidden_size:    1,536
  n_layers:       48          ← 比Maincoder(32)深50%，由vocab节省参数补偿
  n_heads_q:      24
  n_heads_kv:     8           ← GQA 3:1（比Maincoder 4:1更保守，质量更高）
  head_dim:       64          ← 1536/24=64，标准值，FlashAttn最优
  d_ffn:          4,096       ← SwiGLU (2/3 × 4 × 1536 ≈ 4096)
  rope_theta:     1,000,000   ← 与Maincoder相同
  max_seq_len:    16,384      ← 训练上下文（Maincoder仅2048，差8倍）
  norm:           RMSNorm (pre-norm)
  qk_norm:        True
  activation:     SwiGLU
  bias:           False（全部）
  weight_tying:   False

参数估算（非embedding）:
  Attention/层:  1536×1536 + 1536×512 + 1536×512 + 1536×1536
               = 2.36M + 0.79M + 0.79M + 2.36M = 6.3M
  SwiGLU FFN/层: 1536×4096×3 = 18.9M
  RMSNorm×2:    ~0.003M
  每层合计:      ~25.2M
  48层总计:      48 × 25.2M ≈ 1,209M ≈ 1.21B 非embedding参数

Embedding参数:
  Input:  65536 × 1536 = 100M（vs Maincoder的233M，节省133M！）
  Output: 65536 × 1536 = 100M（独立，不共享）
  总参数（含embedding）: ~1.41B
  非embedding: ~1.21B
```

### 1.2 Maincoder 参数效率对比

```
Maincoder-1B:
  embedding 参数: 151,936 × 1536 × 2 = 467M （占总参数 ~47%！）
  实际建模参数:    ~533M（名义1B的一半）

YourCoder-1B:
  embedding 参数: 65,536 × 1536 × 2 = 200M （占总参数 ~14%）
  实际建模参数:    ~1,210M（是Maincoder建模参数的2.3倍）

→ 在同等 "1B" 标签下，我们拥有2.3倍的有效建模容量
```

### 1.3 FIM 特殊 Tokens 设计

```python
# 参考 StarCoder2 / DeepSeek-Coder PSM 格式
FIM_PREFIX  = "<fim_prefix>"
FIM_MIDDLE  = "<fim_middle>"
FIM_SUFFIX  = "<fim_suffix>"
FIM_PAD     = "<fim_pad>"      # padding 用于等长对齐

# PSM 格式（prefix→suffix→middle，主用）
fim_psm = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

# SPM 格式（suffix→prefix→middle，10%比例）
fim_spm = f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"

# Repository level 特殊 tokens
REPO_NAME   = "<|repo_name|>"
FILE_SEP    = "<|file_sep|>"

# 仓库级上下文拼接格式（参考Qwen2.5-Coder）:
# <|repo_name|>owner/repo<|file_sep|>path/to/file.py\n{code}
```

---

## 第二部分：关于 Python-Only 还是多语言的决策

### 2.1 直接回答：不应该纯 Python

**数据充足性不是问题，策略问题才是核心。**

纯 Python 训练的代价与收益：

| 维度 | 纯 Python | 多语言（Python主导） |
|------|-----------|-------------------|
| HumanEval / MBPP+ | 更高（直接优化目标） | 略低 |
| 跨语言泛化 | 研究表明最多34.4%（其他语言） | 完整覆盖 |
| 推理能力 | 有损失（代码多样性降低） | 更强 |
| LiveCodeBench | 竞争题目依赖算法推理 | 更强 |
| BigCodeBench | 依赖库调用多样性 | 更强 |
| 实际部署价值 | 受限 | 全面 |

**关键证据**：
- CRUXEval-X 研究：纯 Python 模型在其他语言最多达到 34.4% pass@1，跨语言泛化有限
- DeepSeek-Coder 使用 87% 多语言代码 + 13% NL，而非纯 Python
- OpenCoder 使用 90% 多语言代码 + 10% NL，HumanEval 表现优秀
- StarCoder 的策略：先多语言基座，再用 35B Python tokens 做 Python 特化 → 证明两步走更有效

### 2.2 最优策略：多语言基座 + Python 权重放大

```
代码语言权重分配（训练时按此比例采样）：
  Python:         40%   ← 大幅权重但非100%
  JavaScript/TS:  12%
  C/C++:          8%
  Java:           7%
  Rust:           5%
  Go:             5%
  Shell/Bash:     4%
  SQL:            3%
  HTML/CSS:       3%
  其他 600+ 语言: 13%
```

Python 40% 已是其在 The Stack v2 实际占比的 3-4 倍（自然占比约 10-15%），足够使模型 Python 专精，同时不失去多样性带来的推理能力。

### 2.3 Python 数据充足性分析

```
Python 数据来源与规模估算：

The Stack v2 dedup 总计: ~900B tokens
  → Python 自然占比 ~12%: ≈ 108B tokens（过滤前）
  → 基础启发式过滤后: ≈ 70B tokens
  → BERT-质量过滤后（10% retention）: ≈ 7B 高质量 token

PyPI 文档 + README: ~5B tokens
GitHub Issues（Python repos）: ~8B tokens
Jupyter Notebooks（Kaggle等）: ~10B tokens
StackOverflow Python: ~6B tokens
Python 官方文档 + PEP: ~1B tokens
合成 Python（LLM生成）: ~20-30B tokens（Annealing阶段）

Phase 1 可用 Python（权重放大后）：
  实际消耗 = (总token budget × Python比例) = 1.3T × 40% = 520B
  → 需要多次 epoch：70B × 7 epoch ≈ 490B（合理，Python数据质量高允许多epoch）
  → 高质量 Python epoch 数不应超过 10，避免过拟合

结论：数据量充足，高质量 Python 需多 epoch，这是可接受的做法
（SmolLM2 用 11T tokens 训练 1.7B，其中数据集有大量重复）
```

---

## 第三部分：两阶段预训练数据策略

### 3.1 Phase 1：主训练（~1.3T tokens，WSD stable 阶段）

**目标**：构建代码理解 + 多语言编程基础

| 数据源 | 比例 | 估计 Token 量 | 说明 |
|--------|------|--------------|------|
| **The Stack v2 dedup（多语言，按权重采样）** | 55% | ~715B | 核心代码语料，含 Python 40%+ 其他语言 |
| **代码相关 Web（Stack Overflow, 文档, 博客）** | 12% | ~156B | FineWeb 中代码相关页面 + 专项爬取 |
| **GitHub Issues + PRs + Discussions** | 8% | ~104B | 工程思维、debug、代码评审 |
| **Jupyter/Kaggle Notebooks** | 6% | ~78B | 数据科学 Python，含 NL+code 混合 |
| **代码文档（ReadTheDocs, 官方文档）** | 5% | ~65B | API 使用方式，实际调用模式 |
| **OpenWebMath + FineMath** | 7% | ~91B | 数学推理，算法分析基础 |
| **FineWeb-Edu（score≥3）** | 4% | ~52B | 通用英语推理能力维持 |
| **Wikipedia（EN） + PeS2o** | 3% | ~39B | 事实锚点，避免完全失去常识 |

> 注：Phase 1 完全没有合成数据，避免污染基础分布

### 3.2 Phase 2：Annealing（~150B tokens，WSD decay 阶段）

**目标**：Python 专精 + 质量压缩（上下文保持 4K）

| 数据源 | 比例 | 估计 Token 量 | 说明 |
|--------|------|--------------|------|
| **高质量 Python（BERT分类器 top-10%）** | 30% | ~45B | 从 The Stack v2 Python 选出精华 |
| **合成 Python（Llama-3.1-70B 重写版）** | 18% | ~27B | 教学化代码，含注释+文档字符串 |
| **Arctic-SnowCoder 风格合成（execution-filtered）** | 10% | ~15B | 仅保留可执行通过 unit test 的代码 |
| **GitHub Issues→代码配对**（Python 专项） | 8% | ~12B | 问题→解决方案，理解代码意图 |
| **Kaggle Gold Notebooks + 竞赛代码** | 8% | ~12B | 高质量 Python 实践代码 |
| **HumanEval 风格合成题库**（非测试集） | 6% | ~9B | 确保对 benchmark 格式适应 |
| **FineMath（score≥4）+ 算法题** | 8% | ~12B | 数学+算法推理强化 |
| **代码文档（Python libs: numpy,pandas,torch...）** | 7% | ~10.5B | 库调用实际模式（BigCodeBench相关） |
| **FineWeb-Edu（score≥4）** | 5% | ~7.5B | 保持自然语言理解 |

---

## 第四部分：数据过滤策略详解

### 4.1 代码质量过滤 Pipeline（三阶段）

```python
# ═══════════════════════════════════
# Stage 1：基础启发式过滤（Phase 1 使用）
# ═══════════════════════════════════

def basic_code_filter(file: dict) -> bool:
    code = file["content"]
    lines = code.split("\n")
    
    # 文件大小
    if len(code) < 100 or len(code) > 1_000_000:
        return False
    
    # 行长度
    if max(len(l) for l in lines) > 1000:  # 过滤 minified
        return False
    if sum(len(l) for l in lines) / max(len(lines), 1) > 150:  # 平均行长
        return False
    
    # Python 特有
    if file["language"] == "Python":
        # 至少含1个函数/类定义
        if "def " not in code and "class " not in code:
            return False
        # 字母比例
        alpha = sum(c.isalpha() for c in code)
        if alpha / len(code) < 0.2:
            return False
    
    # 许可证过滤
    permissive = {"MIT","Apache-2.0","BSD-2-Clause","BSD-3-Clause",
                  "ISC","CC0-1.0","Unlicense","WTFPL","MIT-0"}
    if file.get("license") and file["license"] not in permissive:
        return False
    
    return True

# MinHash 近去重：Jaccard 阈值 0.85（参考 StarCoder2 标准）
# ≈ 去除 40% 近似重复文件（BigCode 实测）
```

```python
# ═══════════════════════════════════
# Stage 2：BERT-style 质量分类器（Phase 2 annealing 使用）
# ═══════════════════════════════════

# 参考 Arctic-SnowCoder：BERT 分类器区分高质量 vs 随机代码
# 正样本来源：
POSITIVE_SOURCES = [
    "CPython 标准库 (Lib/*.py)",            # 最高质量
    "PyTorch/TensorFlow/NumPy core",         # 工业级代码
    "high-star repos (>10K stars) 核心文件", # 经过社区验证
    "Python PEP 实现代码",                   # 标准参考
    "LeetCode 官方题解",                     # 算法规范
]
# 负样本来源：从 Stage 1 数据随机采样（非刻意筛低质量）

# 训练：bert-base-uncased fine-tune 二分类
# 过滤：保留 score > 0.7 的文件（约 top 10%）
# 最终选出 ~7B 高质量 Python tokens

# Stage 2 数据重复次数：× 4 epoch（Arctic-SnowCoder 同策略）
# = 7B × 4 = 28B token budget 来自高质量精选
```

```python
# ═══════════════════════════════════
# Stage 3：合成代码（Annealing 使用）
# ═══════════════════════════════════

# 参考 Arctic-SnowCoder Phase 3 + Magicoder OSS-Instruct
SYNTHESIS_PROMPT = """You are an expert Python programmer and teacher.
Look at this code snippet as inspiration only. 
Write a NEW, complete, well-documented Python function or class that:
1. Has a clear docstring explaining purpose and parameters
2. Uses descriptive variable names
3. Includes inline comments for non-obvious logic
4. Has at least 2 edge case handling examples
5. Is fully original (NOT copied from the inspiration)

Inspiration snippet (do NOT reproduce):
{code_snippet}

Write your new educational Python function below:
"""

# Execution filtering（关键质量门控）：
import subprocess

def execution_filter(code: str, test: str) -> bool:
    """只保留能通过 unit test 的合成代码"""
    try:
        result = subprocess.run(
            ["python", "-c", f"{code}\n{test}"],
            timeout=10,
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False

# 保留率约 40-60%（Magicoder 实测数据）
# 合成产出约 25-30B tokens，过滤后约 10-15B 高质量合成 token
```

### 4.2 Python 特化过滤的额外规则

```python
# HumanEval / MBPP 格式对齐（Annealing Phase 关键）
# 从 Phase 2 数据中筛选出"函数定义+docstring+完整实现"格式

def is_humaneval_aligned(code: str) -> bool:
    """检测是否符合 HumanEval 格式：有函数头+docstring+实现"""
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 有 docstring
                if (ast.get_docstring(node) and
                    # 有实际实现（不只是pass）
                    len(node.body) > 1):
                    return True
    except:
        pass
    return False

# 这类数据在 Annealing 最后 20B token 中占比提高到 30%
# 帮助模型适应 benchmark 的输出格式
```

---

## 第五部分：FIM 预训练策略

### 5.1 FIM 配置

```python
# FIM 是所有主流代码 LLM 的标配
# 参考：StarCoder2, DeepSeek-Coder, Qwen2.5-Coder, CodeLlama
# Maincoder-1B 未提及 FIM → 这是我们的重要优势

FIM_CONFIG = {
    "fim_rate": 0.5,        # 50% 训练 token 使用 FIM，50% 正常 L2R
                             # 研究证明 50-90% 不损害 L2R 生成质量
    "psm_ratio": 0.9,       # 90% 使用 PSM 格式（更常见）
    "spm_ratio": 0.1,       # 10% 使用 SPM 格式（增强多样性）
    "fim_split": "character", # 随机字符级切割（基础）
    # 进阶：AST-FIM（结构感知切割）
    "ast_fim": True,        # Annealing 阶段启用 AST-aware FIM
                            # 在 1B 和 8B 模型上均提升 FIM 性能 5 pts
                            # 来源：Gong et al. 2025
}

# PSM 格式示例（训练时数据格式）：
"""
<fim_prefix>def fibonacci(n):
    """Return nth Fibonacci number."""
    if n <= 1:
<fim_suffix>
    return a
<fim_middle>        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
"""

# Repository level 拼接（增强文件间依赖理解）
"""
<|repo_name|>user/math_utils
<|file_sep|>utils/helpers.py
def add(a, b):
    return a + b

<|file_sep|>main.py
from utils.helpers import add
result = add(1, 2)
"""
```

---

## 第六部分：超参数配置

### 6.1 完整超参数表

```yaml
# ══════════════════════════════
# Phase 1：主训练
# ══════════════════════════════

model:
  hidden_size:       1536
  num_layers:        48
  num_heads_q:       24
  num_heads_kv:      8
  head_dim:          64
  ffn_dim:           4096
  vocab_size:        65536
  max_position:      4096
  rope_theta:        1_000_000
  qk_norm:           true
  rms_norm_eps:      1e-6

optimizer:
  # Linear 层用 Muon，embedding/norm 用 AdamW
  muon:
    lr:              1e-3       # µP 搜索得到的最优 LR（proxy d=128，48层）
    momentum:        0.95
    nesterov:        true
    ns_steps:        5         # Newton-Schulz 迭代次数
    wd:              0.1
  adamw:             # embedding + norm 参数
    lr:              3e-4
    betas:           [0.9, 0.95]
    eps:             1e-8
    wd:              0.1

grad_clip: 1.0

lr_schedule:
  type:              WSD
  warmup_steps:      2000       # ~1% 总步数
  stable_ratio:      0.88       # 88% 稳定阶段
  decay_ratio:       0.12       # 最后12% decay
  decay_shape:       sqrt       # sqrt decay（FG-WSD 2025推荐）
  final_lr:          0          # D2Z

# 4×3090 配置
batch:
  seq_len:           4096       # Phase 1 训练序列长度
  micro_batch:       4          # seqs per GPU
  grad_accum:        64         # 有效 batch = 4GPU × 4 × 4096 × 64 ≈ 4M tokens/step
  # 注：代码模型比通用模型更适合大 batch，因为代码 token 信息密度高

precision:           bfloat16
flash_attention:     2          # flash-attn 2.x
gradient_checkpointing: true    # 节省显存

# FIM 配置
fim_rate:            0.5
fim_format:          [psm, spm]
fim_psm_ratio:       0.9

total_tokens:        1_300_000_000_000   # 1.3T tokens

# ══════════════════════════════
# Phase 2：Annealing（Decay 阶段）
# ══════════════════════════════

phase2:
  total_tokens:      150_000_000_000    # 150B tokens
  seq_len:           4096               # 上下文保持 4K，不做长序列扩展
  micro_batch:       2                  # seq_len加倍，micro_batch减半
  data_switch:       true               # 切换到高质量代码数据
  
  context_extension:
    # 在 Phase 2 开始时做 rope_theta 调整
    rope_theta:      1_000_000          # 保持与 Phase 1 一致，不做长上下文 NTK 扩展
    # 前 10B tokens：seq_len 4096→8192
    # 不做长上下文扩展，整个训练保持 seq_len=4096
    curriculum:      true

  ast_fim:           true              # Annealing 阶段启用 AST-FIM
```

### 6.2 µP 代理模型规格

```python
# 在 proxy 上搜索 LR、WD，迁移到目标 1B 模型
proxy_config = {
    "d_model":   192,    # 1536 的 1/8
    "n_layers":  48,     # 层数保持相同（µP 按宽度缩放，非深度）
    "n_heads_q": 3,      # 24 的 1/8（保持 head_dim=64）
    "n_heads_kv":1,
    "d_ffn":     512,    # 4096 的 1/8
    "vocab":     65536,  # 词表大小不变
}

# 搜索范围
muon_lr_sweep =   [3e-4, 1e-3, 3e-3]
weight_decay  =   [0.05, 0.1, 0.3]
# 在 10B tokens 子集上训练 proxy，取最优组合迁移

# 注意：Everett et al. (Oct 2025) 发现 weight decay 比 LR scaling 更关键
# → 重点扫描 wd，不只看 LR
```

---

## 第七部分：RL 后训练（替代 MCPO 的可复现方案）

### 7.1 SFT 阶段

```yaml
SFT 数据集:
  - OpenCodeInstruct（5M样本，过滤出50K最难）
  - Magicoder-OSS-Instruct（75K高质量Python）
  - Evol-Instruct-Code-80K（复杂度多样）
  - ShareCode（用户真实代码问题）
  
总量: ~200K 高质量样本
格式: ChatML（<|im_start|>system / user / assistant<|im_end|>）
```

### 7.2 GRPO 强化学习（Execution-based Reward）

```python
# GRPO（Group Relative Policy Optimization）代替 MCPO
# 核心：用代码执行结果作为 Reward，无需人工标注

def code_reward(generated_code: str, test_cases: list) -> float:
    """可复现的代码执行奖励"""
    rewards = []
    
    for test in test_cases:
        try:
            # 在沙箱中执行
            result = sandbox_execute(
                code=generated_code,
                test=test["test_code"],
                timeout=10
            )
            if result.passed:
                rewards.append(1.0)
            else:
                # 部分奖励：语法正确但测试失败
                rewards.append(0.2 if result.syntax_ok else 0.0)
        except TimeoutError:
            rewards.append(0.0)
    
    return sum(rewards) / len(rewards) if rewards else 0.0

# 数据来源：
# - LeetCode 题库（非测试集）+ 自动生成 unit test
# - MBPP 训练集（不含 MBPP+ 测试集）
# - 合成算法题（用 GPT-4o 生成 + 人工验证）

GRPO_CONFIG = {
    "group_size":       8,    # 每题生成8个候选，组内对比
    "kl_coef":          0.04,
    "lr":               5e-6,
    "train_steps":      2000,
    "max_new_tokens":   512,
}
```

---

## 第八部分：评测 Benchmark 完整列表

### 8.1 主要 Benchmark（按可信度排序）

| Benchmark | 语言 | 说明 | 可信度 |
|-----------|------|------|--------|
| **LiveCodeBench** | Python | 竞赛题持续更新，抗污染 | ⭐⭐⭐⭐⭐ |
| **BigCodeBench** | Python | 1140题，真实库调用（numpy/pandas等） | ⭐⭐⭐⭐⭐ |
| **HumanEval+** | Python | HumanEval 更严格测试版 | ⭐⭐⭐⭐ |
| **MBPP+** | Python | MBPP 更严格测试版 | ⭐⭐⭐⭐ |
| **CRUXEval** | Python | 代码执行推理（in→out, out→in） | ⭐⭐⭐⭐ |
| **SWE-bench Verified** | Python | 真实 GitHub issue 修复 | ⭐⭐⭐⭐⭐ |
| **HumanEval** | Python | 经典，已饱和/污染 | ⭐⭐⭐ |
| **MBPP** | Python | 经典，参考用 | ⭐⭐⭐ |
| **DS-1000** | Python | 数据科学任务 | ⭐⭐⭐⭐ |
| **SAFIM** | 多语言 | FIM 专项评测（17720题） | ⭐⭐⭐⭐ |

> 注：HumanEval 已高度饱和，Maincoder 的 76% 主要靠 MCPO，
> 真正能力应以 LiveCodeBench 和 BigCodeBench 为主要对标。

### 8.2 对标竞品（最新数据）

| 模型 | HumanEval | HumanEval+ | MBPP+ | 参数 | 特点 |
|------|-----------|------------|-------|------|------|
| Maincoder-1B | 76.22% | 72.56% | 70.90% | 1B | MCPO RL，ctx=2K |
| Qwen2.5-Coder-1.5B | 46.34% | 44.51% | 65.61% | 1.5B | 5.5T tokens |
| DeepSeek-Coder-1.3B | 56.10% | 53.05% | 62.17% | 1.3B | 2T tokens |
| **YourCoder-1B（目标）** | **~70%** | **~65%** | **~72%** | ~1.4B | **ctx=4K, FIM** |

> YourCoder 的优势在于：更大上下文 + FIM 能力 + BigCodeBench（库调用）表现更好
> HumanEval 上 Maincoder 的 MCPO 优势很难复现，但其他 benchmark 可以超越

---

## 第九部分：完整训练流程时间线

```
Week 1-2: 数据准备
  ├── The Stack v2 Python + top-10 语言下载（~2TB）
  ├── 基础过滤 + MinHash 去重（Stage 1）
  ├── Tokenizer 训练（65K vocab，50GB 代码子集）
  └── FineWeb-Edu score≥3 + 数学数据准备

Week 3-4: Proxy 训练 + µP 超参搜索
  ├── d=192, 48层 proxy 模型
  ├── LR × WD 网格搜索（12个组合）
  └── 确认最优超参，预测目标模型性能

Week 5-18: Phase 1 主训练（1.3T tokens）
  ├── 4×3090 DDP + gradient checkpointing
  ├── 约 ~600K steps（batch=4M tokens）
  ├── WSD stable 阶段（88% steps）
  └── 每 50K steps checkpoint + 快速 HumanEval 评估

Week 19-20: Phase 2 Annealing（150B tokens）
  ├── 切换高质量 Python 数据
  ├── 上下文始终保持 4K
  ├── 启用 AST-FIM
  └── WSD sqrt decay → 0

Week 21: SFT
  ├── ChatML 格式转换
  └── 200K 高质量代码指令数据

Week 22: GRPO RL（可选）
  ├── 代码执行奖励
  └── 2000 steps

Week 23: 全面评测
  ├── LiveCodeBench, BigCodeBench, HumanEval+, MBPP+
  ├── SAFIM（FIM 专项）
  └── 与 Maincoder-1B 全面对比
```

---

## 第十部分：关键文献引用

| 发现 | 来源 |
|------|------|
| FIM 50-90% 不损害 L2R | Bavarian et al. 2022，Gong et al. 2024 |
| AST-FIM 提升 FIM 5pts | Gong et al. May 2025 (arxiv 2506.00204) |
| Python-only 跨语言上限 34.4% | CRUXEval-X, 2024 (arxiv 2408.13001) |
| DeepSeek-Coder: 87%代码+13%NL | DeepSeek-Coder github/paper |
| OpenCoder: 90%代码+10%NL | OpenCoder arxiv 2411.04905 |
| Arctic-SnowCoder 3阶段过滤 | Snowflake 2024 (arxiv 2409.02326) |
| Muon > AdamW on 1B | Essential AI 2025 (arxiv 2505.02222) |
| 执行过滤保留率 40-60% | Magicoder 论文 |
| vocab大小对1B影响 | PanGu-π Pro 2024 (arxiv 2402.02791) |
| Maincoder-1B 实际规格 | HuggingFace model card |
| GRPO for code RL | DeepSeek-R1, OpenR1 |
```