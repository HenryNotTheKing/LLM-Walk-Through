# Jupyter Notebook 技术文档编写规范 (Walkie 项目导向)

本文档旨在统一当前工程项目中 `docs/modules/` 子文件夹里所有 Jupyter Notebook 的编写规范，致力于产出具备严谨、连贯且易读的学术/工程并重的“原理-代码”讲解讲义。

## 1. 语言与基调 (Tone & Language)

* **客观严谨的理工科口吻**：使用第三人称，避免过度情绪化、拟人化或引导式的叙述（如避免“让我们来看看”、“嘿，你看”此类语句）。
* **术语准确**：对各类深度学习及工程专有名词（例如 SwiGLU, RMSNorm, RoPE, Grouped Query Attention, Gradient Checkpointing）全英文首字母正确大写拼写。

## 2. 结构组成规范 (Structural Requirements)

每个技术模块（如 Activation, Normalization, Position Encoding）的笔记都应包含以下主体部分：

### 2.1 源码导航与背景
* 在篇头使用 `Markdown` 单元格标明该独立模块的设计出处与路径。
  > 示例：源码导航：[`core/ffn/swiglu.py`](../../../core/ffn/swiglu.py) 中的 `SwiGLUMLP`。
* 一两句客观介绍该技术方案提出的工程及学术背景，如相比传统实现的优势（开销、效果）。

### 2.2 理论推导与数学公式 (Theory Formulation)
* **独立子节**：专门负责将所有的数学定理、公式推导放置在一处，切勿与后面的 Python 函数实现或者应用测试混杂。
* 使用标准的 LaTeX 格式编辑算法模型流程。
  > 示例：将激活函数公式 $\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ 等整理清爽。
* *配图或图表展示*：若涉及到分布函数、激活函数，应提供一小段（纯绘图用途的）无副作用的 Python 画图代码单元来对其特性进行展现（如 ReLU vs SiLU），并包含于此处或紧接着此节之后。

### 2.3 数据测试与单元展示 (Sanity Check)
* 使用极简且极具说服力的随机形状特征，向模型推入假数据检查维度对齐情况。
  > 示例：`torch.randn(Batch, Seq_len, n_embd)` 走完后 assert shape 一致性。

### 2.4 具体源码节选展示与精讲 (Code Extrapolation)
* 将项目里的实际物理 `class` 或 `def` 源码**直接摘录**为 `Markdown` 代码块或者独立的 `Python` 单元格中（只含类宣告，不要执行）。
* 在源码中适量加入注释，将变量与上方所阐述的对应公式符号对齐匹配。这能够帮助阅读该 Notebook 的新成员无缝映射公式到工程代码（例如标明 `bias=False` 的意图、哈达玛乘积对应代码的哪一行）。

## 3. Jupyter Cell 编辑要求
* **原子化原则**：当通过自动化 Agent 编辑时，建议每次通过精准截取出 `Notebook Summary` 中的 Cell ID，按块进行重写（Edit）或增补（Insert）。
* 不要混淆：确保数学推导是在 Markdown 中完成；可执行的逻辑检查在 Code Cell 中。

---

*随着后续对 `norm/` 或 `position/` 的编写，以此规范作为基础校准输出。*