# 文档规范

本目录是项目的"入门正门"。每篇模块文档对应 `core/` 下一个可替换组件，
读者读完应能回答：**这个模块解决什么问题、改了什么、好在哪、代价是什么、如何在本仓库切换、如何验证**。

## 文档层级（**强制**）

文档分为两层：

- **父文档（索引）**：作为某个模块大类的入口，给出该类的全景对照、决策建议、子文档目录。
  - 形式可以是单文件 `docs/0X_xxx.md`，也可以是目录 `docs/0X_xxx/README.md`（推荐——当该类有 ≥ 2 篇子文档时）。
- **子文档（细节）**：单一算法 / 实现的深入说明。文件名以 `0Y_<name>.md` 形式排序。

### 父子双向软链接（**强制**）

> "软链接"在本项目中特指 **Markdown 相对链接**，不依赖 OS 级 symlink。

- **每个子文档**都必须在 **开头第一行** 和 **末尾最后一行** 放置一个返回父文档的链接，形如：

  ```markdown
  > 父文档：[← 分词器总览](README.md)
  ```

- **父文档**必须列出全部子文档（表格或编号列表），并对每个子文档给一句话说明。

例：[`docs/01_tokenizer/README.md`](01_tokenizer/README.md) 是父文档；
[`docs/01_tokenizer/01_bpe.md`](01_tokenizer/01_bpe.md) 是子文档，开头/结尾各一条返回链接。

### 何时拆分

- 当一个模块大类只有一个具体算法时，单文件 `docs/0X_xxx.md` 即可。
- 当出现第二个算法（如分词器从 BPE 扩展到 WordPiece），**必须**重构为目录形式，
  把老的单文件拆为父文档（索引）+ 一篇子文档。

## 三类资源

每篇模块报告的 `## References` 部分必须按下列三类列出：

- `paper`：原始论文 / 技术报告 / 官方博客
- `blog`：高质量博客、课程讲义、可视化解说
- `code`：参考实现仓库（标注它的角色，例如"官方实现"或"教学复刻"）

## Code Map

每篇报告必须有一个 `## Code Map` 表，把概念落到本仓库具体文件：

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
| 例子：Causal MHA | [`core/attention/mha.py`](../core/attention/mha.py) | [`tests/test_gpt2_model.py`](../tests/test_gpt2_model.py) | [`configs/model/gpt2_124m.yaml`](../configs/model/gpt2_124m.yaml) |

至少包含一个源码链接、一个测试链接、一个配置链接。

## 链接约定

- 默认使用 **Markdown 相对链接**（不依赖 OS 级 symlink）。
- 链接到“类/函数”而不是“具体行号”，避免行号漂移失效。
- 链接路径以**当前 `.md` 所在目录**为基准：
  - 平级：`./03_attention_mha.md`
  - 进入子目录：`./01_tokenizer/01_bpe.md`
  - 子文档→父文档（同目录）：`README.md`
  - 顶层文档→源码：`../core/attention/mha.py`
  - 二级文档→源码：`../../core/tokenizer/bpe.py`
  - 文档→测试：`../tests/test_gpt2_model.py` 或 `../../tests/...`
  - 文档→配置：`../configs/model/gpt2_124m.yaml` 或 `../../configs/...`

## 模块报告模板

参见 [`_template_module_report.md`](_template_module_report.md)。
