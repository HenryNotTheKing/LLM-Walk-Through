# LLM Walk-Through 课堂汇报幻灯片

本项目汇报 HTML 幻灯片，基于 [Reveal.js](https://revealjs.com/) 构建，内嵌预渲染的 Jupyter Notebook。

## 快速开始

```bash
cd slides
python -m http.server 8000
```

浏览器访问 `http://localhost:8000` 即可开始演示。

## 操作说明

| 按键 | 功能 |
|------|------|
| `→` / `↓` / `Space` | 下一页 |
| `←` / `↑` | 上一页 |
| `F` | 全屏 |
| `S` | 演讲者视图 |
| `O` | 概览模式 |
| `Esc` | 退出全屏/概览 |

## 幻灯片结构

1. **封面** — 项目名称与标语
2. **目录** — 7 个核心模块一览
3. **模块页 × 7** — 每个模块一页，iframe 内嵌对应 notebook HTML
4. **结尾** — 总结与项目地址

## 包含的 Notebook

| 模块 | 文件 |
|------|------|
| GPT-2 基线 | `notebooks/02_gpt2_baseline.html` |
| 因果多头注意力 | `notebooks/03_attention_mha.html` |
| RoPE 旋转位置编码 | `notebooks/01_rope.html` |
| RMSNorm 归一化 | `notebooks/01_rmsnorm.html` |
| SwiGLU 前馈网络 | `notebooks/01_swiglu.html` |
| GQA + QK-Norm 现代注意力 | `notebooks/04_gqa_qk_norm_rope.html` |
| Walkie 完整模型 | `notebooks/04_code_causal_lm.html` |

## 重新生成 Notebook HTML

若 notebook 内容更新，可重新转换：

```bash
python -m jupyter nbconvert --to html docs/modules/model/02_gpt2_baseline.ipynb --output-dir slides/notebooks --output 02_gpt2_baseline.html
```

## 注意事项

- 必须使用 HTTP 服务器打开（`python -m http.server`），直接用浏览器打开文件会导致 iframe 加载失败。
- 幻灯片字体已调大（28px 基础字号），适合课堂投影。
- iframe 内的 notebook 可独立滚动浏览。
