# 模块名称（例：Rotary Position Embedding）

> 一句话定位：本模块在现代 LLM 架构演进中负责什么。

## 1. 问题背景

它解决了什么真实存在的问题？早期方案（如 sinusoidal / learned PE）有什么不足？

## 2. 经典方案

GPT-2 / Transformer 原版做法。给出公式与张量形状。

## 3. 现代方案

LLaMA 或更新工作中的做法。重点写"改了什么、为什么改"。

## 4. 数学形式

核心公式、复杂度、显存占用、与 attention/FFN 的接口约定。可用 KaTeX：

$$\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

## 5. 工程实现

- 关键类 / 函数（链接到源码）
- 实现路径与 fallback（CPU / MPS / CUDA / Flash）
- 容易踩的坑

## 6. 与本项目主线的关系

本模块在现代 LLM 演进的哪一步被引入（背景动机）；如何在本项目中通过配置切换；切换前后会影响哪些其他模块。

## 7. 实验设计

- 消融对照组
- 关键观测指标（PPL、显存、tok/s、外推长度等）
- 实验脚本：`experiments/...`

## 8. 常见坑

实现时最容易写错的细节。

## 9. References

### paper
- ...

### blog
- ...

### code
- ...

## 10. Code Map

| 概念 | 源码 | 测试 | 配置 |
| --- | --- | --- | --- |
|  |  |  |  |
