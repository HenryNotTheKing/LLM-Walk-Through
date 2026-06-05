## 顶层目录

| 目录 | 作用 |
|---|---|
| `configs/` | 所有 YAML 配置文件，按模型配置与训练配置拆分。 |
| `core/` | 模型核心实现与可替换模块（注意力、前馈、位置编码、分词器等）。 |
| `data/` | 数据下载、编码脚本与本地缓存。 |
| `docs/` | 项目说明、路线图与实验/模块教学 Notebook。 |
| `runs/` | 训练输出目录，通常存放 checkpoint。 |
| `scripts/` | 辅助脚本（如推理生成、安装 PyTorch）。 |
| `tests/` | 单元测试与对齐测试。 |
| `train/` | 训练入口与训练流程实现。 |

## configs/

| 目录 | 作用 |
|---|---|
| `configs/model/` | 模型结构相关配置（如 GPT-2 tiny/124m）。 |
| `configs/train/` | 训练过程配置（数据、优化器、步数、日志等）。 |

## core/

| 目录 | 作用 |
|---|---|
| `core/attention/` | 注意力模块：MHA、MQA、GQA、MLA、滑动窗口、线性注意力等。 |
| `core/residual/` | 跨层残差拓扑：Kimi AttnRes、DeepSeek mHC 等。 |
| `core/ffn/` | 前馈网络：GELU MLP、SwiGLU、GEGLU、MoE 等。 |
| `core/kv_cache/` | KV Cache 相关模块预留/实现。 |
| `core/model/` | 模型组装入口（如 GPT-2 主体）。 |
| `core/norm/` | 归一化层实现（如 LayerNorm）。 |
| `core/position/` | 位置编码实现（当前 learned，后续可扩展 RoPE 等）。 |
| `core/tokenizer/` | 分词器接口与多种算法实现（BPE/Byte BPE/WordPiece/Unigram）。 |
| `core/utils/` | 通用工具（配置加载、设备与分布式辅助）。 |

## data/

| 目录 | 作用 |
|---|---|
| `data/cache/` | 数据集本地缓存与中间文件，按数据集名称分目录。 |
| `data/cache/fineweb_edu_10bt/` | FineWeb-EDU 数据缓存、tokenizer 与快照内容。 |
| `data/cache/smoltalk_chinese_small/` | 中文小数据集缓存、切分文件、tokenizer 与快照内容。 |

## docs/

| 目录 | 作用 |
|---|---|
| `docs/experiments/` | 训练实验 Notebook（从最小闭环到轻量预训练）。 |
| `docs/modules/` | 按模块讲解的 Notebook。 |
| `docs/modules/tokenizer/` | 分词器专题 Notebook（BPE/Byte-level/Unigram/WordPiece）。 |

## runs/

| 目录 | 作用 |
|---|---|
| `runs/fineweb_edu_124m/` | 使用 fineweb_edu 配置训练得到的产物。 |
| `runs/smoltalk_chinese_small/` | 使用中文小数据集训练得到的产物。 |

## 其他协作建议

1. `configs/`、`core/`、`train/`、`tests/` 是主要协作代码区。
2. `data/cache/` 与 `runs/` 多为可再生文件，提交前请确认是否需要纳入版本管理。
3. 若新增模块，建议同步补充 `tests/` 与 `docs/modules/`，保持“实现-验证-说明”一致。
