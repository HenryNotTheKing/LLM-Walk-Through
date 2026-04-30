# 最小预训练闭环

入口：[`train/pretrain.py`](../train/pretrain.py)，配置：[`configs/train/pretrain_tiny.yaml`](../configs/train/pretrain_tiny.yaml)。

## 1. 流程

```
tiny shakespeare 文本
   │
   ▼
教学 BPE 训练 / 加载（缓存到 data/cache/tiny_shakespeare/）
   │
   ▼
编码为 train.bin / val.bin（uint16 / uint32 由分词器决定）
   │
   ▼
    随机采样 batch  ──►  GPT2LMHeadModel  ──► loss
                                                │
                                                ▼
                                       AdamW + cosine LR
                                                │
                                                ▼
                                        eval / checkpoint
```

## 2. 关键配置项（对应 YAML）

| 字段 | 含义 |
| --- | --- |
| `train.batch_size` / `train.block_size` | 输入张量形状 |
| `train.grad_accum_steps` | 大 batch 梯度累积 |
| `train.learning_rate` / `min_lr` / `warmup_steps` | cosine schedule |
| `train.device` / `train.dtype` / `train.amp` | 自动选 cuda/mps/cpu，半精度仅在 CUDA 启用 |
| `distributed.backend` | `none` 单进程；`ddp` 配合 `torchrun` |

## 3. 单卡 / DDP 命令

```powershell
# 单卡
uv run python -m train.pretrain --config configs/train/pretrain_tiny.yaml

# DDP（任意 N 卡）
uv run torchrun --nproc_per_node=N -m train.pretrain `
    --config configs/train/pretrain_tiny.yaml distributed.backend=ddp
```

OmegaConf 的 dotlist 覆盖在 `--config` 之后追加任意 `key=value`。

## 4. 验证

- 单步训练能反传：[`tests/test_training_step.py`](../tests/test_training_step.py)
- 完整 200 步 tiny 训练在 CPU 上几分钟内 loss 应明显下降（~ -2 至 -3 nat）。

## 5. 后续

- DeepSpeed ZeRO（V2）：覆盖 1B 完整预训练。
- KV cache（V1）：让 generate 不再每步全量重算。
- 数据：FineWeb-Edu / OpenCSG / The Stack v2 / MathPile（V2）。
