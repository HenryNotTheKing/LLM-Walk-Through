# Walkie 后训练与评测框架交接说明

本文档用于把本轮新增的 SFT/RL 后训练框架和 HumanEval 风格评测框架交接给后续开发者或服务器上的智能体。当前 Windows 机器只完成代码实现与可运行的纯 Python 验证；真实 vLLM/Ray/torch 训练与评测应在 Linux/CUDA 服务器上完成。

## 1. 本轮完成内容

### 后训练框架

- `train/walkie_sft.py`：Walkie SFT 入口，支持 torchrun/DDP、AMP、AdamW+Muon、checkpoint resume、assistant-only next-token labels。
- `train/walkie_rl.py`：GRPO/DAPO 入口，支持 fake/vLLM rollout、规则奖励、sandbox 执行奖励、ref model、checkpoint resume。
- `configs/train/sft_walkie.yaml`：SFT 默认配置。
- `configs/train/rl_walkie_grpo.yaml`：GRPO 默认配置。
- `configs/train/rl_walkie_dapo.yaml`：DAPO 默认配置。
- `posttrain/data/`：Chat template、tokenizer alias、SFT JSONL/JSON/Parquet iterable dataset。
- `posttrain/rewards/`：reward registry、规则奖励、代码执行奖励与 sandbox runner。
- `posttrain/sandbox/jupyter_client.py`：适配 MultiModal-Jupyter-Sandbox 的 `/run_jupyter` 与 `/clear_session`。
- `posttrain/rollout/`：rollout 抽象、fake backend、vLLM backend。
- `posttrain/rl/`：GRPO/DAPO 数学核心、logprob helpers。
- `posttrain/utils/hf_export.py`：Walkie checkpoint 导出 Qwen3-compatible HF/vLLM 目录。

### 评测框架

- `scripts/evaluate_humaneval.py`：HumanEval JSONL 评测脚本，vLLM 生成，Ray 或 async HTTP sandbox 判题，输出 pass@k。
- `scripts/evaluate_code_dataset.py`：字段可映射的 HumanEval-style 自定义 JSONL 评测脚本。
- `configs/eval/humaneval_vllm.yaml`：HumanEval 评测配置模板。
- `configs/eval/code_dataset_vllm.yaml`：自定义代码数据集评测配置模板。
- `posttrain/eval/`：HumanEval 样本加载、test program 组装、pass@k、vLLM 生成 wrapper、Ray sandbox executor。

### 文档与测试

- `docs/posttrain_server_validation.md`：Linux 单卡服务器验证清单，包含 ckpt 导出、vLLM smoke、sandbox 联调、SFT/RL smoke、HumanEval/自定义数据集评测命令。
- `tests/test_posttrain_template.py`：模板、assistant-only next-token label、tokenizer alias。
- `tests/test_posttrain_rewards.py`：reward registry、sandbox response schema、sandbox runner。
- `tests/test_posttrain_rl_algorithms.py`：GRPO/DAPO 数学核心。
- `tests/test_posttrain_eval.py`：HumanEval-style 评测核心。

## 2. 重要设计决策

- 不扩 Walkie tokenizer 词表：`vocab_size=65536` 和 `uint16` token id 范围保持不变。
- ChatML 支持通过低频/保留 token alias 完成；如果 tokenizer 中找不到安全槽位，训练入口自动回退 `plain_eot`。
- SFT labels 已按 Walkie 训练契约右移：输入 token 预测下一个 token；非 assistant 目标 label 为 `-1`。
- RL 只实现 GRPO/DAPO，不实现 PPO/DPO。
- vLLM rollout 当前采取保守正确策略：每轮 rollout 前导出最新 actor 并重建 vLLM engine，优先保证 on-policy 正确性，后续可优化热同步。
- DDP + vLLM 时只让 rank0 做 rollout/reward/sandbox，并 broadcast 结果，避免每个 rank 重复启动 vLLM。
- 单卡服务器显存紧时可以设置 `ref.offload=cpu`。
- sandbox 执行忽略图片字段，只解析 `stdout/stderr/result/status/execution_time`。

## 3. 服务器迁移后的推荐验证顺序

详细命令见 `docs/posttrain_server_validation.md`。推荐顺序如下：

1. `uv sync --extra walkie --extra posttrain`。
2. 确认 `torch.cuda.is_available()` 为 `True`。
3. 跑纯单元测试：`tests/test_posttrain_template.py`、`tests/test_posttrain_rewards.py`、`tests/test_posttrain_rl_algorithms.py`、`tests/test_posttrain_eval.py`。
4. 跑原有 Walkie checkpoint/optim/schedule 回归测试。
5. 用已有 ckpt 跑 `export_walkie_to_hf`，确认导出目录包含 `config.json`、`tokenizer.json`、`model.safetensors` 或 `pytorch_model.bin`。
6. 用 vLLM 加载导出目录，做一次短生成。
7. 启动 MultiModal-Jupyter-Sandbox，确认 `/run_jupyter` 返回 `ALL TESTS PASSED`。
8. 跑 SFT 单步 smoke。
9. 跑 RL fake rollout 单步 smoke。
10. 跑 RL + vLLM + sandbox 单步 smoke。
11. 跑 resume 验证，确认从 `latest.pt` 继续。
12. 跑 HumanEval 或自定义数据集评测。

## 4. 单卡服务器常用命令

### GRPO fake smoke

```bash
uv run python -m train.walkie_rl \
  --config configs/train/rl_walkie_grpo.yaml \
  --init-from "$CKPT" \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=fake \
  train.total_steps=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  distributed.backend=none
```

### GRPO + vLLM + sandbox smoke

```bash
uv run --extra posttrain python -m train.walkie_rl \
  --config configs/train/rl_walkie_grpo.yaml \
  --init-from "$CKPT" \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=vllm \
  rollout.tensor_parallel_size=1 \
  sandbox.enabled=true \
  sandbox.base_urls=[$SANDBOX_URL] \
  ref.offload=cpu \
  train.total_steps=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  rl.max_completion_length=128 \
  distributed.backend=none
```

### HumanEval 评测

```bash
uv run --extra posttrain python -m scripts.evaluate_code_bench \
  --config configs/eval/walkie_code_bench.yaml \
  --model "$ROOT/runs/walkie_code_0.5b_vllm_hf" \
  --sandbox-url "$SANDBOX_URL" \
  --dataset all
```
