# Walkie 后训练框架服务器验证清单

本文档给迁移到 Linux 服务器后的智能体使用。服务器有 2 张 CUDA 卡，但已经有 Walkie 预训练 checkpoint 可用于验证。不要在 Windows 本机跑 vLLM/Ray 真实训练。

## 0. 约定变量

在服务器上先按实际路径设置这些变量：

```bash
export ROOT=/data/LLM-Walk-Through
export CKPT=$ROOT/runs/walkie_code_0.5b/latest.pt
export TOKENIZER=$ROOT/data/cache/walkie_code/tokenizer.json
export SFT_JSONL=$ROOT/data/posttrain/sft_smoke.jsonl
export RL_JSONL=$ROOT/data/posttrain/rl_smoke.jsonl
export SANDBOX_URL=http://127.0.0.1:18901
cd $ROOT
```

`SFT_JSONL` 至少准备 2 条 `prompt/response` 或 OpenAI `messages` 样本。`RL_JSONL` 至少准备 2 条 `prompt`，若要测代码执行奖励，每条最好带 `tests` 字段，例如：

```jsonl
{"prompt":"Write add(a, b). Return only Python code.","tests":"assert add(1, 2) == 3"}
```

## 1. 依赖与导入检查

```bash
uv sync --extra walkie --extra posttrain
uv run python - <<'PY'
import torch
import posttrain
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY
```

通过标准：能 import `torch` / `posttrain`，`torch.cuda.is_available()` 为 `True`。

## 2. 单元测试

```bash
uv run pytest \
  tests/test_posttrain_template.py \
  tests/test_posttrain_rewards.py \
  tests/test_posttrain_rl_algorithms.py \
  -q
```

通过标准：全部通过。这里会覆盖 assistant-only next-token labels、tokenizer alias、reward registry、sandbox 响应解析、GRPO/DAPO 数学核心。

## 3. Checkpoint 兼容回归

```bash
uv run pytest tests/test_walkie_checkpoint.py tests/test_walkie_optim.py tests/test_walkie_schedule.py -q
```

通过标准：全部通过，确认本次后训练改动没有破坏原有 Walkie checkpoint / optimizer / schedule。

## 4. Tokenizer alias 验证

如果正式采用 ChatML alias，先离线生成不扩词表的 tokenizer：

```bash
uv run python - <<'PY'
from posttrain.data.tokenizer_alias import write_tokenizer_aliases
import os
plan = write_tokenizer_aliases(
  os.environ['TOKENIZER'],
  os.path.join(os.environ['ROOT'], 'data/cache/walkie_code/tokenizer.chatml.json'),
    expected_vocab_size=65536,
)
print(plan)
PY
```

通过标准：`requires_fallback=False`，输出 tokenizer 的 vocab size 仍为 65536。若为 `True`，不要强行 alias，训练命令使用 `data.template=plain_eot`。

## 5. HF/vLLM 导出 smoke

```bash
uv run python - <<'PY'
from posttrain.utils.hf_export import export_walkie_to_hf
import os
export_walkie_to_hf(
  os.environ['CKPT'],
  os.path.join(os.environ['ROOT'], 'runs/posttrain_hf_export'),
  tokenizer_path=os.environ['TOKENIZER'],
)
print('export ok')
PY

uv run --extra posttrain python - <<'PY'
from vllm import LLM, SamplingParams
import os
llm = LLM(model=os.path.join(os.environ['ROOT'], 'runs/posttrain_hf_export'), tensor_parallel_size=1, dtype='auto')
out = llm.generate(['def add(a, b):'], SamplingParams(max_tokens=16, temperature=0.0))
print(out[0].outputs[0].text)
PY
```

通过标准：导出目录包含 `config.json`、`tokenizer.json`、`model.safetensors` 或 `pytorch_model.bin`；vLLM 能加载并生成非空文本。

## 6. Sandbox 联调

先启动 MultiModal-Jupyter-Sandbox Docker 镜像，并确认 `/run_jupyter` 可访问：

```bash
uv run python - <<'PY'
import asyncio
from posttrain.sandbox.jupyter_client import JupyterSandboxClient
import os

async def main():
    client = JupyterSandboxClient([os.environ['SANDBOX_URL']], timeout=10)
    result = await client.run_code("print('ALL TESTS PASSED')")
    print(result)

asyncio.run(main())
PY
```

通过标准：`status='success'`，`stdout` 包含 `ALL TESTS PASSED`，无异常 stderr。

## 7. SFT 单步 smoke

```bash
uv run python -m train.walkie_sft \
  --config configs/train/sft_walkie.yaml \
  --init-from "$CKPT" \
  data.paths=[$SFT_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  train.out_dir=runs/posttrain_sft_smoke \
  train.total_steps=1 \
  train.batch_size=1 \
  train.grad_accum_steps=1 \
  train.ckpt_interval=1 \
  distributed.backend=none
```

通过标准：打印 `[walkie/sft]` 日志或正常结束，并写出 `runs/posttrain_sft_smoke/latest.pt`。若显存紧，保持 `batch_size=1`，不要开 DDP。

## 8. RL fake rollout 单步 smoke

这个测试不启动 vLLM，只验证 actor/ref/logprob/reward/checkpoint 主链路。

```bash
uv run python -m train.walkie_rl \
  --config configs/train/rl_walkie_grpo.yaml \
  --init-from "$CKPT" \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=fake \
  train.out_dir=runs/posttrain_grpo_fake_smoke \
  train.total_steps=1 \
  train.ckpt_interval=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  distributed.backend=none
```

通过标准：打印 `[walkie/rl]` 日志，写出 `latest.pt`。若 OOM，加 `ref.offload=cpu`。

## 9. RL + vLLM + sandbox 单步 smoke

单卡服务器建议先用极小 rollout 参数：

```bash
uv run --extra posttrain python -m train.walkie_rl \
  --config configs/train/rl_walkie_grpo.yaml \
  --init-from "$CKPT" \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=vllm \
  rollout.tensor_parallel_size=1 \
  rollout.export_dir=runs/posttrain_grpo_vllm_smoke/hf_export \
  sandbox.enabled=true \
  sandbox.base_urls=[$SANDBOX_URL] \
  train.out_dir=runs/posttrain_grpo_vllm_smoke \
  train.total_steps=1 \
  train.ckpt_interval=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  rl.max_completion_length=128 \
  distributed.backend=none
```

通过标准：vLLM 成功加载导出目录，sandbox 返回 reward metadata，训练写出 `latest.pt`。若 OOM，按顺序尝试：`ref.offload=cpu`、`rl.num_generations=1`、`rl.max_completion_length=64`、关闭 sandbox 先测 vLLM。

## 10. Resume 验证

在第 8 或第 9 步生成 `latest.pt` 后继续跑 1 步：

```bash
uv run python -m train.walkie_rl \
  --config configs/train/rl_walkie_grpo.yaml \
  --resume runs/posttrain_grpo_fake_smoke/latest.pt \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=fake \
  train.out_dir=runs/posttrain_grpo_fake_smoke \
  train.total_steps=2 \
  train.ckpt_interval=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  distributed.backend=none
```

通过标准：从 `step=1` 继续到 `step=2`，不重新从头初始化优化器；checkpoint extra 中包含 `data_state.prompt_cursor` 和 `rollout_stats`。

## 11. DAPO 单步 smoke

```bash
uv run python -m train.walkie_rl \
  --config configs/train/rl_walkie_dapo.yaml \
  --init-from "$CKPT" \
  data.paths=[$RL_JSONL] \
  data.tokenizer_path=$TOKENIZER \
  data.template=plain_eot \
  rollout.backend=fake \
  train.out_dir=runs/posttrain_dapo_fake_smoke \
  train.total_steps=1 \
  train.ckpt_interval=1 \
  rl.prompt_batch_size=1 \
  rl.num_generations=2 \
  distributed.backend=none
```

通过标准：日志中 `algorithm=dapo`，stats 包含 `loss_normalization`，checkpoint 正常写出。

## 12. 本地代码 Bench 评测

先将 Walkie ckpt 导出成 HF/vLLM 目录，并确认 sandbox 的 `/run_jupyter` 可访问。四个本地评测集位于 `data/bench`，默认使用简单 `user:` / `assistant:` 文本 prompt，不需要新增特殊 token。

```bash
uv run --extra walkie python -m scripts.export_walkie_to_hf \
  --checkpoint "$ROOT/runs/walkie_code_0.5b/latest.pt" \
  --tokenizer "$ROOT/data/cache/walkie_code/tokenizer.json" \
  --output "$ROOT/runs/walkie_code_0.5b_vllm_hf"
```

先做小规模 smoke：

```bash
uv run --extra posttrain python -m scripts.evaluate_code_bench \
  --config configs/eval/walkie_code_bench.yaml \
  --model "$ROOT/runs/walkie_code_0.5b_vllm_hf" \
  --sandbox-url "$SANDBOX_URL" \
  --dataset openai_humaneval \
  --limit 2 \
  --n 1 \
  --pass-at 1 \
  --no-use-ray
```

smoke 通过后再跑四个评测集：

```bash
uv run --extra posttrain python -m scripts.evaluate_code_bench \
  --config configs/eval/walkie_code_bench.yaml \
  --model "$ROOT/runs/walkie_code_0.5b_vllm_hf" \
  --sandbox-url "$SANDBOX_URL" \
  --dataset all \
  --tensor-parallel-size 1 \
  --n 10 \
  --temperature 0.2 \
  --top-p 0.95 \
  --max-tokens 512 \
  --timeout 10 \
  --pass-at 1,5,10
```

单卡服务器如果 Ray 资源紧张，保持 `--no-use-ray`，脚本会使用 async HTTP 并发请求 sandbox。输出：

- `runs/eval/walkie_code_0.5b/<dataset>/results.jsonl`：每个 completion 的代码、stdout/stderr、是否通过。
- `runs/eval/walkie_code_0.5b/<dataset>/summary.json`：`num_tasks`、`pass@1`、`pass@5`、`pass@10`。
- `runs/eval/walkie_code_0.5b/summary.json`：四个数据集汇总。

## 13. 必须记录的结果

智能体完成服务器验证后，请在回复中记录：

- GPU 型号与显存。
- `torch`、`vllm`、`transformers` 版本。
- 使用的 ckpt 路径与 tokenizer 路径。
- 每个测试命令的 exit code。
- SFT/RL smoke 的最后 5 行日志。
- 是否启用了 `ref.offload=cpu`。
- 若失败，保留完整 traceback 和对应命令，不要只写“失败”。