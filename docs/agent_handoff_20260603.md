# Walkie 训练/评测/强化学习交接文档（2026-06-03）

本文档用于把本次会话内已经确认、修改、验证过的内容完整交接给下一个 agent。

目标是让后续 agent 不需要重新翻聊天记录，就能知道：

- 这个仓库里 SFT、评测、RL 分别怎么跑。
- 双卡怎么正确使用。
- remote vLLM 的提速改动是什么、怎么用。
- 哪些目录是干净的，哪些目录已经处于半坏状态，不应继续复用。
- 本次已经修掉了哪些 bug，哪些问题仍然存在。

## 1. 本次会话的关键结论

### 1.1 SFT 数据使用量

- 配置文件 [configs/train/sft_walkie_kodcode_bench.yaml](configs/train/sft_walkie_kodcode_bench.yaml) 使用：
  - `batch_size=4`
  - `grad_accum_steps=8`
  - 双卡 DDP 时 `world_size=2`
- 因此每个 optimizer step 消耗样本数：`4 * 8 * 2 = 64`
- KodCode SFT 清洗后数据总量已核实为 `245932` 条。
- 之前的 `7680` 步 SFT 实际消费样本数：`7680 * 64 = 491520`
- 等价约 `1.9986` 个 epoch，基本把全量数据完整跑了近两遍。

### 1.2 200 step 与 300 step RL 指标对比

- run 目录： [runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter](runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter)
- 200 step full eval：
  - [runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/humaneval_eval/step_00000200_full/summary.json](runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/humaneval_eval/step_00000200_full/summary.json)
  - `pass@1 = 0.35365853658536583`
- 300 step full eval：
  - [runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/humaneval_eval/step_00000300_full/summary.json](runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/humaneval_eval/step_00000300_full/summary.json)
  - `pass@1 = 0.34146341463414637`
- 该回落幅度很小，更像正常波动，不像代码逻辑错误。
- 日志中未发现 NaN、OOM、NCCL mismatch、训练崩溃等硬错误。

### 1.3 remote vLLM 提速结论

- `sync_interval=1` 对当前 remote vLLM 路径过慢，实测主要瓶颈是每步 reload + CUDA graph capture。
- 本次已将 remote 路径改为优先使用：
  - `sync_interval=4`
  - `async_prefetch=true`
  - server 侧 `--enforce-eager`
- 另外已实现：
  - remote rollout client 支持多个 `server_urls`
  - 支持 `request_shards`
  - 支持 `max_concurrent_requests`
  - 训练侧 async prefetch 改为“当前 rollout 完成后立刻预取下一步生成，再跑当前 reward/update”，从而覆盖 remote GPU 空窗。

### 1.4 SFT 双卡与评测的结论

- 最初的 `scripts.run_sft_bench_loop --gpu 0,1` 并不会自动 `torchrun`，不能真正双卡。
- 现在已经修复：
  - [scripts/run_sft_bench_loop.py](scripts/run_sft_bench_loop.py) 会在 `--gpu 0,1` 且 `distributed.backend != none` 时自动生成：
    - `python -m torch.distributed.run --standalone --nproc_per_node=2 -m train.walkie_sft ...`
  - 新增 `--eval-gpu`，支持训练与评测分离到不同 GPU。
- 当前建议：
  - SFT 双卡训练：`--gpu 0,1`
  - 评测单独 GPU：`--eval-gpu 1`

### 1.5 SFT 数据目录崩溃根因与修复

- 训练在 3800 多步附近崩溃，报错：
  - `row must contain messages, prompt/response, or instruction/output fields`
- 根因不是主数据坏了，而是：
  - SFT 数据目录 [data/sft/kodcode_v1_sft_r1_bench_aligned](data/sft/kodcode_v1_sft_r1_bench_aligned)
  - 包含 `manifest.json`
  - 旧的 `iter_sft_rows()` 会把目录下所有 `*.json` 也当成训练数据读入
  - 跑到接近一轮末尾才读到 `manifest.json`，于是崩溃
- 本次已修复 [posttrain/data/sft_dataset.py](posttrain/data/sft_dataset.py)：
  - 目录展开时跳过 `manifest.json`
- 修复后完整验证：
  - 能正常遍历目录并 `normalize_messages`
  - 有效样本数仍为 `245932`

## 2. 本次实际改动

### 2.1 remote vLLM 客户端与训练循环

- [posttrain/rollout/vllm_engine.py](posttrain/rollout/vllm_engine.py)
  - `RemoteVLLMRolloutEngine` 现在支持：
    - 单个 `server_url`
    - 多个 `server_urls`
    - `request_shards`
    - `max_concurrent_requests`
  - `reload()` 可并发发往多个 replica
  - `generate()` 可把 prompt batch 分片到多个 server，并 remap `prompt_index`

- [train/walkie_rl.py](train/walkie_rl.py)
  - async prefetch 从“生成+reward 一起预取”改为“只预取下一步 rollout 生成”
  - 当前步 rollout 完成后，立即发下一步生成；当前步 reward 和 update 再与之重叠
  - remote vLLM 构建函数支持：
    - `rollout.server_urls`
    - `rollout.request_shards`
    - `rollout.max_concurrent_requests`
  - 处理了 OmegaConf `ListConfig` 到 URL 列表的展开问题

- [configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml](configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml)
  - 已加入：
    - `sync_interval: 4`
    - `async_prefetch: true`
    - `server_urls: []`
    - `request_shards: 1`
    - `max_concurrent_requests: 1`

### 2.2 SFT 双卡 loop 与脚本

- [scripts/run_sft_bench_loop.py](scripts/run_sft_bench_loop.py)
  - 新增 `--eval-gpu`
  - 新增双卡 train command 自动生成：
    - `visible_gpu_count()`
    - `effective_distributed_backend()`
    - `build_train_command(..., nproc_per_node=...)`
  - 训练与评测分别使用：
    - `train_env[CUDA_VISIBLE_DEVICES]=args.gpu`
    - `eval_env[CUDA_VISIBLE_DEVICES]=args.eval_gpu or args.gpu`

- [scripts/run_sft_train_loop.sh](scripts/run_sft_train_loop.sh)
  - 已改为通过 `scripts.run_sft_bench_loop` 启动，而不是手写 `for STOP in ...`
  - 已加 `set -euo pipefail`
  - 已使用新的输出目录，避免复用旧的半坏目录

### 2.3 SFT 数据读取修复

- [posttrain/data/sft_dataset.py](posttrain/data/sft_dataset.py)
  - 目录展开时跳过 `manifest.json`

## 3. 已验证内容

### 3.1 编译与测试

本次会话中通过的验证包括：

- `uv run python -m py_compile posttrain/rollout/vllm_engine.py train/walkie_rl.py scripts/serve_vllm_rollout.py`
- `uv run python -m py_compile scripts/run_sft_bench_loop.py train/walkie_sft.py posttrain/data/sft_dataset.py`
- `uv run pytest tests/test_vllm_rollout.py tests/test_walkie_rl_loop.py tests/test_posttrain_rl_algorithms.py`
- `uv run pytest tests/test_walkie_rl_data.py tests/test_walkie_rl_loop.py`

全部通过。

### 3.2 dry-run 验证

已确认当前 SFT bench loop 的 dry-run 会打印真正的双卡命令，例如：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m train.walkie_sft ...
```

### 3.3 数据目录遍历验证

已验证：

```bash
for row in iter_sft_rows(['data/sft/kodcode_v1_sft_r1_bench_aligned']):
    normalize_messages(row)
```

可完整跑过并统计为 `245932` 条有效样本。

## 4. 当前重要目录与状态说明

### 4.1 推荐使用的干净目录

- SFT 新双卡全评目录：
  - [runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2](runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2)
  - 这是当前推荐的 10000 步 / 每 1000 步 full eval 的新目录

### 4.2 不建议继续使用的目录

- [runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full](runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full)
  - 原因：
    - 训练在 `~3840` 步因 `manifest.json` 问题崩过
    - step3000 评测在 GPU0 OOM
    - 手写 shell loop 没有 `set -e`，失败后继续导出/评测
    - 存在 `hf_exports/step_00004000` 这类目录名和真实 checkpoint 步数不一致的问题
    - 当前 `latest.pt` 实际是 step `3750`
  - 结论：不要在这个目录上继续 resume。

### 4.3 RL 相关目录

- DAPO in-process 历史 run：
  - [runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter](runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter)
- remote vLLM 训练配置输出目录（若新开跑可复用配置，但应使用新 out-dir）：
  - [configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml](configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml)

## 5. 如何进行评测

### 5.1 SFT bench loop 内置评测

使用 [scripts/run_sft_bench_loop.py](scripts/run_sft_bench_loop.py)。

如果要 10000 步、每 1000 步 full eval、双卡训练、评测跑 GPU1：

```bash
cd /data/ldyData/LLM-Walk-Through && uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_sft_bench_loop \
  --config configs/train/sft_walkie_kodcode_bench.yaml \
  --init-from runs/walkie_code_0.5b/latest.pt \
  --out-dir runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2 \
  --gpu 0,1 \
  --eval-gpu 1 \
  --segment-steps 1000 \
  --total-steps 10000 \
  --full-eval-interval 1000 \
  --skip-sandbox-smoke
```

说明：

- 训练会自动使用 `torch.distributed.run --nproc_per_node=2`
- 评测导出和 HF generate 会用 `CUDA_VISIBLE_DEVICES=1`
- 只要某次评测成功，结果会写：
  - `bench_eval/step_xxx_full/...`
  - `bench_history.jsonl`
  - 对应 SwanLab bench 指标

### 5.2 单独补评某个 checkpoint

先导出：

```bash
cd /data/ldyData/LLM-Walk-Through && CUDA_VISIBLE_DEVICES=1 uv run --extra walkie --extra flash --extra posttrain python -m scripts.export_walkie_to_hf \
  --checkpoint runs/某个目录/latest.pt \
  --output runs/某个目录/hf_exports/step_XXXXXXXX \
  --tokenizer data/cache/walkie_code/tokenizer.json
```

再评测：

```bash
cd /data/ldyData/LLM-Walk-Through && CUDA_VISIBLE_DEVICES=1 uv run --extra walkie --extra flash --extra posttrain python -m scripts.evaluate_code_bench \
  --model runs/某个目录/hf_exports/step_XXXXXXXX \
  --backend hf \
  --device cuda:0 \
  --attn-implementation flash_attention_2 \
  --dtype bf16 \
  --batch-size 1024 \
  --dataset all \
  --bench-root data/bench \
  --output runs/某个目录/bench_eval/step_XXXXXXXX_full \
  --prompt-style plain_dialog \
  --max-tokens 512 \
  --temperature 0.2 \
  --top-p 0.95 \
  --n 1 \
  --pass-at 1 \
  --timeout 10.0 \
  --max-concurrency 32 \
  --sandbox-url http://127.0.0.1:18901 \
  --skip-sandbox-smoke
```

注意：`--device cuda:0` 是相对当前 `CUDA_VISIBLE_DEVICES` 的局部编号。如果你设置 `CUDA_VISIBLE_DEVICES=1`，那么进程内的 `cuda:0` 实际映射到物理 GPU1。

## 6. 如何进行 SFT

### 6.1 标准 SFT 配置

- 配置文件： [configs/train/sft_walkie_kodcode_bench.yaml](configs/train/sft_walkie_kodcode_bench.yaml)

关键参数：

- `train.batch_size = 4`
- `train.grad_accum_steps = 8`
- `train.total_steps = 7680`（配置默认值，可被 CLI 覆盖）
- `distributed.backend = ddp`

### 6.2 推荐 SFT 启动方式

直接跑脚本：

```bash
cd /data/ldyData/LLM-Walk-Through
bash scripts/run_sft_train_loop.sh
```

该脚本当前等价于第 5.1 节的命令。

### 6.3 为什么现在推荐 bench loop 而不是手写 shell loop

因为内置 loop 有这些好处：

- 训练/评测成功后才写 `bench_history.jsonl`
- 评测结果可以走 SwanLab logging
- `choose_checkpoint_args()` 逻辑稳定，不容易手写错 resume/init-from
- Python loop 抛异常后会停止，不会像旧 shell loop 一样失败后继续写错目录

## 7. 如何进行强化学习（DAPO / remote vLLM）

### 7.1 从 SFT 模型重新开始 DAPO

当前更推荐使用 remote vLLM 路径，而不是双卡 DDP in-process RL。

原因：

- 本次提速改动主要在 `remote_vllm`
- `async_prefetch` 当前只在单训练进程 + remote rollout 路径启用
- 更合理的两卡分工是：
  - GPU0：训练 actor/update
  - GPU1：一个或两个 vLLM rollout server

### 7.2 启动两个 remote vLLM server

终端 1：

```bash
cd /data/ldyData/LLM-Walk-Through

CUDA_VISIBLE_DEVICES=1 uv run --extra posttrain python scripts/serve_vllm_rollout.py \
  --host 127.0.0.1 \
  --port 18080 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 4096 \
  --enforce-eager
```

终端 2：

```bash
cd /data/ldyData/LLM-Walk-Through

CUDA_VISIBLE_DEVICES=1 uv run --extra posttrain python scripts/serve_vllm_rollout.py \
  --host 127.0.0.1 \
  --port 18081 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 4096 \
  --enforce-eager
```

如果显存不够，可把 `0.32` 下调到 `0.28`。

### 7.3 从 SFT 模型重新开始 remote_vllm DAPO

```bash
cd /data/ldyData/LLM-Walk-Through

uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_rl_humaneval_loop \
  --config configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml \
  --init-from runs/walkie_code_0.5b_sft_kodcode_bench/latest.pt \
  --out-dir runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft \
  --gpu 0 \
  --segment-steps 100 \
  --total-steps 1500 \
  --eval-interval 100 \
  --train-override distributed.backend=none \
  --train-override rollout.server_url=http://127.0.0.1:18080,http://127.0.0.1:18081 \
  --train-override rollout.export_dir=runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft/hf_rollout_exports \
  --train-override rollout.sync_interval=4 \
  --train-override rollout.async_prefetch=true \
  --train-override rollout.request_shards=2 \
  --train-override rollout.max_concurrent_requests=2 \
  --train-override rollout.request_timeout=300.0 \
  --train-override rollout.reload_timeout=600.0
```

如果要用新的 SFT 10000-step 跑完后模型作为起点，则把 `--init-from` 改成对应 SFT 新目录的 `latest.pt`。

## 8. 如何使用双卡并行

### 8.1 SFT 双卡

- 训练命令必须出现：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m train.walkie_sft ...
```

- 仅仅 `CUDA_VISIBLE_DEVICES=0,1 python -m train.walkie_sft ...` 不等于双卡。
- 现在 [scripts/run_sft_bench_loop.py](scripts/run_sft_bench_loop.py) 会自动处理这一点。

### 8.2 RL 双卡的正确理解

对于当前最推荐的 remote_vllm 方案：

- 不建议用 `--gpu 0,1` 去做 DDP actor 训练
- 更建议：
  - GPU0：训练
  - GPU1：1~2 个 vLLM rollout server

因为当前 async rollout prefetch 的收益主要在单训练进程 remote rollout 路径。

### 8.3 评测与训练分 GPU

SFT/评测的标准建议：

- 训练：`--gpu 0,1`
- 评测：`--eval-gpu 1`

这样评测不会去抢 GPU0 上那些常驻第三方进程的显存。

## 9. 目录规范与使用建议

### 9.1 顶层目录含义

- [configs](configs)
  - 各类训练/模型配置
- [train](train)
  - 训练入口
- [scripts](scripts)
  - loop、导出、评测、server 启动等辅助脚本
- [posttrain](posttrain)
  - 后训练数据、评测、rollout、reward 等逻辑
- [runs](runs)
  - 训练输出、checkpoint、导出、评测结果
- [swanlog](swanlog)
  - SwanLab 本地记录
- [docs](docs)
  - 文档与 handoff

### 9.2 runs 目录规范建议

建议遵循：

- SFT：
  - `runs/walkie_code_0.5b_sft_*`
- RL in-process：
  - `runs/walkie_code_0.5b_dapo_*_inprocess_*`
- RL remote vLLM：
  - `runs/walkie_code_0.5b_dapo_*_remote_vllm_*`
- full eval 导出：
  - `hf_exports/step_XXXXXXXX`
- full eval 输出：
  - `bench_eval/step_XXXXXXXX_full`
  - RL HumanEval 为 `humaneval_eval/step_XXXXXXXX_full`

### 9.3 输出目录使用原则

- 只要你要“从头重新跑”，必须换一个全新的 `out-dir`
- 只要目录里已有 `latest.pt`，loop 脚本就会优先 `--resume`
- 不要在半坏目录上继续试探性恢复，否则后续评测点与 checkpoint 语义容易错位

## 10. SwanLab 相关说明

### 10.1 为什么会出现“没有上传评测日志”

本次实际情况分两类：

- RL 旧 run：
  - checkpoint 中的 `swanlab_run_id` 正常存在
  - resume 行为可恢复 run id
- SFT 10000/e1000 旧 run：
  - 不是“评测成功但没上传”
  - 而是评测本身没有成功完成，所以不会有 `bench_history.jsonl`，也就不会有对应的 SwanLab bench 指标

### 10.2 已观察到的 SwanLab 异常

- 旧 RL run 中出现过：
  - `COLUMN error: ... too many 500 error responses`
  - 这更像 SwanLab 服务端偶发问题，不是训练主逻辑错误

### 10.3 bench loop 对 SwanLab 的依赖关系

- 只有在：
  - 训练段成功完成
  - export 成功
  - eval 成功
  - summary 成功解析
- 后，才会写 `bench_history.jsonl` 并调用 `log_bench_to_swanlab(...)`

## 11. 本次发现的历史失败案例

### 11.1 旧 SFT shell loop 问题

旧版 [scripts/run_sft_train_loop.sh](scripts/run_sft_train_loop.sh) 的问题有：

- 没有 `set -e`
- 训练失败后仍继续 export/eval
- 评测固定用 GPU0
- 在 GPU0 有大量其它进程时，full eval 易 OOM

这些问题已经在当前版本修掉。

### 11.2 step3000 full eval OOM

旧目录 step3000 full eval 失败栈显示：

- `torch.OutOfMemoryError`
- 评测执行于 GPU0
- GPU0 同时有多个外部 `python3` 进程长期占显存

因此当前推荐把评测迁到 GPU1。

### 11.3 训练接近一轮末尾才炸

这是因为 `manifest.json` 排在训练 shard 后面，旧实现直到读完 `train-00004.jsonl` 才碰到它。

## 12. 后续 agent 最值得优先注意的事项

1. 如果用户要继续 10000/e1000 SFT，直接使用：
   - [scripts/run_sft_train_loop.sh](scripts/run_sft_train_loop.sh)
   - 或第 5.1 节的等价命令
2. 不要继续使用：
   - [runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full](runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full)
3. 如果用户要从 SFT 接 RL，优先用第 7 节的 remote_vllm 方案，而不是双卡 DDP actor。
4. 如果用户说“评测日志没上传”，先判断是：
   - 评测没跑
   - 评测跑了但没落盘
   - 落盘了但 SwanLab API 出错
5. 如果出现训练在高步数才报 schema 错误，优先检查数据目录元数据文件是否被误读，而不是先怀疑主 shard 损坏。

## 13. 建议下个 agent 第一眼检查的文件

- [scripts/run_sft_bench_loop.py](scripts/run_sft_bench_loop.py)
- [scripts/run_sft_train_loop.sh](scripts/run_sft_train_loop.sh)
- [posttrain/data/sft_dataset.py](posttrain/data/sft_dataset.py)
- [train/walkie_sft.py](train/walkie_sft.py)
- [train/walkie_rl.py](train/walkie_rl.py)
- [posttrain/rollout/vllm_engine.py](posttrain/rollout/vllm_engine.py)
- [configs/train/sft_walkie_kodcode_bench.yaml](configs/train/sft_walkie_kodcode_bench.yaml)
- [configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml](configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml)

## 14. 一句话交接摘要

当前仓库已经具备：

- 可用的双卡 SFT bench loop
- 可用的单训练进程 + 双 replica remote vLLM DAPO 路径
- 可分离训练 GPU 和评测 GPU
- 已修复的 SFT manifest 误读问题

后续工作重点应放在：

- 用新目录重新跑干净的 10000/e1000 SFT
- 在需要时从该 SFT 模型切到 remote_vllm DAPO
- 避免继续复用旧的半坏 run 目录# Walkie 训练/评测/RL 交接文档（2026-06-03）

本文档用于把本次会话中已经确认、实现、修复、验证过的重要内容完整交接给下一个 agent。

目标是让后续 agent 不需要翻聊天记录，也能直接知道：

- 当前仓库在 SFT、RL、评测、remote vLLM 上已经做了什么
- 哪些命令是可用的
- 哪些目录可以继续用，哪些目录已经半损坏不建议继续用
- 双卡应该怎么用
- 这次踩过的坑和已修复问题是什么

---

## 1. 当前结论概览

### 1.1 SFT

- KodCode SFT 清洗后实际训练样本数是 `245,932` 条。
- 之前的 `7680` 步 SFT 配置：
  - `batch_size=4`
  - `grad_accum_steps=8`
  - `world_size=2`
  - 每步样本消费量是 `4 x 8 x 2 = 64`
  - 总样本消费量是 `7680 x 64 = 491,520`
  - 等价于把 `245,932` 条数据几乎完整跑了两遍。

### 1.2 RL / DAPO

- `remote_vllm` 路径已经做过提速改动。
- 主要收益来自：
  - `sync_interval >= 4`
  - `rollout.async_prefetch=true`
  - remote vLLM server 使用 `--enforce-eager`
  - 支持 `server_urls + request_shards + max_concurrent_requests` 的多 replica rollout 分片
- 但注意：当前 `async_prefetch` 只在 **非 DDP 单训练进程** 的 `remote_vllm` 路径启用。
- 所以“训练双卡 + remote vLLM 加速”不是 `--gpu 0,1` 这种 DDP 方案，而是：
  - GPU0：训练
  - GPU1：1 到 2 个 remote vLLM server

### 1.3 评测

- 旧的手写 SFT shell loop 里，评测确实启动了，但没有成功完成。
- 不是“评测跑完但 SwanLab 没上传”，而是评测本身 OOM 或被后续错误状态污染，最终没有形成完整 `bench_history.jsonl`。
- 原因包括：
  - 评测固定跑在 GPU0，而 GPU0 上长期有其他进程占显存
  - shell loop 没有 `set -e`，训练或评测失败后仍继续向下执行
  - 导出目录和真实 checkpoint step 发生错位

### 1.4 双卡

- 之前原始的 `scripts.run_sft_bench_loop` 不会根据 `--gpu 0,1` 自动包 `torch.distributed.run`，所以之前那条命令不是真双卡。
- 现在这个问题已经修复。
- 当前 `scripts.run_sft_bench_loop` 已支持：
  - `--gpu 0,1` 自动生成 `torch.distributed.run --standalone --nproc_per_node=2`
  - `--eval-gpu 1`，可以把评测放到指定 GPU 上执行

---

## 2. 本次会话已完成的代码修改

### 2.1 remote vLLM client 并发与分片

修改文件：

- `posttrain/rollout/vllm_engine.py`

已实现：

- `RemoteVLLMRolloutEngine` 支持多 server URL
- 支持 `request_shards`
- 支持 `max_concurrent_requests`
- `reload()` 会对多个 replica 并发 reload
- `generate()` 会把 prompts 分片发到多个 server，并重映射 `prompt_index`

对应配置入口已在：

- `configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml`

新增字段：

- `server_urls`
- `request_shards`
- `max_concurrent_requests`

### 2.2 remote vLLM async prefetch 改进

修改文件：

- `train/walkie_rl.py`

改动内容：

- 之前的 async prefetch 是“下一步 rollout + reward 一起预取”
- 现在改为“当前 rollout 一结束就立刻预取下一步 rollout 生成，reward 留在本步做”

效果：

- 可以让 remote vLLM server GPU 和当前 reward/policy update 重叠，减少空窗期。

### 2.3 SFT 双卡 loop 修复

修改文件：

- `scripts/run_sft_bench_loop.py`
- `scripts/run_sft_train_loop.sh`

已实现：

- `scripts.run_sft_bench_loop` 现在支持自动根据 `--gpu 0,1` 使用 `torch.distributed.run`
- 新增 `--eval-gpu`，使训练和评测可以用不同 GPU
- shell 脚本已改为调用新的 Python loop，而不是手写 for-loop
- shell 脚本加了 `set -euo pipefail`

### 2.4 SFT 数据读取崩溃修复

修改文件：

- `posttrain/data/sft_dataset.py`

问题：

- 数据目录 `data/sft/kodcode_v1_sft_r1_bench_aligned/` 下有 `manifest.json`
- 老逻辑会把目录下所有 `*.json` 也当训练数据读入
- 在一轮数据接近尾部时，读到 `manifest.json`，触发：
  - `row must contain messages, prompt/response, or instruction/output fields`

修复：

- 目录展开时跳过 `manifest.json`

验证结果：

- 当前目录可正常迭代 `245,932` 条有效样本
- 不再在一轮末尾因 metadata 文件崩溃

---

## 3. 关键目录与状态说明

### 3.1 推荐继续使用的目录

- `runs/walkie_code_0.5b_sft_kodcode_bench`
  - 历史 SFT 主目录
- `runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2`
  - 当前推荐的新 10k 步 SFT 目录
- `runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter`
  - 历史 in-process DAPO 主目录
- `runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft`
  - 推荐的新 remote-vLLM DAPO 目录名

### 3.2 不建议继续使用的目录

- `runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full`

原因：

- 这个目录经历过：
  - 非健壮 shell loop
  - step3000 eval OOM
  - step4000 训练失败后脚本仍继续 export/eval
- 已确认：
  - `latest.pt` 实际只有 `step=3750`
  - 但目录中已经出现了 `hf_exports/step_00004000`
  - `bench_history.jsonl` 缺失
- 继续复用会导致 checkpoint step 和评测目录错位。

### 3.3 重要数据目录

- `data/sft/kodcode_v1_sft_r1_bench_aligned`
  - 当前 KodCode SFT 训练目录
  - 文件包括：
    - `train-00000.jsonl` 到 `train-00004.jsonl`
    - `leakage_removed.jsonl`（空文件）
    - `manifest.json`（元数据，不应被当训练样本）

---

## 4. 如何做评测

### 4.1 SFT 分段评测

推荐入口：

- `scripts.run_sft_bench_loop`

推荐参数：

- `--segment-steps N`
- `--full-eval-interval N`
- `--eval-gpu 1`

如果希望“每 1000 步都做 full eval，无抽样评测”，直接用：

- `segment_steps=1000`
- `full_eval_interval=1000`

这样每段结束都会是 full eval，不会进入 smoke eval 分支。

### 4.2 单独评测命令

```bash
cd /data/ldyData/LLM-Walk-Through

CUDA_VISIBLE_DEVICES=1 uv run --extra walkie --extra flash --extra posttrain \
  python -m scripts.evaluate_code_bench \
  --model runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2/hf_exports/step_00001000 \
  --backend hf \
  --device cuda:0 \
  --attn-implementation flash_attention_2 \
  --dtype bf16 \
  --batch-size 1024 \
  --dataset all \
  --bench-root data/bench \
  --output runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2/bench_eval/step_00001000_full \
  --prompt-style plain_dialog \
  --max-tokens 512 \
  --temperature 0.2 \
  --top-p 0.95 \
  --n 1 \
  --pass-at 1 \
  --timeout 10.0 \
  --max-concurrency 32 \
  --sandbox-url http://127.0.0.1:18901 \
  --skip-sandbox-smoke
```

注意：

- 这里的 `--device cuda:0` 是相对于 `CUDA_VISIBLE_DEVICES=1` 的局部索引
- 所以这条命令实际会占物理 GPU1，不是物理 GPU0

### 4.3 评测结果文件

典型落盘位置：

- `runs/.../bench_eval/step_00001000_full/summary.json`
- `runs/.../bench_eval/step_00001000_full/<dataset>/summary.json`
- `runs/.../bench_history.jsonl`

如果没有 `bench_history.jsonl`，优先怀疑：

- 评测命令本身失败
- shell loop 在失败后未中止
- 不是优先怀疑 SwanLab 上传问题

---

## 5. 如何做 SFT

### 5.1 推荐的新 10k 步、双卡、每 1000 步 full eval 的命令

```bash
cd /data/ldyData/LLM-Walk-Through && uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_sft_bench_loop \
  --config configs/train/sft_walkie_kodcode_bench.yaml \
  --init-from runs/walkie_code_0.5b/latest.pt \
  --out-dir runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2 \
  --gpu 0,1 \
  --eval-gpu 1 \
  --segment-steps 1000 \
  --total-steps 10000 \
  --full-eval-interval 1000 \
  --skip-sandbox-smoke
```

含义：

- 训练：双卡 DDP
- 评测：放到 GPU1
- 每 1000 步停一次并做 full eval
- 继续运行时会按 `latest.pt` 自动 resume

### 5.2 对应 shell 脚本

可直接运行：

```bash
cd /data/ldyData/LLM-Walk-Through
bash scripts/run_sft_train_loop.sh
```

当前脚本已经切到干净目录：

- `runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2`

### 5.3 训练失败时优先检查

1. `bench_history.jsonl` 是否增长
2. `latest.pt` 的真实 step
3. `nvidia-smi` 看评测是否跑错 GPU
4. 是否还有其它常驻进程占据 GPU0
5. 是否误用旧目录 `..._full`

---

## 6. 如何做强化学习（DAPO / remote vLLM）

### 6.1 推荐结构

当前最快、也最符合本次改动设计的结构是：

- GPU0：训练进程
- GPU1：1 到 2 个 remote vLLM rollout server

不推荐：

- RL 训练直接 `--gpu 0,1` 做 DDP，同时又想启用 `remote_vllm async_prefetch`

原因：

- 当前 `async_prefetch` 只在非 DDP 单训练进程启用

### 6.2 启动 remote vLLM server

第一个 server：

```bash
cd /data/ldyData/LLM-Walk-Through

CUDA_VISIBLE_DEVICES=1 uv run --extra posttrain python scripts/serve_vllm_rollout.py \
  --host 127.0.0.1 \
  --port 18080 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 4096 \
  --enforce-eager
```

第二个 server：

```bash
cd /data/ldyData/LLM-Walk-Through

CUDA_VISIBLE_DEVICES=1 uv run --extra posttrain python scripts/serve_vllm_rollout.py \
  --host 127.0.0.1 \
  --port 18081 \
  --gpu-memory-utilization 0.32 \
  --max-model-len 4096 \
  --enforce-eager
```

如果 OOM：

- 把 `--gpu-memory-utilization 0.32` 降到 `0.28`

### 6.3 从 SFT 模型重新开始 DAPO

```bash
cd /data/ldyData/LLM-Walk-Through

uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_rl_humaneval_loop \
  --config configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml \
  --init-from runs/walkie_code_0.5b_sft_kodcode_bench/latest.pt \
  --out-dir runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft \
  --gpu 0 \
  --segment-steps 100 \
  --total-steps 1500 \
  --eval-interval 100 \
  --train-override distributed.backend=none \
  --train-override rollout.server_url=http://127.0.0.1:18080,http://127.0.0.1:18081 \
  --train-override rollout.export_dir=runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft/hf_rollout_exports \
  --train-override rollout.sync_interval=4 \
  --train-override rollout.async_prefetch=true \
  --train-override rollout.request_shards=2 \
  --train-override rollout.max_concurrent_requests=2 \
  --train-override rollout.request_timeout=300.0 \
  --train-override rollout.reload_timeout=600.0
```

如果要用新的 SFT 输出作为起点，只需替换：

- `--init-from runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2/latest.pt`

### 6.4 从已有 200 step checkpoint 恢复 DAPO，并恢复 scheduler + SwanLab

这件事之前已经验证过：

- 旧 checkpoint `runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/latest.pt`
  - `step = 200`
  - 有 `schedule`
  - 有 `swanlab_run_id = n9ujkz80zqa76uulzlk8d`
  - 有 `prompt_cursor = 1600`

必须用：

- `--out-dir` 指向已有 checkpoint 目录

不能只用：

- `--init-from latest.pt`

因为那样不会恢复 scheduler 和 SwanLab run state。

恢复命令：

```bash
cd /data/ldyData/LLM-Walk-Through

uv run --extra walkie --extra flash --extra posttrain python -m scripts.run_rl_humaneval_loop \
  --config configs/train/rl_walkie_dapo_kodcode_pass1_remote_vllm.yaml \
  --out-dir runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter \
  --gpu 1 \
  --segment-steps 200 \
  --total-steps 600 \
  --eval-interval 200 \
  --train-override rollout.server_url=http://127.0.0.1:18080 \
  --train-override rollout.export_dir=runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter/hf_rollout_exports \
  --train-override rollout.sync_interval=4 \
  --train-override rollout.async_prefetch=true \
  --train-override rollout.request_timeout=300.0 \
  --train-override rollout.reload_timeout=600.0
```

---

## 7. 双卡并行怎么用

### 7.1 SFT

现在正确的做法是：

- `scripts.run_sft_bench_loop --gpu 0,1`

因为脚本已经会自动转成：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 -m train.walkie_sft ...
```

评测建议再加：

- `--eval-gpu 1`

### 7.2 RL

当前推荐不是 DDP 双卡，而是“训练卡 + rollout 卡”分工：

- GPU0：训练
- GPU1：remote vLLM

原因见上：`async_prefetch` 不在 DDP 下启用。

### 7.3 如何判断是否真双卡

只看 `world_size=2` 不够。

应该同时确认：

1. 训练命令是否由 `torch.distributed.run --nproc_per_node=2` 启动
2. `train.walkie_sft` 日志中 `world_size=2`
3. `nvidia-smi` 上两张卡都出现对应训练进程显存占用

---

## 8. 目录规范与运行规范

### 8.1 runs 目录命名建议

- SFT：
  - `runs/walkie_code_0.5b_sft_kodcode_bench_10000_e1000_full_v2`
- DAPO in-process：
  - `runs/walkie_code_0.5b_dapo_kodcode_pass1_inprocess_execfilter`
- DAPO remote-vLLM：
  - `runs/walkie_code_0.5b_dapo_kodcode_pass1_remote_vllm_fresh_from_sft`

命名建议包含：

- 模型规模
- 训练阶段（sft / dapo / grpo）
- 数据集来源（kodcode / pass1）
- 是否 remote_vllm / inprocess
- 是否 fresh / resume

### 8.2 不要复用已损坏目录

如果出现以下任意情况，直接新建目录，不要硬续：

- `latest.pt` step 与 `hf_exports/step_xxx` 不一致
- `bench_history.jsonl` 缺失
- shell loop 训练/评测中途失败但仍继续
- SwanLab run id 和本地 checkpoint 不一致

### 8.3 shell 运行规范

自定义 shell loop 必须有：

```bash
set -euo pipefail
```

否则训练、导出、评测任何一步失败，都可能导致后续目录错标。

---

## 9. 本次确认过的旧问题与修复结论

### 9.1 step3000 评测为什么没有上传 SwanLab

不是“上传失败为主”，而是评测先 OOM 了。

tmux 已见到的错误：

- `torch.OutOfMemoryError`

当时评测命令：

- 固定跑在 GPU0

而 GPU0 本来就有其它常驻进程占显存，所以 full eval 没能完成，自然也不会有完整评测日志进入 `bench_history.jsonl` 再上传 SwanLab。

### 9.2 step4000 训练为什么失败

不是模型数值炸掉，而是数据读取碰到了 `manifest.json`。

错误：

- `row must contain messages, prompt/response, or instruction/output fields`

根因已经修复。

### 9.3 旧 shell loop 为什么继续往后跑

因为没有：

- `set -e`

所以：

- step4000 训练失败
- 仍继续 export
- 仍继续 evaluate
- 最终出现目录名与真实 step 不一致

---

## 10. 验证记录

本次会话已验证：

- `scripts.run_sft_bench_loop` dry-run 会生成：
  - `python -m torch.distributed.run --standalone --nproc_per_node=2 -m train.walkie_sft ...`
- `posttrain/data/sft_dataset.py` 修复后可完整遍历：
  - `245932` 条有效样本
- `py_compile` 通过
- 以下测试通过：
  - `tests/test_walkie_rl_loop.py`
  - `tests/test_walkie_rl_data.py`
  - `tests/test_vllm_rollout.py`
  - `tests/test_posttrain_rl_algorithms.py`

---

## 11. 后续 agent 的建议行动顺序

### 如果目标是继续 SFT

1. 不要碰旧目录 `..._full`
2. 直接启动 `..._full_v2`
3. 优先看：
   - `bench_history.jsonl`
   - `latest.pt step`
   - `bench_eval/step_xxx_full/summary.json`

### 如果目标是继续 RL

1. 先起 remote vLLM server
2. 用单训练进程 + remote_vllm 路径
3. 不要把 DDP 双卡和 `async_prefetch` 混在一起期待同样收益

### 如果目标是排查性能

优先看：

1. `sync_interval`
2. 是否使用 `--enforce-eager`
3. 是否启用了 `request_shards/max_concurrent_requests`
4. 训练卡和 rollout 卡是否分离

---

## 12. 一句话总结

当前仓库已经具备：

- 可用的双卡 SFT 分段训练/全量评测 loop
- 可用的 remote-vLLM DAPO 提速路径
- 可恢复 scheduler/SwanLab 的 RL resume 方式

但必须使用本次修复后的入口和新目录，不能继续沿用旧的半坏 SFT 输出目录。