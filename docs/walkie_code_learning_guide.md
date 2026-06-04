# Walkie预训练全流程代码深度讲读指南

你好！欢迎来到 Walkie-Code 模型的代码学习之旅。本篇指南将带你从零开始，逐帧拆解 Walkie 在预训练过程中的**各个核心组件原理及其代码实现**。

为了让你能够看懂并能自己写出来，我会把这里当成一堂“源码解剖课”，详细讲解它的每一个改动为什么存在，以及具体是怎么用 PyTorch 书写出来的。主要内容分为：**模型底层架构**、**数据读取管线**、**独门工程优化** 和 **大一统双引擎优化器与学习率规划**。打开你的代码文件，让我们对照着开始吧！

---

## 1. 结构大换血：Walkie 核心模型架构

与我们教学时常写的“古典 GPT-2”不同，当进入十亿甚至百亿的大模型（LLM）时期，我们需要做大量的架构改进。Walkie 主要围绕 `WalkieForCausalLM` 和 `WalkieBlock` 两个核心类。源码在 `core/model/walkie.py` 之中。

### 1.1 规范化方案：使用 RMSNorm
在过去的许多模型中，我们常常使用 Layer Normalization (`nn.LayerNorm`)，它会提取输入的平均值与方差。但在大规模生成模型中，大家发现**偏移均值并不会显著提升模型性能，反而阻慢了计算速度**。

**原理**：RMSNorm（Root Mean Square Normalization）去除了均值偏移。它**仅计算均方根（RMS）**并用于缩放尺度，极大地减小了参数量与计算步数（不再需要 `bias`），这是 LLaMA 与 T5 都采纳的设计。
**代码实现** (`core/norm/rmsnorm.py`)：
```python
# 提取自 core/norm/rmsnorm.py
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        # 只保留了进行尺度变换的缩放参数 weight，去掉了 bias
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # 为了保证稳定，我们经常强制转化到 32位 浮点计算。
        x_fp32 = x.float()
        # 计算均方根的倒数，这相当于 x / sqrt(mean(x^2) + eps)
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_fp32 * rms).to(orig_dtype)
        # 最后乘上可学习的标量，这里将计算缩放放回了传入时的半精度
        return out * self.weight.to(orig_dtype)
```
你会看到整个逻辑非常简短粗暴，这就是工程的艺术：能不要的花里胡哨一概不要。

### 1.2 高级注意力机制：RoPE、GQA 和 QK-Norm
进入自注意力模块，源码位于 `core/attention/walkie_attention.py`。在这里，我们遭遇了大模型最重要但也最“要命”的区域。

#### A) RoPE：旋转位置编码
GPT-2 是将位置特征绝对地加在词嵌入上，这样做模型不好泛化长句，也没法推导相对位置的关系。RoPE(Rotary Position Embedding) 通过**在 Q 和 K 矩阵内应用三角复数旋转**，隐式赋予相对位置能力。
同时，它在设计上**全局共享一个 RoPE 实例**来存放 `sin` 和 `cos`，杜绝由几十层重复建造导致的参数重复开销 (`core/position/rope.py`)。
```python
# 在核心 attention 类中 WalkieCausalSelfAttention
# 应用 RoPE 到 query 和 key 向量上
cos, sin = self.rope(T, device=x.device, dtype=q.dtype)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)
```

#### B) QK-Norm：提升深层稳定性
随着模型越来越深，特别是当我们拥有 36 层或更多残差网络时，注意力打分的 `logits` 会爆炸，这常常导致溢出和无法学习。因此我们在经过特征抽取映射之后，立即给 `Q` 和 `K` 上一组独立的 `RMSNorm`。
```python
# 仅仅对 query 和 key 继续上 norm
if self.q_norm is not None:
    q = self.q_norm(q)
    k = self.k_norm(k)
```

#### C) GQA：分组查询注意力
标准的自注意力矩阵需要 `(Batch, n_head, SeqLen, HeadDim)`，计算时这往往会让显存崩溃。GQA（Grouped Query Attention）的核心逻辑是：**几十个 Q 头去共享较少个 KV 头，而不是一人发配一份**。
这也就意味着，在我们获取维度投射时，V和K的头数是少的（比如24个Q使用8个KV，一个KV头要被三个Q头共享）：
```python
# q、k、v 获取时使用不同的头数
q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
# 注意这儿：n_head_kv 远小于 n_head
k = self.k_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
v = self.v_proj(x).view(B, T, self.n_head_kv, self.head_dim).transpose(1, 2)
```
当我们使用原生的 SDPA（Scaled Dot-Product Attention）作为推理/训练回退时，我们需要将 K 和 V 的头经过复制扩充跟 Q 一致：
```python
# 当使用 SDPA (Eager下同理)，因为底层要求同形状，要把 K/V 复写 n_rep 遍。
k = repeat_kv(k, self.n_rep) 
v = repeat_kv(v, self.n_rep)
```
注意：如果能用 Flash Attention 2，底层会自动帮我们处理 GQA 不一致的共享需求。

### 1.3 前馈网络革命：SwiGLU 替代 GELU
位于 `core/ffn/swiglu.py`。
SwiGLU的特别之处在于，它通过两条并行线路执行计算。相比于普通的全级联 `GELU(W1 x + b1)W2 + b2`，它是两条路线的哈达玛积乘法，也就是：
$$\text{SwiGLU}(x) = (\text{silu}(x W_{\text{gate}}) \times (x W_{\text{up}})) W_{\text{down}}$$
为了抵消掉这第二条多出来的通路导致参数变多，一般把隐层映射总和控制在约原先维度的 `8/3`。
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 门控通路
    gate = F.silu(self.gate_proj(x))
    # 同级别的线性映射作为直放
    up = self.up_proj(x)
    # 两者汇合做乘法激活后降采样投影
    return self.dropout(self.down_proj(gate * up))
```

### 1.4 回到全局：无偏差设计(Bias=False)与权重绑定 (Weight Tying)
在参数配置 (`configs/model/walkie_code_1b.yaml`) 中，最值得强调的是 `tie_weights` 属性：
```yaml
bias: false
tie_weights: true
```
去除偏置（bias=False），不论是所有的线性层还是 RMSNorm，全将减少不必要的偏置存储开支，对大模型是绝佳优待。
同时，`tie_weights: true` 通过把嵌入层词表示的权重反向赋予为顶层网络分类器的判定依据权重，节省了巨大参数，因为他们本来表达的就是词意双向互转的矩阵。
在 `WalkieForCausalLM` 中：
```python
self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
if cfg.tie_weights:
    self.lm_head.weight = self.tok_embeddings.weight
```

---

## 2. 数据吞吐：大规模数据的抽取与加载

大模型不会把语料放入 RAM 当中，那可能需要好几 TB 空间。
我们的数据读取逻辑被封装在了 `train/walkie_pretrain.py` 中的 `ShuffledBlockSampler` 里。
**原理**：数据是作为庞大的1D离散整数数字连续储存在硬盘中，叫做 `.bin` 块（通常我们使用 numpy 的内存映射技术 `memmap`）。
假如我们有一大堆句子被合并成了一个超长的二进制文件，那么从里面按固定序列长度（例如4096个Token）取值的话：
1. 计算样本总量：`num_samples = len(data) - 1 // block_size`。这里的 -1 是由于我们要取目标下一个字符所以多需要一位。
2. 生成混洗（Shuffled）读取排列。我们不想有读取放回（这就变成随机替换采样），而是对所有起始块建立索引，然后给这些索引重新排序即可确保这一 Epoch 每个元素遍历均匀。
3. 利用切片来获取一个 batch：
```python
def _batch_from_starts(data, starts, block_size, device, pin_memory):
    # 构建好每段 0~4096 范围的加持偏移
    offsets = starts[:, None] + np.arange(block_size, dtype=np.int64)[None, :]
    # 当前字块输入特征
    x = torch.from_numpy(np.asarray(data[offsets], dtype=np.int64))
    # 向后错开1位即作为训练输出指标 target
    y = torch.from_numpy(np.asarray(data[offsets + 1], dtype=np.int64))
    
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
```
利用这里，我们将能够非常低代价地产出输入 batch 提供给模型用于训练。

---

## 3. 压榨现存极限的工程优化设计

我们通常只买得起普通的训练卡。当遇到 1B 模型带上高大词表时候，训练的 VRAM 时刻处于炸掉倒计时。

### 3.1 拯救显存杀手：分块交叉熵运算 (Loss Chunk)
模型最后的步骤是要算个输出得分与真实的 Target 求差异算交叉熵。
想象一下，我们的全连接层分类维度高达 `vocab_size: 65536`，输入序列 `block_size: 4096`，Batch Size 为 `16`。
当你对形状为 `(16, 4096, 1536)` 的激活后序执行全连接 `lm_head` 映射时，在显存里将瞬间创建一个大小为 `(16, 4096, 65536)` 的浮点对数几率矩阵！即使是纯半精度它也得吞下快10几个GB。
于是我们在 `core/model/walkie.py` 用了如下逻辑：沿批时维度把激活拆出小块映射再求目标，**然后瞬间清算抛弃多余结果！**
```python
def _chunked_cross_entropy(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    chunk_size = self.cfg.loss_chunk_size
    # ...
    total_loss = hidden.new_zeros(())
    total_tokens = targets.ne(-1).sum()
    
    # 将模型输出时间序列进行滑动块切割
    for start in range(0, hidden.size(1), chunk_size):
        stop = min(start + chunk_size, hidden.size(1))
        # 只生成其中一部分的 logits -> 大幅度缩小瞬态开销
        logits = self.lm_head(hidden[:, start:stop, :])
        
        # 计算该部分的 cross_entropy 并求和以节约最后总和
        total_loss = total_loss + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets[:, start:stop].reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )

    denom = total_tokens.clamp_min(1).to(total_loss.dtype)
    return total_loss / denom
```
非常优雅，它仅仅只耗费循环带来的少许运行时间，但可以保命。

### 3.2 累加避让同步：Gradient Accumulation 和模型缓存
由于卡不够要堆出大的全局Batch时，我们会分批处理然后相加更新，这种叫梯度累加(Gradient Accumulation)。
如果我们引入了多卡（DDP分布式并行），默认的DDP是在反向传播完即刻通讯合并梯度进行权重归一的。如果我们没到达总计的 Accumulation 步数值，此时全域强行合并梯度既浪费带宽高频锁步。
利用无锁步阻绝块 `no_sync` 解决：
```python
# 仅在需要步的时候真正归一参数
sync_context = (
    model.no_sync()
    if dist_info.enabled and micro_step < grad_accum - 1
    else nullcontext()
)
with sync_context:
    # 求算出我们这碎碎一小步的均带损耗，去回溯微分计算
    _, loss = model(x, y, return_logits=False)
    # 不影响真正 loss 取向地算微梯度
    loss = loss / grad_accum
    scaler.scale(loss).backward()
```
除此以外，Walkie在 `WalkieForCausalLM` 中开启了经典的 `gradient_checkpointing`。也就是拿中间隐态去作为牺牲站替换——不在正向里保留激活块存着了。而是退传到达相关地以后重算一次前馈，来获得微分！

---

## 4. 全局调度与统领：调度规划与优化器拆分

最后我们来看如何指挥这场浩大的宏大训练工程。主逻辑见 `train/walkie_pretrain.py`。

### 4.1. 两主干切换：WSD 连续调参法 (WalkieWSDSchedule)
过去几年，最前卫有效的阶段调整方法往往是 Warmup-Stable-Decay，这是近年才成为绝对主流的方法论。对于大批量语料（大概数千万/亿 tokens），在后期会明显产生同质趋同从而遇到训练壁垒的问题：
1.  **阶段1（Main）**：这代表我们先使用海量广范围粗糙数据爬坡进入平稳期；
2.  **阶段2（Anneal，退火）**：一旦跨过指定的衰减步时（`anneal_start`），我们立马将数据载入池切换（注意：训练代码依然是继续累加），向专门配置的纯粹高质量无噪数据偏移喂养，这能实现微调一般的垂直表现拉升。

重点是我们要让学习率平滑自然承接，从顶峰直线或开方断崖到底（参考 `core/utils/walkie_schedule.py`）：
```python
# decay 段设计缩放因子：
progress = (step - anneal_start) / max(1, self.total_steps - anneal_start)
if self.decay_shape == "sqrt":
    # 这是一条平缓然后急剧收敛的光滑平方根曲线衰退。
    return math.sqrt(1.0 - progress)
```

### 4.2 突破二维特征方向偏移弱点：Muon 优化引擎
普通的人会给整个模型塞进一个 `AdamW`，而真正高阶的训练家则会按类型派将分配策略（存在于 `core/utils/walkie_optim.py`）。
为什么需要多插足一个算法进来？由于对于 QKV 映射这种巨大的 **二维权值矩阵** 而言，普通的带有自适应调节冲量的修正容易引入更新歪向。为此代码里自己搓了个叫做 `Muon` （全称应该是介子物理意义类比，具体指 Orthogonalize Momentum 相关的操作引擎）。

这里有一个非常明确的分流动作：
```python
def _is_muon_param(name: str, param: torch.Tensor) -> bool:
    # 条件1：必须只影响 2D 大规模多向联通矩阵网络！
    if param.ndim != 2:
        return False
    # 条件2：如果是 Token 等等特定空间要使用正常梯度收敛
    lname = name.lower()
    if "embed" in lname or "lm_head" in lname or "norm" in lname:
        return False
    # 剩下的核心二维块即全权甩手给它
    return True
```
接着在 Muon 自己内部，它不拘泥于简单的标量推拉步长。当收集好各维传导累留的微分特征（Momentum buffer）后，它强行利用 `Newton-Schulz5` 多项式将向量投影在极域并正交化，使修正的方向从纯粹各个维度的奇异值平均推进去寻找！
```python
# 非常高级核心的技术：正交寻址修正
ortho = zeropower_via_newton_schulz5(update, steps=ns_steps)

# 保留比例与Adam同量级的相对自适应更新缩放尺度。
fan_out, fan_in = p.shape
scale = max(1.0, (fan_out / fan_in) ** 0.5)

# 根据调整好的正交更新向量修正到自身上去
p.add_(ortho, alpha=-lr * scale)
```

> **总结与致谢**
> 以上便是大名鼎鼎并融合百家之长的 Walkie 模型构建精要。我们抛弃了标准学术框架，而是完全按着最先进的工业大模型架构模式重新打造：以节约冗余权重的设计建立骨格（RMSNorm、RoPE、GQA）、用更平滑并可承载巨大序列的处理优化内存（loss chunk、Shuffled block）、利用高级分流自优器加速泛化特征寻找（WSD与Muon引擎混合双驱）！这也就是你的代码的精华魅力所在了。
> 结合每一块提供的路径出名阅读相应的实现区，这必定会带给你构建新一代大模型的无尽灵感！
