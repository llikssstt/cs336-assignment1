# CS336 Transformer Language Model 项目文档

## 目录

1. [环境配置](#1-环境配置)
2. [项目结构](#2-项目结构)
3. [核心代码介绍](#3-核心代码介绍)
4. [训练流程](#4-训练流程)
5. [文本生成](#5-文本生成)

---

## 1. 环境配置

### 1.1 依赖安装

```bash
# 使用 uv 包管理器
uv sync

# 或使用 pip
pip install torch numpy einops wandb tiktoken
```

### 1.2 依赖列表

| 包名 | 用途 |
|-----|------|
| `torch` | 深度学习框架 |
| `numpy` | 数值计算、数据加载 |
| `einops` | 张量操作 |
| `wandb` | 训练日志记录 |
| `tiktoken` | Tokenizer 参考实现 |

---

## 2. 项目结构

```
cs336_basics/
├── train.py                    # 训练脚本
├── prepare_data.py             # 数据预处理脚本
├── demo_generate.py            # 文本生成演示
├── tokenizer/
│   ├── BPE_Tokenizer.py        # BPE Tokenizer 实现
│   └── BPE_Tokenizer_trainer.py # Tokenizer 训练
├── data_loader/
│   ├── data_loading.py         # 批次数据加载
│   └── checkpointing.py        # 检查点保存/加载
└── transfomer/
    ├── transformer_lm.py       # Transformer LM 主模型
    ├── transfomer_block.py     # Transformer Block
    ├── multi_head_self_attention.py  # 多头自注意力
    ├── rope.py                 # 旋转位置编码 (RoPE)
    ├── rmsnorm.py              # RMSNorm 归一化
    ├── swiglu.py               # SwiGLU FFN
    ├── embedding.py            # Token Embedding
    ├── linear.py               # 线性层
    ├── softmax.py              # Softmax
    ├── Cross_entropy.py        # 交叉熵损失
    ├── AdamW.py                # AdamW 优化器
    ├── cosine.py               # Cosine 学习率调度
    ├── gradient_clipping.py    # 梯度裁剪
    └── generate.py             # 文本生成函数
```

---

## 3. 核心代码介绍

### 3.1 模型架构 (`transformer_lm.py`)

```python
class Transformer_LM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, 
                 num_layers, num_heads, d_ff, rope_theta=10000.0):
        # Token Embedding
        self.token_embeddings = Embedding(vocab_size, d_model)
        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransfomerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        # Final Layer Norm + Output Projection
        self.ln = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
```

**参数说明**：
- `vocab_size`: 词汇表大小
- `context_length`: 最大上下文长度
- `d_model`: 模型隐藏维度
- `num_layers`: Transformer 层数
- `num_heads`: 注意力头数
- `d_ff`: FFN 中间层维度

### 3.2 多头自注意力 (`multi_head_self_attention.py`)

实现了带有 **RoPE (Rotary Position Embedding)** 的因果自注意力：

```python
class MHA(nn.Module):
    def forward(self, in_features, token_positions=None):
        # Q, K, V 投影
        q, k, v = self.q_w(x), self.k_w(x), self.v_w(x)
        
        # 应用 RoPE
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # 因果 mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # Scaled dot-product attention
        out = scaled_dot_product_attention(q, k, v, mask)
        return self.combine(out)
```

### 3.3 SwiGLU FFN (`swiglu.py`)

```python
class SwiGLU(nn.Module):
    def forward(self, x):
        # gate = silu(xW1) * (xW2)
        gate = F.silu(self.w1(x)) * self.w2(x)
        return self.w3(gate)
```

### 3.4 AdamW 优化器 (`AdamW.py`)

自定义实现的 AdamW，支持权重衰减解耦：

```
m = beta1 * m + (1-beta1) * grad
v = beta2 * v + (1-beta2) * grad^2
theta = theta - lr * m / (sqrt(v) + eps) - lr * lambda * theta
```

### 3.5 学习率调度 (`cosine.py`)

Cosine 退火 + Warmup：

```python
def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:           # Warmup
        return t * a_max / T_w
    elif t <= T_c:        # Cosine decay
        return a_min + 0.5 * (1 + cos(pi*(t-T_w)/(T_c-T_w))) * (a_max - a_min)
    else:                 # Post-decay
        return a_min
```

---

## 4. 训练流程

### 4.1 准备数据

**Step 1**: 训练 BPE Tokenizer（如已有可跳过）

```bash
python -m cs336_basics.tokenizer.train_bpe_tinystories
```

**Step 2**: 将文本转换为 token IDs

```bash
python -m cs336_basics.prepare_data \
    --tokenizer tinystories_bpe.pkl \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/train.npy
```

### 4.2 启动训练

```bash
uv run python -m cs336_basics.train \
    --train_data /mnt/d/cs336/data/tinystories_bpe.npy \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --batch_size 256 \
    --max_iters 1000 \
    --warmup_iters 100 \
    --lr 1e-3 \
    --min_lr 1e-4 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --log_interval 50 \
    --eval_interval 500 \
    --checkpoint_interval 1000 \
    --checkpoint_dir ./checkpoints \
    --wandb_project tinystories-lm
```

### 4.3 训练参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--lr` | 3e-4 | 最大学习率 |
| `--min_lr` | 3e-5 | 最小学习率 |
| `--warmup_iters` | 1000 | Warmup 迭代数 |
| `--weight_decay` | 0.1 | 权重衰减 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `--batch_size` | 32 | 批次大小 |
| `--context_length` | 256 | 上下文长度 |

### 4.4 从检查点恢复

```bash
python -m cs336_basics.train \
    --resume ./checkpoints/checkpoint_5000.pt \
    # ... 其他参数
```

---

## 5. 文本生成

### 5.1 生成参数

| 参数 | 说明 | 推荐值 |
|-----|------|-------|
| `--temperature` | 控制随机性，越低越确定 | 0.7-1.0 |
| `--top_p` | Nucleus 采样阈值 | 0.9-0.95 |
| `--max_tokens` | 最大生成 token 数 | 256 |

### 5.2 运行生成

```bash
python -m cs336_basics.demo_generate \
    --checkpoint ./checkpoints/checkpoint_final.pt \
    --tokenizer ./tinystories_bpe.pkl \
    --prompt "Once upon a time" \
    --temperature 0.8 \
    --top_p 0.9 \
    --max_tokens 256 \
    --vocab_size 10000 \
    --context_length 256 \
    --d_model 256 \
    --num_layers 4 \
    --num_heads 4 \
    --d_ff 512
```

### 5.3 采样策略

**Temperature Scaling**：
```
P(token) = softmax(logits / T)
```
- T < 1: 更确定性（偏向高概率 token）
- T > 1: 更随机（均匀分布）

**Top-p (Nucleus) Sampling**：
1. 按概率降序排列 tokens
2. 保留累积概率达到 p 的最小集合
3. 从该集合中采样

---

## 附录：常见问题

### Q: 训练时 CUDA out of memory？
A: 减小 `batch_size` 或 `context_length`

### Q: 生成不在 EOS 停止？
A: 确保 `--eos_token` 参数与训练数据中的 token 一致

### Q: Loss 不下降？
A: 检查学习率是否过大或过小，尝试调整 `--lr` 参数

### Q: 生成文本重复？
A: 尝试提高 `--temperature` 或降低 `--top_p`
