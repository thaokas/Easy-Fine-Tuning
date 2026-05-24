# LoRA：低秩适配学习笔记

> 适用目标：理解如何以很少的可训练参数完成 SFT 或偏好训练，是个人复现大模型微调最直接的技术基础。  
> 资料核验日期：2026-05-24

## 1. 一句话理解

LoRA（Low-Rank Adaptation）冻结预训练模型权重，不直接训练完整权重改变量，而是假设任务适配所需的改变量可以用两个小的低秩矩阵近似表示，只训练这两个矩阵。

## 2. 数学直觉

全量微调会把一个线性层权重 `W_0` 更新为 `W_0 + ΔW`。LoRA 将增量写成：

$$
W = W_0 + \Delta W,\qquad
\Delta W = \frac{\alpha}{r}BA
$$

其中：

- `W_0 in R^(d x k)` 冻结；
- `A in R^(r x k)`、`B in R^(d x r)` 可训练；
- `r` 远小于 `d` 和 `k`，称为 rank；
- `alpha/r` 控制适配分支的缩放。

原来需要训练 `d*k` 个参数，现在只需要训练 `r*(d+k)` 个参数。例如 `d=k=4096, r=16` 时，单层由约 1678 万个可更新参数降为约 13 万个。

### 初始化

论文中的典型设置使 LoRA 分支在训练开始时贡献为零，例如一个矩阵随机初始化、另一个初始化为零。于是刚开始训练时模型行为等同于原基础模型，再逐步学习任务增量。

## 3. 它解决了什么问题

| 全量微调痛点 | LoRA 的应对 |
| --- | --- |
| 每个任务都需要保存整份模型 | 每个任务只保存小适配器 |
| 优化器状态和梯度显存昂贵 | 主权重冻结，可训练参数显著减少 |
| 多任务部署成本高 | 同一基础模型可切换多个 adapter |
| Adapter 可能增加推理层结构 | LoRA 权重可 merge 回主权重 |

需要特别区分：LoRA 减少的是**可训练参数及相关训练开销**，不是自动将基础模型所有激活显存消除；上下文长度、batch size 与模型前向仍影响显存。

## 4. 放在哪些层

原始 LoRA 论文重点研究了 Transformer attention 投影。实际使用中常见目标模块为：

```text
q_proj, v_proj                   最简配置
q_proj, k_proj, v_proj, o_proj   attention 全投影
gate_proj, up_proj, down_proj    再覆盖 MLP，容量更强但更贵
```

选层不应只背一个固定答案：应依据模型架构、任务难度、可训练预算和验证集表现比较。Hugging Face PEFT 支持通过 `target_modules` 配置目标模块，也提供将 LoRA 应用于线性层的便捷配置。

## 5. Merge、多个适配器与 QLoRA

### Merge

训练完成后，可把 `alpha/r * BA` 合入原权重。合并后不需要在推理中额外走 LoRA 支路，因此适合部署固定任务模型。若要动态切换任务，通常保留独立 adapter。

### QLoRA

QLoRA 不是“量化 LoRA 小矩阵”，而是把冻结的基础模型以 4-bit 形式加载，并在其上训练 LoRA adapter。Dettmers et al. 提出的关键组件包括 NF4、double quantization 与 paged optimizers，使大模型微调显存门槛显著降低。

### 变体阅读方向

| 方法 | 关注问题 |
| --- | --- |
| QLoRA | 显存更低的 LoRA 微调 |
| AdaLoRA | 在参数预算下动态分配不同层的 rank |
| DoRA | 将权重幅值和方向拆开，增强低秩更新表现 |
| rsLoRA | 在较高 rank 下更稳定的缩放策略 |

## 6. 与公开课内容相连的澄清

- LoRA 是参数高效微调方法（PEFT），它可以承载 SFT、DPO，甚至部分 RL 训练；它本身不是数据集或损失函数。
- “合并后没有 LoRA 支路引起的额外推理开销”成立；合并过程会生成完整权重，存储和量化部署仍要单独处理。
- rank 更大不等于效果单调变好，过拟合、预算和 target modules 均会影响结果。
- “LoRA 只适合风格、不适合知识”过于绝对。它能学习领域事实，但对于大规模、频繁变化、要求可溯源的知识，SFT/LoRA 通常不是最理想的单一方案。

## 7. 最小实践思路

下面是需要理解的 PEFT 配置形状，实际 model 和模块名应与所选模型匹配：

```python
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

建议实验：

1. 固定 SFT 数据集，比较 `r=8/16/32` 的验证表现和 adapter 大小。
2. 固定 rank，比较只覆盖 `q/v` 与覆盖全部 attention 投影。
3. 在显存有限时尝试 QLoRA，并记录训练吞吐与输出质量。
4. 测试未合并 adapter 与 merge 后模型生成结果是否一致到可接受误差。

## 8. 推荐阅读资料

### 必读

1. Hu et al., 2021/ICLR 2022, *LoRA: Low-Rank Adaptation of Large Language Models*。原始论文，重点阅读方法与实验中的 rank/目标矩阵分析。  
   链接：https://openreview.net/forum?id=nZeVKeeFYf9
2. Hugging Face PEFT, *LoRA* 官方概念指南。适合将矩阵直觉映射到实际配置、merge 和 LoRA 变体。  
   链接：https://huggingface.co/docs/peft/main/en/conceptual_guides/lora

### 扩展

3. Dettmers et al., 2023, *QLoRA: Efficient Finetuning of Quantized Language Models*。  
   链接：https://arxiv.org/abs/2305.14314
4. Zhang et al., 2023, *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning*。  
   链接：https://arxiv.org/abs/2303.10512
5. Liu et al., 2024, *DoRA: Weight-Decomposed Low-Rank Adaptation*。  
   链接：https://arxiv.org/abs/2402.09353

## 9. 阅读完成后的自测题

1. 为什么 `r*(d+k)` 远小于 `d*k` 时仍可能有效？
2. LoRA 的 `r` 与 `alpha` 分别在控制什么？
3. SFT + LoRA 与 DPO + LoRA 的相同点和不同点是什么？
4. QLoRA 中被 4-bit 量化的是哪部分参数？
5. 什么场景应保留 adapter 动态加载，而不是 merge？
