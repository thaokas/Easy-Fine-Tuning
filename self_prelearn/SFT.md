# SFT：监督微调学习笔记

> 适用目标：理解大模型从“续写模型”转向“能按指令回答”的第一步，并为 LoRA、DPO、PPO/GRPO 后训练建立基础。  
> 资料核验日期：2026-05-24

## 1. 一句话理解

SFT（Supervised Fine-Tuning，监督微调）是在已经预训练好的语言模型上，用高质量的 `(输入/指令, 期望回答)` 样本继续做监督学习，让模型学习目标任务的回答方式、格式、语气和行为边界。

在对话模型中，它通常是后训练管线的起点：

```text
预训练模型 -> SFT 指令模型 -> 偏好对齐（DPO 或 RLHF/PPO）-> 可选的推理强化（GRPO 等）
```

注意：这是一条常见工程路线，不是逻辑上的硬性规定。例如 DeepSeek-R1-Zero 探索了不先做 SFT、直接进行强化学习的路线。

## 2. 核心训练目标

给定指令或上下文 `x` 和目标回答 token 序列 `y=(y_1,...,y_T)`，SFT 最小化目标回答的负对数似然：

$$
\mathcal{L}_{SFT}(\theta) =
-\sum_{t=1}^{T}\log \pi_\theta(y_t \mid x,y_{<t})
$$

直觉是：训练数据给模型一份“标准答案”，模型逐 token 学会提高标准答案出现的概率。

### Completion-only loss

聊天数据经模板拼接后往往包含 system、user 和 assistant 内容。进行指令跟随 SFT 时，常见做法是只对 assistant 的回答 token 计算损失，对 system/user token 做 mask：

```text
<system>你是助手。</system>       loss mask
<user>解释二叉搜索树。</user>     loss mask
<assistant>二叉搜索树是...</assistant> 计算 loss
```

这样模型优化的是“如何回答输入”，而不是重复背诵提问文本。实践中应以所用框架和数据格式是否支持该设置为准。

## 3. 数据是 SFT 的中心问题

### 常见数据形式

```json
{
  "messages": [
    {"role": "system", "content": "你是一名编程助教。"},
    {"role": "user", "content": "用 Python 实现快速排序。"},
    {"role": "assistant", "content": "def quicksort(arr): ..."}
  ]
}
```

也可使用 `{"prompt": "...", "completion": "..."}` 格式。Hugging Face TRL 的 `SFTTrainer` 官方文档明确支持 language modeling、prompt-completion，以及标准或 conversational 数据格式，并会对 conversational 数据应用 chat template。

### 数据检查清单

1. 是否与部署时使用同一种 chat template 和特殊 token。
2. answer 中是否存在事实错误、危险回答、矛盾风格或泄漏信息。
3. 长样本截断后，是否误删答案而只保留问题。
4. 多轮对话中，模型是否应当学习所有 assistant 回合。
5. 训练集、验证集和人工评测题是否有近重复污染。

### 公开课中要讲清的边界

- SFT 很擅长教输出格式、指令遵循、风格和特定任务解法。
- SFT 可以改变模型对知识的调用方式，也可以让模型记住部分新事实；但把它概括为“大规模可靠注入新知识”的首选并不稳妥。持续预训练或 RAG 常更适合大量新知识和可更新知识。
- “数据质量优于数量”是重要经验，但效果仍取决于任务覆盖、多样性、基础模型能力和评测方法，不能把某个固定样本数量当作普适阈值。

## 4. 与 LoRA 的关系

SFT 说的是**训练目标/数据范式**，LoRA 说的是**更新参数的方式**，两者不是竞争关系：

```text
SFT + 全参数更新：更新模型全部参数
SFT + LoRA：冻结主模型，只训练低秩适配器
SFT + QLoRA：冻结量化后的主模型，只训练 LoRA 适配器
```

因此公开课中说“用 LoRA 做 SFT”是准确的；说“LoRA 取代 SFT”则不准确。

## 5. 训练与评测实践

### 可先掌握的参数

| 项目 | 需要理解的影响 |
| --- | --- |
| 学习率 | 太大容易破坏原能力，太小则适配不足 |
| Epoch / steps | 数据较小时过训会迅速出现 |
| 最大序列长度 | 影响显存、吞吐及长回答覆盖 |
| Packing | 把短样本拼成完整序列，提高训练利用率 |
| Loss mask | 决定模型是否只学习回答部分 |
| PEFT 配置 | 决定训练参数量与显存预算 |

### 不能只看训练 loss

应保留一组未参与训练的 prompts，至少观察：

- 指令遵循率和目标格式正确率；
- 事实性、拒答边界和安全行为；
- 与基础模型相比，通用能力是否下降；
- 回答是否出现模板化、重复、过度冗长等过拟合迹象。

## 6. 建议动手路线

1. 阅读 TRL 的 SFT 数据格式和 loss 说明，理解 chat template 与 mask。
2. 用小模型和几十到几百条自建格式任务样本运行一次 SFT/LoRA。
3. 准备固定测试题，对比 base model 与 SFT model 的格式遵循和回答质量。
4. 将相同 SFT 数据切换为 LoRA 或 QLoRA，比较显存、训练时间与输出差异。
5. 完成 SFT 后，再进入 DPO，直观看到“标准答案学习”与“偏好对比较准”的差异。

## 7. 推荐阅读资料

### 必读

1. Ouyang et al., 2022, *Training language models to follow instructions with human feedback*（InstructGPT）。阅读重点：第一个阶段如何用 demonstrations 做 SFT，以及它如何连接 RM 和 PPO。  
   链接：https://arxiv.org/abs/2203.02155
2. Hugging Face TRL, *SFT Trainer* 官方文档。阅读重点：数据格式、chat template、token-level loss 与可运行示例。  
   链接：https://huggingface.co/docs/trl/main/en/sft_trainer

### 扩展

3. Wei et al., 2021, *Finetuned Language Models Are Zero-Shot Learners*（FLAN）。理解 instruction tuning 如何改善未见任务泛化。  
   链接：https://arxiv.org/abs/2109.01652
4. Taori et al., 2023, *Stanford Alpaca*。理解指令数据生成和低成本复现路径，同时注意其数据质量与许可讨论。  
   链接：https://crfm.stanford.edu/2023/03/13/alpaca.html

## 8. 阅读完成后的自测题

1. SFT 和预训练都用 next-token prediction，它们最关键的数据差异是什么？
2. 为什么 LoRA 可以用于 SFT，却不等于 SFT？
3. 为什么聊天 SFT 中常只计算 assistant token 的 loss？
4. 什么任务更适合通过 RAG 或持续预训练解决，而不是简单扩大 SFT 数据？
5. 若训练 loss 降低而通用问答明显退化，你会优先检查哪些因素？
