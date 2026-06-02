# DPO：直接偏好优化学习笔记

> 适用目标：理解如何不运行完整强化学习循环，直接用偏好对训练对齐后的语言模型。  
> 资料核验日期：2026-05-24

## 1. 一句话理解

DPO（Direct Preference Optimization）把偏好对齐转化为一个监督式分类损失：给定同一 prompt 的 preferred/chosen 回答和 dispreferred/rejected 回答，它训练 policy 相对参考模型更偏向 chosen，而无需先训练显式 reward model 再执行 PPO。

DPO 由 Rafailov et al. 提出，并发表于 NeurIPS 2023。

## 2. 它解决 RLHF 的哪一步复杂度

典型 RLHF：

```text
SFT -> 收集偏好比较 -> 训练 reward model -> PPO 在线采样优化 policy
```

DPO：

```text
SFT/reference policy -> 偏好比较数据 -> 直接优化 policy
```

DPO 的优点不是“没有偏好数据”，而是从固定偏好对直接学习，不需要显式 reward model 和 RL rollouts。

## 3. 训练数据

每条训练记录包含：

```json
{
  "prompt": "如何解释过拟合？",
  "chosen": "过拟合是模型过度贴合训练数据...",
  "rejected": "过拟合就是训练时间太长，越短越好。"
}
```

高质量 DPO 数据的关键：

- `chosen` 与 `rejected` 确实存在清晰、稳定的偏好差异；
- 差异应与目标对齐维度相关，如正确性、有用性、安全性或表达品质；
- 若 rejected 过于差劲，任务会过易，模型未必学到精细偏好；
- 数据生成模型、标注规则和评测 prompt 需要防止泄漏与风格偏置。

## 4. DPO 损失的直觉与公式

RLHF 的 KL 约束最优策略可与奖励建立解析关系。DPO 借此把隐式奖励差值写成 policy 与 reference policy 的 log probability 差值。其常见损失为：

$$
\mathcal{L}_{DPO} =
-\mathbb{E}_{(x,y_w,y_l)}
\log\sigma\left(
\beta
\left[
\log \frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}
-
\log \frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]
\right)
$$

其中：

- `y_w` 是 chosen，`y_l` 是 rejected；
- `pi_theta` 是正在训练的 policy；
- `pi_ref` 通常是训练开始时冻结的参考模型；
- `beta` 关联对参考策略偏离的约束强度与优化尺度。

最重要的直觉：

```text
不是无条件增加 chosen 的概率，
而是让 chosen 相对于 rejected、且相对于 reference 更占优。
```

## 5. 与 SFT、PPO 的关系

| 维度 | SFT | DPO | PPO-RLHF |
| --- | --- | --- | --- |
| 数据 | 标准回答 | chosen/rejected 对 | 在线生成 + reward |
| 学习目标 | 模仿答案 | 学习相对偏好 | 最大化奖励并受约束探索 |
| 显式 reward model | 否 | 否 | 通常是 |
| 在线探索 | 否 | 否 | 是 |
| 实现复杂度 | 低 | 中低 | 高 |

DPO 一般从一个已有可用回答能力的 policy 开始，例如 SFT 模型。原因很实际：偏好训练主要在已有回答空间中调整选择倾向，而不是从零教会模型对话能力。不过，这属于典型实践而非不可打破的数学前提。

## 6. DPO 的优势与局限

### 优势

- 实现类似监督训练，通常比 PPO-RLHF 简单；
- 不需单独维护 reward model 与 value model；
- 使用离线偏好数据，实验更容易重复；
- 可结合 LoRA/QLoRA 降低训练成本。

### 局限

- 不会主动探索数据集中没有的新解法；
- 结果高度依赖偏好对质量与覆盖面；
- 可能过度优化某些容易标注的表面偏好；
- reference、beta、长度处理与数据混合策略仍会显著影响训练行为。

## 7. 公开课应避免的过度表述

- 可说：DPO 在固定偏好数据的对齐训练中，是比 PPO-RLHF 更简洁的重要选择。
- 不宜说：DPO 在所有后训练任务上“替代”PPO。需要在线探索或环境反馈时，两者承担的任务不同。
- 可说：DPO 不需要显式训练 reward model。
- 不宜说：DPO 没有 reward 假设；其推导本身恰恰利用了隐式奖励建模关系。

## 8. 最小实践路线

Hugging Face TRL 提供 `DPOTrainer`，官方文档覆盖 standard 与 conversational preference 数据格式。建议按以下顺序练习：

1. 从 SFT 后模型和一个小规模偏好数据集开始。
2. 使用 LoRA 运行 DPO，保留 reference policy 与训练日志。
3. 使用固定 prompts 比较 SFT 与 DPO 输出，重点看语气、安全和答题优选，而不只是 loss。
4. 改变 `beta` 与数据难度，观察模型是否变得保守或过度倾向某种风格。
5. 在讲课中将相同偏好数据与 PPO 的“在线生成 + reward”路线作流程比较。

## 9. 推荐阅读资料

### 必读

1. Rafailov et al., 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*（NeurIPS 论文页面，含论文与补充材料）。  
   链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html
2. 论文的 arXiv 版本，便于持续查看修订。  
   链接：https://arxiv.org/abs/2305.18290
3. Hugging Face TRL, *DPO Trainer* 官方文档。重点阅读数据格式、reference model、loss variants 与示例。  
   链接：https://huggingface.co/docs/trl/main/en/dpo_trainer

### 扩展

4. Ethayarajh et al., 2024, *KTO: Model Alignment as Prospect Theoretic Optimization*，了解不要求成对偏好标签的方向。  
   链接：https://arxiv.org/abs/2402.01306
5. Meng et al., 2024, *SimPO: Simple Preference Optimization with a Reference-Free Reward*，了解 reference-free 变体。  
   链接：https://arxiv.org/abs/2405.14734

## 10. 阅读完成后的自测题

1. DPO 为什么仍需要 chosen/rejected 偏好数据？
2. reference model 在 DPO 损失中起到什么作用？
3. 为什么 DPO 不等于 PPO 的在线探索？
4. DPO 与 LoRA 是互斥方法吗？
5. 当偏好数据的 rejected 太差时，训练可能错失什么学习机会？
