# GRPO：组相对策略优化学习笔记

> 适用目标：理解现代推理强化学习中如何通过同题多答案比较，减少 value model 开销并利用可验证奖励。  
> 资料核验日期：2026-05-24

## 1. 一句话理解

GRPO（Group Relative Policy Optimization）对每个问题一次采样一组回答，用这组回答的相对奖励估计 advantage，从而省去 PPO 常用的独立 critic/value model；它尤其适合数学、代码等可以自动检查结果的推理任务。

历史上应准确区分：

- GRPO 作为算法由 DeepSeekMath 论文（2024）提出并使用；
- DeepSeek-R1 技术报告（2025）进一步让 GRPO 因大规模推理强化学习而广为人知。

## 2. 从 PPO 的成本说起

PPO-RLHF 一般需要训练 value/critic 来估计 advantage。在大模型场景中，value model 会增加内存、计算和工程复杂度。

GRPO 的思路是：对于同一个 prompt `q`，生成 `G` 个候选答案：

```text
q -> {o_1, o_2, ..., o_G} -> {r_1, r_2, ..., r_G}
```

若同题的部分回答明显优于其他回答，那么“同组平均水平”就可充当 baseline，而不必让另一个大型网络预测价值。

## 3. Group-relative advantage

最常见的入门表示是将组内奖励标准化：

$$
\hat{A}_i =
\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\varepsilon}
$$

直觉：

- 同题中分数高于组平均的回答获得正 advantage；
- 低于组平均的回答获得负 advantage；
- 所有回答都一样好或一样差时，这一组提供的区分训练信号很弱。

DeepSeekMath 的 GRPO 目标还使用类似 PPO 的 probability ratio clipping，并加入相对于 reference policy 的 KL 正则，以限制 policy 过度漂移。不同代码库的 KL 表达和 loss 配置可能不同，动手时应以对应实现文档为准。

## 4. 奖励从哪里来

GRPO **不等同于**“只有可验证奖励”。它是一种相对策略优化方法，奖励可以来自 reward model 或规则函数。不过，推理任务中可验证奖励特别有价值：

| 任务 | 奖励例子 |
| --- | --- |
| 数学题 | 抽取最终答案并与标准答案比对 |
| 代码题 | 运行单元测试、编译或静态检查 |
| 格式任务 | XML/JSON 格式及字段规则检查 |
| 通用对话 | reward model 或 judge 评分，需警惕偏差 |

自动验证器便宜、一致、可规模化，但也容易带来“只迎合验证规则”的 reward hacking，因此验证器设计本身是训练系统的一部分。

## 5. DeepSeekMath 与 DeepSeek-R1 中的角色

### DeepSeekMath

DeepSeekMath 论文将 GRPO 用于数学推理训练，并把它描述为 PPO 的变体：不训练 critic，而从对同一问题采样的多个输出得分中估计 baseline。

### DeepSeek-R1

DeepSeek-R1 报告了 `R1-Zero`：从基础模型直接进行 RL，在推理任务中观察到自我验证、反思与更长推理行为；之后的 `R1` 路线加入冷启动数据及多阶段训练，以改善可读性、语言混杂等问题。

这给公开课提供了一个重要讲点：

```text
纯 RL 可以探索推理行为，但可用产品模型仍可能需要 SFT/冷启动和后续对齐。
```

## 6. GRPO 与 PPO、DPO 对比

| 维度 | PPO-RLHF | DPO | GRPO |
| --- | --- | --- | --- |
| 数据产生 | 在线生成 | 离线偏好对 | 在线为每个 prompt 生成一组答案 |
| 显式 critic/value | 通常有 | 无 | 无 |
| reference/KL | 通常有 | 有 reference 形式 | 常有 KL 正则/参考策略 |
| 探索新回答 | 有 | 无 | 有，且同题多样本比较 |
| 典型用途 | 通用偏好 RLHF | 离线偏好对齐 | 可验证推理 RL |

一个容易忽略的代价是：GRPO 虽去掉 critic，但每个 prompt 需要生成多条 completion。长推理链的 rollout 成本可能非常高，不能简单等价为“总计算一定更省”。

## 7. 实践中的关键问题

1. **Group size**：越大越有机会形成有效排序，但 rollout 成本也越高。
2. **奖励稀疏**：若一组全部错误，标准化奖励几乎不给出有效方向，可考虑课程中进一步介绍难度调度或动态采样。
3. **长度与格式奖励**：过度奖励格式或长度可能扭曲真正的推理质量。
4. **KL 控制**：防止模型为了通过验证器而严重偏离语言质量。
5. **评测泄漏**：数学/代码训练集与评测题重复会夸大推理提升。

## 8. 建议动手路线

1. 先掌握 PPO 的 ratio clipping 与 DPO 的离线区别。
2. 阅读 DeepSeekMath 的 GRPO 章节，画出“同一题采样多解 -> 打分 -> 组内 advantage -> policy update”流程。
3. 阅读 DeepSeek-R1 中 `R1-Zero` 与冷启动版本的区别，理解纯 RL 与可读输出之间的张力。
4. 使用 TRL `GRPOTrainer` 的最小示例，在有自动判分的小型任务上观察 reward、completion length 与 KL/训练日志。
5. 自己设计一个失败案例：例如只检查最终答案而不检查输出格式或代码副作用，思考模型会如何钻奖励规则的空子。

## 9. 推荐阅读资料

### 必读

1. Shao et al., 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*。GRPO 最重要的算法源头，重点阅读强化学习部分。  
   链接：https://arxiv.org/abs/2402.03300
2. DeepSeek-AI et al., 2025, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*。重点阅读 GRPO、R1-Zero 与 cold-start 多阶段训练。  
   链接：https://arxiv.org/abs/2501.12948
3. DeepSeek-R1 官方 GitHub 仓库，包含模型说明、引用信息和更新入口。  
   链接：https://github.com/deepseek-ai/DeepSeek-R1

### 实作参考

4. Hugging Face TRL, *GRPO Trainer* 官方文档。重点看 reward functions、数据格式和 trainer 参数。  
   链接：https://huggingface.co/docs/trl/main/en/grpo_trainer
5. Yu et al., 2025, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*。作为 GRPO 后进一步处理 clip、采样与长回答问题的扩展阅读。  
   链接：https://arxiv.org/abs/2503.14476

## 10. 阅读完成后的自测题

1. GRPO 去掉了 PPO 流程中的哪一个核心模型组件？
2. 为什么同一个 prompt 要采样多个回答？
3. GRPO 是否只能使用可验证奖励？为什么可验证奖励在推理任务上特别受欢迎？
4. 为什么去掉 critic 不意味着 rollout 训练总成本必然很低？
5. DeepSeek-R1-Zero 与最终 DeepSeek-R1 的训练路线给 SFT 与 RL 的关系带来了什么启示？
